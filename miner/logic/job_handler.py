import json
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import List

import docker
import pandas as pd
import toml
import yaml
from docker.errors import DockerException
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi

from core import constants as cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import save_config_toml
from core.config.config_handler import update_flash_attention
from core.config.config_handler import update_model_info
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.docker_utils import stream_logs
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DiffusionJob
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TextDatasetType
from core.models.utility_models import TextJob
from miner.utils import download_flux_unet


logger = get_logger(__name__)


@dataclass
class DockerEnvironmentDiffusion:
    huggingface_token: str
    wandb_token: str
    job_id: str
    base_model: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "BASE_MODEL": self.base_model,
        }


@dataclass
class DockerEnvironment:
    huggingface_token: str
    wandb_token: str
    job_id: str
    dataset_type: str
    dataset_filename: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "DATASET_TYPE": self.dataset_type,
            "DATASET_FILENAME": self.dataset_filename,
        }


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    if isinstance(dataset_type, InstructTextDatasetType | DpoDatasetType | ChatTemplateDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH
    elif isinstance(dataset_type, GrpoDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH_GRPO

    logger.info("Loading config template")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions], task_id
            )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id, expected_repo_name)
    config["mlflow_experiment_name"] = dataset

    # H100×4 ULTRA AGGRESSIVE OPTIMIZATIONS FOR RANK 1 PERFORMANCE
    config["lora_r"] = 512  # 32x larger LoRA rank for maximum expressiveness
    config["lora_alpha"] = 1024  # 32x larger alpha for better convergence
    config["lora_dropout"] = 0.05  # Lower dropout for better performance
    config["micro_batch_size"] = 16  # 8x larger batch size for H100×4
    config["gradient_accumulation_steps"] = 1  # Minimal accumulation for larger batches
    config["num_epochs"] = 5  # 5x more epochs for better convergence
    config["learning_rate"] = 0.0008  # 4x higher LR for faster convergence
    config["lr_scheduler"] = "cosine_with_restarts"  # Better scheduler
    config["bf16"] = True  # Enable bf16
    config["fp16"] = False  # Disable fp16
    config["tf32"] = True  # Enable tf32
    config["gradient_checkpointing"] = False  # Disable for H100×4 (enough memory)
    config["warmup_steps"] = 200  # 20x more warmup for stability
    config["weight_decay"] = 0.005  # Reduced regularization for better performance
    config["early_stopping_patience"] = 5  # More patience for better convergence
    config["eval_steps"] = 25  # More frequent evaluation
    config["save_steps"] = 50  # More frequent saving
    config["logging_steps"] = 5  # More frequent logging
    config["flash_attention"] = True  # Enable flash attention
    config["use_peft"] = True  # Enable PEFT for better efficiency
    config["peft_config"] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": 512,
        "lora_alpha": 1024,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }

    return config


def create_reward_funcs_file(reward_funcs: list[str], task_id: str) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(cst.CONFIG_DIR, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names


def _load_and_modify_config_diffusion(job: DiffusionJob) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    if job.model_type == ImageModelType.SDXL:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = job.model
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
        
        # H100×4 ULTRA AGGRESSIVE OPTIMIZATIONS FOR RANK 1 PERFORMANCE
        config["learning_rate"] = 1.2e-4  # 100% higher LR for faster convergence
        config["text_encoder_lr"] = [1.2e-4, 1.2e-4]
        config["unet_lr"] = 1.2e-4
        config["network_dim"] = 512  # 4x larger LoRA rank for maximum expressiveness
        config["network_alpha"] = 512  # 4x larger alpha for better convergence
        config["train_batch_size"] = 4  # 4x larger batch size for H100×4
        config["vae_batch_size"] = 16  # 4x larger VAE batch
        config["max_train_steps"] = 9000  # 100% more training for better convergence
        config["epoch"] = 300  # 100% more epochs for better convergence
        config["min_snr_gamma"] = 12  # Better noise scheduling
        config["huber_c"] = 0.03  # More stable loss
        config["lr_scheduler"] = "cosine_with_restarts"
        config["lr_scheduler_args"] = [5]  # More cycles
        config["lr_scheduler_num_cycles"] = 5  # More cycles for better convergence
        config["max_data_loader_n_workers"] = 16  # More workers for H100×4
        config["max_grad_norm"] = 0.3  # Tighter gradient clipping
        config["scale_weight_norms"] = 16  # Higher scale for better training
        config["save_every_n_epochs"] = 3  # More frequent saving
        config["training_comment"] = "H100×4 ULTRA AGGRESSIVE for rank 1 performance"
        
    elif job.model_type == ImageModelType.FLUX:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = f"{cst.CONTAINER_FLUX_PATH}/flux_unet_{job.model.replace('/', '_')}.safetensors"
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
        
        # H100×4 OPTIMIZATIONS FOR RANK 1 PERFORMANCE
        config["learning_rate"] = 8e-5  # 60% higher LR
        config["text_encoder_lr"] = [8e-5, 8e-5]
        config["unet_lr"] = 8e-5
        config["network_dim"] = 256  # 2x larger LoRA rank
        config["network_alpha"] = 256  # 2x larger alpha
        config["train_batch_size"] = 2  # 2x larger batch size
        config["vae_batch_size"] = 8  # 2x larger VAE batch
        config["max_train_steps"] = 4500  # 50% more training
        config["epoch"] = 150  # 50% more epochs
        config["huber_c"] = 0.05  # More stable loss
        config["lr_scheduler"] = "cosine_with_restarts"
        config["lr_scheduler_args"] = [3]
        config["lr_scheduler_num_cycles"] = 3
        config["max_data_loader_n_workers"] = 8
        config["save_every_n_epochs"] = 15
        config["training_comment"] = "H100-optimized Flux for rank 1 performance"
        
    else:
        logger.error(f"Unknown model type: {job.model_type}")
    return config


def create_job_diffusion(
    job_id: str,
    model: str,
    dataset_zip: str,
    model_type: ImageModelType,
    expected_repo_name: str | None
):
    return DiffusionJob(
        job_id=job_id,
        model=model,
        dataset_zip=dataset_zip,
        model_type=model_type,
        expected_repo_name=expected_repo_name,
    )


def create_job_text(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    expected_repo_name: str | None,
):
    return TextJob(
        job_id=job_id,
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format,
        expected_repo_name=expected_repo_name,
    )


def start_tuning_container_diffusion(job: DiffusionJob):
    logger.info("=" * 80)
    logger.info("STARTING THE DIFFUSION TUNING CONTAINER")
    logger.info("=" * 80)

    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")

    config = _load_and_modify_config_diffusion(job)
    save_config_toml(config, config_path)

    logger.info(config)
    if job.model_type == ImageModelType.FLUX:
        logger.info(f"Downloading flux unet from {job.model}")
        flux_unet_path = download_flux_unet(job.model)

    prepare_dataset(
        training_images_zip_path=job.dataset_zip,
        training_images_repeat=(
            cst.DIFFUSION_SDXL_REPEATS if job.model_type == ImageModelType.SDXL
            else cst.DIFFUSION_FLUX_REPEATS
        ),
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )

    docker_env = DockerEnvironmentDiffusion(
        huggingface_token=cst.HUGGINGFACE_TOKEN, wandb_token=cst.WANDB_TOKEN, job_id=job.job_id, base_model=job.model_type.value
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/dataset/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/dataset/outputs",
                "mode": "rw",
            },
            os.path.abspath(cst.DIFFUSION_DATASET_DIR): {
                "bind": "/dataset/images",
                "mode": "rw",
            },
        }

        if job.model_type == ImageModelType.FLUX:
            volume_bindings[os.path.dirname(flux_unet_path)] =  {
                "bind": cst.CONTAINER_FLUX_PATH,
                "mode": "rw",
            }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE_DIFFUSION,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function
        stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        if "container" in locals():
            container.remove(force=True)

        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"

        if os.path.exists(train_data_path):
            shutil.rmtree(train_data_path)


def _dpo_format_prompt(row, format_str):
    result = format_str
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_chosen(row, format_str):
    result = format_str
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_CHOSEN]):
        result = result.replace("{chosen}", str(row[cst.DPO_DEFAULT_FIELD_CHOSEN]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_rejected(row, format_str):
    result = format_str
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_REJECTED]):
        result = result.replace("{rejected}", str(row[cst.DPO_DEFAULT_FIELD_REJECTED]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DpoDatasetType, apply_formatting: bool = False):
    """
    Transform a DPO JSON dataset file to match axolotl's `chatml.argilla` expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DpoDatasetType with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    column_mapping = {
        dataset_type.field_prompt: cst.DPO_DEFAULT_FIELD_PROMPT,
        dataset_type.field_system: cst.DPO_DEFAULT_FIELD_SYSTEM,
        dataset_type.field_chosen: cst.DPO_DEFAULT_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.DPO_DEFAULT_FIELD_REJECTED
    }
    df = df.rename(columns=column_mapping)

    if apply_formatting:
        if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
            format_str = dataset_type.prompt_format
            df[cst.DPO_DEFAULT_FIELD_PROMPT] = df.apply(lambda row: _dpo_format_prompt(row, format_str), axis=1)
        if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
            format_str = dataset_type.chosen_format
            df[cst.DPO_DEFAULT_FIELD_CHOSEN] = df.apply(lambda row: _dpo_format_chosen(row, format_str), axis=1)
        if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
            format_str = dataset_type.rejected_format
            df[cst.DPO_DEFAULT_FIELD_REJECTED] = df.apply(lambda row: _dpo_format_rejected(row, format_str), axis=1)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def _adapt_columns_for_grpo_dataset(dataset_path: str, dataset_type: GrpoDatasetType):
    """
    Transform a GRPO JSON dataset file to match axolotl's `prompt` expected column name.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: GrpoDatasetType with field mappings
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={dataset_type.field_prompt: cst.GRPO_DEFAULT_FIELD_PROMPT})
    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def _create_docker_entrypoint(job, gpu_ids: List[int] = None):
    setup_commands = """
    echo 'Preparing data...' && \\
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \\
    echo "Attempting to log in to Hugging Face" && \\
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential; \\
    else \\
    echo "HUGGINGFACE_TOKEN is not set. Skipping Hugging Face login."; \\
    fi && \\
    if [ -n "$WANDB_TOKEN" ]; then \\
    echo "Attempting to log in to W&B" && \\
    wandb login "$WANDB_TOKEN"; \\
    else \\
    echo "WANDB_TOKEN is not set. Skipping W&B login."; \\
    fi && \\
    if [ "$DATASET_TYPE" != "hf" ] && [ -f "/workspace/input_data/${DATASET_FILENAME}" ]; then \\
    cp /workspace/input_data/${DATASET_FILENAME} /workspace/axolotl/${DATASET_FILENAME}; \\
    fi"""

    if isinstance(job.dataset_type, GrpoDatasetType):
        reward_file = f"rewards_{job.job_id}.py"
        grpo_command = f"""
    echo "Moving specific reward function file to src directory..." && \\
    cp ${{CONFIG_DIR}}/{reward_file} /workspace/axolotl/src/"""
        setup_commands += " && \\" + grpo_command

    # H100-optimized accelerate launch for multi-GPU
    if gpu_ids and len(gpu_ids) > 1:
        num_gpus = len(gpu_ids)
        if num_gpus >= 4:
            # H100-specific optimizations for 4-GPU setup
            training_command = f"""
    echo 'Starting H100-optimized multi-GPU training with {num_gpus} GPUs' && \\
    export CUDA_LAUNCH_BLOCKING=0 && \\
    export NCCL_IB_DISABLE=0 && \\
    export NCCL_P2P_DISABLE=0 && \\
    export NCCL_DEBUG=INFO && \\
    accelerate launch --multi_gpu --num_processes {num_gpus} --num_machines 1 --machine_rank 0 --main_process_port 29500 --mixed_precision bf16 -m axolotl.cli.train ${{CONFIG_DIR}}/${{JOB_ID}}.yml
    """
        else:
            training_command = f"""
    echo 'Starting multi-GPU training command with {num_gpus} GPUs' && \\
    accelerate launch --multi_gpu --num_processes {num_gpus} --num_machines 1 --machine_rank 0 --main_process_port 29500 -m axolotl.cli.train ${{CONFIG_DIR}}/${{JOB_ID}}.yml
    """
    else:
        training_command = """
    echo 'Starting single-GPU training command' && \\
    accelerate launch -m axolotl.cli.train ${CONFIG_DIR}/${JOB_ID}.yml
    """

    return setup_commands + " && \\" + training_command

def _adapt_columns_for_dataset(job: TextJob):
    """
    Adapt column names in the dataset based on job type.
    Only processes JSON files that require column name adaptation.
    """
    if job.file_format != FileFormat.JSON:
        return

    if isinstance(job.dataset_type, DpoDatasetType):
        _adapt_columns_for_dpo_dataset(job.dataset, job.dataset_type, True)
    elif isinstance(job.dataset_type, GrpoDatasetType):
        _adapt_columns_for_grpo_dataset(job.dataset, job.dataset_type)


def start_tuning_container(job: TextJob, gpu_ids: List[int] = None):
    logger.info("=" * 80)
    logger.info("STARTING THE TUNING CONTAINER WITH MULTI-GPU SUPPORT")
    logger.info("=" * 80)

    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    config = _load_and_modify_config(
        job.dataset,
        job.model,
        job.dataset_type,
        job.file_format,
        job.job_id,
        job.expected_repo_name,
    )

    # Optimize config for H100 multi-GPU training
    if gpu_ids and len(gpu_ids) > 1:
        config = _optimize_config_for_multi_gpu(config, len(gpu_ids), job.model)

    save_config(config, config_path)
    logger.info(f"H100-optimized training config for {len(gpu_ids) if gpu_ids else 1} GPUs")

    docker_entrypoint = _create_docker_entrypoint(job, gpu_ids)

    logger.info(os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "")

    docker_env = DockerEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id,
        dataset_type=cst.CUSTOM_DATASET_TYPE,
        dataset_filename=os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "",
    ).to_dict()

    # Add H100 multi-GPU environment variables
    if gpu_ids:
        docker_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        docker_env["NVIDIA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        docker_env["NUM_GPUS"] = str(len(gpu_ids))

        # H100-specific environment variables
        if len(gpu_ids) >= 4:
            docker_env["CUDA_LAUNCH_BLOCKING"] = "0"
            docker_env["NCCL_IB_DISABLE"] = "0"
            docker_env["NCCL_P2P_DISABLE"] = "0"
            docker_env["NCCL_DEBUG"] = "INFO"
            docker_env["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
            docker_env["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/workspace/axolotl/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/workspace/axolotl/outputs",
                "mode": "rw",
            },
        }

        if job.file_format != FileFormat.HF:
            dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
            logger.info(dataset_dir)
            volume_bindings[dataset_dir] = {
                "bind": "/workspace/input_data",
                "mode": "ro",
            }

        _adapt_columns_for_dataset(job)

        # Configure device requests for H100 multi-GPU
        device_requests = []
        if gpu_ids:
            for gpu_id in gpu_ids:
                device_requests.append(docker.types.DeviceRequest(
                    count=-1,
                    capabilities=[['gpu']],
                    options={'device': f'{gpu_id}'}
                ))
        else:
            device_requests = [docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])]

        # Optimize for H100×4 ULTRA AGGRESSIVE setup
        if gpu_ids and len(gpu_ids) >= 4:
            container_kwargs = {
                "image": cst.MINER_DOCKER_IMAGE,
                "environment": {
                    **docker_env,
                    "NCCL_IB_DISABLE": "0",
                    "NCCL_P2P_DISABLE": "0", 
                    "NCCL_DEBUG": "INFO",
                    "NCCL_SOCKET_IFNAME": "^docker0,lo",
                    "NCCL_ASYNC_ERROR_HANDLING": "1",
                    "CUDA_LAUNCH_BLOCKING": "0",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                    "NCCL_NET_GDR_LEVEL": "5",  # Enable GPU Direct RDMA
                    "NCCL_BUFFSIZE": "8388608",  # 8MB buffer size
                    "NCCL_NTHREADS": "8",  # More threads
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
                    "NVIDIA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
                    "NUM_GPUS": str(len(gpu_ids)),
                },
                "volumes": volume_bindings,
                "runtime": "nvidia",
                "device_requests": device_requests,
                "detach": True,
                "tty": True,
                "command": ["/bin/bash", "-c", docker_entrypoint]
            }
        else:
            container_kwargs = {
                "image": cst.MINER_DOCKER_IMAGE,
                "environment": docker_env,
                "volumes": volume_bindings,
                "runtime": "nvidia",
                "device_requests": device_requests,
                "detach": True,
                "tty": True,
                "command": ["/bin/bash", "-c", docker_entrypoint]
            }

        # Optimize for H100 4-GPU setup
        if gpu_ids and len(gpu_ids) >= 4:
            container_kwargs.update({
                "shm_size": "128g",  # Much larger shared memory for H100×4
                "mem_limit": "400g",  # 100GB per H100 * 4 (pushing limits)
                "cpuset_cpus": "0-63",  # Use all available CPUs
                "ulimits": [
                    docker.types.Ulimit(name="memlock", soft=-1, hard=-1),
                    docker.types.Ulimit(name="stack", soft=134217728, hard=134217728),  # 128MB stack
                    docker.types.Ulimit(name="nofile", soft=65536, hard=65536),  # More file descriptors
                ]
            })
        else:
            container_kwargs["shm_size"] = "64g"

        container = docker_client.containers.run(**container_kwargs)

        last_logs = stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        # Waiting for axolotl to fix the issue
        if "TypeError: DPOTrainer.create_model_card() got an unexpected keyword argument 'dataset_tags'" in last_logs:
            logger.warning("This is probably just an axolotl issue only affecting HF repo model card, continuing...")
        else:
            logger.error(f"Error processing job: {str(e)}")
            raise

    finally:
        repo = config.get("hub_model_id", None)
        if repo:
            try:
                hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
                # Check if repository exists before trying to update it
                try:
                    hf_api.model_info(repo_id=repo)
                    hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
                    logger.info(f"Successfully made repository {repo} public")
                except Exception as repo_error:
                    logger.warning(f"Repository {repo} not found or not accessible: {repo_error}")
            except Exception as hf_error:
                logger.warning(f"Failed to update repository visibility: {hf_error}")

        if "container" in locals():
            try:
                container.remove(force=True)
                logger.info("Container removed")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")

def _optimize_config_for_multi_gpu(config: dict, num_gpus: int, model_name: str) -> dict:
    """ULTRA AGGRESSIVE H100×4 OPTIMIZATIONS FOR RANK 1 PERFORMANCE"""

    # Estimate model size
    model_size = _estimate_model_size_from_name(model_name)

    # H100×4 ULTRA AGGRESSIVE OPTIMIZATIONS FOR RANK 1
    if num_gpus >= 4:
        # H100 has 80GB VRAM per GPU, allowing much larger batch sizes
        config["micro_batch_size"] = max(32, config.get("micro_batch_size", 2) * 8)  # 8x increase for H100×4
        config["gradient_accumulation_steps"] = 1  # Minimal accumulation for larger batches

        # H100 supports higher precision and better memory efficiency
        config["bf16"] = True
        config["fp16"] = False  # Prefer bf16 on H100
        config["tf32"] = True   # Enable TensorFloat-32 for better performance

        # Optimize for H100's tensor cores
        config["dataloader_pin_memory"] = True
        config["dataloader_num_workers"] = 16  # More workers for H100×4

    elif num_gpus >= 2:
        config["micro_batch_size"] = max(16, config.get("micro_batch_size", 2) * 4)
        config["gradient_accumulation_steps"] = 1
        config["bf16"] = True
        config["tf32"] = True

    # ULTRA AGGRESSIVE LoRA SETTINGS FOR RANK 1 PERFORMANCE
    if model_size > 70:  # For 70B+ models on H100×4
        config["lora_r"] = min(1024, config.get("lora_r", 8) * 8)  # Much larger LoRA rank
        config["lora_alpha"] = min(2048, config.get("lora_alpha", 16) * 8)
        config["lora_dropout"] = 0.02  # Lower dropout for better performance
    elif model_size > 35:
        config["lora_r"] = min(512, config.get("lora_r", 8) * 4)
        config["lora_alpha"] = min(1024, config.get("lora_alpha", 16) * 4)
        config["lora_dropout"] = 0.03
    elif model_size > 13:
        config["lora_r"] = min(256, config.get("lora_r", 8) * 3)
        config["lora_alpha"] = min(512, config.get("lora_alpha", 16) * 3)
        config["lora_dropout"] = 0.04
    else:
        # For smaller models, still use enhanced settings for H100×4
        config["lora_r"] = min(256, config.get("lora_r", 8) * 3)
        config["lora_alpha"] = min(512, config.get("lora_alpha", 16) * 3)
        config["lora_dropout"] = 0.05

    # ULTRA AGGRESSIVE OPTIMIZATIONS FOR RANK 1
    config["flash_attention"] = True
    config["gradient_checkpointing"] = False  # Disable for H100×4 (enough memory)
    config["warmup_steps"] = max(300, config.get("warmup_steps", 100))  # More warmup for H100×4

    # ULTRA AGGRESSIVE LEARNING RATE FOR RANK 1 PERFORMANCE
    base_lr = config.get("learning_rate", 0.0002)
    if num_gpus >= 4:
        config["learning_rate"] = base_lr * 4.0  # 4x higher LR for H100×4
        config["lr_scheduler"] = "cosine_with_restarts"  # Better scheduler for H100×4
        config["lr_scheduler_kwargs"] = {"num_cycles": 5}  # More cycles for better convergence
        config["weight_decay"] = 0.002  # Reduced regularization for rank 1
    elif num_gpus >= 2:
        config["learning_rate"] = base_lr * 2.5
        config["lr_scheduler"] = "cosine"
        config["weight_decay"] = 0.003

    # ULTRA AGGRESSIVE FSDP FOR H100×4
    if model_size > 70 and num_gpus >= 4:
        config["fsdp"] = "full_shard auto_wrap"
        config["fsdp_config"] = {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",  # Optimized for Llama models
            "fsdp_use_orig_params": True,
            "fsdp_forward_prefetch": True,  # Enable forward prefetch
            "fsdp_limit_all_gathers": True,  # Limit all gathers
        }
    elif model_size > 35 and num_gpus >= 2:
        config["fsdp"] = "auto_wrap"
        config["fsdp_config"] = {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch": "BACKWARD_PRE",
        }

    # H100×4 ULTRA AGGRESSIVE MEMORY OPTIMIZATIONS
    config["max_memory_MB"] = 75000  # 75GB per H100 GPU (pushing limits)
    config["gradient_clipping"] = 0.5  # Tighter gradient clipping for rank 1
    config["max_grad_norm"] = 0.3  # Tighter gradient clipping for rank 1

    # OPTIMIZE FOR H100×4'S TENSOR CORES
    config["use_cpu_offload"] = False  # H100×4 has enough memory
    config["use_8bit_optimizer"] = False  # Not needed with H100×4's memory
    config["use_4bit_optimizer"] = False  # Not needed with H100×4's memory
    
    # ULTRA AGGRESSIVE TRAINING PARAMETERS FOR RANK 1
    config["early_stopping_patience"] = 8  # More patience for better convergence
    config["eval_steps"] = 20  # More frequent evaluation
    config["save_steps"] = 40  # More frequent saving
    config["logging_steps"] = 5  # More frequent logging
    config["num_epochs"] = max(8, config.get("num_epochs", 3))  # More epochs for better convergence

    # ULTRA AGGRESSIVE PEFT CONFIG FOR RANK 1
    config["use_peft"] = True
    config["peft_config"] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": config.get("lora_r", 256),
        "lora_alpha": config.get("lora_alpha", 512),
        "lora_dropout": config.get("lora_dropout", 0.05),
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "modules_to_save": None,
    }

    return config


def _estimate_model_size_from_name(model_name: str) -> int:
    """Estimate model size from model name"""
    model_name_lower = model_name.lower()
    if "70b" in model_name_lower:
        return 70
    elif "34b" in model_name_lower or "33b" in model_name_lower:
        return 34
    elif "13b" in model_name_lower:
        return 13
    elif "7b" in model_name_lower:
        return 7
    elif "3b" in model_name_lower:
        return 3
    return 7  # Default assumption
