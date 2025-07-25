"""
Calculates and schedules weights every SCORING_PERIOD
"""

import asyncio
import os
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from dotenv import load_dotenv

from core.models.utility_models import TaskType
from validator.db.sql.auditing import store_latest_scores_url
from validator.db.sql.submissions_and_scoring import get_aggregate_scores_since, get_aggregate_scores_for_leaderboard_since


load_dotenv(os.getenv("ENV_FILE", ".vali.env"))

import json
from uuid import UUID

from fiber.chain import fetch_nodes
from fiber.chain import weights
from fiber.chain.chain_utils import query_substrate
from fiber.chain.models import Node
from substrateinterface import SubstrateInterface

import validator.core.constants as cts
from core import constants as ccst
from core.constants import BUCKET_NAME
from validator.core.config import Config
from validator.core.config import load_config
from validator.core.models import PeriodScore
from validator.core.models import TaskResults
from validator.db.sql.nodes import get_vali_node_id
from validator.evaluation.scoring import get_period_scores_from_results
from validator.utils.logging import get_logger
from validator.utils.util import save_json_to_temp_file
from validator.utils.util import try_db_connections
from validator.utils.util import upload_file_to_minio


logger = get_logger(__name__)


TIME_PER_BLOCK: int = 500


def get_organic_proportion(task_results: list[TaskResults], task_types: TaskType | set[TaskType], days: int) -> float:
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    if isinstance(task_types, set):
        type_set = task_types
    else:
        type_set = {task_types}

    specific_type_tasks = [
        i for i in task_results
        if i.task.created_at > cutoff_date and i.task.task_type in type_set
    ]

    organic_count = sum(1 for task in specific_type_tasks if task.task.is_organic)
    total_count = len(specific_type_tasks)

    logger.info(f'The total count is {total_count} with organic_count {organic_count} for types {type_set}')
    organic_proportion = organic_count / total_count if total_count > 0 else 0.0
    logger.info(f'THE ORGANIC PROPORTION RIGHT NOW IS {organic_proportion}')
    return organic_proportion


def detect_suspicious_nodes(task_results: list[TaskResults], task_types: TaskType | set[TaskType], days: int = 7) -> set[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    if isinstance(task_types, set):
        type_set = task_types
    else:
        type_set = {task_types}

    period_tasks_organic = [
        task for task in task_results
        if task.task.task_type in type_set
        and task.task.is_organic
        and task.task.created_at > cutoff
    ]

    period_tasks_synth = [
        task for task in task_results
        if task.task.task_type in type_set
        and not task.task.is_organic
        and task.task.created_at > cutoff
    ]

    # Get scores for comparison
    organic_scores = get_period_scores_from_results(
        period_tasks_organic,
        weight_multiplier=1.0  # Temporary multiplier just for comparison
    )

    synth_scores = get_period_scores_from_results(
        period_tasks_synth,
        weight_multiplier=1.0  # Temporary multiplier just for comparison
    )

    # Count synth jobs per hotkey
    synth_job_counts = {}
    for task in period_tasks_synth:
        for node_score in task.node_scores:
            if node_score.hotkey not in synth_job_counts:
                synth_job_counts[node_score.hotkey] = 0
            synth_job_counts[node_score.hotkey] += 1

    suspicious_hotkeys = set()
    synth_by_hotkey = {score.hotkey: score for score in synth_scores}

    for organic_score in organic_scores:
        hotkey = organic_score.hotkey
        synth_job_count = synth_job_counts.get(hotkey, 0)
        
        min_required_synth_jobs = cts.MIN_SYNTH_JOBS_REQUIRED_PER_DAY * days
        if synth_job_count < min_required_synth_jobs:
            logger.info(
                f"Node {hotkey} has only {synth_job_count} synth jobs (requires {min_required_synth_jobs} for {days} days) "
                f"for {type_set} in {days}-day period - flagging as suspicious"
            )
            suspicious_hotkeys.add(hotkey)
        elif hotkey in synth_by_hotkey:
            synth_score = synth_by_hotkey[hotkey]
            if organic_score.average_score > (synth_score.average_score + 0.5 * synth_score.std_score):
                logger.info(
                    f"Node {hotkey} has a much higher organic vs synth score "
                    f"for {type_set} in {days}-day period - flagging as suspicious"
                )
                suspicious_hotkeys.add(hotkey)
        else:
            logger.info(
                f"Node {hotkey} has organic scores but no synth scores "
                f"for {task_types} in {days}-day period - flagging as suspicious"
            )
            suspicious_hotkeys.add(hotkey)

    return suspicious_hotkeys

def get_period_scores_from_task_results(task_results: list[TaskResults]) -> list[PeriodScore]:
    """Process task results into period scores with appropriate filtering and weighting."""
    if not task_results:
        logger.info("There were no results to be scored")
        return []

    task_types = [
        {"type": {TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK}, "weight_key": "INSTRUCT_TEXT_TASK_SCORE_WEIGHT"},
        {"type": TaskType.DPOTASK, "weight_key": "DPO_TASK_SCORE_WEIGHT"},
        {"type": TaskType.IMAGETASK, "weight_key": "IMAGE_TASK_SCORE_WEIGHT"},
        {"type": TaskType.GRPOTASK, "weight_key": "GRPO_TASK_SCORE_WEIGHT"},
    ]

    organic_proportions = {}
    suspicious_hotkeys = {}

    for task_config in task_types:
        task_types_raw = task_config["type"]
        weight_key = task_config["weight_key"]

        task_type_list = task_types_raw if isinstance(task_types_raw, set) else [task_types_raw]

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])
        organic_proportions[task_types_key] = get_organic_proportion(task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7)

        suspicious_hotkeys[task_types_key] = detect_suspicious_nodes(task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7)
        logger.info(f"Found {len(suspicious_hotkeys[task_types_key])} suspicious nodes for {task_types_key}")

    filtered_tasks = {}

    for task_config in task_types:
        task_types_raw = task_config["type"]
        task_type_list = task_types_raw if isinstance(task_types_raw, set) else [task_types_raw]

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

        organic_tasks = []
        synth_tasks = []
        for task_type in task_type_list:
            organic_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=True))
            synth_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=False))

        filtered_tasks[f"{task_types_key}_organic"] = organic_tasks
        filtered_tasks[f"{task_types_key}_synth"] = synth_tasks


    periods = {
        "one_day": {
            "cutoff": datetime.now(timezone.utc) - timedelta(days=1),
            "weight": cts.ONE_DAY_SCORE_WEIGHT
        },
        "three_day": {
            "cutoff": datetime.now(timezone.utc) - timedelta(days=3),
            "weight": cts.THREE_DAY_SCORE_WEIGHT
        },
        "seven_day": {
            "cutoff": datetime.now(timezone.utc) - timedelta(days=7),
            "weight": cts.SEVEN_DAY_SCORE_WEIGHT
        }
    }

    all_period_scores = []

    for period_name, period_config in periods.items():
        cutoff = period_config["cutoff"]
        period_weight = period_config["weight"]

        for task_config in task_types:
            raw_types = task_config["type"]
            task_type_list = raw_types if isinstance(raw_types, set) else [raw_types]

            weight_key = task_config["weight_key"]
            task_weight = getattr(cts, weight_key)

            task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

            organic_proportion = organic_proportions[task_types_key]
            synth_proportion = 1 - organic_proportion

            if organic_proportion > 0:
                period_tasks_organic = filter_tasks_by_period(filtered_tasks[f"{task_types_key}_organic"], cutoff)
                scores_organic = get_period_scores_from_results(
                        period_tasks_organic,
                        weight_multiplier=period_weight * task_weight * organic_proportion
                    )

                for organic_score in scores_organic:
                        if organic_score.hotkey in suspicious_hotkeys[task_types_key]:
                            logger.info(f"Setting {task_types_key} organic score to zero for suspicious node {organic_score.hotkey} in {period_name} period")
                            organic_score.weight_multiplier = 0.0
                
                all_period_scores.extend(scores_organic)

            if synth_proportion > 0:
                period_tasks_synth = filter_tasks_by_period(filtered_tasks[f"{task_types_key}_synth"], cutoff)
                scores_synth = get_period_scores_from_results(
                        period_tasks_synth,
                        weight_multiplier=period_weight * task_weight * synth_proportion
                    )

                all_period_scores.extend(scores_synth)

    return all_period_scores

def filter_tasks_by_period(tasks: list[TaskResults], cutoff_time: datetime) -> list[TaskResults]:
    return [task for task in tasks if task.task.created_at > cutoff_time]


def filter_tasks_by_type(
    tasks: list[TaskResults],
    task_type: TaskType,
    is_organic: bool | None = None
) -> list[TaskResults]:
    if is_organic is None:
        return [task for task in tasks if task.task.task_type == task_type]
    return [
        task for task in tasks
        if task.task.task_type == task_type and task.task.is_organic == is_organic
    ]


async def _get_weights_to_set(config: Config) -> tuple[list[PeriodScore], list[TaskResults]]:
    """
    Retrieve task results from the database and score multiple periods independently.
    This ensures a fairer ramp-up for new miners.

    In the future, as miners become more stable, we aim to encourage long-term stability.
    This means not giving new miners more weight than necessary, while still allowing them
    the potential to reach the top position without being deregistered.

    Period scores are calculated completely independently
    """
    date = datetime.now() - timedelta(days=cts.SCORING_WINDOW)
    task_results: list[TaskResults] = await get_aggregate_scores_since(date, config.psql_db)

    all_period_scores = get_period_scores_from_task_results(task_results)

    return all_period_scores, task_results


async def _get_leaderboard_data(config: Config) -> tuple[list[PeriodScore], list[TaskResults]]:
    """
    Retrieve task results from the database for leaderboard/analytics purposes.
    This includes ALL scores (including zeros) for accurate counting and statistics.
    This is separate from _get_weights_to_set which filters for weight calculations.
    """
    date = datetime.now() - timedelta(days=cts.SCORING_WINDOW)
    task_results: list[TaskResults] = await get_aggregate_scores_for_leaderboard_since(date, config.psql_db)

    all_period_scores = get_period_scores_from_task_results(task_results)

    return all_period_scores, task_results


async def _upload_results_to_s3(config: Config, task_results: list[TaskResults]) -> None:
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            return super().default(obj)

    scores_json = json.dumps([result.model_dump() for result in task_results], indent=2, cls=DateTimeEncoder)

    temp_file, _ = await save_json_to_temp_file(scores_json, "latest_scores", dump_json=False)
    datetime_of_upload = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    presigned_url = await upload_file_to_minio(temp_file, BUCKET_NAME, f"latest_scores_{datetime_of_upload}.json")
    os.remove(temp_file)
    await store_latest_scores_url(presigned_url, config)
    return presigned_url


def get_miner_performance_breakdown(hotkey: str, task_results: list[TaskResults]) -> dict:
    """Get detailed performance breakdown for a specific miner"""

    task_type_configs = [
        {"type": {TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK}, "weight_key": "INSTRUCT_TEXT_TASK_SCORE_WEIGHT"},
        {"type": TaskType.DPOTASK, "weight_key": "DPO_TASK_SCORE_WEIGHT"},
        {"type": TaskType.IMAGETASK, "weight_key": "IMAGE_TASK_SCORE_WEIGHT"},
        {"type": TaskType.GRPOTASK, "weight_key": "GRPO_TASK_SCORE_WEIGHT"},
    ]

    periods = {
        "one_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=1), "weight": cts.ONE_DAY_SCORE_WEIGHT},
        "three_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=3), "weight": cts.THREE_DAY_SCORE_WEIGHT},
        "seven_day": {"cutoff": datetime.now(timezone.utc) - timedelta(days=7), "weight": cts.SEVEN_DAY_SCORE_WEIGHT}
    }

    organic_proportions = {}
    suspicious_hotkeys = {}

    for task_config in task_type_configs:
        raw_types = task_config["type"]
        task_type_list = raw_types if isinstance(raw_types, set) else [raw_types]

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])
        organic_proportions[task_types_key] = get_organic_proportion(task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7)
        suspicious_hotkeys[task_types_key] = detect_suspicious_nodes(task_results, set(task_type_list) if len(task_type_list) > 1 else task_type_list[0], days=7)

    breakdown = {"task_types": {}, "period_totals": {}, "all_scores": []}

    for task_config in task_type_configs:
        raw_types = task_config["type"]
        task_type_list = raw_types if isinstance(raw_types, set) else [raw_types]

        task_weight = getattr(cts, task_config["weight_key"])

        task_types_key = str(sorted(task_type_list)) if len(task_type_list) > 1 else str(task_type_list[0])

        organic_tasks = []
        synthetic_tasks = []
        for task_type in task_type_list:
            organic_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=True))
            synthetic_tasks.extend(filter_tasks_by_type(task_results, task_type, is_organic=False))

        miner_organic_tasks = [tr for tr in organic_tasks if any(ns.hotkey == hotkey for ns in tr.node_scores)]
        miner_synthetic_tasks = [tr for tr in synthetic_tasks if any(ns.hotkey == hotkey for ns in tr.node_scores)]

        type_data = {
            "task_weight": task_weight,
            "organic_proportion": organic_proportions[task_types_key],
            "is_suspicious": hotkey in suspicious_hotkeys[task_types_key],
            "periods": {}
        }

        for period_name, period_config in periods.items():
            period_weight = period_config["weight"]
            cutoff = period_config["cutoff"]

            period_organic = filter_tasks_by_period(miner_organic_tasks, cutoff)
            period_synthetic = filter_tasks_by_period(miner_synthetic_tasks, cutoff)

            organic_mult = period_weight * task_weight * organic_proportions[task_types_key]
            synth_mult = period_weight * task_weight * (1 - organic_proportions[task_types_key])

            organic_scores = get_period_scores_from_results(period_organic, weight_multiplier=organic_mult) if period_organic else []
            synth_scores = get_period_scores_from_results(period_synthetic, weight_multiplier=synth_mult) if period_synthetic else []

            miner_organic_score = next((s for s in organic_scores if s.hotkey == hotkey), None)
            miner_synth_score = next((s for s in synth_scores if s.hotkey == hotkey), None)

            if miner_organic_score and hotkey in suspicious_hotkeys[task_types_key]:
                miner_organic_score.weight_multiplier = 0.0

            type_data["periods"][period_name] = {
                "organic": {
                    "score": miner_organic_score,
                    "task_count": len(period_organic),
                    "weighted_contribution": (miner_organic_score.normalised_score * miner_organic_score.weight_multiplier) if miner_organic_score and miner_organic_score.normalised_score else 0
                },
                "synthetic": {
                    "score": miner_synth_score,
                    "task_count": len(period_synthetic),
                    "weighted_contribution": (miner_synth_score.normalised_score * miner_synth_score.weight_multiplier) if miner_synth_score and miner_synth_score.normalised_score else 0
                }
            }

            breakdown["all_scores"].extend([s for s in [miner_organic_score, miner_synth_score] if s])

        type_data["total_organic_tasks"] = len(miner_organic_tasks)
        type_data["total_synthetic_tasks"] = len(miner_synthetic_tasks)

        breakdown["task_types"][task_types_key] = type_data

        for period_name in periods:
            total = sum(
                breakdown["task_types"][tt]["periods"][period_name]["organic"]["weighted_contribution"] +
                breakdown["task_types"][tt]["periods"][period_name]["synthetic"]["weighted_contribution"]
                for tt in breakdown["task_types"]
            )
            breakdown["period_totals"][period_name] = total

    return breakdown


async def get_node_weights_from_period_scores(
    substrate: SubstrateInterface, netuid: int, node_results: list[PeriodScore]
) -> tuple[list[int], list[float]]:
    """
    Get the node ids and weights from the node results.
    """
    all_nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(substrate, netuid)

    hotkey_to_node_id = {node.hotkey: node.node_id for node in all_nodes}

    all_node_ids = [node.node_id for node in all_nodes]
    all_node_weights = [0.0 for _ in all_nodes]

    logger.info("=== NODE WEIGHT CALCULATIONS ===")
    for node_result in node_results:
        if node_result.normalised_score is not None:
            node_id = hotkey_to_node_id.get(node_result.hotkey)
            if node_id is not None:
                contribution = node_result.normalised_score * node_result.weight_multiplier
                all_node_weights[node_id] = all_node_weights[node_id] + contribution
                logger.info(f"Node ID {node_id} (hotkey: {node_result.hotkey[:8]}...): "
                           f"normalized_score={node_result.normalised_score:.6f}, "
                           f"weight_multiplier={node_result.weight_multiplier:.6f}, "
                           f"contribution={contribution:.6f}, "
                           f"total_weight={all_node_weights[node_id]:.6f}")

    tournament_weight_reduction = 1 - cts.WEIGHT_FOR_TOURN
    for i in range(len(all_node_weights)):
        all_node_weights[i] = all_node_weights[i] * tournament_weight_reduction
    
    if cts.EMISSION_BURN_HOTKEY:
        burn_node_id = hotkey_to_node_id.get(cts.EMISSION_BURN_HOTKEY)
        if burn_node_id is not None:
            all_node_weights[burn_node_id] = cts.WEIGHT_FOR_TOURN
            logger.info(f"Assigned tournament weight {cts.WEIGHT_FOR_TOURN} to burn hotkey {cts.EMISSION_BURN_HOTKEY} (node ID: {burn_node_id})")
        else:
            logger.warning(f"EMISSION_BURN_HOTKEY {cts.EMISSION_BURN_HOTKEY} not found in node list")
    else:
        logger.warning("EMISSION_BURN_HOTKEY not configured")

    logger.info("=== FINAL NODE WEIGHTS ===")
    for node_id, weight in enumerate(all_node_weights):
        if weight > 0:
            logger.info(f"Node ID {node_id}: final_weight={weight:.6f}")

    logger.info(f"Node ids: {all_node_ids}")
    logger.info(f"Node weights: {all_node_weights}")
    logger.info(f"Number of non zero node weights: {sum(1 for weight in all_node_weights if weight != 0)}")
    logger.info(f"Everything going in is {all_node_ids} {all_node_weights} {netuid} {ccst.VERSION_KEY}")
    return all_node_ids, all_node_weights


async def set_weights(config: Config, all_node_ids: list[int], all_node_weights: list[float], validator_node_id: int) -> bool:
    try:
        success = await asyncio.to_thread(
            weights.set_node_weights,
            substrate=config.substrate,
            keypair=config.keypair,
            node_ids=all_node_ids,
            node_weights=all_node_weights,
            netuid=config.netuid,
            version_key=ccst.VERSION_KEY,
            validator_node_id=int(validator_node_id),
            wait_for_inclusion=False,
            wait_for_finalization=False,
            max_attempts=3,
        )
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
        return False

    if success:
        logger.info("Weights set successfully.")

        return True
    else:
        logger.error("Failed to set weights :(")
        return False


async def _get_and_set_weights(config: Config, validator_node_id: int) -> bool:
    node_results, task_results = await _get_weights_to_set(config)
    if node_results is None:
        logger.info("No weights to set. Skipping weight setting.")
        return False
    if len(node_results) == 0:
        logger.info("No nodes to set weights for. Skipping weight setting.")
        return False

    all_node_ids, all_node_weights = await get_node_weights_from_period_scores(config.substrate, config.netuid, node_results)
    logger.info("Weights calculated, about to set...")

    success = await set_weights(config, all_node_ids, all_node_weights, validator_node_id)
    if success and task_results:
        url = await _upload_results_to_s3(config, task_results)
        logger.info(f"Uploaded the scores to s3 for auditing - url: {url}")

    return success


async def _set_metagraph_weights(config: Config) -> None:
    nodes: list[Node] = fetch_nodes.get_nodes_for_netuid(config.substrate, config.netuid)
    node_ids = [node.node_id for node in nodes]
    node_weights = [node.incentive for node in nodes]
    validator_node_id = await get_vali_node_id(config.substrate, config.keypair.ss58_address)
    if validator_node_id is None:
        raise ValueError("Validator node id not found")

    await asyncio.to_thread(
        weights.set_node_weights,
        substrate=config.substrate,
        keypair=config.keypair,
        node_ids=node_ids,
        node_weights=node_weights,
        netuid=config.netuid,
        version_key=ccst.VERSION_KEY,
        validator_node_id=int(validator_node_id),
        wait_for_inclusion=False,
        wait_for_finalization=False,
        max_attempts=3,
    )


#


# To improve: use activity cutoff & The epoch length to set weights at the perfect times
async def set_weights_periodically(config: Config, just_once: bool = False) -> None:
    substrate = config.substrate
    substrate, uid = query_substrate(
        substrate,
        "SubtensorModule",
        "Uids",
        [config.netuid, config.keypair.ss58_address],
        return_value=True,
    )

    if uid is None:
        raise ValueError(f"Can't find hotkey {config.keypair.ss58_address} for our keypair on netuid: {config.netuid}.")

    consecutive_failures = 0
    while True:
        substrate, current_block = query_substrate(substrate, "System", "Number", [], return_value=True)
        substrate, last_updated_value = query_substrate(
            substrate, "SubtensorModule", "LastUpdate", [config.netuid], return_value=False
        )
        updated: int = current_block - last_updated_value[uid]
        substrate, weights_set_rate_limit = query_substrate(
            substrate, "SubtensorModule", "WeightsSetRateLimit", [config.netuid], return_value=True
        )
        logger.info(
            f"My Validator Node ID: {uid}. Last updated {updated} blocks ago. Weights set rate limit: {weights_set_rate_limit}."
        )

        if updated < weights_set_rate_limit:
            logger.info("Sleeping for a bit as we set recently...")
            await asyncio.sleep((weights_set_rate_limit - updated + 1) * 12)
            continue

        if os.getenv("ENV", "prod").lower() == "dev":
            success = await _get_and_set_weights(config, uid)
        else:
            try:
                success = await _get_and_set_weights(config, uid)
            except Exception as e:
                logger.error(f"Failed to set weights with error: {e}")
                logger.exception(e)
                success = False

        if success:
            consecutive_failures = 0
            logger.info("Successfully set weights! Sleeping for 25 blocks before next check...")
            if just_once:
                return
            await asyncio.sleep(12 * 25)
            continue

        consecutive_failures += 1
        if just_once:
            logger.info("Failed to set weights, will try again...")
            await asyncio.sleep(12 * 1)
        else:
            logger.info(f"Failed to set weights {consecutive_failures} times in a row - sleeping for a bit...")
            await asyncio.sleep(12 * 25)  # Try again in 25 blocks

        if consecutive_failures == 1 or updated < 3000:
            continue

        if just_once or config.set_metagraph_weights_with_high_updated_to_not_dereg:
            logger.warning("Setting metagraph weights as our updated value is getting too high!")
            if just_once:
                logger.warning("Please exit if you do not want to do this!!!")
                await asyncio.sleep(4)
            try:
                success = await _set_metagraph_weights(config)
            except Exception as e:
                logger.error(f"Failed to set metagraph weights: {e}")
                success = False

            if just_once:
                return

            if success:
                consecutive_failures = 0
                continue


async def main():
    config = load_config()
    await try_db_connections(config)
    await set_weights_periodically(config)


if __name__ == "__main__":
    asyncio.run(main())
