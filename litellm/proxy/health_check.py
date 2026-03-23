# This file runs a health check for the LLM, used on litellm/proxy

import asyncio
import logging
import random
import sys
import threading
import time
from typing import List, Literal, Optional

import litellm

logger = logging.getLogger(__name__)
from litellm.constants import DEFAULT_HEALTH_CHECK_PROMPT, HEALTH_CHECK_TIMEOUT_SECONDS

ILLEGAL_DISPLAY_PARAMS = [
    "messages",
    "api_key",
    "prompt",
    "input",
    "vertex_credentials",
    "aws_access_key_id",
    "aws_secret_access_key",
]

MINIMAL_DISPLAY_PARAMS = ["model", "mode_error"]
HEALTH_CHECK_STRATEGY_NON_STREAM_ONLY = "non_stream_only"
HEALTH_CHECK_STRATEGY_STREAM_ONLY = "stream_only"
HEALTH_CHECK_STRATEGY_NON_STREAM_THEN_STREAM = "non_stream_then_stream"
SUPPORTED_HEALTH_CHECK_STRATEGIES = {
    HEALTH_CHECK_STRATEGY_NON_STREAM_ONLY,
    HEALTH_CHECK_STRATEGY_STREAM_ONLY,
    HEALTH_CHECK_STRATEGY_NON_STREAM_THEN_STREAM,
}
HealthCheckMode = Optional[
    Literal[
        "chat",
        "completion",
        "embedding",
        "audio_speech",
        "audio_transcription",
        "image_generation",
        "video_generation",
        "batch",
        "rerank",
        "realtime",
        "responses",
        "ocr",
    ]
]


def _get_process_rss_mb() -> Optional[float]:
    """
    Get process RSS memory in MB.
    On Linux, ru_maxrss is in KB. On macOS, ru_maxrss is in bytes.
    """
    try:
        import resource

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(ru_maxrss) / (1024 * 1024)
        return float(ru_maxrss) / 1024
    except Exception:
        return None


def _rss_mb_for_log() -> str:
    rss_mb = _get_process_rss_mb()
    if rss_mb is None:
        return "unknown"
    return f"{rss_mb:.2f}"


def _get_random_llm_message():
    """
    Get a random message from the LLM.
    """
    messages = ["Hey how's it going?", "What's 1 + 1?"]

    return [{"role": "user", "content": random.choice(messages)}]


def _clean_endpoint_data(endpoint_data: dict, details: Optional[bool] = True):
    """
    Clean the endpoint data for display to users.
    """
    endpoint_data.pop("litellm_logging_obj", None)
    return (
        {k: v for k, v in endpoint_data.items() if k not in ILLEGAL_DISPLAY_PARAMS}
        if details is not False
        else {k: v for k, v in endpoint_data.items() if k in MINIMAL_DISPLAY_PARAMS}
    )


def filter_deployments_by_id(
    model_list: List,
) -> List:
    seen_ids = set()
    filtered_deployments = []

    for deployment in model_list:
        _model_info = deployment.get("model_info") or {}
        _id = _model_info.get("id") or None
        if _id is None:
            continue

        if _id not in seen_ids:
            seen_ids.add(_id)
            filtered_deployments.append(deployment)

    return filtered_deployments


async def run_with_timeout(task, timeout):
    try:
        return await asyncio.wait_for(task, timeout)
    except asyncio.TimeoutError:
        # `asyncio.wait_for()` already cancels only the awaited task on timeout.
        # Do not cancel unrelated sibling health check tasks.
        return {"error": "Timeout exceeded"}


def _resolve_health_check_strategy(model_info: dict) -> str:
    strategy = model_info.get("health_check_strategy")
    if strategy in SUPPORTED_HEALTH_CHECK_STRATEGIES:
        return strategy

    health_check_stream = model_info.get("health_check_stream", None)
    if health_check_stream is True:
        return HEALTH_CHECK_STRATEGY_STREAM_ONLY
    if health_check_stream is False:
        return HEALTH_CHECK_STRATEGY_NON_STREAM_ONLY

    return HEALTH_CHECK_STRATEGY_NON_STREAM_THEN_STREAM


def _resolve_health_check_mode(model_info: dict) -> HealthCheckMode:
    mode = model_info.get("mode", None)
    if mode == "chat":
        return "chat"
    if mode == "completion":
        return "completion"
    if mode == "embedding":
        return "embedding"
    if mode == "audio_speech":
        return "audio_speech"
    if mode == "audio_transcription":
        return "audio_transcription"
    if mode == "image_generation":
        return "image_generation"
    if mode == "video_generation":
        return "video_generation"
    if mode == "batch":
        return "batch"
    if mode == "rerank":
        return "rerank"
    if mode == "realtime":
        return "realtime"
    if mode == "responses":
        return "responses"
    if mode == "ocr":
        return "ocr"
    return None


def _get_health_check_attempts(strategy: str) -> list[tuple[str, bool]]:
    if strategy == HEALTH_CHECK_STRATEGY_STREAM_ONLY:
        return [("stream", True)]
    if strategy == HEALTH_CHECK_STRATEGY_NON_STREAM_ONLY:
        return [("non_stream", False)]
    return [("non_stream", False), ("stream", True)]


def _is_successful_health_check_result(result: object) -> bool:
    return isinstance(result, dict) and "error" not in result


def _add_health_check_metadata(
    result: dict,
    strategy: str,
    attempt_mode: Optional[str],
    attempted_modes: list[str],
) -> dict:
    return {
        **result,
        "health_check_strategy": strategy,
        "health_check_attempted_modes": attempted_modes,
        "health_check_result_mode": attempt_mode,
    }


async def _run_single_health_check_attempt(
    litellm_params: dict,
    mode: HealthCheckMode,
    stream: bool,
    timeout: float,
) -> dict:
    try:
        return await run_with_timeout(
            litellm.ahealth_check(
                litellm_params,
                mode=mode,
                prompt=DEFAULT_HEALTH_CHECK_PROMPT,
                input=["test from litellm"],
                stream=stream,
            ),
            timeout,
        )
    except Exception as exc:
        return {"error": str(exc), "mode_error": str(exc)}


async def _run_model_health_check(model: dict):
    litellm_params = model["litellm_params"]
    model_info = model.get("model_info", {})
    mode = _resolve_health_check_mode(model_info)
    litellm_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    timeout = model_info.get("health_check_timeout") or HEALTH_CHECK_TIMEOUT_SECONDS

    strategy = _resolve_health_check_strategy(model_info)
    attempts = _get_health_check_attempts(strategy)
    attempted_modes: list[str] = []
    last_result: dict = {"error": "Health check did not run"}
    start_time = time.monotonic()

    for attempt_mode, stream in attempts:
        attempted_modes.append(attempt_mode)
        elapsed = time.monotonic() - start_time
        remaining_timeout = timeout - elapsed

        if remaining_timeout <= 0:
            last_result = {
                "error": "Timeout exceeded",
                "mode_error": "Timeout exceeded before fallback completed",
            }
            break

        result = await _run_single_health_check_attempt(
            litellm_params=litellm_params,
            mode=mode,
            stream=stream,
            timeout=remaining_timeout,
        )

        if _is_successful_health_check_result(result):
            return _add_health_check_metadata(
                result=result,
                strategy=strategy,
                attempt_mode=attempt_mode,
                attempted_modes=attempted_modes,
            )

        last_result = result

    return _add_health_check_metadata(
        result=last_result,
        strategy=strategy,
        attempt_mode=None,
        attempted_modes=attempted_modes,
    )


async def _run_health_checks_with_bounded_concurrency(
    models: list, concurrency_limit: int
) -> tuple[list, int]:
    """
    Run health checks with at most `concurrency_limit` active tasks.
    Preserves result ordering to match `models`.
    """
    results: list = [None] * len(models)
    tasks_to_index: dict[asyncio.Task, int] = {}
    model_iter = iter(enumerate(models))
    peak_in_flight = 0

    def _schedule_next() -> bool:
        nonlocal peak_in_flight
        try:
            idx, next_model = next(model_iter)
        except StopIteration:
            return False
        task = asyncio.create_task(_run_model_health_check(next_model))
        tasks_to_index[task] = idx
        peak_in_flight = max(peak_in_flight, len(tasks_to_index))
        return True

    for _ in range(min(concurrency_limit, len(models))):
        _schedule_next()

    while tasks_to_index:
        done, _ = await asyncio.wait(
            set(tasks_to_index.keys()),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            idx = tasks_to_index.pop(task)
            try:
                results[idx] = task.result()
            except Exception as e:
                results[idx] = e
            _schedule_next()

    return results, peak_in_flight


async def _perform_health_check(
    model_list: list,
    details: Optional[bool] = True,
    max_concurrency: Optional[int] = None,
    instrumentation_context: Optional[dict] = None,
):
    """
    Perform a health check for each model in the list.

    max_concurrency: Optional limit on concurrent health check requests.
    """

    instrumentation_context = instrumentation_context or {}
    instrumentation_enabled = bool(instrumentation_context.get("enabled", False))
    cycle_id = instrumentation_context.get("cycle_id", "unknown")
    source = instrumentation_context.get("source", "unknown")

    dispatch_mode = "unbounded"
    peak_in_flight = 0
    if isinstance(max_concurrency, int) and max_concurrency > 0:
        dispatch_mode = "bounded"
        results, peak_in_flight = await _run_health_checks_with_bounded_concurrency(
            model_list, max_concurrency
        )
    else:
        tasks = [
            asyncio.create_task(_run_model_health_check(model)) for model in model_list
        ]
        peak_in_flight = len(tasks)
        results = await asyncio.gather(*tasks, return_exceptions=True)

    if instrumentation_enabled:
        logger.debug(
            "health_check_dispatch_summary source=%s cycle_id=%s mode=%s model_count=%d max_concurrency=%s peak_in_flight=%d thread_count=%d rss_mb=%s",
            source,
            cycle_id,
            dispatch_mode,
            len(model_list),
            max_concurrency,
            peak_in_flight,
            threading.active_count(),
            _rss_mb_for_log(),
        )

    healthy_endpoints = []
    unhealthy_endpoints = []

    for is_healthy, model in zip(results, model_list):
        litellm_params = model["litellm_params"]

        if isinstance(is_healthy, dict) and "error" not in is_healthy:
            healthy_endpoints.append(
                _clean_endpoint_data({**litellm_params, **is_healthy}, details)
            )
        elif isinstance(is_healthy, dict):
            unhealthy_endpoints.append(
                _clean_endpoint_data({**litellm_params, **is_healthy}, details)
            )
        else:
            unhealthy_endpoints.append(_clean_endpoint_data(litellm_params, details))

    return healthy_endpoints, unhealthy_endpoints


def _update_litellm_params_for_health_check(
    model_info: dict, litellm_params: dict
) -> dict:
    """
    Update the litellm params for health check.

    - gets a short `messages` param for health check
    - updates the `model` param with the `health_check_model` if it exists Doc: https://docs.litellm.ai/docs/proxy/health#wildcard-routes
    - updates the `voice` param with the `health_check_voice` for `audio_speech` mode if it exists Doc: https://docs.litellm.ai/docs/proxy/health#text-to-speech-models
    - for Bedrock models with region routing (bedrock/region/model), strips the litellm routing prefix but preserves the model ID
    """
    litellm_params["messages"] = _get_random_llm_message()
    _health_check_max_tokens = model_info.get("health_check_max_tokens", None)
    if _health_check_max_tokens is not None:
        litellm_params["max_tokens"] = _health_check_max_tokens
    elif "*" not in (
        model_info.get("health_check_model") or litellm_params.get("model") or ""
    ):
        litellm_params["max_tokens"] = 1

    _health_check_model = model_info.get("health_check_model", None)
    if _health_check_model is not None:
        litellm_params["model"] = _health_check_model
    if model_info.get("mode", None) == "audio_speech":
        litellm_params["voice"] = model_info.get("health_check_voice", "alloy")

    # Handle Bedrock region routing format: bedrock/region/model
    # This is needed because health checks bypass get_llm_provider() for the model param
    # Issue #15807: Without this, health checks send "region/model" as the model ID to AWS
    # which causes: "bedrock-runtime.../model/us-west-2/mistral.../invoke" (region in model ID)
    #
    # However, we must preserve cross-region inference profile prefixes like "us.", "eu.", etc.
    # Issue: Stripping these breaks AWS requirement for inference profile IDs
    #
    # Must also preserve route prefixes (converse/, invoke/) and handlers (llama/, deepseek_r1/, etc.)
    if litellm_params["model"].startswith("bedrock/"):
        from litellm.llms.bedrock.common_utils import BedrockModelInfo

        model = litellm_params["model"]
        # Strip only the bedrock/ prefix (preserve routes like converse/, invoke/)
        if model.startswith("bedrock/"):
            model = model[8:]  # len("bedrock/") = 8

        # Now check for region routing and strip it if present
        # Need to handle formats like:
        # - "us-west-2/model" → "model"
        # - "converse/us-west-2/model" → "converse/model"
        # - "llama/arn:..." → "llama/arn:..." (preserve handler)
        #
        # Strategy: Check each path segment, remove regions, preserve everything else
        parts = model.split("/")
        filtered_parts = []

        for part in parts:
            # Skip AWS regions, keep everything else
            if part not in BedrockModelInfo.all_global_regions:
                filtered_parts.append(part)

        model = "/".join(filtered_parts)
        litellm_params["model"] = model

    return litellm_params


async def perform_health_check(
    model_list: list,
    model: Optional[str] = None,
    cli_model: Optional[str] = None,
    details: Optional[bool] = True,
    model_id: Optional[str] = None,
    max_concurrency: Optional[int] = None,
    instrumentation_context: Optional[dict] = None,
):
    """
    Perform a health check on the system.

    When model_id is provided, only the deployment with that id is checked
    (so models that share the same name but have different ids are checked separately).
    When model (name) is provided, all deployments matching that name are checked.

    Returns:
        (bool): True if the health check passes, False otherwise.
    """
    instrumentation_context = instrumentation_context or {}
    instrumentation_enabled = bool(instrumentation_context.get("enabled", False))
    cycle_id = instrumentation_context.get("cycle_id", "unknown")
    source = instrumentation_context.get("source", "unknown")

    if not model_list:
        if cli_model:
            model_list = [
                {"model_name": cli_model, "litellm_params": {"model": cli_model}}
            ]
        else:
            if instrumentation_enabled:
                logger.debug(
                    "health_check_cycle_skipped source=%s cycle_id=%s reason=no_models",
                    source,
                    cycle_id,
                )
            return [], []

    cycle_start_time = time.monotonic()
    requested_model_count = len(model_list)

    # Filter by model_id first so a single deployment is checked when id is specified
    if model_id is not None:
        _by_id = [
            x for x in model_list if (x.get("model_info") or {}).get("id") == model_id
        ]
        if _by_id:
            model_list = _by_id
    elif model is not None:
        _new_model_list = [
            x for x in model_list if x["litellm_params"]["model"] == model
        ]
        if _new_model_list == []:
            _new_model_list = [x for x in model_list if x["model_name"] == model]
        model_list = _new_model_list

    post_filter_model_count = len(model_list)
    model_list = filter_deployments_by_id(
        model_list=model_list
    )  # filter duplicate deployments (e.g. when model alias'es are used)
    deduped_model_count = len(model_list)

    if instrumentation_enabled:
        logger.debug(
            "health_check_cycle_start source=%s cycle_id=%s requested_model_count=%d post_model_filter_count=%d deduped_model_count=%d max_concurrency=%s thread_count=%d rss_mb=%s",
            source,
            cycle_id,
            requested_model_count,
            post_filter_model_count,
            deduped_model_count,
            max_concurrency,
            threading.active_count(),
            _rss_mb_for_log(),
        )

    try:
        healthy_endpoints, unhealthy_endpoints = await _perform_health_check(
            model_list,
            details,
            max_concurrency=max_concurrency,
            instrumentation_context=instrumentation_context,
        )
    except Exception:
        if instrumentation_enabled:
            logger.exception(
                "health_check_cycle_failed source=%s cycle_id=%s model_count=%d duration_ms=%.2f thread_count=%d rss_mb=%s",
                source,
                cycle_id,
                deduped_model_count,
                (time.monotonic() - cycle_start_time) * 1000,
                threading.active_count(),
                _rss_mb_for_log(),
            )
        raise

    if instrumentation_enabled:
        logger.debug(
            "health_check_cycle_complete source=%s cycle_id=%s model_count=%d healthy_count=%d unhealthy_count=%d duration_ms=%.2f thread_count=%d rss_mb=%s",
            source,
            cycle_id,
            deduped_model_count,
            len(healthy_endpoints),
            len(unhealthy_endpoints),
            (time.monotonic() - cycle_start_time) * 1000,
            threading.active_count(),
            _rss_mb_for_log(),
        )

    return healthy_endpoints, unhealthy_endpoints
