# This file runs a health check for the LLM, used on litellm/proxy

import asyncio
import logging
import random
import sys
import threading
import time
from typing import List, Optional

import litellm

logger = logging.getLogger(__name__)
from litellm.constants import (
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS,
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING,
    DEFAULT_HEALTH_CHECK_PROMPT,
    HEALTH_CHECK_TIMEOUT_SECONDS,
)

ILLEGAL_DISPLAY_PARAMS = [
    "messages",
    "api_key",
    "prompt",
    "input",
    "vertex_credentials",
    "aws_access_key_id",
    "aws_secret_access_key",
    "exception",  # internal; not JSON-serializable, never for display
    "litellm_metadata",  # internal tracking metadata with auth objects; not for display
]
# Provider routing fields. Allowed for proxy admins so they can see which
# region/version a deployment is checking; gated at the endpoint layer for
# non-admin callers (see _strip_admin_only_fields_from_health_result).
ADMIN_ONLY_HEALTH_DISPLAY_PARAMS = ("api_base", "api_version")

MINIMAL_DISPLAY_PARAMS = ["model", "mode_error"]

# Health-check modes that forward `reasoning_effort` to the provider (chat-style calls).
_HEALTH_CHECK_MODES_SUPPORTING_REASONING_EFFORT = frozenset(
    (None, "chat", "completion")
)


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


def health_check_filter_kwargs_from_general_settings(
    general_settings: Optional[dict],
) -> dict:
    """
    Build kwargs for ``perform_health_check`` from ``general_settings``.

    When ``health_check_skip_disabled_background_models`` is true, deployments with
    ``model_info.disable_background_health_check`` are omitted from health runs
    (including on-demand ``GET /health``), matching the background loop behavior.
    """
    g = general_settings or {}
    return {
        "health_check_skip_disabled_background_models": bool(
            g.get("health_check_skip_disabled_background_models", False)
        ),
    }


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
        timeout_exception = litellm.Timeout(
            message="Health check timeout exceeded",
            model="",
            llm_provider="",
        )
        return {"error": "Timeout exceeded", "exception": timeout_exception}


async def _run_model_health_check(model: dict):
    litellm_params = model["litellm_params"]
    model_info = model.get("model_info", {})
    mode = model_info.get("mode", None)
    litellm_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    timeout = model_info.get("health_check_timeout") or HEALTH_CHECK_TIMEOUT_SECONDS

    return await run_with_timeout(
        litellm.ahealth_check(
            litellm_params,
            mode=mode,
            prompt=DEFAULT_HEALTH_CHECK_PROMPT,
            input=["test from litellm"],
        ),
        timeout,
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
    # Exceptions keyed by model_id; returned separately so callers can use
    # them for cooldown integration without risking JSON-serialization errors
    # in the /health response.
    exceptions_by_model_id: dict = {}

    for is_healthy, model in zip(results, model_list):
        litellm_params = model["litellm_params"]
        _model_id = (model.get("model_info") or {}).get("id")

        if isinstance(is_healthy, dict) and "error" not in is_healthy:
            cleaned = _clean_endpoint_data({**litellm_params, **is_healthy}, details)
            if _model_id:
                cleaned["model_id"] = _model_id
            healthy_endpoints.append(cleaned)
        elif isinstance(is_healthy, dict):
            cleaned = _clean_endpoint_data({**litellm_params, **is_healthy}, details)
            if _model_id:
                cleaned["model_id"] = _model_id
                if "exception" in is_healthy:
                    exc = is_healthy["exception"]
                    exceptions_by_model_id[_model_id] = exc
                    # Store integer status code so shared-cache readers can
                    # reconstruct the transient-error filter without the exception object.
                    cleaned["exception_status"] = getattr(exc, "status_code", 500)
            unhealthy_endpoints.append(cleaned)
        else:
            cleaned = _clean_endpoint_data(litellm_params, details)
            if _model_id:
                cleaned["model_id"] = _model_id
                if isinstance(is_healthy, Exception):
                    exceptions_by_model_id[_model_id] = is_healthy
                    cleaned["exception_status"] = getattr(
                        is_healthy, "status_code", 500
                    )
            unhealthy_endpoints.append(cleaned)

    return healthy_endpoints, unhealthy_endpoints, exceptions_by_model_id


def build_deployment_health_states(
    healthy_endpoints: list,
    unhealthy_endpoints: list,
) -> dict:
    """
    Build a dict mapping deployment_id -> DeploymentHealthStateValue from
    health check endpoint results.

    Each endpoint dict includes a 'model_id' field (added by _perform_health_check)
    that maps back to the deployment's model_info.id.

    Used by the background health check loop to feed health state into
    the router's DeploymentHealthCache for health-check-driven routing.
    """
    now = time.time()
    states: dict = {}

    for ep in healthy_endpoints:
        model_id = ep.get("model_id")
        if model_id:
            states[model_id] = {
                "is_healthy": True,
                "timestamp": now,
                "reason": "",
            }

    for ep in unhealthy_endpoints:
        model_id = ep.get("model_id")
        if model_id:
            states[model_id] = {
                "is_healthy": False,
                "timestamp": now,
                "reason": "background_health_check_failed",
            }

    return states


def _deployment_model_string_for_health_check(litellm_params: dict) -> str:
    """Deployment model from litellm_params (before Bedrock rewrite).

    Used for reasoning vs non-reasoning max_tokens and wildcard detection only.
    Does not use ``health_check_model``; that override applies later to the request.
    """
    return litellm_params.get("model") or ""


def _health_check_deployment_is_wildcard(litellm_params: dict) -> bool:
    return "*" in _deployment_model_string_for_health_check(litellm_params)


def _resolve_health_check_max_tokens(
    model_info: dict, litellm_params: dict
) -> Optional[int]:
    """
    Pick max_tokens for the health check request.

    Priority:
    1. model_info.health_check_max_tokens (explicit override)
    2. For non-wildcard routes: health_check_max_tokens_reasoning / _non_reasoning
       from model_info based on litellm.supports_reasoning(litellm_params["model"])
    3. For non-wildcard reasoning routes: BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING
       from env (if set)
    4. BACKGROUND_HEALTH_CHECK_MAX_TOKENS (global, any route including wildcards)
    5. Non-wildcard default: 5
    6. Wildcard and nothing from (1)(4): leave unset (caller omits max_tokens)
    """
    explicit = model_info.get("health_check_max_tokens", None)
    if explicit is not None:
        return int(explicit)

    is_wildcard = _health_check_deployment_is_wildcard(litellm_params)
    deployment_model = _deployment_model_string_for_health_check(litellm_params)

    if not is_wildcard:
        try:
            is_reasoning = litellm.supports_reasoning(deployment_model)
        except Exception:
            is_reasoning = False
        tokens_reasoning = model_info.get("health_check_max_tokens_reasoning", None)
        tokens_non_reasoning = model_info.get(
            "health_check_max_tokens_non_reasoning", None
        )
        if tokens_reasoning is not None or tokens_non_reasoning is not None:
            if is_reasoning and tokens_reasoning is not None:
                return int(tokens_reasoning)
            if not is_reasoning and tokens_non_reasoning is not None:
                return int(tokens_non_reasoning)
        if is_reasoning and BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING is not None:
            return int(BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING)

    if BACKGROUND_HEALTH_CHECK_MAX_TOKENS is not None:
        return int(BACKGROUND_HEALTH_CHECK_MAX_TOKENS)

    if not is_wildcard:
        return 5

    return None


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
    _resolved_max_tokens = _resolve_health_check_max_tokens(model_info, litellm_params)
    if _resolved_max_tokens is not None:
        litellm_params["max_tokens"] = _resolved_max_tokens

    # Per-model reasoning effort for health checks only (e.g. reasoning_effort=none).
    if model_info.get("mode", None) in _HEALTH_CHECK_MODES_SUPPORTING_REASONING_EFFORT:
        _hc_reasoning_effort = model_info.get("health_check_reasoning_effort", None)
        if _hc_reasoning_effort is not None:
            litellm_params["reasoning_effort"] = _hc_reasoning_effort

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
    health_check_skip_disabled_background_models: bool = False,
):
    """
    Perform a health check on the system.

    When model_id is provided, only the deployment with that id is checked
    (so models that share the same name but have different ids are checked separately).
    When model (name) is provided, all deployments matching that name are checked.

    When ``health_check_skip_disabled_background_models`` is True (via
    ``general_settings.health_check_skip_disabled_background_models``), deployments
    with ``model_info.disable_background_health_check: true`` are omitted from
    this run (including targeted ``/health`` queries), consistent with the
    background health loop.

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
            return [], [], {}

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

    if health_check_skip_disabled_background_models:
        model_list = [
            x
            for x in model_list
            if not (x.get("model_info") or {}).get(
                "disable_background_health_check", False
            )
        ]
    if not model_list:
        if instrumentation_enabled:
            logger.debug(
                "health_check_cycle_skipped source=%s cycle_id=%s reason=no_models_after_filter",
                source,
                cycle_id,
            )
        return [], [], {}

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
        (
            healthy_endpoints,
            unhealthy_endpoints,
            exceptions_by_model_id,
        ) = await _perform_health_check(
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

    return healthy_endpoints, unhealthy_endpoints, exceptions_by_model_id
