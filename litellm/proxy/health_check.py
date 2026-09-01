# This file runs a health check for the LLM, used on litellm/proxy

import asyncio
import logging
import random
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar

from pydantic import TypeAdapter, ValidationError

import litellm

if TYPE_CHECKING:
    from litellm.router import Router

logger: Final = logging.getLogger(__name__)
_DeploymentT: Final = TypeVar("_DeploymentT", bound=Mapping[str, object])
from litellm.constants import (
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS,
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING,
    DEFAULT_HEALTH_CHECK_PROMPT,
    HEALTH_CHECK_TIMEOUT_SECONDS,
)
from litellm.router_utils.auto_router_model_naming import (
    StrategyRouterDependency,
    classify_strategy_router_model,
    strategy_router_dependencies,
)

ILLEGAL_DISPLAY_PARAMS: Final = [
    "messages",
    "api_key",
    "prompt",
    "input",
    "client_secret",
    "azure_ad_token",
    "azure_username",
    "azure_password",
    "vertex_credentials",
    "vertex_ai_credentials",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_web_identity_token",
    "extra_headers",
    "headers",
    "exception",  # internal; not JSON-serializable, never for display
    "litellm_metadata",  # internal tracking metadata with auth objects; not for display
]
# Provider routing fields. Allowed for proxy admins so they can see which
# region/version a deployment is checking; gated at the endpoint layer for
# non-admin callers (see _strip_admin_only_fields_from_health_result).
ADMIN_ONLY_HEALTH_DISPLAY_PARAMS: Final = ("api_base", "api_version")

MINIMAL_DISPLAY_PARAMS: Final = ["model", "mode_error"]

# Modes whose health-check probe is a chat-style completion call and
# therefore accept `max_tokens`. Other modes (embedding, image_generation,
# audio_*, rerank, video_generation, ocr, search, moderation, ...) hit
# endpoints that reject unknown fields with 400 "Unknown parameter:
# 'max_tokens'". Allow-list so new modes are safe by default.
# Per-deployment override: `model_info.health_check_supports_max_tokens`.
_MAX_TOKEN_SUPPORT_MODES: Final[frozenset[str]] = frozenset({"chat", "completion", "responses"})


def _resolve_health_check_mode(model_info: Mapping[str, object], litellm_params: Mapping[str, object]) -> str | None:
    """
    Effective mode for a deployment's health-check probe.

    Prefers operator-set `model_info.mode`; otherwise resolves it from the model
    cost map, which understands `bedrock/` and cross-region inference-profile
    prefixes (`us.`, `eu.`, `apac.`). Without this, non-chat Bedrock deployments
    (e.g. embeddings) are probed as chat, so `max_tokens` is injected and the
    request 400s on "extraneous key [max_tokens]".
    """
    explicit_mode: Final = model_info.get("mode")
    if isinstance(explicit_mode, str):
        return explicit_mode
    model: Final = litellm_params.get("model")
    if not isinstance(model, str):
        return None
    try:
        return litellm.get_model_info(model=model).get("mode")
    except Exception:
        return None


def _should_inject_health_check_max_tokens(model_info: Mapping[str, object], mode: str | None) -> bool:
    """
    Whether the health-check probe should include `max_tokens`.

    Order:
      1. `model_info.health_check_supports_max_tokens` (operator override).
      2. `_MAX_TOKEN_SUPPORT_MODES`. An unresolvable mode is treated as `chat`
         for backward compatibility.
    """
    explicit: Final = model_info.get("health_check_supports_max_tokens")
    if explicit is not None:
        return bool(explicit)
    return (mode or "chat") in _MAX_TOKEN_SUPPORT_MODES


# Health-check modes that forward `reasoning_effort` to the provider (chat-style calls).
_HEALTH_CHECK_MODES_SUPPORTING_REASONING_EFFORT: Final = frozenset((None, "chat", "completion"))


def _get_process_rss_mb() -> float | None:
    """
    Get process RSS memory in MB.
    On Linux, ru_maxrss is in KB. On macOS, ru_maxrss is in bytes.
    """
    try:
        import resource

        ru_maxrss: Final = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(ru_maxrss) / (1024 * 1024)
        return float(ru_maxrss) / 1024
    except Exception:
        return None


def _rss_mb_for_log() -> str:
    rss_mb: Final = _get_process_rss_mb()
    if rss_mb is None:
        return "unknown"
    return f"{rss_mb:.2f}"


def _get_random_llm_message():
    """
    Get a random message from the LLM.
    """
    messages: Final = ["Hey how's it going?", "What's 1 + 1?"]

    return [{"role": "user", "content": random.choice(messages)}]


def _clean_endpoint_data(endpoint_data: dict, details: bool | None = True):
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
    general_settings: dict | None,
) -> dict:
    """
    Build kwargs for ``perform_health_check`` from ``general_settings``.

    When ``health_check_skip_disabled_background_models`` is true, deployments with
    ``model_info.disable_background_health_check`` are omitted from health runs
    (including on-demand ``GET /health``), matching the background loop behavior.
    """
    g: Final = general_settings or {}
    return {
        "health_check_skip_disabled_background_models": bool(
            g.get("health_check_skip_disabled_background_models", False)
        ),
    }


def parse_background_health_check_model_groups(
    general_settings: Mapping[str, object] | None,
) -> frozenset[str] | None:
    """
    Read ``general_settings.background_health_check_model_groups``.

    ``None`` means the allowlist is unset and every deployment participates
    (legacy behavior). A list scopes background health checks and health-check
    routing to deployments whose ``model_name`` is listed. A malformed value
    raises so the proxy fails at startup instead of silently probing everything.
    """
    raw: Final = (general_settings or {}).get("background_health_check_model_groups")
    if raw is None:
        return None
    try:
        return frozenset(TypeAdapter(list[str]).validate_python(raw))
    except ValidationError as e:
        raise ValueError(
            "general_settings.background_health_check_model_groups must be a list of model group names"
        ) from e


def filter_deployments_to_model_groups(
    model_list: Sequence[_DeploymentT],
    model_groups: AbstractSet[str] | None,
) -> tuple[_DeploymentT, ...]:
    """Deployments whose ``model_name`` is in ``model_groups``; all of them when unset."""
    if model_groups is None:
        return tuple(model_list)
    return tuple(x for x in model_list if x.get("model_name") in model_groups)


def filter_deployments_by_id(
    model_list: Sequence[Mapping[str, object]],
) -> list:
    seen_ids: Final = set()
    filtered_deployments: Final = []

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
        timeout_exception: Final = litellm.Timeout(
            message="Health check timeout exceeded",
            model="",
            llm_provider="",
        )
        return {"error": "Timeout exceeded", "exception": timeout_exception}


def _skips_health_checks(deployment: Mapping[str, object]) -> bool:
    info: Final = deployment.get("model_info")
    return bool(info.get("disable_background_health_check", False)) if isinstance(info, Mapping) else False


def _health_check_eligible(
    model_list: Sequence[Mapping[str, object]], skip_disabled: bool
) -> tuple[Mapping[str, object], ...]:
    """Deployments this run is allowed to contact.

    The one eligibility gate, applied to the requested set and to the pool a router's
    dependencies are drawn from alike, so an opted-out deployment cannot re-enter through a
    router that depends on it.
    """
    return tuple(x for x in model_list if not (skip_disabled and _skips_health_checks(x)))


def _deployment_model(deployment: Mapping[str, object]) -> str | None:
    params: Final = deployment.get("litellm_params")
    return params.get("model") if isinstance(params, Mapping) else None


def _narrow_to_target(
    model_list: Sequence[Mapping[str, object]], model: str | None, model_id: str | None
) -> tuple[Mapping[str, object], ...]:
    """Narrow to the requested deployment. An id matching nothing keeps the whole list."""
    if model_id is not None:
        by_id: Final = tuple(x for x in model_list if _deployment_id(x) == model_id)
        return by_id or tuple(model_list)
    if model is None:
        return tuple(model_list)
    by_param: Final = tuple(x for x in model_list if _deployment_model(x) == model)
    return by_param or tuple(x for x in model_list if x.get("model_name") == model)


def _is_strategy_router_deployment(litellm_params: Mapping[str, object]) -> bool:
    """True for strategy-router deployments."""
    model: Final[object] = litellm_params.get("model", "")
    return isinstance(model, str) and classify_strategy_router_model(model) is not None


def _is_marker(deployment: Mapping[str, object]) -> bool:
    params: Final = deployment.get("litellm_params")
    return isinstance(params, Mapping) and _is_strategy_router_deployment(params)


def _deployment_id(deployment: Mapping[str, object]) -> str | None:
    info: Final = deployment.get("model_info")
    ident: Final = info.get("id") if isinstance(info, Mapping) else None
    return str(ident) if ident else None


def _resolved_deployment_ids(router: "Router", model_name: str) -> frozenset[str] | None:
    """Deployment ids backing `model_name`, or None when the name resolves to nothing.

    `get_model_list` composes every channel the request path itself uses (exact name,
    model_group_alias, routing groups, wildcards); a mirror of any one channel would call a
    working tier broken. An alias whose target is gone resolves to nothing, which fails a
    request exactly like an unknown name.
    """
    resolved: Final = router.get_model_list(model_name=model_name)
    if not resolved:
        return None
    return frozenset(ident for entry in resolved if (ident := _deployment_id(entry)))


def _dependency_failure(
    dependency: StrategyRouterDependency,
    router: "Router",
    unhealthy_ids: frozenset[str],
) -> str | None:
    """Why this dependency makes its router unable to serve, or None when it does not.

    A name reds its router only when *every* deployment behind it is known unhealthy. One
    replica this run never judged, hidden from the caller or opted out of health checks, can
    still serve what the dead one drops, so partial evidence leaves the verdict green.
    """
    resolved: Final = _resolved_deployment_ids(router, dependency.model_name)
    if resolved is None:
        return f"{dependency.role} model '{dependency.model_name}' matches no deployment on this proxy"
    if not resolved or not resolved <= unhealthy_ids:
        return None
    return f"{dependency.role} model '{dependency.model_name}' has no healthy deployment"


def _strategy_router_dependency_error(
    deployment: Mapping[str, object],
    router: "Router",
    unhealthy_ids: frozenset[str],
) -> str | None:
    """The first dependency fault that makes this router unable to serve, if any."""
    params: Final = deployment.get("litellm_params")
    if not isinstance(params, Mapping):
        return None
    return next(
        (
            failure
            for dependency in strategy_router_dependencies(params)
            if (failure := _dependency_failure(dependency, router, unhealthy_ids))
        ),
        None,
    )


def _deployments_by_id(
    universe: Sequence[Mapping[str, object]], ids: frozenset[str]
) -> tuple[Mapping[str, object], ...]:
    """The deployments for `ids`, one row per id.

    Reuses the requested set's own dedupe rule, so an alias that duplicates a row cannot get
    it probed twice or split a single id's verdict across two disagreeing results.
    """
    matched: Final = tuple(d for d in universe if (uid := _deployment_id(d)) and uid in ids)
    return tuple(filter_deployments_by_id(model_list=matched))


def _dependency_deployments_to_probe(
    checked: Sequence[Mapping[str, object]],
    universe: Sequence[Mapping[str, object]],
    router: "Router",
) -> tuple[Mapping[str, object], ...]:
    """Deployments backing the checked routers' dependencies that are not already checked.

    Empty on a full-list run, which therefore gains no probe; it is the targeted
    `/health?model_id=<router>` call the dashboard makes per deployment that needs them,
    since a router's verdict is a statement about models the request never named. Drawn from
    `universe`, the caller's access-filtered list, so no deployment is probed that the caller
    was not already granted. Expansion follows routers through routers, one hop per round,
    because a child router's own models must be probed for the parent to fail; stopping when
    a round adds nothing is what makes a router cycle terminate.
    """
    checked_ids: Final = frozenset(cid for d in checked if (cid := _deployment_id(d)))
    reached = checked_ids  # rebind-ok: the sweep's cursor, one hop wider per round
    frontier = tuple(checked)  # rebind-ok: the routers whose dependencies the next round expands
    for _ in range(len(universe)):
        names = frozenset(
            dependency.model_name
            for deployment in frontier
            if isinstance(params := deployment.get("litellm_params"), Mapping)
            for dependency in strategy_router_dependencies(params)
        )
        fresh_ids = (
            frozenset(ident for name in names for ident in (_resolved_deployment_ids(router, name) or ())) - reached
        )
        if not fresh_ids:
            break
        frontier = _deployments_by_id(universe, fresh_ids)
        reached = reached | fresh_ids
    return _deployments_by_id(universe, reached - checked_ids)


def _strategy_router_verdicts(
    healthy_endpoints: Sequence[Mapping[str, object]],
    unhealthy_endpoints: Sequence[Mapping[str, object]],
    checked: Sequence[Mapping[str, object]],
    router: "Router",
) -> Mapping[str, str]:
    """The dependency fault, per model id, for every strategy router that cannot serve.

    A marker is filed healthy by `_run_model_health_check` returning `{}`, which says only
    that nothing was probed. This is where that placeholder becomes a verdict, derived from
    this run's own results rather than a re-probe or a cache that is empty unless
    `enable_health_check_routing` is on. A marker never fails a probe of its own, so verdicts
    settle over rounds, each feeding the last round's reds back in as unhealthy; without that
    the parent of a red child would stay green. Bounded by the marker count, which is what
    makes a router cycle terminate green rather than spin.
    """
    by_id: Final = MappingProxyType({i: d for d in checked if (i := _deployment_id(d))})
    markers: Final = MappingProxyType(
        {
            marker_id: by_id[marker_id]
            for endpoint in healthy_endpoints
            if isinstance(marker_id := endpoint.get("model_id"), str) and marker_id in by_id
            if _is_marker(by_id[marker_id])
        }
    )
    probe_failures: Final = frozenset(
        ident for endpoint in unhealthy_endpoints if isinstance(ident := endpoint.get("model_id"), str)
    )
    settled: Mapping[str, str] = MappingProxyType({})  # rebind-ok: the fixed point, a round's verdicts at a time
    for _ in range(len(markers)):
        fresh = MappingProxyType(
            {
                marker_id: error
                for marker_id, deployment in markers.items()
                if marker_id not in settled
                if (error := _strategy_router_dependency_error(deployment, router, probe_failures | frozenset(settled)))
            }
        )
        if not fresh:
            break
        settled = MappingProxyType({**settled, **fresh})
    return settled


def _finalize_strategy_router_endpoints(
    healthy_endpoints: Sequence[Mapping[str, object]],
    unhealthy_endpoints: Sequence[Mapping[str, object]],
    checked: Sequence[Mapping[str, object]],
    router: "Router | None",
    dependency_probes: Sequence[Mapping[str, object]],
) -> tuple[Sequence[Mapping[str, object]], Sequence[Mapping[str, object]]]:
    """Apply router verdicts, then drop the deployments probed only to reach them.

    The probes exist to judge the routers that depend on them; reporting them would answer a
    targeted request with deployments the caller never asked about.
    """
    verdicts: Final = (
        _strategy_router_verdicts(healthy_endpoints, unhealthy_endpoints, checked, router)
        if router is not None
        else MappingProxyType({})
    )
    dropped: Final = frozenset(i for d in dependency_probes if (i := _deployment_id(d)))

    def keep(endpoint: Mapping[str, object]) -> bool:
        model_id: Final = endpoint.get("model_id")
        return not (isinstance(model_id, str) and model_id in dropped)

    def verdict_for(endpoint: Mapping[str, object]) -> str | None:
        model_id: Final = endpoint.get("model_id")
        return verdicts.get(model_id) if isinstance(model_id, str) else None

    kept_healthy: Final = tuple(e for e in healthy_endpoints if keep(e))
    return (
        tuple(e for e in kept_healthy if verdict_for(e) is None),
        tuple(e for e in unhealthy_endpoints if keep(e))
        + tuple(
            dict(e, error=error)  # mutable-ok: the /health payload must stay a plain JSON-serializable dict
            for e in kept_healthy
            if (error := verdict_for(e)) is not None
        ),
    )


async def _run_model_health_check(model: dict):
    litellm_params = model["litellm_params"]
    model_info: Final = model.get("model_info", {})

    if _is_strategy_router_deployment(litellm_params):
        return {}

    mode: Final = _resolve_health_check_mode(
        model_info,
        litellm_params,  # any-ok: untyped router config dict
    )
    litellm_params = _update_litellm_params_for_health_check(model_info, litellm_params)
    timeout: Final = model_info.get("health_check_timeout") or HEALTH_CHECK_TIMEOUT_SECONDS

    return await run_with_timeout(
        litellm.ahealth_check(
            litellm_params,
            mode=mode,
            prompt=DEFAULT_HEALTH_CHECK_PROMPT,
            input=["test from litellm"],
        ),
        timeout,
    )


async def _run_health_checks_with_bounded_concurrency(models: list, concurrency_limit: int) -> tuple[list, int]:
    """
    Run health checks with at most `concurrency_limit` active tasks.
    Preserves result ordering to match `models`.
    """
    results: Final[list] = [None] * len(models)
    tasks_to_index: Final[dict[asyncio.Task, int]] = {}
    model_iter: Final = iter(enumerate(models))
    peak_in_flight = 0

    def _schedule_next() -> bool:
        nonlocal peak_in_flight
        try:
            idx, next_model = next(model_iter)
        except StopIteration:
            return False
        task: Final = asyncio.create_task(_run_model_health_check(next_model))
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
    details: bool | None = True,
    max_concurrency: int | None = None,
    instrumentation_context: dict | None = None,
):
    """
    Perform a health check for each model in the list.

    max_concurrency: Optional limit on concurrent health check requests.
    """

    instrumentation_context = instrumentation_context or {}
    instrumentation_enabled: Final = bool(instrumentation_context.get("enabled", False))
    cycle_id: Final = instrumentation_context.get("cycle_id", "unknown")
    source: Final = instrumentation_context.get("source", "unknown")

    dispatch_mode = "unbounded"
    peak_in_flight = 0
    if isinstance(max_concurrency, int) and max_concurrency > 0:
        dispatch_mode = "bounded"
        results, peak_in_flight = await _run_health_checks_with_bounded_concurrency(model_list, max_concurrency)
    else:
        tasks: Final = [asyncio.create_task(_run_model_health_check(model)) for model in model_list]
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

    healthy_endpoints: Final = []
    unhealthy_endpoints: Final = []
    # Exceptions keyed by model_id; returned separately so callers can use
    # them for cooldown integration without risking JSON-serialization errors
    # in the /health response.
    exceptions_by_model_id: Final[dict] = {}

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
                    cleaned["exception_status"] = getattr(is_healthy, "status_code", 500)
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
    now: Final = time.time()
    states: Final[dict] = {}

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


def _resolve_health_check_max_tokens(model_info: dict, litellm_params: dict) -> int | None:
    """
    Pick max_tokens for the health check request.

    Priority:
    1. model_info.health_check_max_tokens (explicit override)
    2. For non-wildcard routes: health_check_max_tokens_reasoning / _non_reasoning
       from model_info based on litellm.supports_reasoning(litellm_params["model"])
    3. For non-wildcard reasoning routes: BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING
       from env (if set)
    4. BACKGROUND_HEALTH_CHECK_MAX_TOKENS (global, any route including wildcards)
    5. Non-wildcard default: 16
    6. Wildcard and nothing from (1)(4): leave unset (caller omits max_tokens)
    """
    explicit: Final = model_info.get("health_check_max_tokens", None)
    if explicit is not None:
        return int(explicit)

    is_wildcard: Final = _health_check_deployment_is_wildcard(litellm_params)
    deployment_model: Final = _deployment_model_string_for_health_check(litellm_params)

    if not is_wildcard:
        try:
            is_reasoning = litellm.supports_reasoning(deployment_model)
        except Exception:
            is_reasoning = False
        tokens_reasoning: Final = model_info.get("health_check_max_tokens_reasoning", None)
        tokens_non_reasoning: Final = model_info.get("health_check_max_tokens_non_reasoning", None)
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
        return 16

    return None


def _update_litellm_params_for_health_check(model_info: dict, litellm_params: dict) -> dict:
    """
    Update the litellm params for health check.

    - merges `model_info.health_check_params` into the probe request, so a deployment whose provider
      requires a payload field litellm does not synthesize (e.g. `mediaSource` for Bedrock TwelveLabs
      Pegasus) can supply it. The dedicated knobs below are applied afterwards and win on conflict.
    - gets a short `messages` param for health check
    - adds a bounded `max_tokens` when the deployment is a chat-style mode
      (`chat`, `completion`, `responses`) or the operator explicitly opts in
      via `model_info.health_check_supports_max_tokens`. Non-chat endpoints
      (image, embedding, audio_*, rerank, video, ocr, search, moderation, ...)
      reject unknown fields with 400 "Unknown parameter: 'max_tokens'".
    - updates the `model` param with the `health_check_model` if it exists Doc: https://docs.litellm.ai/docs/proxy/health#wildcard-routes
    - updates the `voice` param with the `health_check_voice` for `audio_speech` mode if it exists Doc: https://docs.litellm.ai/docs/proxy/health#text-to-speech-models
    - for Bedrock models with region routing (bedrock/region/model), strips the litellm routing prefix but preserves the model ID, and pins `custom_llm_provider` to `bedrock` (only when the deployment hasn't already set one, so an explicit `bedrock_converse` survives) so the bare model id still resolves to the provider (e.g. cross-region ids like `us.cohere.embed-v4:0`)
    """
    mode: Final = _resolve_health_check_mode(
        model_info,
        litellm_params,  # any-ok: untyped router config dict
    )
    _health_check_params: Final = model_info.get("health_check_params", None)
    if isinstance(_health_check_params, dict):
        litellm_params.update(_health_check_params)
    elif _health_check_params is not None:
        logger.warning(
            "health_check_params for model %s is a %s, expected a dict. Ignoring it.",
            litellm_params.get("model"),
            type(_health_check_params).__name__,
        )

    litellm_params["messages"] = _get_random_llm_message()
    if _should_inject_health_check_max_tokens(
        model_info,
        mode,  # any-ok: untyped router config dict
    ):
        _resolved_max_tokens: Final = _resolve_health_check_max_tokens(model_info, litellm_params)
        if _resolved_max_tokens is not None:
            litellm_params["max_tokens"] = _resolved_max_tokens

    # Per-model reasoning effort for health checks only (e.g. reasoning_effort=none).
    if mode in _HEALTH_CHECK_MODES_SUPPORTING_REASONING_EFFORT:
        _hc_reasoning_effort: Final = model_info.get("health_check_reasoning_effort", None)
        if _hc_reasoning_effort is not None:
            litellm_params["reasoning_effort"] = _hc_reasoning_effort

    _health_check_model: Final = model_info.get("health_check_model", None)
    if _health_check_model is not None:
        litellm_params["model"] = _health_check_model
    if mode == "audio_speech":
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
        model = model.removeprefix("bedrock/")  # len("bedrock/") = 8

        # Now check for region routing and strip it if present
        # Need to handle formats like:
        # - "us-west-2/model" → "model"
        # - "converse/us-west-2/model" → "converse/model"
        # - "llama/arn:..." → "llama/arn:..." (preserve handler)
        #
        # Strategy: Check each path segment, remove regions, preserve everything else
        parts: Final = model.split("/")
        filtered_parts: Final = []

        for part in parts:
            # Skip AWS regions, keep everything else
            if part not in BedrockModelInfo.all_global_regions:
                filtered_parts.append(part)

        model = "/".join(filtered_parts)
        litellm_params["model"] = model
        if not litellm_params.get("custom_llm_provider"):  # any-ok: untyped router dict
            litellm_params["custom_llm_provider"] = (  # any-ok: untyped router dict
                "bedrock"
            )

    return litellm_params


async def perform_health_check(
    model_list: list,
    model: str | None = None,
    cli_model: str | None = None,
    details: bool | None = True,
    model_id: str | None = None,
    max_concurrency: int | None = None,
    instrumentation_context: dict | None = None,
    health_check_skip_disabled_background_models: bool = False,
    router: "Router | None" = None,
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
    instrumentation_enabled: Final = bool(instrumentation_context.get("enabled", False))
    cycle_id: Final = instrumentation_context.get("cycle_id", "unknown")
    source: Final = instrumentation_context.get("source", "unknown")

    if not model_list:
        if cli_model:
            model_list = [{"model_name": cli_model, "litellm_params": {"model": cli_model}}]
        else:
            if instrumentation_enabled:
                logger.debug(
                    "health_check_cycle_skipped source=%s cycle_id=%s reason=no_models",
                    source,
                    cycle_id,
                )
            return [], [], {}

    cycle_start_time: Final = time.monotonic()
    requested_model_count: Final = len(model_list)
    skip_disabled: Final = health_check_skip_disabled_background_models
    narrowed: Final = _health_check_eligible(_narrow_to_target(model_list, model, model_id), skip_disabled)
    if not narrowed:
        if instrumentation_enabled:
            logger.debug(
                "health_check_cycle_skipped source=%s cycle_id=%s reason=no_models_after_filter",
                source,
                cycle_id,
            )
        return [], [], {}

    post_filter_model_count: Final = len(narrowed)
    requested: Final = filter_deployments_by_id(model_list=narrowed)
    deduped_model_count: Final = len(requested)

    dependency_probes: Final = (
        _dependency_deployments_to_probe(requested, _health_check_eligible(model_list, skip_disabled), router)
        if router is not None
        else ()
    )
    checked: Final = requested + list(dependency_probes)  # mutable-ok: _perform_health_check takes a list

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
            probed_healthy,
            probed_unhealthy,
            exceptions_by_model_id,
        ) = await _perform_health_check(
            checked,
            details,
            max_concurrency=max_concurrency,
            instrumentation_context=instrumentation_context,
        )
        graded_healthy, graded_unhealthy = _finalize_strategy_router_endpoints(
            probed_healthy, probed_unhealthy, checked, router, dependency_probes
        )
        healthy_endpoints: Final = list(graded_healthy)
        unhealthy_endpoints: Final = list(graded_unhealthy)
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
