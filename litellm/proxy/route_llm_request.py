import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from fastapi import HTTPException, status

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.router_utils.common_utils import _is_proxy_admin_request, resolve_model_group_alias
from litellm.router_utils.fallback_event_handlers import (
    _check_non_standard_fallback_format,
    creates_provider_scoped_resource,
    get_authenticated_team_context,
    get_fallback_model_group,
    preserve_authenticated_team_context,
    references_provider_scoped_resource,
)

# Client-supplied params that make the router or the call path fabricate a
# failure or a delay instead of calling the provider. The ``mock_testing_*``
# names are kept in sync with ``litellm.types.router.MockRouterTestingParams``
# by ``test_gated_mock_params_cover_mock_router_testing_params``. Hardcoding
# (rather than deriving via ``dataclasses.fields(MockRouterTestingParams)`` at
# import time) avoids a cyclic import: ``litellm.types.router`` imports
# back into proxy modules before this module finishes loading.
GATED_MOCK_PARAM_NAMES: Final[tuple[str, ...]] = (
    "mock_testing_fallbacks",
    "mock_testing_context_fallbacks",
    "mock_testing_content_policy_fallbacks",
    "mock_testing_rate_limit_error",
    "mock_timeout",
    "mock_delay",
)

MOCK_TESTING_CONFIG_KEY: Final = "dangerously_allow_mock_testing_request_params"

EVAL_ROUTE_TYPES: Final = frozenset(
    {
        "acreate_eval",
        "alist_evals",
        "aget_eval",
        "aupdate_eval",
        "adelete_eval",
        "acancel_eval",
        "acreate_run",
        "alist_runs",
        "aget_run",
        "acancel_run",
        "adelete_run",
    }
)

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def _route_user_config_request(data: dict, route_type: str):
    """Route a request using the user-provided router config."""
    router_config: Final = data.pop("user_config")

    # Filter router_config to only include valid Router.__init__ arguments
    # This prevents TypeError when invalid parameters are stored in the database
    valid_args: Final = litellm.Router.get_valid_args()
    filtered_config: Final = {k: v for k, v in router_config.items() if k in valid_args}

    user_router: Final = litellm.Router(**filtered_config)
    ret_val: Final = getattr(user_router, f"{route_type}")(**data)
    user_router.discard()
    return ret_val


def _is_a2a_agent_model(model_name: Any) -> bool:
    """Check if the model name is for an A2A agent (a2a/ prefix)."""
    return isinstance(model_name, str) and model_name.startswith("a2a/")


def _deployment_ids(deployments: Any) -> set[str]:
    """Return deployment IDs from either Router healthy-deployment shape."""
    deployment_list = [deployments] if isinstance(deployments, Mapping) else deployments
    if not isinstance(deployment_list, list):
        return set()

    deployment_ids: set[str] = set()
    for deployment in deployment_list:
        if not isinstance(deployment, Mapping):
            continue
        model_info = deployment.get("model_info")
        if not isinstance(model_info, Mapping):
            continue
        deployment_id = model_info.get("id")
        if isinstance(deployment_id, str) and deployment_id:
            deployment_ids.add(deployment_id)
    return deployment_ids


def _expand_fallback_target(fallback_target: Any) -> list[str | dict[str, Any]]:
    """Expand a list-valued direct fallback into ordered concrete candidates."""
    if isinstance(fallback_target, str):
        return [fallback_target]
    if not isinstance(fallback_target, Mapping):
        return []

    fallback_model = fallback_target.get("model")
    if isinstance(fallback_model, str):
        return [dict(fallback_target)]
    if not isinstance(fallback_model, list) or not fallback_model:
        return []
    if not all(isinstance(candidate, str) and candidate for candidate in fallback_model):
        return []

    return [{**fallback_target, "model": candidate} for candidate in fallback_model]


def _expand_fallback_tail(fallback_targets: list[Any]) -> list[Any]:
    """Expand list-valued direct targets before handing the trusted tail to runtime."""
    expanded_targets: list[Any] = []
    for fallback_target in fallback_targets:
        candidates = _expand_fallback_target(fallback_target)
        if candidates:
            expanded_targets.extend(candidates)
        else:
            expanded_targets.append(fallback_target)
    return expanded_targets


async def _get_available_fallback_request(
    llm_router: LitellmRouter,
    model_name: str,
    team_id: str | None,
    request_data: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a trusted request targeting the first executable server fallback."""
    if request_data.get("disable_fallbacks") is True:
        return None

    max_fallbacks: Final = request_data.get("max_fallbacks", getattr(llm_router, "max_fallbacks", None))
    if max_fallbacks == 0:
        return None

    fallbacks: Final = getattr(llm_router, "fallbacks", None)
    if not isinstance(fallbacks, list):
        return None

    router_model_list: Final = getattr(llm_router, "model_list", None)
    if not isinstance(router_model_list, list):
        return None
    router_deployment_ids: Final = _deployment_ids(router_model_list)
    # Runtime enforcement below is fail-closed: every Router deployment must have
    # an ID so an unvalidated target cannot escape the exclusion boundary.
    if len(router_deployment_ids) != len(router_model_list):
        return None

    if _check_non_standard_fallback_format(fallbacks):
        fallback_model_group = fallbacks
    else:
        fallback_model_group, _ = get_fallback_model_group(
            fallbacks=fallbacks,
            model_group=model_name,
        )
    if not isinstance(fallback_model_group, list):
        return None

    same_model_group_only: Final = references_provider_scoped_resource(
        request_data
    ) or creates_provider_scoped_resource(request_data)
    alias_map: Final = getattr(llm_router, "model_group_alias", None)
    canonical_model_group: Final = resolve_model_group_alias(alias_map, model_name) or model_name
    _, authenticated_team_bucket = get_authenticated_team_context(request_data)

    for fallback_index, fallback_target in enumerate(fallback_model_group):
        fallback_candidates = _expand_fallback_target(fallback_target)
        for candidate_index, candidate_target in enumerate(fallback_candidates):
            # Both copies start without caller-supplied fallbacks. ``fallback_request``
            # is disposable preflight state; ``runtime_request`` is the trusted request
            # we will hand to the Router if this server-configured target is eligible.
            runtime_request: dict[str, Any] = dict(request_data)
            runtime_request.pop("fallbacks", None)
            fallback_request: dict[str, Any] = dict(runtime_request)

            if isinstance(candidate_target, str):
                fallback_model = candidate_target
                canonical_fallback_model = resolve_model_group_alias(alias_map, fallback_model) or fallback_model
                if canonical_fallback_model == canonical_model_group:
                    continue
            else:
                fallback_model = candidate_target["model"]
                canonical_fallback_model = resolve_model_group_alias(alias_map, fallback_model) or fallback_model
                fallback_request.update(candidate_target)
                runtime_request.update(candidate_target)

            if same_model_group_only and canonical_fallback_model != canonical_model_group:
                continue

            fallback_request["model"] = fallback_model
            runtime_request["model"] = fallback_model
            preserve_authenticated_team_context(
                request_kwargs=fallback_request,
                authenticated_team_id=team_id,
                source_bucket=authenticated_team_bucket,
            )
            preserve_authenticated_team_context(
                request_kwargs=runtime_request,
                authenticated_team_id=team_id,
                source_bucket=authenticated_team_bucket,
            )

            try:
                fallback_messages = fallback_request.get("messages")
                fallback_input = fallback_request.get("input")
                pre_routing_hook_response = await llm_router.async_pre_routing_hook(
                    model=fallback_model,
                    request_kwargs=fallback_request,
                    messages=fallback_messages,
                    input=fallback_input,
                    specific_deployment=False,
                )
                if pre_routing_hook_response is not None:
                    fallback_model = pre_routing_hook_response.model
                    fallback_messages = pre_routing_hook_response.messages
                    if pre_routing_hook_response.litellm_params is not None:
                        fallback_request.update(pre_routing_hook_response.litellm_params)
                        preserve_authenticated_team_context(
                            request_kwargs=fallback_request,
                            authenticated_team_id=team_id,
                            source_bucket=authenticated_team_bucket,
                        )

                healthy_deployments = await llm_router.async_get_healthy_deployments(
                    model=fallback_model,
                    request_kwargs=fallback_request,
                    messages=fallback_messages,
                    input=fallback_input,
                )
            except Exception:
                continue

            validated_deployment_ids: Final = _deployment_ids(healthy_deployments)
            if not validated_deployment_ids:
                continue

            # ``async_get_available_deployment`` invokes the pre-routing hook again at
            # runtime. A stateful strategy can therefore return a different target on
            # the second invocation. Router already supports a one-shot exclusion list
            # that is consumed by the next healthy-deployment lookup; restrict that
            # lookup to deployments that actually passed this preflight validation.
            existing_exclusions = runtime_request.get("_excluded_deployment_ids")
            excluded_deployment_ids = (
                {deployment_id for deployment_id in existing_exclusions if isinstance(deployment_id, str)}
                if isinstance(existing_exclusions, (list, tuple, set, frozenset))
                else set()
            )
            excluded_deployment_ids.update(router_deployment_ids - validated_deployment_ids)
            runtime_request["_excluded_deployment_ids"] = sorted(excluded_deployment_ids)

            # We are consuming the first fallback hop here instead of letting the
            # blocked primary enter normal Router selection. Preserve the remaining
            # candidates from a list-valued target before later trusted fallbacks,
            # unless the selected server-side dict supplied its own fallback chain.
            if "fallbacks" not in runtime_request:
                remaining_candidates = fallback_candidates[candidate_index + 1 :]
                later_fallbacks = _expand_fallback_tail(fallback_model_group[fallback_index + 1 :])
                runtime_request["fallbacks"] = [*remaining_candidates, *later_fallbacks]
            fallback_depth = request_data.get("fallback_depth", 0)
            runtime_request["fallback_depth"] = fallback_depth + 1 if isinstance(fallback_depth, int) else 1
            return runtime_request

    return None


async def _has_available_fallback(
    llm_router: LitellmRouter,
    model_name: str,
    team_id: str | None,
    request_data: Mapping[str, Any],
) -> bool:
    """Return whether a server-configured fallback is eligible for this request at runtime."""
    return (
        await _get_available_fallback_request(
            llm_router=llm_router,
            model_name=model_name,
            team_id=team_id,
            request_data=request_data,
        )
        is not None
    )


async def _raise_if_model_fully_blocked(
    llm_router: LitellmRouter,
    model_name: Any,
    team_id: str | None,
    request_data: dict[str, Any],
    *,
    allow_router_fallback: bool = True,
) -> None:
    if not isinstance(model_name, str) or not model_name:
        return
    if not isinstance(llm_router, litellm.Router):
        return
    deployments: Final = llm_router.get_model_list(model_name=model_name, team_id=team_id) or []
    if not llm_router._are_all_deployments_blocked(deployments):
        return

    fallback_request = (
        await _get_available_fallback_request(
            llm_router=llm_router,
            model_name=model_name,
            team_id=team_id,
            request_data=request_data,
        )
        if allow_router_fallback
        else None
    )
    if fallback_request is None:
        raise litellm.PermissionDeniedError(
            message="Model is blocked",
            model=model_name,
            llm_provider="",
            response=httpx.Response(
                status_code=403,
                request=httpx.Request(method="POST", url="https://github.com/BerriAI/litellm"),
            ),
        )

    # Never send the fully blocked primary back through normal Router selection:
    # a request-dependent pre-routing strategy/plugin could otherwise rewrite it
    # into a non-fallback model that was never authorized. The request now targets
    # the first executable server-configured fallback validated above.
    request_data.clear()
    request_data.update(fallback_request)


ROUTE_ENDPOINT_MAPPING: Final = {
    "acompletion": "/chat/completions",
    "atext_completion": "/completions",
    "aembedding": "/embeddings",
    "aimage_generation": "/image/generations",
    "aspeech": "/audio/speech",
    "atranscription": "/audio/transcriptions",
    "amoderation": "/moderations",
    "arerank": "/rerank",
    "aresponses": "/responses",
    "_aresponses_websocket": "/responses",
    "alist_input_items": "/responses/{response_id}/input_items",
    "aimage_edit": "/images/edits",
    "acancel_responses": "/responses/{response_id}/cancel",
    "acompact_responses": "/responses/compact",
    "aocr": "/ocr",
    "asearch": "/search",
    "avideo_generation": "/videos",
    "avideo_list": "/videos",
    "avideo_status": "/videos/{video_id}",
    "avideo_content": "/videos/{video_id}/content",
    "avideo_remix": "/videos/{video_id}/remix",
    "avideo_create_character": "/videos/characters",
    "avideo_get_character": "/videos/characters/{character_id}",
    "avideo_edit": "/videos/edits",
    "avideo_extension": "/videos/extensions",
    "acreate_realtime_client_secret": "/realtime/client_secrets",
    "arealtime_calls": "/realtime/calls",
    "acreate_realtime_transcription_session": "/realtime/transcription_sessions",
    "acreate_container": "/containers",
    "alist_containers": "/containers",
    "aretrieve_container": "/containers/{container_id}",
    "adelete_container": "/containers/{container_id}",
    # Auto-generated container file routes
    "aupload_container_file": "/containers/{container_id}/files",
    "alist_container_files": "/containers/{container_id}/files",
    "aretrieve_container_file": "/containers/{container_id}/files/{file_id}",
    "adelete_container_file": "/containers/{container_id}/files/{file_id}",
    "aretrieve_container_file_content": "/containers/{container_id}/files/{file_id}/content",
    "acreate_skill": "/skills",
    "alist_skills": "/skills",
    "aget_skill": "/skills/{skill_id}",
    "adelete_skill": "/skills/{skill_id}",
    "aingest": "/rag/ingest",
    # Google Interactions API routes
    "acreate_interaction": "/interactions",
    "aget_interaction": "/interactions/{interaction_id}",
    "adelete_interaction": "/interactions/{interaction_id}",
    "acancel_interaction": "/interactions/{interaction_id}/cancel",
    # Google Managed Agents API routes
    "acreate_agent": "/v1beta/agents",
    "alist_agents": "/v1beta/agents",
    "aget_agent": "/v1beta/agents/{name}",
    "adelete_agent": "/v1beta/agents/{name}",
    "alist_agent_versions": "/v1beta/agents/{name}/versions",
    # OpenAI Evals API routes
    "acreate_eval": "/evals",
    "alist_evals": "/evals",
    "aget_eval": "/evals/{eval_id}",
    "aupdate_eval": "/evals/{eval_id}",
    "adelete_eval": "/evals/{eval_id}",
    "acancel_eval": "/evals/{eval_id}/cancel",
    # OpenAI Evals Runs API routes
    "acreate_run": "/evals/{eval_id}/runs",
    "alist_runs": "/evals/{eval_id}/runs",
    "aget_run": "/evals/{eval_id}/runs/{run_id}",
    "acancel_run": "/evals/{eval_id}/runs/{run_id}/cancel",
    "adelete_run": "/evals/{eval_id}/runs/{run_id}",
    "acreate_batch": "/batches",
}


class ProxyModelNotFoundError(HTTPException):
    def __init__(self, route: str, model_name: str, retryable_with_model_read_through: bool = True):
        self.retryable_with_model_read_through: Final = retryable_with_model_read_through
        detail: Final = {
            "error": f"{route}: Invalid model name passed in model={model_name}. Call `/v1/models` to view available models for your key."
        }
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


REQUIRED_BODY_PARAMS_BY_ROUTE: Final[Mapping[str, tuple[str, ...]]] = {
    "acompletion": ("messages",),
    "aembedding": ("input",),
    "acreate_batch": ("input_file_id", "endpoint", "completion_window"),
}


class ProxyMissingRequiredParamError(ProxyException):
    def __init__(self, route: str, param: str):
        super().__init__(
            message=f"{route}: Missing required parameter: '{param}'.",
            type="invalid_request_error",
            param=param,
            code=status.HTTP_400_BAD_REQUEST,
        )


def raise_if_required_body_param_missing(route_type: str, data: Mapping[str, object]) -> None:
    missing_param: Final = next(
        (param for param in REQUIRED_BODY_PARAMS_BY_ROUTE.get(route_type, ()) if data.get(param) is None),
        None,
    )
    if missing_param is None:
        return
    raise ProxyMissingRequiredParamError(
        route=ROUTE_ENDPOINT_MAPPING.get(route_type, route_type),
        param=missing_param,
    )


class MockTestingParamsDisabledError(HTTPException):
    def __init__(self, params: tuple[str, ...]):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={  # mutable-ok: HTTPException.detail has no immutable form; same shape as the sibling errors here
                "error": (
                    f"Mock testing request params are disabled on this proxy: {', '.join(params)}. "
                    f"An admin can enable them by setting `general_settings.{MOCK_TESTING_CONFIG_KEY}: true` "
                    "in config.yaml. This setting cannot be changed from the Admin UI or the API."
                )
            },
        )


def raise_if_mock_testing_params_disallowed(data: Mapping[str, object], *, allowed: bool) -> None:
    """Reject client-supplied mock testing params unless an admin opted in.

    Rejecting (rather than silently dropping) keeps a request that asked for a
    synthetic failure from returning a normal success, which reads as a passing
    fallback test that never ran.
    """
    if allowed:
        return
    present: Final = tuple(name for name in GATED_MOCK_PARAM_NAMES if name in data)
    if present:
        raise MockTestingParamsDisabledError(params=present)


def mock_testing_params_allowed() -> bool:
    """Read the opt-in from the running proxy's ``general_settings``."""
    from litellm.proxy import proxy_server

    return proxy_server.general_settings.get(MOCK_TESTING_CONFIG_KEY, False) is True


def get_team_id_from_data(data: dict) -> str | None:
    """
    Get the team id from the data's metadata or litellm_metadata params.
    """
    if "metadata" in data and data["metadata"] is not None and "user_api_key_team_id" in data["metadata"]:
        return data["metadata"].get("user_api_key_team_id")
    elif (
        "litellm_metadata" in data
        and data["litellm_metadata"] is not None
        and "user_api_key_team_id" in data["litellm_metadata"]
    ):
        return data["litellm_metadata"].get("user_api_key_team_id")
    return None


_shared_session_lock: asyncio.Lock | None = None


def _get_shared_session_lock() -> asyncio.Lock:
    """Lazily create the shared session lock (must be called within a running event loop).

    WARNING: Do not reset _shared_session_lock to None while any coroutine may be
    executing the session-recovery path; doing so breaks the double-checked locking
    guarantee and can cause duplicate session creation.
    """
    global _shared_session_lock
    if _shared_session_lock is None:
        _shared_session_lock = asyncio.Lock()
    return _shared_session_lock


async def add_shared_session_to_data(data: dict) -> None:
    """
    Add shared aiohttp session for connection reuse (prevents cold starts).
    If the session was closed (e.g. due to network interruption or idle timeout),
    automatically recreates it so connection pooling is restored.
    Uses an asyncio.Lock to prevent race conditions where multiple concurrent
    requests could each create a new session, leaking intermediate ones.
    Silently continues without session reuse if import fails or session is unavailable.

    Args:
        data: Dictionary to add the shared session to
    """
    try:
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy import proxy_server

        session = proxy_server.shared_aiohttp_session

        if session is not None and not session.closed:
            data["shared_session"] = session
            verbose_proxy_logger.info("SESSION REUSE: Attached shared aiohttp session to request (ID: %s)", id(session))
        elif session is not None and session.closed:
            # Session was created at startup but has since closed — recreate it
            # Use lock to prevent concurrent recreation (avoids session/connector leak)
            lock: Final = _get_shared_session_lock()
            async with lock:
                # Double-check under lock — another coroutine may have already recreated it
                session = proxy_server.shared_aiohttp_session
                if session is not None and not session.closed:
                    data["shared_session"] = session
                    return

                # session could be None here (if another coroutine set it to None)
                # or closed — either way we need to recreate
                if session is not None:
                    verbose_proxy_logger.warning(
                        "SESSION REUSE: Shared aiohttp session is closed (ID: %s), recreating...", id(session)
                    )
                else:
                    verbose_proxy_logger.warning(
                        "SESSION REUSE: Shared aiohttp session is None after re-check, recreating..."
                    )
                try:
                    new_session = await proxy_server._initialize_shared_aiohttp_session()
                except Exception:
                    verbose_proxy_logger.exception("SESSION REUSE: Exception during shared session recreation")
                    new_session = None
                if new_session is not None:
                    proxy_server.shared_aiohttp_session = new_session
                    data["shared_session"] = new_session
                else:
                    verbose_proxy_logger.info(
                        "SESSION REUSE: Failed to recreate shared session, continuing without session reuse"
                    )
        else:
            verbose_proxy_logger.info("SESSION REUSE: No shared session available for this request")
    except Exception:
        # Continue without session reuse — this outer handler covers import failures
        # and other unexpected errors to avoid breaking the request path.
        # Inner recovery logic has its own specific exception handling.
        try:
            from litellm._logging import verbose_proxy_logger

            verbose_proxy_logger.debug(
                "SESSION REUSE: Unexpected error in session setup, continuing without reuse",
                exc_info=True,
            )
        except Exception:
            pass


RouteType = Literal[
    "acompletion",
    "atext_completion",
    "aembedding",
    "aimage_generation",
    "aspeech",
    "atranscription",
    "amoderation",
    "arerank",
    "aresponses",
    "aget_responses",
    "adelete_responses",
    "acancel_responses",
    "acompact_responses",
    "acreate_response_reply",
    "alist_input_items",
    "_arealtime",  # private function for realtime API
    "acreate_realtime_client_secret",
    "arealtime_calls",
    "acreate_realtime_transcription_session",
    "_aresponses_websocket",  # private function for responses WebSocket mode
    "aimage_edit",
    "agenerate_content",
    "agenerate_content_stream",
    "allm_passthrough_route",
    "acreate_batch",
    "aretrieve_batch",
    "alist_batches",
    "afile_content",
    "afile_retrieve",
    "acreate_fine_tuning_job",
    "acancel_fine_tuning_job",
    "alist_fine_tuning_jobs",
    "aretrieve_fine_tuning_job",
    "avector_store_search",
    "avector_store_create",
    "avector_store_retrieve",
    "avector_store_list",
    "avector_store_update",
    "avector_store_delete",
    "avector_store_file_create",
    "avector_store_file_list",
    "avector_store_file_retrieve",
    "avector_store_file_content",
    "avector_store_file_update",
    "avector_store_file_delete",
    "aocr",
    "asearch",
    "avideo_generation",
    "avideo_list",
    "avideo_status",
    "avideo_content",
    "avideo_remix",
    "avideo_create_character",
    "avideo_get_character",
    "avideo_edit",
    "avideo_extension",
    "acreate_container",
    "alist_containers",
    "aretrieve_container",
    "adelete_container",
    "aupload_container_file",
    "alist_container_files",
    "aretrieve_container_file",
    "adelete_container_file",
    "aretrieve_container_file_content",
    "acreate_skill",
    "alist_skills",
    "aget_skill",
    "adelete_skill",
    "aingest",
    "anthropic_messages",
    "acreate_interaction",
    "aget_interaction",
    "adelete_interaction",
    "acancel_interaction",
    "acreate_agent",
    "alist_agents",
    "aget_agent",
    "adelete_agent",
    "alist_agent_versions",
    "asend_message",
    "call_mcp_tool",
    "acancel_batch",
    "afile_delete",
    "acreate_eval",
    "alist_evals",
    "aget_eval",
    "aupdate_eval",
    "adelete_eval",
    "acancel_eval",
    "acreate_run",
    "alist_runs",
    "aget_run",
    "acancel_run",
    "adelete_run",
]


async def route_request(
    data: dict,
    llm_router: LitellmRouter | None,
    user_model: str | None,
    route_type: RouteType,
    user_api_key_dict: UserAPIKeyAuth | None = None,
):
    """
    Common helper to route the request
    """
    try:
        return await _route_request_single_attempt(
            data=data,
            llm_router=llm_router,
            user_model=user_model,
            route_type=route_type,
            user_api_key_dict=user_api_key_dict,
        )
    except ProxyModelNotFoundError as e:
        requested_model: Final = data.get("model", "")
        if not e.retryable_with_model_read_through or not isinstance(requested_model, str) or not requested_model:
            raise
        from litellm.proxy import proxy_server
        from litellm.proxy.common_utils.registry_read_through import (
            model_registry_read_through,
        )

        if not await model_registry_read_through.attempt(requested_model):
            raise
        return await _route_request_single_attempt(
            data=data,
            llm_router=proxy_server.llm_router,
            user_model=user_model,
            route_type=route_type,
            user_api_key_dict=user_api_key_dict,
        )


async def _route_request_single_attempt(  # noqa: ANN202  # returns unawaited provider coroutines; the inferred union keeps route_request's callers typed
    data: dict,  # mutable-ok: request body is the proxy-wide mutable dict contract shared with route_request
    llm_router: LitellmRouter | None,
    user_model: str | None,
    route_type: RouteType,
    user_api_key_dict: UserAPIKeyAuth | None = None,
):
    raise_if_required_body_param_missing(route_type=route_type, data=data)

    await add_shared_session_to_data(data)

    raise_if_mock_testing_params_disallowed(data, allowed=mock_testing_params_allowed())

    data.pop("enable_tag_filtering", None)

    team_id: Final = get_team_id_from_data(data)
    router_model_names: Final = llm_router.model_names if llm_router is not None else []
    is_proxy_admin_without_team: Final = team_id is None and _is_proxy_admin_request(data)

    # Preprocess Google GenAI generate content requests
    if route_type in ["agenerate_content", "agenerate_content_stream"]:
        # Map generationConfig to config parameter for Google GenAI compatibility
        if "generationConfig" in data and "config" not in data:
            data["config"] = data.pop("generationConfig")
    if "api_key" in data or "api_base" in data:
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**data)
        else:
            return getattr(litellm, f"{route_type}")(**data)

    elif (
        route_type == "acompletion"
        and data.get("model", "") is not None
        and "," in data.get("model", "")
        and llm_router is not None
    ):
        # Handle batch completions with comma-separated models BEFORE user_config check
        # This ensures batch completion logic is applied even when user_config is set
        if data.get("fastest_response", False):
            return llm_router.abatch_completion_fastest_response(**data)
        else:
            models: Final = [model.strip() for model in data.pop("model").split(",")]
            return llm_router.abatch_completion(models=models, **data)

    elif "user_config" in data:
        return _route_user_config_request(data, route_type)

    elif "router_settings_override" in data:
        # Apply per-request router settings overrides from key/team config
        # Instead of creating a new Router (expensive), merge settings into kwargs
        # The Router already supports per-request overrides for these settings
        override_settings: Final = data.pop("router_settings_override")

        # Settings that the Router accepts as per-request kwargs
        # These override the global router settings for this specific request
        per_request_settings: Final = [
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
            "num_retries",
            "timeout",
            "model_group_retry_policy",
            "routing_strategy",
            "enable_tag_filtering",
        ]

        # Merge override settings into data (only if not already set in request)
        for key in per_request_settings:
            if key in override_settings and key not in data:
                data[key] = override_settings[key]
        # Use main router with overridden kwargs
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**data)
        else:
            return getattr(litellm, f"{route_type}")(**data)
    elif llm_router is not None:
        await _raise_if_model_fully_blocked(
            llm_router=llm_router,
            model_name=data.get("model"),
            team_id=team_id,
            request_data=data,
            allow_router_fallback=route_type not in EVAL_ROUTE_TYPES,
        )
        # Evals API: always route to litellm directly (not through router)
        # But extract model credentials if a model is provided
        if route_type in EVAL_ROUTE_TYPES:
            # If a model is provided, get its credentials from the router
            model: Final = data.get("model")
            if model and llm_router:
                try:
                    # Try to get deployment credentials for this model
                    deployment_creds = llm_router.get_deployment_credentials(model_id=model)
                    if not deployment_creds:
                        # Try by model group name
                        deployment: Final = llm_router.get_deployment_by_model_group_name(model_group_name=model)
                        if (
                            deployment
                            and deployment.litellm_params
                            and not llm_router._is_deployment_blocked(deployment)
                        ):
                            deployment_creds = deployment.litellm_params.model_dump(exclude_none=True)

                    # If we found credentials, merge them into data (but don't override user-provided values)
                    if deployment_creds:
                        data.update(deployment_creds)
                except Exception:
                    # If we can't get deployment creds, continue without them
                    pass

            return getattr(litellm, f"{route_type}")(**data)
        # Skip model-based routing for container operations
        if route_type in [
            "acreate_container",
            "alist_containers",
            "aretrieve_container",
            "adelete_container",
            "aupload_container_file",
            "alist_container_files",
            "aretrieve_container_file",
            "adelete_container_file",
            "aretrieve_container_file_content",
        ]:
            return getattr(llm_router, f"{route_type}")(**data)
        # Interactions API: create with agent, get/delete/cancel don't need model routing
        if route_type in [
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
        ]:
            return getattr(llm_router, f"{route_type}")(**data)
        # Managed Agents API: these don't need model routing
        if route_type in [
            "acreate_agent",
            "alist_agents",
            "aget_agent",
            "adelete_agent",
            "alist_agent_versions",
        ]:
            return getattr(llm_router, f"{route_type}")(**data)
        if route_type in [
            "avideo_list",
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "avideo_create_character",
            "avideo_get_character",
            "avideo_edit",
            "avideo_extension",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_delete",
            "acreate_skill",
            "alist_skills",
            "aget_skill",
            "adelete_skill",
            "aingest",
        ] and (data.get("model") is None or data.get("model") == ""):
            # These endpoints don't need a model, use custom_llm_provider directly
            return getattr(litellm, f"{route_type}")(**data)

        team_model_name: Final = llm_router.map_team_model(data["model"], team_id) if team_id is not None else None
        if team_model_name is not None:
            data["model"] = team_model_name
            return getattr(llm_router, f"{route_type}")(**data)

        elif (
            is_proxy_admin_without_team
            and data["model"] not in router_model_names
            and data["model"] in llm_router.team_public_model_names
        ) or llm_router.is_recognized_model(data["model"]):
            return getattr(llm_router, f"{route_type}")(**data)

        elif data["model"] not in router_model_names:
            # Check wildcards before checking deployment_names
            # Priority: 1. Exact model_name match, 2. Wildcard match, 3. deployment_names match
            if llm_router.router_general_settings.pass_through_all_models:
                return getattr(litellm, f"{route_type}")(**data)
            elif llm_router.default_deployment is not None or len(llm_router.pattern_router.patterns) > 0:
                return getattr(llm_router, f"{route_type}")(**data)
            elif data["model"] in llm_router.deployment_names:
                # Only match deployment_names if no wildcard matched
                return getattr(llm_router, f"{route_type}")(**data, specific_deployment=True)
            elif route_type in [
                "amoderation",
                "aget_responses",
                "adelete_responses",
                "acancel_responses",
                "alist_input_items",
                "avector_store_create",
                "avector_store_search",
                "avector_store_retrieve",
                "avector_store_list",
                "avector_store_update",
                "avector_store_delete",
                "avector_store_file_create",
                "avector_store_file_list",
                "avector_store_file_retrieve",
                "avector_store_file_content",
                "avector_store_file_update",
                "avector_store_file_delete",
                "asearch",
                "acreate_container",
                "alist_containers",
                "aretrieve_container",
                "adelete_container",
                "aupload_container_file",
                "alist_container_files",
                "aretrieve_container_file",
                "adelete_container_file",
                "aretrieve_container_file_content",
            ]:
                # These endpoints can work with or without model parameter
                return getattr(llm_router, f"{route_type}")(**data)
            elif route_type in [
                "avideo_status",
                "avideo_content",
                "avideo_remix",
                "avideo_create_character",
                "avideo_get_character",
                "avideo_edit",
                "avideo_extension",
            ]:
                # Video endpoints: If model is provided (e.g., from decoded video_id or target_model_names),
                # try router first to allow for multi-deployment load balancing
                try:
                    return getattr(llm_router, f"{route_type}")(**data)
                except Exception:
                    # If router fails (e.g., model not found in router), fall back to direct call
                    return getattr(litellm, f"{route_type}")(**data)
            elif _is_a2a_agent_model(data.get("model", "")):
                from litellm.proxy.agent_endpoints.a2a_routing import (
                    route_a2a_agent_request,
                )

                result: Final = await route_a2a_agent_request(data, route_type, user_api_key_dict=user_api_key_dict)
                if result is not None:
                    return result
                # Fall through to raise exception below if result is None

    elif user_model is not None or route_type == "allm_passthrough_route":
        return getattr(litellm, f"{route_type}")(**data)

    # if no route found then it's a bad request
    route_name: Final = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
    raise ProxyModelNotFoundError(
        route=route_name,
        model_name=data.get("model", ""),
    )
