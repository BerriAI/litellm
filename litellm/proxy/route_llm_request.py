import asyncio
import os
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

from fastapi import HTTPException, status

import litellm
from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


_USER_CONFIG_ROUTER_CACHE_MAX_SIZE = max(
    1, int(os.getenv("LITELLM_USER_CONFIG_ROUTER_CACHE_MAX_SIZE", "64"))
)
_USER_CONFIG_ROUTER_CACHE_TTL_SECONDS = max(
    1, int(os.getenv("LITELLM_USER_CONFIG_ROUTER_CACHE_TTL_SECONDS", "300"))
)
_USER_CONFIG_ROUTER_CACHE: "OrderedDict[Tuple[Any, ...], Tuple[LitellmRouter, float]]" = (
    OrderedDict()
)
_USER_CONFIG_ROUTER_CACHE_LOCK = threading.Lock()


def _discard_router_safely(router: LitellmRouter) -> None:
    try:
        router.discard()
    except Exception:
        pass


def _freeze_user_config_cache_key(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_user_config_cache_key(val))
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, list):
        return tuple(_freeze_user_config_cache_key(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_user_config_cache_key(item) for item in value)
    if isinstance(value, set):
        frozen_items = [_freeze_user_config_cache_key(item) for item in value]
        return tuple(
            sorted(
                frozen_items,
                key=lambda item: (type(item).__name__, repr(item)),
            )
        )
    return value


def _get_user_config_router_cache_key(filtered_config: dict) -> Tuple[Any, ...]:
    frozen_config = _freeze_user_config_cache_key(filtered_config)
    if isinstance(frozen_config, tuple):
        return frozen_config
    return (frozen_config,)


def _prune_expired_user_config_routers(now: float) -> None:
    expired_keys = [
        key
        for key, (_, expires_at) in _USER_CONFIG_ROUTER_CACHE.items()
        if expires_at <= now
    ]
    for key in expired_keys:
        _USER_CONFIG_ROUTER_CACHE.pop(key)


def _clear_user_config_router_cache() -> None:
    with _USER_CONFIG_ROUTER_CACHE_LOCK:
        while _USER_CONFIG_ROUTER_CACHE:
            _USER_CONFIG_ROUTER_CACHE.popitem(last=False)


def _get_or_create_user_config_router(filtered_config: dict) -> LitellmRouter:
    cache_key = _get_user_config_router_cache_key(filtered_config)
    with _USER_CONFIG_ROUTER_CACHE_LOCK:
        now = time.monotonic()
        _prune_expired_user_config_routers(now)
        cached_entry = _USER_CONFIG_ROUTER_CACHE.get(cache_key)
        if cached_entry is not None:
            router, expires_at = cached_entry
            if expires_at > now:
                _USER_CONFIG_ROUTER_CACHE[cache_key] = (
                    router,
                    now + _USER_CONFIG_ROUTER_CACHE_TTL_SECONDS,
                )
                _USER_CONFIG_ROUTER_CACHE.move_to_end(cache_key)
                return router
            _USER_CONFIG_ROUTER_CACHE.pop(cache_key, None)

    new_router = litellm.Router(**filtered_config)
    inserted_into_cache = False
    evicted_count = 0
    try:
        with _USER_CONFIG_ROUTER_CACHE_LOCK:
            now = time.monotonic()
            _prune_expired_user_config_routers(now)
            cached_entry = _USER_CONFIG_ROUTER_CACHE.get(cache_key)
            if cached_entry is not None:
                router, expires_at = cached_entry
                if expires_at > now:
                    _USER_CONFIG_ROUTER_CACHE[cache_key] = (
                        router,
                        now + _USER_CONFIG_ROUTER_CACHE_TTL_SECONDS,
                    )
                    _USER_CONFIG_ROUTER_CACHE.move_to_end(cache_key)
                    _discard_router_safely(new_router)
                    return router
                _USER_CONFIG_ROUTER_CACHE.pop(cache_key, None)

            _USER_CONFIG_ROUTER_CACHE[cache_key] = (
                new_router,
                now + _USER_CONFIG_ROUTER_CACHE_TTL_SECONDS,
            )
            inserted_into_cache = True
            _USER_CONFIG_ROUTER_CACHE.move_to_end(cache_key)

            while len(_USER_CONFIG_ROUTER_CACHE) > _USER_CONFIG_ROUTER_CACHE_MAX_SIZE:
                _USER_CONFIG_ROUTER_CACHE.popitem(last=False)
                evicted_count += 1
    except Exception:
        with _USER_CONFIG_ROUTER_CACHE_LOCK:
            if inserted_into_cache:
                cached_entry = _USER_CONFIG_ROUTER_CACHE.get(cache_key)
                if cached_entry is not None and cached_entry[0] is new_router:
                    _USER_CONFIG_ROUTER_CACHE.pop(cache_key, None)
        _discard_router_safely(new_router)
        raise
    if evicted_count > 0:
        verbose_proxy_logger.warning(
            "user_config Router cache full (evicted %d entries). "
            "Increase LITELLM_USER_CONFIG_ROUTER_CACHE_MAX_SIZE if this is frequent.",
            evicted_count,
        )
    return new_router


async def _route_user_config_request(data: dict, user_config: dict, route_type: str):
    """
    Route a request using the user-provided router config.

    This is `async def` and returns the final response when awaited.
    `route_request()` deliberately returns this coroutine without awaiting it,
    preserving the existing double-await call-site pattern:
    `llm_call = await route_request(...); response = await llm_call`.
    """
    # Filter router_config to only include valid Router.__init__ arguments
    # This prevents TypeError when invalid parameters are stored in the database
    valid_args = litellm.Router.get_valid_args()
    filtered_config = {k: v for k, v in user_config.items() if k in valid_args}

    user_router = await asyncio.to_thread(
        _get_or_create_user_config_router, filtered_config
    )

    # Handle batch completions with comma-separated models for user-provided routers
    if (
        route_type == "acompletion"
        and data.get("model", "") is not None
        and "," in data.get("model", "")
    ):
        if data.get("fastest_response", False):
            models = [m.strip() for m in str(data.get("model", "")).split(",")]
            kwargs = dict(_kwargs_for_llm(data))
            kwargs.pop("model", None)
            kwargs.pop("fastest_response", None)
            return await user_router.abatch_completion_fastest_response(
                models=models, **kwargs
            )

        models = [m.strip() for m in str(data.get("model", "")).split(",")]
        kwargs = dict(_kwargs_for_llm(data))
        kwargs.pop("model", None)
        kwargs.pop("fastest_response", None)
        return await user_router.abatch_completion(models=models, **kwargs)

    return await getattr(user_router, f"{route_type}")(**_kwargs_for_llm(data))


def _is_a2a_agent_model(model_name: Any) -> bool:
    """Check if the model name is for an A2A agent (a2a/ prefix)."""
    return isinstance(model_name, str) and model_name.startswith("a2a/")


ROUTE_ENDPOINT_MAPPING = {
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
}


class ProxyModelNotFoundError(HTTPException):
    def __init__(self, route: str, model_name: str):
        detail = {
            "error": f"{route}: Invalid model name passed in model={model_name}. Call `/v1/models` to view available models for your key."
        }
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def get_team_id_from_data(data: dict) -> Optional[str]:
    """
    Get the team id from the data's metadata or litellm_metadata params.
    """
    if (
        "metadata" in data
        and data["metadata"] is not None
        and "user_api_key_team_id" in data["metadata"]
    ):
        return data["metadata"].get("user_api_key_team_id")
    elif (
        "litellm_metadata" in data
        and data["litellm_metadata"] is not None
        and "user_api_key_team_id" in data["litellm_metadata"]
    ):
        return data["litellm_metadata"].get("user_api_key_team_id")
    return None


_shared_session_lock: Optional[asyncio.Lock] = None


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


# Keys that are only for proxy/guardrail use and must not be sent to the LLM API
_INTERNAL_REQUEST_KEYS = frozenset(
    {"_presidio_pii_tokens", "fastest_response", "user_config"}
)


def _kwargs_for_llm(data: dict) -> dict:
    """
    Strip internal proxy keys so they are not sent to the LLM provider.

    NOTE: Returns `data` by reference on the fast path when no internal keys are
    present. Callers that need to mutate the result (for example `pop("model")`)
    must wrap it in `dict(...)` first.
    """
    if _INTERNAL_REQUEST_KEYS.isdisjoint(data):
        return data
    return {k: v for k, v in data.items() if k not in _INTERNAL_REQUEST_KEYS}


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
        import litellm.proxy.proxy_server as proxy_server
        from litellm._logging import verbose_proxy_logger

        session = proxy_server.shared_aiohttp_session

        if session is not None and not session.closed:
            data["shared_session"] = session
            verbose_proxy_logger.info(
                f"SESSION REUSE: Attached shared aiohttp session to request (ID: {id(session)})"
            )
        elif session is not None and session.closed:
            # Session was created at startup but has since closed — recreate it
            # Use lock to prevent concurrent recreation (avoids session/connector leak)
            lock = _get_shared_session_lock()
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
                        f"SESSION REUSE: Shared aiohttp session is closed (ID: {id(session)}), recreating..."
                    )
                else:
                    verbose_proxy_logger.warning(
                        "SESSION REUSE: Shared aiohttp session is None after re-check, recreating..."
                    )
                try:
                    new_session = (
                        await proxy_server._initialize_shared_aiohttp_session()
                    )
                except Exception:
                    verbose_proxy_logger.exception(
                        "SESSION REUSE: Exception during shared session recreation"
                    )
                    new_session = None
                if new_session is not None:
                    proxy_server.shared_aiohttp_session = new_session
                    data["shared_session"] = new_session
                else:
                    verbose_proxy_logger.info(
                        "SESSION REUSE: Failed to recreate shared session, continuing without session reuse"
                    )
        else:
            verbose_proxy_logger.info(
                "SESSION REUSE: No shared session available for this request"
            )
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


async def route_request(  # noqa: PLR0915 - Complex routing function, refactoring tracked separately
    data: dict,
    llm_router: Optional[LitellmRouter],
    user_model: Optional[str],
    route_type: Literal[
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
    ],
):
    """
    Common helper to route the request
    """
    await add_shared_session_to_data(data)

    team_id = get_team_id_from_data(data)
    router_model_names = llm_router.model_names if llm_router is not None else []

    # Preprocess Google GenAI generate content requests
    if route_type in ["agenerate_content", "agenerate_content_stream"]:
        # Map generationConfig to config parameter for Google GenAI compatibility
        if "generationConfig" in data and "config" not in data:
            data["config"] = data.pop("generationConfig")
    if "api_key" in data or "api_base" in data:
        kwargs = _kwargs_for_llm(data)
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**kwargs)
        else:
            return getattr(litellm, f"{route_type}")(**kwargs)

    elif "user_config" in data:
        # user_config stays authoritative for routing, including comma-separated
        # batch model requests. Keep this check ahead of the main-router batch
        # branch so user-scoped router resolution is not bypassed.
        user_config = data.pop("user_config")
        return _route_user_config_request(data, user_config, route_type)

    elif (
        route_type == "acompletion"
        and data.get("model", "") is not None
        and "," in data.get("model", "")
        and llm_router is not None
    ):
        if data.get("fastest_response", False):
            models = [model.strip() for model in str(data.get("model", "")).split(",")]
            kwargs = dict(_kwargs_for_llm(data))
            kwargs.pop("model", None)
            kwargs.pop("fastest_response", None)
            return llm_router.abatch_completion_fastest_response(
                models=models, **kwargs
            )
        else:
            models = [model.strip() for model in str(data.get("model", "")).split(",")]
            kwargs = dict(_kwargs_for_llm(data))
            kwargs.pop("model", None)
            kwargs.pop("fastest_response", None)
            return llm_router.abatch_completion(models=models, **kwargs)

    elif "router_settings_override" in data:
        # Apply per-request router settings overrides from key/team config
        # Instead of creating a new Router (expensive), merge settings into kwargs
        # The Router already supports per-request overrides for these settings
        override_settings = data.pop("router_settings_override")

        # Settings that the Router accepts as per-request kwargs
        # These override the global router settings for this specific request
        per_request_settings = [
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
            "num_retries",
            "timeout",
            "model_group_retry_policy",
        ]

        # Merge override settings into data (only if not already set in request)
        for key in per_request_settings:
            if key in override_settings and key not in data:
                data[key] = override_settings[key]

        # Use main router with overridden kwargs
        kwargs = _kwargs_for_llm(data)
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**kwargs)
        else:
            return getattr(litellm, f"{route_type}")(**kwargs)

    elif llm_router is not None:
        # Evals API: always route to litellm directly (not through router)
        # But extract model credentials if a model is provided
        if route_type in [
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
            "acreate_realtime_client_secret",
            "arealtime_calls",
        ]:
            # If a model is provided, get its credentials from the router
            model = data.get("model")
            if model and llm_router:
                try:
                    # Try to get deployment credentials for this model
                    deployment_creds = llm_router.get_deployment_credentials(
                        model_id=model
                    )
                    if not deployment_creds:
                        # Try by model group name
                        deployment = llm_router.get_deployment_by_model_group_name(
                            model_group_name=model
                        )
                        if deployment and deployment.litellm_params:
                            deployment_creds = deployment.litellm_params.model_dump(
                                exclude_none=True
                            )

                    # If we found credentials, merge them into data (but don't override user-provided values)
                    if deployment_creds:
                        data.update(deployment_creds)
                except Exception:
                    # If we can't get deployment creds, continue without them
                    pass

            return getattr(litellm, f"{route_type}")(**_kwargs_for_llm(data))
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
            return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))
        # Interactions API: create with agent, get/delete/cancel don't need model routing
        if route_type in [
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
        ]:
            return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))
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
            return getattr(litellm, f"{route_type}")(**_kwargs_for_llm(data))

        team_model_name = (
            llm_router.map_team_model(data["model"], team_id)
            if team_id is not None
            else None
        )
        if team_model_name is not None:
            data["model"] = team_model_name
            return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))

        elif data["model"] in router_model_names or llm_router.has_model_id(
            data["model"]
        ):
            return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))

        elif (
            llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):
            return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))

        elif data["model"] not in router_model_names:
            # Check wildcards before checking deployment_names
            # Priority: 1. Exact model_name match, 2. Wildcard match, 3. deployment_names match
            if llm_router.router_general_settings.pass_through_all_models:
                return getattr(litellm, f"{route_type}")(**_kwargs_for_llm(data))
            elif (
                llm_router.default_deployment is not None
                or len(llm_router.pattern_router.patterns) > 0
            ):
                return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))
            elif data["model"] in llm_router.deployment_names:
                # Only match deployment_names if no wildcard matched
                return getattr(llm_router, f"{route_type}")(
                    **_kwargs_for_llm(data), specific_deployment=True
                )
            elif route_type in [
                "amoderation",
                "aget_responses",
                "adelete_responses",
                "acancel_responses",
                "alist_input_items",
                "avector_store_create",
                "avector_store_search",
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
                return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))
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
                    return getattr(llm_router, f"{route_type}")(**_kwargs_for_llm(data))
                except Exception:
                    # If router fails (e.g., model not found in router), fall back to direct call
                    return getattr(litellm, f"{route_type}")(**_kwargs_for_llm(data))
            elif _is_a2a_agent_model(data.get("model", "")):
                from litellm.proxy.agent_endpoints.a2a_routing import (
                    route_a2a_agent_request,
                )

                result = route_a2a_agent_request(data, route_type)
                if result is not None:
                    return result
                # Fall through to raise exception below if result is None

    elif user_model is not None:
        return getattr(litellm, f"{route_type}")(**_kwargs_for_llm(data))
    elif route_type == "allm_passthrough_route":
        return getattr(litellm, f"{route_type}")(**_kwargs_for_llm(data))

    # if no route found then it's a bad request
    route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
    raise ProxyModelNotFoundError(
        route=route_name,
        model_name=data.get("model", ""),
    )
