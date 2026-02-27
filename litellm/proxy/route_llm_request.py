from typing import TYPE_CHECKING, Any, Literal, Optional

from fastapi import HTTPException, status

import litellm

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def _route_user_config_request(data: dict, route_type: str):
    """Route a request using the user-provided router config."""
    router_config = data.pop("user_config")

    # Filter router_config to only include valid Router.__init__ arguments
    # This prevents TypeError when invalid parameters are stored in the database
    valid_args = litellm.Router.get_valid_args()
    filtered_config = {k: v for k, v in router_config.items() if k in valid_args}

    user_router = litellm.Router(**filtered_config)
    ret_val = getattr(user_router, f"{route_type}")(**data)
    user_router.discard()
    return ret_val


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


def add_shared_session_to_data(data: dict) -> None:
    """
    Add shared aiohttp session for connection reuse (prevents cold starts).
    Silently continues without session reuse if import fails or session is unavailable.

    Args:
        data: Dictionary to add the shared session to
    """
    try:
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy.proxy_server import shared_aiohttp_session

        if shared_aiohttp_session is not None and not shared_aiohttp_session.closed:
            data["shared_session"] = shared_aiohttp_session
            verbose_proxy_logger.info(
                f"SESSION REUSE: Attached shared aiohttp session to request (ID: {id(shared_aiohttp_session)})"
            )
        else:
            verbose_proxy_logger.info(
                "SESSION REUSE: No shared session available for this request"
            )
    except Exception:
        # Silently continue without session reuse if import fails or session unavailable
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
        "aimage_edit",
        "agenerate_content",
        "agenerate_content_stream",
        "allm_passthrough_route",
        "avector_store_search",
        "avector_store_create",
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
    add_shared_session_to_data(data)

    team_id = get_team_id_from_data(data)
    router_model_names = llm_router.model_names if llm_router is not None else []

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
            models = [model.strip() for model in data.pop("model").split(",")]
            return llm_router.abatch_completion(models=models, **data)

    elif "user_config" in data:
        return _route_user_config_request(data, route_type)

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
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**data)
        else:
            return getattr(litellm, f"{route_type}")(**data)
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
        ]:
            # If a model is provided, get its credentials from the router
            model = data.get("model")
            if model and llm_router:
                try:
                    # Try to get deployment credentials for this model
                    deployment_creds = llm_router.get_deployment_credentials(model_id=model)
                    if not deployment_creds:
                        # Try by model group name
                        deployment = llm_router.get_deployment_by_model_group_name(model_group_name=model)
                        if deployment and deployment.litellm_params:
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
        if route_type in [
            "avideo_list",
            "avideo_status",
            "avideo_content",
            "avideo_remix",
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

        team_model_name = (
            llm_router.map_team_model(data["model"], team_id)
            if team_id is not None
            else None
        )
        if team_model_name is not None:
            data["model"] = team_model_name
            return getattr(llm_router, f"{route_type}")(**data)

        elif data["model"] in router_model_names or llm_router.has_model_id(
            data["model"]
        ):
            return getattr(llm_router, f"{route_type}")(**data)

        elif (
            llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):
            return getattr(llm_router, f"{route_type}")(**data)

        elif data["model"] not in router_model_names:
            # Check wildcards before checking deployment_names
            # Priority: 1. Exact model_name match, 2. Wildcard match, 3. deployment_names match
            if llm_router.router_general_settings.pass_through_all_models:
                return getattr(litellm, f"{route_type}")(**data)
            elif (
                llm_router.default_deployment is not None
                or len(llm_router.pattern_router.patterns) > 0
            ):
                return getattr(llm_router, f"{route_type}")(**data)
            elif data["model"] in llm_router.deployment_names:
                # Only match deployment_names if no wildcard matched
                return getattr(llm_router, f"{route_type}")(
                    **data, specific_deployment=True
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
                return getattr(llm_router, f"{route_type}")(**data)
            elif route_type in [
                "avideo_status",
                "avideo_content",
                "avideo_remix",
            ]:
                # Video endpoints: If model is provided (e.g., from decoded video_id), try router first
                try:
                    return getattr(llm_router, f"{route_type}")(**data)
                except Exception:
                    # If router fails (e.g., model not found in router), fall back to direct call
                    return getattr(litellm, f"{route_type}")(**data)
            elif _is_a2a_agent_model(data.get("model", "")):
                from litellm.proxy.agent_endpoints.a2a_routing import (
                    route_a2a_agent_request,
                )
                
                result = route_a2a_agent_request(data, route_type)
                if result is not None:
                    return result
                # Fall through to raise exception below if result is None

    elif user_model is not None:
        return getattr(litellm, f"{route_type}")(**data)
    elif route_type == "allm_passthrough_route":
        return getattr(litellm, f"{route_type}")(**data)

    # if no route found then it's a bad request
    route_name = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
    raise ProxyModelNotFoundError(
        route=route_name,
        model_name=data.get("model", ""),
    )
