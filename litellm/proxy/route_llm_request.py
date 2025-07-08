from typing import TYPE_CHECKING, Any, Literal, Optional

from fastapi import HTTPException, status

import litellm

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


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


async def route_request(
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
        "alist_input_items",
        "_arealtime",  # private function for realtime API
        "aimage_edit",
        "agenerate_content",
        "agenerate_content_stream",
        "allm_passthrough_route",
    ],
):
    """
    Common helper to route the request
    """
    team_id = get_team_id_from_data(data)
    router_model_names = llm_router.model_names if llm_router is not None else []
    if "api_key" in data or "api_base" in data:
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**data)
        else:
            return getattr(litellm, f"{route_type}")(**data)

    elif "user_config" in data:
        router_config = data.pop("user_config")
        user_router = litellm.Router(**router_config)
        ret_val = getattr(user_router, f"{route_type}")(**data)
        user_router.discard()
        return ret_val

    elif (
        route_type == "acompletion"
        and data.get("model", "") is not None
        and "," in data.get("model", "")
        and llm_router is not None
    ):
        if data.get("fastest_response", False):
            return llm_router.abatch_completion_fastest_response(**data)
        else:
            models = [model.strip() for model in data.pop("model").split(",")]
            return llm_router.abatch_completion(models=models, **data)
    elif llm_router is not None:
        team_model_name = (
            llm_router.map_team_model(data["model"], team_id)
            if team_id is not None
            else None
        )
        if team_model_name is not None:
            data["model"] = team_model_name
            return getattr(llm_router, f"{route_type}")(**data)

        elif (
            data["model"] in router_model_names
            or data["model"] in llm_router.get_model_ids()
        ):

            return getattr(llm_router, f"{route_type}")(**data)

        elif (
            llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):

            return getattr(llm_router, f"{route_type}")(**data)

        elif data["model"] in llm_router.deployment_names:

            return getattr(llm_router, f"{route_type}")(
                **data, specific_deployment=True
            )

        elif data["model"] not in router_model_names:

            if llm_router.router_general_settings.pass_through_all_models:
                return getattr(litellm, f"{route_type}")(**data)
            elif (
                llm_router.default_deployment is not None
                or len(llm_router.pattern_router.patterns) > 0
            ):
                return getattr(llm_router, f"{route_type}")(**data)
            elif route_type in [
                "amoderation",
                "aget_responses",
                "adelete_responses",
                "alist_input_items",
            ]:
                # moderation endpoint does not require `model` parameter
                return getattr(llm_router, f"{route_type}")(**data)

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
