from collections.abc import Callable, Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from litellm.llms.chatgpt.search.transformation import (
    ChatGPTSearchRequest,
    is_chatgpt_search_response_header,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.types.utils import LlmProviders

router: Final = APIRouter()


class _ChatGPTSearchRouteData(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    model: str
    method: str = "POST"
    endpoint: str = "alpha/search"
    request_json: ChatGPTSearchRequest = Field(serialization_alias="json")
    custom_llm_provider: str = LlmProviders.CHATGPT.value
    required_custom_llm_provider: str = LlmProviders.CHATGPT.value


class _ProxyRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, strict=True)

    general_settings: Mapping[str, object]
    select_data_generator: Callable[..., object]
    user_model: str | None
    user_temperature: float | None
    user_request_timeout: float | None
    user_max_tokens: int | None
    user_api_base: str | None


@router.post("/v1/alpha/search")
@router.post("/alpha/search")
async def chatgpt_alpha_search(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    from litellm.proxy import proxy_server
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    try:
        search_request: Final = ChatGPTSearchRequest.model_validate(await request.json())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The alpha search request requires a JSON object with a non-empty string `model`",
        ) from exc

    route_data: Final = _ChatGPTSearchRouteData(model=search_request.model, request_json=search_request)
    data: Final = route_data.model_dump(by_alias=True)
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    proxy_runtime: Final = _ProxyRuntime.model_validate(vars(proxy_server))

    try:
        result: Final = await processor.base_passthrough_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_server.proxy_logging_obj,
            general_settings=proxy_runtime.general_settings,
            proxy_config=proxy_server.proxy_config,
            select_data_generator=proxy_runtime.select_data_generator,
            llm_router=proxy_server.llm_router,
            model=search_request.model,
            user_model=proxy_runtime.user_model,
            user_temperature=proxy_runtime.user_temperature,
            user_request_timeout=proxy_runtime.user_request_timeout,
            user_max_tokens=proxy_runtime.user_max_tokens,
            user_api_base=proxy_runtime.user_api_base,
            version=proxy_server.version,
        )
    except Exception as exc:  # noqa: BLE001  # every failure must run the shared proxy exception lifecycle
        raise await processor._handle_llm_api_exception(  # pyright: ignore[reportPrivateUsage]  # shared proxy lifecycle
            e=exc,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_server.proxy_logging_obj,
            version=proxy_server.version,
        )

    for name in tuple(result.headers):
        if not is_chatgpt_search_response_header(name):
            del result.headers[name]
    return result
