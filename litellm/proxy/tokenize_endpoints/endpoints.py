from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.custom_openapi_spec import inline_request_body
from litellm.proxy.common_utils.http_parsing_utils import (
    read_request_body as read_tokenize_request_body,  # pyright: ignore[reportUnknownVariableType]  # shared parser result is validated below
)
from litellm.types.tokenize import TokenizeRequest, TokenizeResponse

router: Final = APIRouter()
_REQUEST_DATA_ADAPTER: Final = TypeAdapter(dict[str, object])
_REQUEST_BODY: Final = inline_request_body(TokenizeRequest, {"model": "tokenizer", "prompt": "Hello world"})


class _ProxySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    general_settings: Mapping[str, object]
    user_model: str | None
    user_api_base: str | None


@router.post(
    "/v1/tokenize",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenizeResponse,
    openapi_extra={"requestBody": _REQUEST_BODY},
    tags=["tokenize"],
)
@router.post(
    "/tokenize",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenizeResponse,
    openapi_extra={"requestBody": _REQUEST_BODY},
    tags=["tokenize"],
)
async def tokenize(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> object:
    """Tokenize a string with a configured model, returning its token count and available tokens."""
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_request_timeout,
        version,
    )

    settings: Final = _ProxySettings.model_validate(vars(proxy_server))
    try:
        data: Final = _REQUEST_DATA_ADAPTER.validate_python(await read_tokenize_request_body(request))
        TokenizeRequest.model_validate(data)
    except ValidationError as error:
        raise HTTPException(
            status_code=400, detail="Tokenize request requires a JSON object with string model and prompt fields."
        ) from error
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="atokenize",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=dict(settings.general_settings),
            proxy_config=proxy_config,
            select_data_generator=None,
            model=None,
            user_model=settings.user_model,
            user_temperature=None,
            user_request_timeout=user_request_timeout,
            user_max_tokens=None,
            user_api_base=settings.user_api_base,
            version=version,
        )
    except Exception as error:  # noqa: BLE001  # normalize all provider and routing failures at the HTTP boundary
        raise await processor.handle_llm_api_exception(
            e=error,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
