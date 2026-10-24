from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, model_validator

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.tokenize.routing import TokenizeAdmissionContext
from litellm.types.tokenize import TokenizeRequest, TokenizeResponse

router: Final = APIRouter()


class ProxyTokenizeRequest(TokenizeRequest):
    model_config = ConfigDict(frozen=True, extra="allow", strict=True)

    @model_validator(mode="after")
    def reserved_fields(self) -> "ProxyTokenizeRequest":
        if any(name.startswith("_tokenize_") for name in self.model_extra or {}):
            raise ValueError("_tokenize_ fields are reserved")
        return self


class _ProxySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    general_settings: Mapping[str, object]
    user_model: str | None
    user_api_base: str | None


@router.post(
    "/v1/tokenize",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenizeResponse,
    tags=["tokenize"],
)
@router.post(
    "/tokenize",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenizeResponse,
    tags=["tokenize"],
)
async def tokenize(
    body: ProxyTokenizeRequest,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> object:
    """Tokenize raw text and return its count with explicitly requested representations."""
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_request_timeout,
        version,
    )

    settings: Final = _ProxySettings.model_validate(vars(proxy_server))
    processor: Final = ProxyBaseLLMRequestProcessing(
        data=body.model_dump(exclude_unset=True),
        tokenize_admission_context=TokenizeAdmissionContext(),
    )
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
