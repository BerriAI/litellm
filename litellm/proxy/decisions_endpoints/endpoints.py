from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse  # pyright: ignore[reportDeprecated]  # required endpoint contract
from pydantic import TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.types.decisions import DecisionsRequestBody, OpenAIDecisionRequestBody

router: Final = APIRouter()
_REQUEST_DATA_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_DECISIONS_REQUEST_BODY_ADAPTER: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_OPENAI_DECISION_REQUEST_BODY_ADAPTER: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(
    OpenAIDecisionRequestBody
)
_GENERAL_SETTINGS_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_OPTIONAL_STRING_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)
_OPTIONAL_FLOAT_ADAPTER: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)


async def _invalid_request(
    raw_data: Mapping[str, object], error: ValidationError, user_api_key_dict: UserAPIKeyAuth
) -> Exception:
    from litellm.proxy.proxy_server import proxy_logging_obj, version

    return await ProxyBaseLLMRequestProcessing(data=dict(raw_data))._handle_llm_api_exception(
        e=BadRequestError(
            message=f"Invalid Decisions request: {error}",
            model=str(raw_data.get("model", "")),
            llm_provider="",
        ),
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
        version=version,
    )


async def _request_data(request: Request, user_api_key_dict: UserAPIKeyAuth) -> dict[str, object]:
    body: Final = await request.body()
    try:
        return _REQUEST_DATA_ADAPTER.validate_json(body)
    except ValidationError as error:
        raise await _invalid_request(raw_data={}, error=error, user_api_key_dict=user_api_key_dict)


async def _process_decisions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    body_adapter: TypeAdapter[DecisionsRequestBody] | TypeAdapter[OpenAIDecisionRequestBody],
) -> object:
    from litellm.proxy.proxy_server import (
        general_settings as proxy_general_settings,
    )
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_max_tokens,
        user_request_timeout,
        version,
    )
    from litellm.proxy.proxy_server import (
        user_api_base as proxy_user_api_base,
    )
    from litellm.proxy.proxy_server import (
        user_model as proxy_user_model,
    )
    from litellm.proxy.proxy_server import (
        user_temperature as proxy_user_temperature,
    )

    data: Final = await _request_data(request, user_api_key_dict)
    try:
        body_adapter.validate_python(data)
    except ValidationError as error:
        raise await _invalid_request(raw_data=data, error=error, user_api_key_dict=user_api_key_dict)
    general_settings: Final = _GENERAL_SETTINGS_ADAPTER.validate_python(proxy_general_settings)
    user_api_base: Final = _OPTIONAL_STRING_ADAPTER.validate_python(proxy_user_api_base)
    user_model: Final = _OPTIONAL_STRING_ADAPTER.validate_python(proxy_user_model)
    user_temperature: Final = _OPTIONAL_FLOAT_ADAPTER.validate_python(proxy_user_temperature)
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adecisions",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=None,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as error:
        raise await processor._handle_llm_api_exception(
            e=error,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/systemone",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
)
@router.post(
    "/systemone",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
)
async def systemone(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    return await _process_decisions(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        body_adapter=_DECISIONS_REQUEST_BODY_ADAPTER,
    )


@router.post(
    "/v1/decisions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
)
@router.post(
    "/decisions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
)
async def decisions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    return await _process_decisions(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        body_adapter=_OPENAI_DECISION_REQUEST_BODY_ADAPTER,
    )
