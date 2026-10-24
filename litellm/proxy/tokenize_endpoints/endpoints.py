from collections import deque
from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.proxy.auth.auth_utils import (
    _allow_model_level_clientside_configurable_parameters,  # pyright: ignore[reportPrivateUsage]  # reuse existing proxy authorization policy
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.custom_openapi_spec import inline_request_body
from litellm.proxy.common_utils.http_parsing_utils import (
    read_request_body as read_tokenize_request_body,  # pyright: ignore[reportUnknownVariableType]  # shared parser result is validated below
)
from litellm.tokenize.routing import TokenizeAdmissionContext
from litellm.types.tokenize import TokenizeRequest, TokenizeResponse

router: Final = APIRouter()
_REQUEST_DATA_ADAPTER: Final = TypeAdapter(dict[str, object])
_REQUEST_ARRAY_ADAPTER: Final = TypeAdapter(list[object])
_REQUEST_BODY: Final = inline_request_body(TokenizeRequest, {"model": "tokenizer", "prompt": "Hello world"})
_FORBIDDEN_TOKENIZE_FIELDS: Final = frozenset(
    {
        "artifact",
        "artifact_path",
        "artifact_url",
        "authorization",
        "backend",
        "local_config",
        "local_tokenizer",
        "local_tokenizer_config",
        "proxy_server_request",
        "source",
        "tokenizer",
        "tokenizer_config",
    }
)
_UNSUPPORTED_INPUT_FIELDS: Final = frozenset(
    {"output", "messages", "tools", "image", "images", "chat_template", "add_generation_prompt"}
)


class _ProxySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    general_settings: Mapping[str, object]
    user_model: str | None
    user_api_base: str | None


def _contains_forbidden_tokenize_field(value: object) -> bool:
    pending: Final[deque[object]] = deque((value,))  # mutable-ok: queue required for iterative nested traversal
    while pending:
        (current,) = (pending.popleft(),)
        if isinstance(current, Mapping):
            (current_mapping,) = (_REQUEST_DATA_ADAPTER.validate_python(current),)
            if any(key.startswith("_tokenize_") or key in _FORBIDDEN_TOKENIZE_FIELDS for key in current_mapping):
                return True
            pending.extend(current_mapping.values())
            continue
        if isinstance(current, list):
            pending.extend(_REQUEST_ARRAY_ADAPTER.validate_python(current))
    return False


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
    try:
        data: Final = _REQUEST_DATA_ADAPTER.validate_python(await read_tokenize_request_body(request))
        if _contains_forbidden_tokenize_field(data) or _UNSUPPORTED_INPUT_FIELDS.intersection(data):
            raise ValueError("Tokenize request contains server-owned or unsupported fields.")
        TokenizeRequest.model_validate(
            {name: value for name, value in data.items() if name in TokenizeRequest.model_fields}
        )
    except ValidationError as error:
        fields: Final = ", ".join(".".join(str(part) for part in item["loc"]) for item in error.errors())
        raise HTTPException(status_code=400, detail=f"Invalid tokenize fields: {fields}") from error
    except ValueError as error:
        raise HTTPException(
            status_code=400, detail="Tokenize request requires a JSON object with string model and prompt fields."
        ) from error
    authorized_fields: Final = frozenset(
        param
        for param in data
        if not param.startswith("_tokenize_")
        and param not in _FORBIDDEN_TOKENIZE_FIELDS
        and (
            settings.general_settings.get("allow_client_side_credentials") is True
            or _allow_model_level_clientside_configurable_parameters(
                model=str(data["model"]),
                param="api_base" if param == "base_url" else param,
                request_body_value=data[param],
                llm_router=llm_router,
            )
        )
    )
    processor: Final = ProxyBaseLLMRequestProcessing(
        data=data,
        tokenize_admission_context=TokenizeAdmissionContext(authorized_fields=authorized_fields),
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
