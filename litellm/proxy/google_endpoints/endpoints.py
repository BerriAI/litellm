from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, ORJSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, field_validator

from litellm.litellm_core_utils.json_validation_rule import normalize_json_schema_types
from litellm.proxy._types import *
from litellm.proxy._types import ProxyException, TokenCountRequest
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    log_llm_api_exception,
    proxy_exception_from_http_exception,
    resolve_litellm_call_id,
)
from litellm.proxy.common_utils.http_parsing_utils import (  # noqa: F401  # legacy module exports
    _read_request_body,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    read_request_body,
)
from litellm.proxy.common_utils.token_count_settings import count_source_fields
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse


class _GoogleCountTokensTextPart(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str


class _GoogleCountTokensSystemInstruction(BaseModel):
    model_config = ConfigDict(frozen=True)
    parts: tuple[_GoogleCountTokensTextPart, ...]

    @field_validator("parts", mode="before")
    @classmethod
    def normalize_parts(cls, parts: object) -> object:
        return (parts,) if isinstance(parts, Mapping) else parts


class _GoogleCountTokensFunctionDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    name: str
    description: str | None = None
    parameters: JsonValue | None = None
    parameters_json_schema: JsonValue | None = Field(default=None, alias="parametersJsonSchema")


class _GoogleCountTokensTool(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)
    function_declarations: tuple[_GoogleCountTokensFunctionDeclaration, ...] = Field(
        default=(), alias="functionDeclarations"
    )


class _GoogleCountTokensRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)
    contents: tuple[Mapping[str, JsonValue], ...] = ()
    system_instruction: _GoogleCountTokensSystemInstruction | None = Field(default=None, alias="systemInstruction")
    tools: tuple[_GoogleCountTokensTool, ...] | None = ()


router: Final = APIRouter(
    tags=["google genai endpoints"],
)


def _input_schema(declaration: _GoogleCountTokensFunctionDeclaration) -> object:
    if declaration.parameters_json_schema is not None:
        return declaration.parameters_json_schema
    if declaration.parameters is not None:
        return normalize_json_schema_types(declaration.parameters)
    return {"type": "object", "properties": {}}


_GOOGLE_RPC_STATUS: Final = MappingProxyType(
    {
        400: "INVALID_ARGUMENT",
        401: "UNAUTHENTICATED",
        403: "PERMISSION_DENIED",
        404: "NOT_FOUND",
        409: "ABORTED",
        429: "RESOURCE_EXHAUSTED",
        499: "CANCELLED",
        501: "UNIMPLEMENTED",
        503: "UNAVAILABLE",
        504: "DEADLINE_EXCEEDED",
    }
)


def _google_error_response(request: Request, error: ProxyException) -> JSONResponse:
    from litellm.proxy.proxy_server import (
        _close_dangling_otel_server_span,  # pyright: ignore[reportPrivateUsage]  # proxy_server keeps the span-close helper private; error JSONResponses returned by the route must stamp the OTel server span like the global ProxyException handler does
    )

    status_code: Final = int(error.code) if error.code is not None and error.code.isdigit() else 500
    _close_dangling_otel_server_span(request, status_code, exc=error)
    status: Final = _GOOGLE_RPC_STATUS.get(status_code, "INTERNAL" if status_code >= 500 else "FAILED_PRECONDITION")
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": status_code, "message": error.message, "status": status}},
        headers=error.headers,
    )


def _google_count_tokens_request(model_name: str, data: object) -> TokenCountRequest:
    from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter

    validated_data: Final = TypeAdapter(_GoogleCountTokensRequest).validate_python(data)
    contents: Final = [dict(content) for content in validated_data.contents]
    messages: Final = (
        GoogleGenAIAdapter().translate_generate_content_to_completion(model_name, contents).get("messages", [])
    )
    system: Final = (
        [{"type": "text", "text": part.text} for part in validated_data.system_instruction.parts]
        if validated_data.system_instruction is not None
        else None
    )
    declarations: Final = chain.from_iterable(tool.function_declarations for tool in validated_data.tools or ())
    tools: Final = [
        {
            "name": declaration.name,
            **({"description": declaration.description} if declaration.description is not None else {}),
            "input_schema": _input_schema(declaration),
        }
        for declaration in declarations
    ]
    return TokenCountRequest(
        model=model_name,
        contents=contents,
        messages=messages,
        system=system,
        tools=tools or None,
    )


@router.post(
    "/v1beta/models/{model_name:path}:generateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name:path}:generateContent",
    dependencies=[Depends(user_api_key_auth)],
)
async def google_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = await read_request_body(request=request)
    if "model" not in data:
        data["model"] = model_name

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="agenerate_content",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model_name,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1beta/models/{model_name:path}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
@router.post(
    "/models/{model_name:path}:streamGenerateContent",
    dependencies=[Depends(user_api_key_auth)],
)
async def google_stream_generate_content(
    request: Request,
    model_name: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = await read_request_body(request=request)
    if "model" not in data:
        data["model"] = model_name
    data["stream"] = True
    # google-genai SDK (?alt=sse) must not receive OpenAI's data: [DONE] terminator.
    data["_litellm_skip_openai_stream_done"] = True
    data["_litellm_raw_sse_stream"] = True

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="agenerate_content_stream",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model_name,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1beta/models/{model_name:path}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountDetailsResponse,
)
@router.post(
    "/models/{model_name:path}:countTokens",
    dependencies=[Depends(user_api_key_auth)],
    response_model=TokenCountDetailsResponse,
)
async def google_count_tokens(request: Request, model_name: str):
    """
    ```json
    return {
        "totalTokens": 31,
        "totalBillableCharacters": 96,
        "promptTokensDetails": [
            {
            "modality": "TEXT",
            "tokenCount": 31
            }
        ]
    }
    ```
    """
    from litellm.proxy.common_utils.http_parsing_utils import read_request_body
    from litellm.proxy.proxy_server import token_counter as internal_token_counter

    data: Final = await read_request_body(request=request)
    try:
        token_request: Final = _google_count_tokens_request(model_name, data)
    except ValidationError:
        return _google_error_response(
            request,
            ProxyException(
                message="Invalid Google countTokens request body", type="invalid_request_error", param=None, code=400
            ),
        )

    try:
        token_response: Final = await internal_token_counter(request=token_request, call_endpoint=True)
    except ProxyException as error:
        return _google_error_response(request, error)
    except HTTPException as error:
        return _google_error_response(request, proxy_exception_from_http_exception(error, {}))
    except Exception as error:
        log_llm_api_exception(error, resolve_litellm_call_id(request.headers.get("x-litellm-call-id")))
        return _google_error_response(
            request,
            ProxyException(message="Internal server error", type="internal_server_error", param=None, code=500),
        )
    original_response: Final[dict] = token_response.original_response or {}
    counted: Final[TokenCountDetailsResponse] = {
        "totalTokens": original_response.get("totalTokens") or token_response.total_tokens or 0,
        "promptTokensDetails": original_response.get("promptTokensDetails", []),
        **count_source_fields(token_response),
    }
    return counted


# ============================================================
# Google Interactions API Endpoints
# Per OpenAPI spec: https://ai.google.dev/static/api/interactions.openapi.json
# ============================================================


@router.post(
    "/v1beta/interactions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.post(
    "/interactions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def create_interaction(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new interaction using Google's Interactions API.
    
    Per OpenAPI spec: POST /{api_version}/interactions
    
    Supports both model interactions and agent interactions:
    - Model: Provide `model` parameter (e.g., "gemini-2.5-flash")
    - Agent: Provide `agent` parameter (e.g., "deep-research-pro-preview-12-2025")
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1beta/interactions" \
        -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gemini/gemini-2.5-flash",
            "input": "Hello, how are you?"
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = await read_request_body(request=request)

    # Default to gemini provider for interactions
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = "gemini"

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=data.get("model"),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1beta/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.get(
    "/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def get_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get an interaction by ID.

    Per OpenAPI spec: GET /{api_version}/interactions/{interaction_id}
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = {"interaction_id": interaction_id, "custom_llm_provider": "gemini"}

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1beta/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.delete(
    "/interactions/{interaction_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def delete_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete an interaction by ID.

    Per OpenAPI spec: DELETE /{api_version}/interactions/{interaction_id}
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = {"interaction_id": interaction_id, "custom_llm_provider": "gemini"}

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1beta/interactions/{interaction_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
@router.post(
    "/interactions/{interaction_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["interactions"],
)
async def cancel_interaction(
    request: Request,
    interaction_id: str,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel an interaction by ID.

    Per OpenAPI spec: POST /{api_version}/interactions/{interaction_id}:cancel
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = {"interaction_id": interaction_id, "custom_llm_provider": "gemini"}

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_interaction",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
