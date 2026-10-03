"""
Anthropic Skills API endpoints - /v1/skills
"""

from functools import partial
from types import MappingProxyType
from typing import Annotated, Final, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from typing_extensions import ReadOnly, TypedDict, assert_never

import litellm
from litellm.llms.litellm_proxy.skills.skill_search import (
    DEFAULT_SKILL_SEARCH_TOP_K,
    SkillSearchEmbeddingFailed,
    SkillSearchHits,
    SkillSearchNotConfigured,
    SkillSearchUnsupportedProvider,
    global_skill_search_index,
    search_hosted_skills,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    convert_upload_files_to_file_data,
    get_request_body,
    resolve_inference_model,
)
from litellm.proxy.openai_files_endpoints.common_utils import (
    _extract_model_param as extract_model_param,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import HttpPassThroughEndpointHelpers
from litellm.types.llms.anthropic_skills import ListSkillsResponse
from litellm.types.router import CredentialLiteLLMParams

router: Final = APIRouter()


class _SkillSearchErrorDetail(TypedDict):
    error: ReadOnly[str]
    message: ReadOnly[str]


def _skill_search_error(status_code: int, error: str, message: str) -> HTTPException:
    detail: Final[_SkillSearchErrorDetail] = {"error": error, "message": message}
    return HTTPException(status_code=status_code, detail=detail)


async def _search_skills(
    custom_llm_provider: str | None, query: str, top_k: int, user_api_key_dict: UserAPIKeyAuth
) -> ListSkillsResponse:
    from litellm.llms.litellm_proxy.skills.transformation import (
        LiteLLMSkillsTransformationHandler,
    )
    from litellm.proxy.proxy_server import llm_router, proxy_logging_obj

    outcome: Final = await search_hosted_skills(
        custom_llm_provider=custom_llm_provider,
        query=query,
        top_k=top_k,
        router=llm_router,
        embedding_model=litellm.skill_search_embedding_model,
        index=global_skill_search_index,
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
    )
    to_response: Final = LiteLLMSkillsTransformationHandler().db_skill_to_response
    match outcome:
        case SkillSearchHits(hits):
            skills: Final = [
                to_response(hit.skill).model_copy(update=MappingProxyType({"search_score": hit.score})) for hit in hits
            ]
            return ListSkillsResponse(data=skills, has_more=False, next_page=None)
        case SkillSearchUnsupportedProvider(reason):
            raise _skill_search_error(400, "skill_search_unsupported_provider", reason)
        case SkillSearchNotConfigured(reason):
            raise _skill_search_error(400, "skill_search_not_configured", reason)
        case SkillSearchEmbeddingFailed(reason):
            raise _skill_search_error(503, "skill_search_unavailable", reason)
        case _:
            assert_never(outcome)


@router.post(
    "/v1/skills",
    tags=["[beta] Anthropic Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_skill(
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new skill on Anthropic.
    
    Requires `?beta=true` query parameter.
    
    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: claude-account-1`
    - Pass model via query: `?model=claude-account-1`
    - Pass model via form field: `model=claude-account-1`
    
    Example usage:
    ```bash
    # Basic usage
    curl -X POST "http://localhost:4000/v1/skills?beta=true" \
      -H "Content-Type: multipart/form-data" \
      -H "Authorization: Bearer your-key" \
      -F "display_title=My Skill" \
      -F "files[]=@skill.zip"
    
    # With model-based routing
    curl -X POST "http://localhost:4000/v1/skills?beta=true" \
      -H "Content-Type: multipart/form-data" \
      -H "Authorization: Bearer your-key" \
      -H "x-litellm-model: claude-account-1" \
      -F "display_title=My Skill" \
      -F "files[]=@skill.zip"
    ```
    
    Returns: Skill object with id, display_title, etc.
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

    data: Final = await _skill_request_data(request, "create", custom_llm_provider)

    # Process request using ProxyBaseLLMRequestProcessing
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_skill",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=extract_model_param(request, data),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/skills",
    tags=["[beta] Anthropic Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_skills(
    fastapi_response: Response,
    request: Request,
    limit: int | None = 10,
    after_id: str | None = None,
    before_id: str | None = None,
    custom_llm_provider: str | None = "anthropic",
    query: Annotated[
        str | None,
        Query(
            min_length=1,
            description="Describe what you need in natural language to rank the skills you can access by "
            "semantic similarity over their title and description. Each result carries a search_score. "
            "Only supported for custom_llm_provider=litellm_proxy. Requires "
            "litellm_settings.skill_search_embedding_model.",
        ),
    ] = None,
    top_k: Annotated[
        int,
        Query(ge=1, le=100, description="With query: the maximum number of ranked skills to return."),
    ] = DEFAULT_SKILL_SEARCH_TOP_K,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List skills on Anthropic.

    Requires `?beta=true` query parameter.

    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: claude-account-1`
    - Pass model via query: `?model=claude-account-1`
    - Pass model via body: `{"model": "claude-account-1"}`

    Example usage:
    ```bash
    # Basic usage
    curl "http://localhost:4000/v1/skills?beta=true&limit=10" \
      -H "Authorization: Bearer your-key"

    # With model-based routing
    curl "http://localhost:4000/v1/skills?beta=true&limit=10" \
      -H "Authorization: Bearer your-key" \
      -H "x-litellm-model: claude-account-1"
    ```

    Pass `?custom_llm_provider=litellm_proxy&query=<task>` to rank the LiteLLM-hosted skills you can
    access by semantic similarity instead of paging through the whole registry:
    ```bash
    curl "http://localhost:4000/v1/skills?custom_llm_provider=litellm_proxy&query=summarize+a+pdf&top_k=5" \
      -H "Authorization: Bearer your-key"
    ```

    Returns: ListSkillsResponse with list of skills
    """
    if query is not None:
        return await _search_skills(
            custom_llm_provider=custom_llm_provider, query=query, top_k=top_k, user_api_key_dict=user_api_key_dict
        )

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

    data: Final = await _skill_request_data(request, "list", custom_llm_provider)

    # Use query params if not in body
    if "limit" not in data and limit is not None:
        data["limit"] = limit
    if "after_id" not in data and after_id is not None:
        data["after_id"] = after_id
    if "before_id" not in data and before_id is not None:
        data["before_id"] = before_id

    # Process request using ProxyBaseLLMRequestProcessing
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_skills",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=extract_model_param(request, data),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/skills/{skill_id}",
    tags=["[beta] Anthropic Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_skill(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a specific skill by ID from Anthropic.
    
    Requires `?beta=true` query parameter.
    
    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: claude-account-1`
    - Pass model via query: `?model=claude-account-1`
    - Pass model via body: `{"model": "claude-account-1"}`
    
    Example usage:
    ```bash
    # Basic usage
    curl "http://localhost:4000/v1/skills/skill_123?beta=true" \
      -H "Authorization: Bearer your-key"
    
    # With model-based routing
    curl "http://localhost:4000/v1/skills/skill_123?beta=true" \
      -H "Authorization: Bearer your-key" \
      -H "x-litellm-model: claude-account-1"
    ```
    
    Returns: Skill object
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

    data: Final = await _skill_request_data(request, "get", custom_llm_provider)

    # Process request using ProxyBaseLLMRequestProcessing
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_skill",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=extract_model_param(request, data),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.delete(
    "/v1/skills/{skill_id}",
    tags=["[beta] Anthropic Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_skill(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a skill by ID from Anthropic.
    
    Requires `?beta=true` query parameter.
    
    Note: Anthropic does not allow deleting skills with existing versions.
    
    Model-based routing (for multi-account support):
    - Pass model via header: `x-litellm-model: claude-account-1`
    - Pass model via query: `?model=claude-account-1`
    - Pass model via body: `{"model": "claude-account-1"}`
    
    Example usage:
    ```bash
    # Basic usage
    curl -X DELETE "http://localhost:4000/v1/skills/skill_123?beta=true" \
      -H "Authorization: Bearer your-key"
    
    # With model-based routing
    curl -X DELETE "http://localhost:4000/v1/skills/skill_123?beta=true" \
      -H "Authorization: Bearer your-key" \
      -H "x-litellm-model: claude-account-1"
    ```
    
    Returns: DeleteSkillResponse with type="skill_deleted"
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

    data: Final = await _skill_request_data(request, "delete", custom_llm_provider)

    # Process request using ProxyBaseLLMRequestProcessing
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_skill",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=extract_model_param(request, data),
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


SkillRouteType = Literal["acreate_skill", "alist_skills", "aget_skill", "adelete_skill"]
_MODEL_CREDENTIAL_PARAMS: Final = frozenset(CredentialLiteLLMParams.model_fields) | {
    "client",
    "extra_headers",
    "organization",
    "azure_ad_token_provider",
    "litellm_credential_name",
    "configurable_clientside_auth_params",
}


async def _skill_request_data(
    request: Request, operation: str, default_provider: str | None
) -> dict[str, object]:  # mutable-ok: proxy processing mutates routing data
    from litellm.proxy.proxy_server import general_settings_view, user_model

    try:
        raw_body: Final[object] = (
            await get_request_body(request) if request.method == "POST" else await _read_request_body(request)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not isinstance(raw_body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    body: Final = await convert_upload_files_to_file_data(raw_body)
    model: Final = body.get("model") if body.get("model") not in (None, "") else extract_model_param(request, body)
    resolved_model: Final = resolve_inference_model(model, general_settings_view(), user_model, model)
    provider: Final = (
        body.get("custom_llm_provider")
        if body.get("custom_llm_provider") not in (None, "")
        else request.query_params.get("custom_llm_provider") or default_provider
    )
    if resolved_model is None and provider is not None and not isinstance(provider, str):
        raise HTTPException(status_code=400, detail="custom_llm_provider must be a string")
    request_params: Final = {**request.query_params, **body}
    return {
        **{
            key: value
            for key, value in request_params.items()
            if resolved_model is None or key not in _MODEL_CREDENTIAL_PARAMS
        },
        **request.path_params,
        "model": model,
        "custom_llm_provider": provider,
        "_skill_operation": operation,
        "_skill_single_file_upload": isinstance(raw_body.get("files"), UploadFile),
    }


async def _native_skill_endpoint(
    operation: str,
    route_type: SkillRouteType,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> object:
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_model,
        user_request_timeout,
        version,
    )

    data: Final = await _skill_request_data(request, operation, "openai")
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result: Final[object] = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type=route_type,
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            model=extract_model_param(request, data),
            user_model=user_model,
            user_request_timeout=user_request_timeout,
            user_api_base=user_api_base,
            version=version,
        )
        if operation not in ("content", "version_content"):
            return result
        response: Final = getattr(result, "response", None)
        if not isinstance(response, httpx.Response):
            raise TypeError("Skills content response did not contain an HTTP response")
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                response.headers, custom_headers=dict(fastapi_response.headers)
            ),
        )
    except Exception as e:  # noqa: BLE001  # proxy maps provider errors to the public exception contract
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


_NATIVE_SKILL_ROUTES: Final[tuple[tuple[str, str, str, SkillRouteType], ...]] = (
    ("POST", "/v1/skills/{skill_id}", "update", "acreate_skill"),
    ("GET", "/v1/skills/{skill_id}/content", "content", "aget_skill"),
    ("POST", "/v1/skills/{skill_id}/versions", "create_version", "acreate_skill"),
    ("GET", "/v1/skills/{skill_id}/versions", "list_versions", "alist_skills"),
    ("GET", "/v1/skills/{skill_id}/versions/{version}", "version", "aget_skill"),
    ("DELETE", "/v1/skills/{skill_id}/versions/{version}", "delete_version", "adelete_skill"),
    ("GET", "/v1/skills/{skill_id}/versions/{version}/content", "version_content", "aget_skill"),
)

for method, path, operation, route_type in _NATIVE_SKILL_ROUTES:
    router.add_api_route(
        path,
        partial(_native_skill_endpoint, operation, route_type),
        methods=[method],
        name=f"{operation}_skill",
        description=f"Native skill operation: {operation.replace('_', ' ')}",
        response_model=None,
        response_class=Response if operation in ("content", "version_content") else JSONResponse,
        openapi_extra={
            "parameters": [
                {"name": field, "in": "path", "required": True, "schema": {"type": "string"}}
                for field in ("skill_id", "version")
                if f"{{{field}}}" in path
            ]
        },
        tags=["[beta] OpenAI Skills API"],
    )
