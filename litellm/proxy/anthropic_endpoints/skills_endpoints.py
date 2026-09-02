"""
Anthropic Skills API endpoints - /v1/skills
"""

from types import MappingProxyType
from typing import Annotated, Final, assert_never

import orjson
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.llms.litellm_proxy.skills.skill_search import (
    DEFAULT_SKILL_SEARCH_TOP_K,
    SkillSearchEmbeddingFailed,
    SkillSearchHits,
    SkillSearchNotConfigured,
    global_skill_search_index,
    search_skills,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    convert_upload_files_to_file_data,
    get_form_data,
)
from litellm.types.llms.anthropic_skills import (
    DeleteSkillResponse,
    ListSkillsResponse,
    Skill,
)
from litellm.types.utils import LlmProviders

router: Final = APIRouter()


class _SkillSearchErrorDetail(TypedDict):
    error: ReadOnly[str]
    message: ReadOnly[str]


def _skill_search_error(status_code: int, error: str, message: str) -> HTTPException:
    detail: Final[_SkillSearchErrorDetail] = {"error": error, "message": message}
    return HTTPException(status_code=status_code, detail=detail)


async def _search_litellm_skills(query: str, top_k: int, user_api_key_dict: UserAPIKeyAuth) -> ListSkillsResponse:
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
    from litellm.llms.litellm_proxy.skills.transformation import (
        LiteLLMSkillsTransformationHandler,
    )
    from litellm.proxy.proxy_server import llm_router

    db_skills: Final = await LiteLLMSkillsHandler.list_skills_for_search(user_api_key_dict=user_api_key_dict)
    outcome: Final = await search_skills(
        query=query,
        skills=db_skills,
        top_k=top_k,
        router=llm_router,
        embedding_model=litellm.skill_search_embedding_model,
        index=global_skill_search_index,
        user_api_key_dict=user_api_key_dict,
    )
    to_response: Final = LiteLLMSkillsTransformationHandler().db_skill_to_response
    match outcome:
        case SkillSearchHits(hits):
            skills: Final = [  # mutable-ok: ListSkillsResponse.data requires list[Skill]; never mutated after
                to_response(hit.skill).model_copy(update=MappingProxyType({"search_score": hit.score})) for hit in hits
            ]
            return ListSkillsResponse(data=skills, has_more=False, next_page=None)
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
    response_model=Skill,
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

    # Read form data and convert UploadFile objects to file data tuples
    form_data: Final = await get_form_data(request)
    data: Final = await convert_upload_files_to_file_data(form_data)

    # Extract model for routing (header > query > body)
    model: Final = data.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")
    if model:
        data["model"] = model

    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

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
            model=data.get("model"),
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
    response_model=ListSkillsResponse,
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
        if custom_llm_provider != LlmProviders.LITELLM_PROXY.value:
            raise _skill_search_error(
                400,
                "skill_search_unsupported_provider",
                "query is only supported for custom_llm_provider=litellm_proxy",
            )
        return await _search_litellm_skills(query=query, top_k=top_k, user_api_key_dict=user_api_key_dict)

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

    # Read request body
    body: Final = await request.body()
    data: Final = orjson.loads(body) if body else {}

    # Use query params if not in body
    if "limit" not in data and limit is not None:
        data["limit"] = limit
    if "after_id" not in data and after_id is not None:
        data["after_id"] = after_id
    if "before_id" not in data and before_id is not None:
        data["before_id"] = before_id

    # Extract model for routing (header > query > body)
    model: Final = data.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")
    if model:
        data["model"] = model

    # Set custom_llm_provider: body > query param > default
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

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
            model=data.get("model"),
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
    response_model=Skill,
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

    # Read request body
    body: Final = await request.body()
    data: Final = orjson.loads(body) if body else {}

    # Set skill_id from path parameter
    data["skill_id"] = skill_id

    # Extract model for routing (header > query > body)
    model: Final = data.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")
    if model:
        data["model"] = model

    # Set custom_llm_provider: body > query param > default
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

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
            model=data.get("model"),
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
    response_model=DeleteSkillResponse,
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

    # Read request body
    body: Final = await request.body()
    data: Final = orjson.loads(body) if body else {}

    # Set skill_id from path parameter
    data["skill_id"] = skill_id

    # Extract model for routing (header > query > body)
    model: Final = data.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")
    if model:
        data["model"] = model

    # Set custom_llm_provider: body > query param > default
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider

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
            model=data.get("model"),
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
