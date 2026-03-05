"""
Skills API endpoints - /v1/skills

Provider defaults:
- Original CRUD endpoints (create, list, get, delete) default to "anthropic" for
  backward compatibility — these endpoints existed before OpenAI Skills support.
- OpenAI-only endpoints (update, content, versions CRUD) default to "openai" because
  these operations are only supported by OpenAI's Skills API.

All endpoints accept a custom_llm_provider query param to override the default.
"""

import base64
from typing import Optional

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response

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
from litellm.utils import ProviderConfigManager

router = APIRouter()


def _validate_provider_supports_operation(
    operation: str, custom_llm_provider: str
) -> None:
    """Raise 400 if the provider does not support the requested operation."""
    config = ProviderConfigManager.get_provider_skills_api_config(
        LlmProviders(custom_llm_provider)
    )
    if config is None:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{custom_llm_provider}' does not support the Skills API.",
        )
    if operation not in config.supported_operations:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The '{operation}' operation is not supported by "
                f"'{custom_llm_provider}'. "
                f"Supported operations: {sorted(config.supported_operations)}."
            ),
        )


@router.post(
    "/v1/skills",
    tags=["[beta] Anthropic Skills API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=Skill,
)
async def create_skill(
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "anthropic",
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
    form_data = await get_form_data(request)
    data = await convert_upload_files_to_file_data(form_data)
    
    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    
    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
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
    limit: Optional[int] = 10,
    after_id: Optional[str] = None,
    before_id: Optional[str] = None,
    custom_llm_provider: Optional[str] = "anthropic",
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
    
    Returns: ListSkillsResponse with list of skills
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
    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )
    
    # Use query params if not in body
    if "limit" not in data and limit is not None:
        data["limit"] = limit
    if "after_id" not in data and after_id is not None:
        data["after_id"] = after_id
    if "before_id" not in data and before_id is not None:
        data["before_id"] = before_id
    # Map Anthropic-style param names to SDK param names
    if "after_id" in data:
        data.setdefault("page", data.pop("after_id"))
    if "before_id" in data:
        data.setdefault("before", data.pop("before_id"))
    
    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    # Set custom_llm_provider: body > query param > default
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    
    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
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
    custom_llm_provider: Optional[str] = "anthropic",
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
    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )
    
    # Set skill_id from path parameter
    data["skill_id"] = skill_id
    
    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    # Set custom_llm_provider: body > query param > default
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    
    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
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
    custom_llm_provider: Optional[str] = "anthropic",
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
    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )
    
    # Set skill_id from path parameter
    data["skill_id"] = skill_id
    
    # Extract model for routing (header > query > body)
    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    # Set custom_llm_provider: body > query param > default
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    
    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
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


@router.post(
    "/v1/skills/{skill_id}",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_skill_endpoint(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Update a skill (e.g. set default_version). OpenAI-specific."""
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

    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )

    data["skill_id"] = skill_id

    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    _validate_provider_supports_operation("update", data["custom_llm_provider"])

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aupdate_skill",
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
    "/v1/skills/{skill_id}/content",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_skill_content_endpoint(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Get skill content. OpenAI-specific."""
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

    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )

    data["skill_id"] = skill_id

    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    _validate_provider_supports_operation("content", data["custom_llm_provider"])

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_skill_content",
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
        # Binary content: return raw bytes with proper media type
        if isinstance(result, dict) and "content_type" in result and "json" not in result.get("content_type", ""):
            return Response(
                content=base64.b64decode(result["content"]),
                media_type=result["content_type"],
            )
        return result
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/skills/{skill_id}/versions",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_skill_version_endpoint(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Create a new skill version. OpenAI-specific."""
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

    form_data = await get_form_data(request)
    data = await convert_upload_files_to_file_data(form_data)

    data["skill_id"] = skill_id

    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    _validate_provider_supports_operation("create_version", data["custom_llm_provider"])

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acreate_skill_version",
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
    "/v1/skills/{skill_id}/versions",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_skill_versions_endpoint(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    limit: Optional[int] = 20,
    after: Optional[str] = None,
    before: Optional[str] = None,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List skill versions. OpenAI-specific."""
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

    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )

    data["skill_id"] = skill_id
    if "limit" not in data and limit is not None:
        data["limit"] = limit
    if "after" not in data and after is not None:
        data["after"] = after
    if "before" not in data and before is not None:
        data["before"] = before

    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    _validate_provider_supports_operation("list_versions", data["custom_llm_provider"])

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_skill_versions",
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
    "/v1/skills/{skill_id}/versions/{version}",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_skill_version_endpoint(
    skill_id: str,
    version: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Get a specific skill version. OpenAI-specific."""
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
        version as proxy_version,
    )

    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )

    data["skill_id"] = skill_id
    data["version"] = version

    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    _validate_provider_supports_operation("get_version", data["custom_llm_provider"])

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_skill_version",
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
            version=proxy_version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=proxy_version,
        )


@router.delete(
    "/v1/skills/{skill_id}/versions/{version}",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_skill_version_endpoint(
    skill_id: str,
    version: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Delete a skill version. OpenAI-specific."""
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
        version as proxy_version,
    )

    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )

    data["skill_id"] = skill_id
    data["version"] = version

    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    _validate_provider_supports_operation("delete_version", data["custom_llm_provider"])

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_skill_version",
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
            version=proxy_version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=proxy_version,
        )


@router.get(
    "/v1/skills/{skill_id}/versions/{version}/content",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_skill_version_content_endpoint(
    skill_id: str,
    version: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "openai",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Get skill version content. OpenAI-specific."""
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
        version as proxy_version,
    )

    body = await request.body()
    try:
        data = orjson.loads(body) if body else {}
    except orjson.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON body. Ensure the request Content-Type is application/json.",
        )

    data["skill_id"] = skill_id
    data["version"] = version

    model = (
        data.get("model")
        or request.query_params.get("model")
        or request.headers.get("x-litellm-model")
    )
    if model:
        data["model"] = model
    
    if "custom_llm_provider" not in data:
        data["custom_llm_provider"] = custom_llm_provider
    _validate_provider_supports_operation("get_version_content", data["custom_llm_provider"])

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_skill_version_content",
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
            version=proxy_version,
        )
        # Binary content: return raw bytes with proper media type
        if isinstance(result, dict) and "content_type" in result and "json" not in result.get("content_type", ""):
            return Response(
                content=base64.b64decode(result["content"]),
                media_type=result["content_type"],
            )
        return result
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=proxy_version,
        )

