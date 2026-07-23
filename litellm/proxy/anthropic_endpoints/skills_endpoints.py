"""
Anthropic Skills API endpoints - /v1/skills
"""

from typing import Any, Optional

import orjson
from fastapi import APIRouter, Depends, Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    convert_upload_files_to_file_data,
    get_form_data,
)
from litellm.types.llms.openai_skills import (
    OpenAIDeletedSkillVersion,
    OpenAISkill,
    OpenAISkillVersion,
    OpenAISkillVersionList,
)

router = APIRouter()


async def _process_skill_request(
    *,
    data: dict[str, Any],
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    route_type: str,
) -> Any:
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

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        result = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type=route_type,
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
    except Exception as e:  # noqa: BLE001  # proxy exception handling maps all provider failures
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )

    if hasattr(result, "response") and hasattr(result.response, "content"):
        return Response(
            content=result.response.content,
            status_code=result.response.status_code,
            headers={
                key: value
                for key, value in result.response.headers.items()
                if key.lower() in {"content-type", "content-disposition"}
            },
        )
    return result


def _set_skill_route_params(data: dict[str, Any], request: Request, skill_id: str | None = None) -> dict[str, Any]:
    from litellm.litellm_core_utils.skill_id_utils import decode_skill_id

    if skill_id is not None:
        data["skill_id"] = skill_id
        decoded = decode_skill_id(skill_id)
        if decoded is not None:
            data["skill_id"] = decoded["id"]
            data["model"] = decoded["model"]
    if "model" not in data:
        model = request.query_params.get("model") or request.headers.get("x-litellm-model")
        if model:
            data["model"] = model
    return data


@router.post(
    "/v1/skills",
    tags=["[beta] Anthropic Skills API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=None,
)
async def create_skill(
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008, RUF100  # FastAPI dependency injection
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
    model = data.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")
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
    response_model=None,
)
async def list_skills(
    fastapi_response: Response,
    request: Request,
    limit: Optional[int] = 10,
    after: str | None = None,
    order: str | None = None,
    after_id: Optional[str] = None,
    before_id: Optional[str] = None,
    custom_llm_provider: Optional[str] = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008, RUF100  # FastAPI dependency injection
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
    data = orjson.loads(body) if body else {}

    # Use query params if not in body
    if "limit" not in data and limit is not None:
        data["limit"] = limit
    if "after" not in data and after is not None:
        data["after"] = after
    if "order" not in data and order is not None:
        data["order"] = order
    if "after_id" not in data and after_id is not None:
        data["after_id"] = after_id
    if "before_id" not in data and before_id is not None:
        data["before_id"] = before_id

    # Extract model for routing (header > query > body)
    model = data.get("model") or request.query_params.get("model") or request.headers.get("x-litellm-model")
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
    response_model=None,
)
async def get_skill(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008, RUF100  # FastAPI dependency injection
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
    data = orjson.loads(body) if body else {}

    _set_skill_route_params(data, request, skill_id)

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
    response_model=None,
)
async def delete_skill(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: Optional[str] = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008, RUF100  # FastAPI dependency injection
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
    data = orjson.loads(body) if body else {}

    _set_skill_route_params(data, request, skill_id)

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
    response_model=OpenAISkill,
)
async def update_skill(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008, RUF100  # FastAPI dependency injection
):
    body = await request.body()
    data = orjson.loads(body) if body else {}
    _set_skill_route_params(data, request, skill_id)
    data.setdefault("custom_llm_provider", custom_llm_provider)
    return await _process_skill_request(
        data=data,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        route_type="aupdate_skill",
    )


@router.get(
    "/v1/skills/{skill_id}/content",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_skill_content(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008, RUF100  # FastAPI dependency injection
):
    data = _set_skill_route_params({}, request, skill_id)
    data.setdefault("custom_llm_provider", custom_llm_provider)
    return await _process_skill_request(
        data=data,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        route_type="aget_skill_content",
    )


@router.post(
    "/v1/skills/{skill_id}/versions",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OpenAISkillVersion,
)
async def create_skill_version(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008, RUF100  # FastAPI dependency injection
):
    form_data = await get_form_data(request)
    data = await convert_upload_files_to_file_data(form_data)
    _set_skill_route_params(data, request, skill_id)
    if isinstance(data.get("default"), str):
        data["default"] = data["default"].lower() == "true"
    data.setdefault("custom_llm_provider", custom_llm_provider)
    return await _process_skill_request(
        data=data,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        route_type="acreate_skill_version",
    )


@router.get(
    "/v1/skills/{skill_id}/versions",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OpenAISkillVersionList,
)
async def list_skill_versions(
    skill_id: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    data: dict[str, Any] = {
        key: value for key, value in request.query_params.items() if key in {"after", "limit", "order", "model"}
    }
    _set_skill_route_params(data, request, skill_id)
    if "limit" in data:
        data["limit"] = int(data["limit"])
    data.setdefault("custom_llm_provider", custom_llm_provider)
    return await _process_skill_request(
        data=data,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        route_type="alist_skill_versions",
    )


@router.get(
    "/v1/skills/{skill_id}/versions/{version}",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OpenAISkillVersion,
)
async def get_skill_version(
    skill_id: str,
    version: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    data = _set_skill_route_params({}, request, skill_id)
    data["version"] = version
    data.setdefault("custom_llm_provider", custom_llm_provider)
    return await _process_skill_request(
        data=data,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        route_type="aget_skill_version",
    )


@router.get(
    "/v1/skills/{skill_id}/versions/{version}/content",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_skill_version_content(
    skill_id: str,
    version: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    data = _set_skill_route_params({}, request, skill_id)
    data["version"] = version
    data.setdefault("custom_llm_provider", custom_llm_provider)
    return await _process_skill_request(
        data=data,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        route_type="aget_skill_version_content",
    )


@router.delete(
    "/v1/skills/{skill_id}/versions/{version}",
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=OpenAIDeletedSkillVersion,
)
async def delete_skill_version(
    skill_id: str,
    version: str,
    fastapi_response: Response,
    request: Request,
    custom_llm_provider: str | None = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    data = _set_skill_route_params({}, request, skill_id)
    data["version"] = version
    data.setdefault("custom_llm_provider", custom_llm_provider)
    return await _process_skill_request(
        data=data,
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        route_type="adelete_skill_version",
    )
