"""
Skills API endpoints - /v1/skills

Supports two modes controlled by litellm_settings.skills_mode:
- "litellm": Skills stored in LiteLLM DB, works with any model provider
- "passthrough": Pass-through to Anthropic API (requires Anthropic model)
"""

from typing import Literal, Optional

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import NewSkillRequest, UserAPIKeyAuth
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

router = APIRouter()


def get_skills_mode() -> Literal["litellm", "passthrough"]:
    """
    Get the skills_mode from litellm_settings.

    Returns:
        "litellm" - Skills managed by LiteLLM (stored in DB, works with any provider)
        "passthrough" - Pass-through to Anthropic API (default for backwards compatibility)
    """
    from litellm.proxy.proxy_server import general_settings

    # Check general_settings for skills_mode
    skills_mode = general_settings.get("skills_mode")

    if skills_mode is None:
        return "passthrough"

    if skills_mode not in ("litellm", "passthrough"):
        verbose_proxy_logger.warning(
            f"Invalid skills_mode '{skills_mode}', defaulting to 'passthrough'"
        )
        return "passthrough"

    return skills_mode


async def _handle_litellm_create_skill(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
) -> Skill:
    """Handle skill creation in LiteLLM mode (local DB storage)."""
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
    from litellm.proxy.skills_endpoints.validation import validate_skill_files

    # Parse form data
    form_data = await get_form_data(request)

    # Get display_title override if provided
    display_title_override = form_data.get("display_title")

    # Get files from form data
    files_data = form_data.get("files[]", [])
    if not files_data:
        files_data = form_data.get("files", [])

    if not files_data:
        raise HTTPException(
            status_code=400,
            detail="No files provided. SKILL.md is required.",
        )

    # Normalize to list if single file
    if not isinstance(files_data, list):
        files_data = [files_data]

    # Read file contents
    file_tuples = []
    for file_item in files_data:
        if isinstance(file_item, UploadFile):
            content = await file_item.read()
            filename = file_item.filename or "unknown"
            file_tuples.append((filename, content))
        elif isinstance(file_item, tuple) and len(file_item) >= 2:
            filename, content = file_item[0], file_item[1]
            if isinstance(content, str):
                content = content.encode("utf-8")
            file_tuples.append((filename, content))

    if not file_tuples:
        raise HTTPException(
            status_code=400,
            detail="No valid files provided. SKILL.md is required.",
        )

    # Validate files and create ZIP
    zip_content, frontmatter, body, errors = validate_skill_files(file_tuples)

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"errors": errors},
        )

    assert zip_content is not None
    assert frontmatter is not None

    # Create skill request
    skill_request = NewSkillRequest(
        display_title=display_title_override or frontmatter.name,
        description=frontmatter.description,
        instructions=body,
        file_content=zip_content,
        file_name="skill.zip",
        file_type="application/zip",
    )

    # Create skill in DB
    skill_record = await LiteLLMSkillsHandler.create_skill(
        data=skill_request,
        user_id=user_api_key_dict.user_id,
    )

    verbose_proxy_logger.debug(f"Created LiteLLM skill: {skill_record.skill_id}")

    return Skill(
        id=skill_record.skill_id,
        display_title=skill_record.display_title,
        source=skill_record.source,
        latest_version=skill_record.latest_version,
        created_at=skill_record.created_at.isoformat()
        if skill_record.created_at
        else "",
        updated_at=skill_record.updated_at.isoformat()
        if skill_record.updated_at
        else "",
    )


async def _handle_litellm_list_skills(
    limit: int = 20,
    page: Optional[str] = None,
) -> ListSkillsResponse:
    """Handle skill listing in LiteLLM mode (local DB)."""
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    # Clamp limit
    limit = max(1, min(limit, 100))

    # Parse page to offset
    offset = 0
    if page:
        try:
            offset = int(page)
        except ValueError:
            pass

    # Fetch from DB
    skills = await LiteLLMSkillsHandler.list_skills(
        limit=limit + 1,
        offset=offset,
    )

    has_more = len(skills) > limit
    if has_more:
        skills = skills[:limit]

    skill_responses = [
        Skill(
            id=s.skill_id,
            display_title=s.display_title,
            source=s.source,
            latest_version=s.latest_version,
            created_at=s.created_at.isoformat() if s.created_at else "",
            updated_at=s.updated_at.isoformat() if s.updated_at else "",
        )
        for s in skills
    ]

    return ListSkillsResponse(
        data=skill_responses,
        has_more=has_more,
        next_page=str(offset + limit) if has_more else None,
    )


async def _handle_litellm_get_skill(skill_id: str) -> Skill:
    """Handle skill retrieval in LiteLLM mode (local DB)."""
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    try:
        skill = await LiteLLMSkillsHandler.get_skill(skill_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_id}")

    return Skill(
        id=skill.skill_id,
        display_title=skill.display_title,
        source=skill.source,
        latest_version=skill.latest_version,
        created_at=skill.created_at.isoformat() if skill.created_at else "",
        updated_at=skill.updated_at.isoformat() if skill.updated_at else "",
    )


async def _handle_litellm_delete_skill(skill_id: str) -> DeleteSkillResponse:
    """Handle skill deletion in LiteLLM mode (local DB)."""
    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

    try:
        result = await LiteLLMSkillsHandler.delete_skill(skill_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_id}")

    return DeleteSkillResponse(id=result["id"], type=result["type"])


@router.post(
    "/v1/skills",
    tags=["[beta] Skills API"],
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
    Create a new skill.

    Behavior depends on `litellm_settings.skills_mode`:
    - "litellm": Stores skill in LiteLLM DB, works with any provider
    - "passthrough": Creates skill on Anthropic (requires Anthropic model)

    SKILL.md must have YAML frontmatter (for litellm mode):
    ```yaml
    ---
    name: My Skill (max 64 chars)
    description: What this skill does (max 1024 chars, optional)
    ---
    ```

    Example usage:
    ```bash
    curl -X POST "http://localhost:4000/v1/skills" \\
      -H "Content-Type: multipart/form-data" \\
      -H "Authorization: Bearer your-key" \\
      -F "display_title=My Skill" \\
      -F "files[]=@SKILL.md"
    ```

    Returns: Skill object with id, display_title, etc.
    """
    # Check skills mode
    skills_mode = get_skills_mode()

    if skills_mode == "litellm":
        return await _handle_litellm_create_skill(request, user_api_key_dict)

    # Passthrough mode - forward to Anthropic
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
    tags=["[beta] Skills API"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListSkillsResponse,
)
async def list_skills(
    fastapi_response: Response,
    request: Request,
    limit: Optional[int] = 20,
    page: Optional[str] = None,
    after_id: Optional[str] = None,
    before_id: Optional[str] = None,
    custom_llm_provider: Optional[str] = "anthropic",
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List skills.

    Behavior depends on `litellm_settings.skills_mode`:
    - "litellm": Lists skills from LiteLLM DB
    - "passthrough": Lists skills from Anthropic

    Query parameters:
    - limit: Number of results (default 20, max 100)
    - page: Pagination token (litellm mode only)

    Example usage:
    ```bash
    curl "http://localhost:4000/v1/skills?limit=10" \\
      -H "Authorization: Bearer your-key"
    ```

    Returns: ListSkillsResponse with list of skills
    """
    # Check skills mode
    skills_mode = get_skills_mode()

    if skills_mode == "litellm":
        return await _handle_litellm_list_skills(limit=limit or 20, page=page)

    # Passthrough mode - forward to Anthropic
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
    if "after_id" not in data and after_id is not None:
        data["after_id"] = after_id
    if "before_id" not in data and before_id is not None:
        data["before_id"] = before_id

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
    tags=["[beta] Skills API"],
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
    Get a specific skill by ID.

    Behavior depends on `litellm_settings.skills_mode`:
    - "litellm": Gets skill from LiteLLM DB
    - "passthrough": Gets skill from Anthropic

    Example usage:
    ```bash
    curl "http://localhost:4000/v1/skills/litellm_skill_123" \\
      -H "Authorization: Bearer your-key"
    ```

    Returns: Skill object
    """
    # Check skills mode
    skills_mode = get_skills_mode()

    if skills_mode == "litellm":
        return await _handle_litellm_get_skill(skill_id)

    # Passthrough mode - forward to Anthropic
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
    tags=["[beta] Skills API"],
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
    Delete a skill by ID.

    Behavior depends on `litellm_settings.skills_mode`:
    - "litellm": Deletes skill from LiteLLM DB
    - "passthrough": Deletes skill from Anthropic

    Example usage:
    ```bash
    curl -X DELETE "http://localhost:4000/v1/skills/litellm_skill_123" \\
      -H "Authorization: Bearer your-key"
    ```

    Returns: DeleteSkillResponse with type="skill_deleted"
    """
    # Check skills mode
    skills_mode = get_skills_mode()

    if skills_mode == "litellm":
        return await _handle_litellm_delete_skill(skill_id)

    # Passthrough mode - forward to Anthropic
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
