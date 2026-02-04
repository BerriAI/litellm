"""
CRUD ENDPOINTS FOR PROMPTS
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.prompts.init_prompts import (
    ListPromptsResponse,
    PromptInfo,
    PromptInfoResponse,
    PromptLiteLLMParams,
    PromptSpec,
    PromptTemplateBase,
)
from litellm.types.proxy.prompt_endpoints import TestPromptRequest

router = APIRouter()


def get_base_prompt_id(prompt_id: str) -> str:
    """
    Extract the base prompt ID by stripping the version suffix if present.

    Args:
        prompt_id: Prompt ID that may include version suffix (e.g., "jack_success.v1" or "jack_success_v1")

    Returns:
        Base prompt ID without version suffix (e.g., "jack_success")

    Examples:
        >>> get_base_prompt_id("jack_success.v1")
        "jack_success"
        >>> get_base_prompt_id("jack_success_v1")
        "jack_success"
        >>> get_base_prompt_id("jack_success")
        "jack_success"
    """
    # Try dot separator first (.v)
    if ".v" in prompt_id:
        return prompt_id.split(".v")[0]
    # Try underscore separator (_v)
    if "_v" in prompt_id:
        return prompt_id.split("_v")[0]
    return prompt_id


def get_version_number(prompt_id: str) -> int:
    """
    Extract the version number from a versioned prompt ID.

    Args:
        prompt_id: Prompt ID that may include version suffix (e.g., "jack_success.v2" or "jack_success_v2")

    Returns:
        Version number (defaults to 1 if no version suffix or invalid format)

    Examples:
        >>> get_version_number("jack_success.v2")
        2
        >>> get_version_number("jack_success_v2")
        2
        >>> get_version_number("jack_success")
        1
    """
    # Try dot separator first (.v)
    if ".v" in prompt_id:
        version_str = prompt_id.split(".v")[1]
        try:
            return int(version_str)
        except ValueError:
            pass

    # Try underscore separator (_v)
    if "_v" in prompt_id:
        version_str = prompt_id.split("_v")[1]
        try:
            return int(version_str)
        except ValueError:
            pass

    return 1


def construct_versioned_prompt_id(prompt_id: str, version: Optional[int] = None) -> str:
    """
    Construct a versioned prompt ID from a base prompt_id and version number.

    Args:
        prompt_id: Base prompt ID (e.g., "jack_success")
        version: Version number (if None, returns the base prompt_id unchanged)

    Returns:
        Versioned prompt ID (e.g., "jack_success.v4")

    Examples:
        >>> construct_versioned_prompt_id("jack_success", 4)
        "jack_success.v4"
        >>> construct_versioned_prompt_id("jack_success", None)
        "jack_success"
        >>> construct_versioned_prompt_id("jack_success.v2", 4)
        "jack_success.v4"
    """
    if version is None:
        return prompt_id

    # Strip any existing version suffix first
    base_id = get_base_prompt_id(prompt_id)
    return f"{base_id}.v{version}"


def get_latest_version_prompt_id(prompt_id: str, all_prompt_ids: Dict[str, Any]) -> str:
    """
    Find the latest version of a prompt from available prompt IDs.

    Args:
        prompt_id: Base prompt ID or versioned prompt ID (e.g., "jack_success" or "jack_success.v2")
        all_prompt_ids: Dictionary of all available prompt IDs (keys are prompt IDs)

    Returns:
        The prompt ID with the highest version number, or the original prompt_id if no versions exist

    Examples:
        >>> all_ids = {"jack.v1": {}, "jack.v2": {}, "jack.v3": {}}
        >>> get_latest_version_prompt_id("jack", all_ids)
        "jack.v3"
        >>> get_latest_version_prompt_id("jack.v1", all_ids)
        "jack.v3"
        >>> all_ids = {"simple": {}}
        >>> get_latest_version_prompt_id("simple", all_ids)
        "simple"
    """
    base_id = get_base_prompt_id(prompt_id=prompt_id)

    # Find all versions of this prompt
    matching_versions = []
    for stored_prompt_id in all_prompt_ids.keys():
        if get_base_prompt_id(prompt_id=stored_prompt_id) == base_id:
            version_num = get_version_number(prompt_id=stored_prompt_id)
            matching_versions.append((version_num, stored_prompt_id))

    # Use the highest version number
    if matching_versions:
        matching_versions.sort(reverse=True)
        return matching_versions[0][1]
    else:
        # No versioned prompts found, use the base ID as-is
        return prompt_id


def get_latest_prompt_versions(prompts: List[PromptSpec]) -> List[PromptSpec]:
    """
    Filter a list of prompts to return only the latest version of each unique prompt.

    Args:
        prompts: List of PromptSpec objects

    Returns:
        List of PromptSpec objects with only the latest version of each prompt
    """
    latest_prompts: Dict[str, PromptSpec] = {}

    for prompt in prompts:
        base_id = get_base_prompt_id(prompt_id=prompt.prompt_id)
        version = get_version_number(prompt_id=prompt.prompt_id)

        # Keep the prompt with the highest version number
        if base_id not in latest_prompts:
            latest_prompts[base_id] = prompt
        else:
            existing_version = get_version_number(
                prompt_id=latest_prompts[base_id].prompt_id
            )
            if version > existing_version:
                latest_prompts[base_id] = prompt

    return list(latest_prompts.values())


async def get_next_version_for_prompt(prisma_client, prompt_id: str) -> int:
    """
    Get the next version number for a prompt.

    Args:
        prisma_client: Prisma database client
        prompt_id: Base prompt ID

    Returns:
        Next version number (1 if no versions exist, max_version + 1 otherwise)
    """
    existing_prompts = await prisma_client.db.litellm_prompttable.find_many(
        where={"prompt_id": prompt_id}
    )

    if existing_prompts:
        max_version = max(p.version for p in existing_prompts)
        return max_version + 1
    else:
        return 1


def create_versioned_prompt_spec(db_prompt) -> PromptSpec:
    """
    Helper function to create a PromptSpec with versioned prompt_id from a DB prompt entry.

    Args:
        db_prompt: The DB prompt object (from prisma)

    Returns:
        PromptSpec with versioned prompt_id (e.g., "chat_prompt.v1")
    """
    import json

    from litellm.types.prompts.init_prompts import PromptLiteLLMParams

    prompt_dict = db_prompt.model_dump()
    base_prompt_id = prompt_dict["prompt_id"]
    version = prompt_dict.get("version", 1)

    # Parse litellm_params
    litellm_params_data = prompt_dict.get("litellm_params")
    if isinstance(litellm_params_data, str):
        litellm_params_data = json.loads(litellm_params_data)
    litellm_params = PromptLiteLLMParams(**litellm_params_data)

    # Parse prompt_info
    prompt_info_data = prompt_dict.get("prompt_info")
    if prompt_info_data:
        if isinstance(prompt_info_data, str):
            prompt_info_data = json.loads(prompt_info_data)
        prompt_info = PromptInfo(**prompt_info_data)
    else:
        prompt_info = PromptInfo(prompt_type="db")

    # Create versioned prompt_id
    versioned_prompt_id = f"{base_prompt_id}.v{version}"

    return PromptSpec(
        prompt_id=versioned_prompt_id,
        litellm_params=litellm_params,
        prompt_info=prompt_info,
        created_at=prompt_dict.get("created_at"),
        updated_at=prompt_dict.get("updated_at"),
    )


class Prompt(BaseModel):
    prompt_id: str
    litellm_params: PromptLiteLLMParams
    prompt_info: Optional[PromptInfo] = None


class PatchPromptRequest(BaseModel):
    litellm_params: Optional[PromptLiteLLMParams] = None
    prompt_info: Optional[PromptInfo] = None


@router.get(
    "/prompts/list",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListPromptsResponse,
)
async def list_prompts(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List the prompts that are available on the proxy server

    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/prompts/list" -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "prompts": [
            {
                "prompt_id": "my_prompt_id",
                "litellm_params": {
                    "prompt_id": "my_prompt_id",
                    "prompt_integration": "dotprompt",
                    "prompt_directory": "/path/to/prompts"
                },
                "prompt_info": {
                    "prompt_type": "config"
                },
                "created_at": "2023-11-09T12:34:56.789Z",
                "updated_at": "2023-11-09T12:34:56.789Z"
            }
        ]
    }
    ```
    """
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    # check key metadata for prompts
    key_metadata = user_api_key_dict.metadata
    if key_metadata is not None:
        prompts = cast(Optional[List[str]], key_metadata.get("prompts", None))
        if prompts is not None:
            prompt_list = []
            for prompt_id in prompts:
                if prompt_id in IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS:
                    original_prompt = IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS[
                        prompt_id
                    ]
                    # Create a copy with base prompt_id (without version suffix)
                    prompt_copy = PromptSpec(
                        prompt_id=get_base_prompt_id(
                            prompt_id=original_prompt.prompt_id
                        ),
                        litellm_params=original_prompt.litellm_params,
                        prompt_info=original_prompt.prompt_info,
                        created_at=original_prompt.created_at,
                        updated_at=original_prompt.updated_at,
                    )
                    prompt_list.append(prompt_copy)
            return ListPromptsResponse(prompts=prompt_list)
    # check if user is proxy admin - show all prompts
    if user_api_key_dict.user_role is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        # Get all prompts and filter to show only the latest version of each
        all_prompts = list(IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.values())
        latest_prompts = get_latest_prompt_versions(prompts=all_prompts)
        # Create copies with base prompt_id (without version suffix) for display
        prompts_for_display = []
        for original_prompt in latest_prompts:
            prompt_copy = PromptSpec(
                prompt_id=get_base_prompt_id(prompt_id=original_prompt.prompt_id),
                litellm_params=original_prompt.litellm_params,
                prompt_info=original_prompt.prompt_info,
                created_at=original_prompt.created_at,
                updated_at=original_prompt.updated_at,
            )
            prompts_for_display.append(prompt_copy)
        return ListPromptsResponse(prompts=prompts_for_display)
    else:
        return ListPromptsResponse(prompts=[])


@router.get(
    "/prompts/{prompt_id}/versions",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListPromptsResponse,
)
async def get_prompt_versions(
    prompt_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get all versions of a specific prompt by base prompt ID
    
    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)
    
    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/prompts/jack_success/versions" \\
        -H "Authorization: Bearer <your_api_key>"
    ```
    
    Example Response:
    ```json
    {
        "prompts": [
            {
                "prompt_id": "jack_success.v1",
                "litellm_params": {...},
                "prompt_info": {"prompt_type": "db"},
                "created_at": "2023-11-09T12:34:56.789Z",
                "updated_at": "2023-11-09T12:34:56.789Z"
            },
            {
                "prompt_id": "jack_success.v2",
                "litellm_params": {...},
                "prompt_info": {"prompt_type": "db"},
                "created_at": "2023-11-09T13:45:12.345Z",
                "updated_at": "2023-11-09T13:45:12.345Z"
            }
        ]
    }
    ```
    """
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    # Only allow proxy admins to view version history
    if user_api_key_dict.user_role is None or (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
    ):
        raise HTTPException(
            status_code=403, detail="Only proxy admins can view prompt versions"
        )

    # Strip version suffix if provided (e.g., "jack_success.v1" -> "jack_success")
    base_prompt_id = get_base_prompt_id(prompt_id=prompt_id)

    # Get all prompts and filter by base_prompt_id
    all_prompts = list(IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.values())
    prompt_versions = [
        prompt
        for prompt in all_prompts
        if get_base_prompt_id(prompt_id=prompt.prompt_id) == base_prompt_id
    ]

    if not prompt_versions:
        raise HTTPException(
            status_code=404, detail=f"No versions found for prompt ID {base_prompt_id}"
        )

    # Create response with explicit version field for each prompt
    versioned_prompts = []
    for prompt in prompt_versions:
        # Extract version number from the root prompt_id which has version suffix
        # (e.g., "jack-sparrow.v3" -> 3)
        version_number = get_version_number(prompt_id=prompt.prompt_id)

        # Strip version from prompt_id for clean display
        base_prompt_id = get_base_prompt_id(prompt_id=prompt.prompt_id)

        # Create a copy with explicit version field and clean prompt_id
        versioned_prompt = PromptSpec(
            prompt_id=base_prompt_id,  # Clean ID without version (e.g., "jack-sparrow")
            litellm_params=prompt.litellm_params,
            prompt_info=prompt.prompt_info,
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
            version=version_number,  # Explicit version field (e.g., 3)
        )
        versioned_prompts.append(versioned_prompt)

    # Sort by version number (descending - newest first)
    versioned_prompts.sort(key=lambda p: p.version or 1, reverse=True)

    return ListPromptsResponse(prompts=versioned_prompts)


@router.get(
    "/prompts/{prompt_id}",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptInfoResponse,
)
@router.get(
    "/prompts/{prompt_id}/info",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptInfoResponse,
)
async def get_prompt_info(
    prompt_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get detailed information about a specific prompt by ID, including prompt content

    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)

    Example Request:
    ```bash
    curl -X GET "http://localhost:4000/prompts/my_prompt_id/info" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "prompt_id": "my_prompt_id",
        "litellm_params": {
            "prompt_id": "my_prompt_id",
            "prompt_integration": "dotprompt",
            "prompt_directory": "/path/to/prompts"
        },
        "prompt_info": {
            "prompt_type": "config"
        },
        "created_at": "2023-11-09T12:34:56.789Z",
        "updated_at": "2023-11-09T12:34:56.789Z",
        "content": "System: You are a helpful assistant.\n\nUser: {{user_message}}"
    }
    ```
    """
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY

    ## CHECK IF USER HAS ACCESS TO PROMPT
    prompts: Optional[List[str]] = None
    if user_api_key_dict.metadata is not None:
        prompts = cast(
            Optional[List[str]], user_api_key_dict.metadata.get("prompts", None)
        )
        if prompts is not None and prompt_id not in prompts:
            raise HTTPException(status_code=400, detail=f"Prompt {prompt_id} not found")
    if user_api_key_dict.user_role is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        pass
    else:
        raise HTTPException(
            status_code=403,
            detail=f"You are not authorized to access this prompt. Your role - {user_api_key_dict.user_role}, Your key's prompts - {prompts}",
        )

    # Try to get prompt directly first
    prompt_spec = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(prompt_id)

    # If not found, try to find the latest version
    if prompt_spec is None:
        latest_prompt_id = get_latest_version_prompt_id(
            prompt_id=prompt_id,
            all_prompt_ids=IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS,
        )
        prompt_spec = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(latest_prompt_id)

    if prompt_spec is None:
        raise HTTPException(status_code=400, detail=f"Prompt {prompt_id} not found")

    # Extract version number from the prompt_id
    version_number = get_version_number(prompt_id=prompt_spec.prompt_id)

    # Create a copy of the prompt spec with the base prompt ID (stripped of version)
    # and explicit version field for consistency with list_prompts and versions endpoints
    prompt_spec_response = PromptSpec(
        prompt_id=get_base_prompt_id(prompt_id=prompt_spec.prompt_id),
        litellm_params=prompt_spec.litellm_params,  # This preserves the versioned ID
        prompt_info=prompt_spec.prompt_info,
        created_at=prompt_spec.created_at,
        updated_at=prompt_spec.updated_at,
        version=version_number,  # Explicit version field
    )

    # Get prompt content from the callback
    prompt_template: Optional[PromptTemplateBase] = None
    try:
        prompt_callback = IN_MEMORY_PROMPT_REGISTRY.get_prompt_callback_by_id(
            prompt_spec.prompt_id
        )
        if prompt_callback is not None:
            # Extract content based on integration type
            integration_name = prompt_callback.integration_name

            if integration_name == "dotprompt":
                # For dotprompt integration, get content from the prompt manager
                from litellm.integrations.dotprompt.dotprompt_manager import (
                    DotpromptManager,
                )

                if isinstance(prompt_callback, DotpromptManager):
                    template = prompt_callback.prompt_manager.get_all_prompts_as_json()
                    if template is not None and len(template) == 1:
                        template_id = list(template.keys())[0]
                        prompt_template = PromptTemplateBase(
                            litellm_prompt_id=template_id,  # id sent to prompt management tool
                            content=template[template_id]["content"],
                            metadata=template[template_id]["metadata"],
                        )

    except Exception:
        # If content extraction fails, continue without content
        pass

    # Create response with content
    return PromptInfoResponse(
        prompt_spec=prompt_spec_response,
        raw_prompt_template=prompt_template,
    )


@router.post(
    "/prompts",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_prompt(
    request: Prompt,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new prompt

    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/prompts" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "prompt_id": "my_prompt",
            "litellm_params": {
                "prompt_id": "json_prompt",
                "prompt_integration": "dotprompt",
                ### EITHER prompt_directory OR prompt_data MUST BE PROVIDED
                "prompt_directory": "/path/to/dotprompt/folder",
                "prompt_data": {"json_prompt": {"content": "This is a prompt", "metadata": {"model": "gpt-4"}}}
            },
            "prompt_info": {
                "prompt_type": "config"
            }
        }'
    ```
    """

    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.proxy.proxy_server import prisma_client

    # Only allow proxy admins to create prompts
    if user_api_key_dict.user_role is None or (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
    ):
        raise HTTPException(
            status_code=403, detail="Only proxy admins can create prompts"
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Get next version number
        new_version = await get_next_version_for_prompt(
            prisma_client=prisma_client, prompt_id=request.prompt_id
        )

        # Store prompt in db with version
        prompt_db_entry = await prisma_client.db.litellm_prompttable.create(
            data={
                "prompt_id": request.prompt_id,
                "version": new_version,
                "litellm_params": request.litellm_params.model_dump_json(),
                "prompt_info": (
                    request.prompt_info.model_dump_json()
                    if request.prompt_info
                    else PromptInfo(prompt_type="db").model_dump_json()
                ),
            }
        )

        # Create versioned prompt spec
        prompt_spec = create_versioned_prompt_spec(db_prompt=prompt_db_entry)

        # Initialize the prompt
        initialized_prompt = IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
            prompt=prompt_spec, config_file_path=None
        )

        if initialized_prompt is None:
            raise HTTPException(status_code=500, detail="Failed to initialize prompt")

        return initialized_prompt

    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/prompts/{prompt_id}",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_prompt(
    prompt_id: str,
    request: Prompt,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing prompt

    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)

    Example Request:
    ```bash
    curl -X PUT "http://localhost:4000/prompts/my_prompt_id" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "prompt_id": "my_prompt",
            "litellm_params": {
                "prompt_id": "my_prompt",
                    "prompt_integration": "dotprompt",
                    "prompt_directory": "/path/to/prompts"
                },
                "prompt_info": {
                    "prompt_type": "config"
                }
            }
        }'
    ```
    """
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.proxy.proxy_server import prisma_client

    # Only allow proxy admins to update prompts
    if user_api_key_dict.user_role is None or (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
    ):
        raise HTTPException(
            status_code=403, detail="Only proxy admins can update prompts"
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Strip version suffix from prompt_id if present (e.g., "jack_success.v1" -> "jack_success")
        base_prompt_id = get_base_prompt_id(prompt_id=prompt_id)

        # Check if any version exists
        existing_prompts = await prisma_client.db.litellm_prompttable.find_many(
            where={"prompt_id": base_prompt_id}
        )

        if not existing_prompts:
            raise HTTPException(
                status_code=404, detail=f"Prompt with ID {base_prompt_id} not found"
            )

        # Check if it's a config prompt
        existing_in_memory = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(prompt_id)
        if (
            existing_in_memory
            and existing_in_memory.prompt_info.prompt_type == "config"
        ):
            raise HTTPException(
                status_code=400,
                detail="Cannot update config prompts.",
            )

        # Get next version number (UPDATE creates a new version)
        new_version = await get_next_version_for_prompt(
            prisma_client=prisma_client, prompt_id=base_prompt_id
        )

        # Store new version in db
        prompt_db_entry = await prisma_client.db.litellm_prompttable.create(
            data={
                "prompt_id": base_prompt_id,
                "version": new_version,
                "litellm_params": request.litellm_params.model_dump_json(),
                "prompt_info": (
                    request.prompt_info.model_dump_json()
                    if request.prompt_info
                    else PromptInfo(prompt_type="db").model_dump_json()
                ),
            }
        )

        # Create versioned prompt spec
        prompt_spec = create_versioned_prompt_spec(db_prompt=prompt_db_entry)

        # Initialize the new version
        initialized_prompt = IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
            prompt=prompt_spec, config_file_path=None
        )

        if initialized_prompt is None:
            raise HTTPException(status_code=500, detail="Failed to update prompt")

        return initialized_prompt

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/prompts/{prompt_id}",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_prompt(
    prompt_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a prompt

    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)

    Example Request:
    ```bash
    curl -X DELETE "http://localhost:4000/prompts/my_prompt_id" \\
        -H "Authorization: Bearer <your_api_key>"
    ```

    Example Response:
    ```json
    {
        "message": "Prompt my_prompt_id deleted successfully"
    }
    ```
    """
    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.proxy.proxy_server import prisma_client

    # Only allow proxy admins to delete prompts
    if user_api_key_dict.user_role is None or (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
    ):
        raise HTTPException(
            status_code=403, detail="Only proxy admins can delete prompts"
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Try to get prompt directly first
        existing_prompt = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(prompt_id)

        # If not found, try to find the latest version
        if existing_prompt is None:
            latest_prompt_id = get_latest_version_prompt_id(
                prompt_id=prompt_id,
                all_prompt_ids=IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS,
            )
            existing_prompt = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(
                latest_prompt_id
            )
            # Use the resolved prompt_id for deletion
            prompt_id = latest_prompt_id

        if existing_prompt is None:
            raise HTTPException(
                status_code=404, detail=f"Prompt with ID {prompt_id} not found"
            )

        if existing_prompt.prompt_info.prompt_type == "config":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete config prompts.",
            )

        # Get the base prompt ID (without version suffix) for database deletion
        base_prompt_id = get_base_prompt_id(prompt_id=prompt_id)

        # Delete all versions of the prompt from the database
        await prisma_client.db.litellm_prompttable.delete_many(
            where={"prompt_id": base_prompt_id}
        )

        # Remove all versions of the prompt from memory
        IN_MEMORY_PROMPT_REGISTRY.delete_prompts_by_base_id(base_prompt_id)

        return {"message": f"Prompt {base_prompt_id} deleted successfully"}

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/prompts/{prompt_id}",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def patch_prompt(
    prompt_id: str,
    request: PatchPromptRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Partially update an existing prompt

    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)

    This endpoint allows updating specific fields of a prompt without sending the entire object.
    Only the following fields can be updated:
    - litellm_params: LiteLLM parameters for the prompt
    - prompt_info: Additional information about the prompt

    Example Request:
    ```bash
    curl -X PATCH "http://localhost:4000/prompts/my_prompt_id" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "prompt_info": {
                "prompt_type": "db"
            }
        }'
    ```
    """

    from litellm.proxy.prompts.prompt_registry import IN_MEMORY_PROMPT_REGISTRY
    from litellm.proxy.proxy_server import prisma_client

    # Only allow proxy admins to patch prompts
    if user_api_key_dict.user_role is None or (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
    ):
        raise HTTPException(
            status_code=403, detail="Only proxy admins can patch prompts"
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        # Check if prompt exists and get current data
        existing_prompt = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(prompt_id)
        if existing_prompt is None:
            raise HTTPException(
                status_code=404, detail=f"Prompt with ID {prompt_id} not found"
            )

        if existing_prompt.prompt_info.prompt_type == "config":
            raise HTTPException(
                status_code=400,
                detail="Cannot update config prompts.",
            )

        # Update fields if provided
        updated_litellm_params = (
            request.litellm_params
            if request.litellm_params is not None
            else existing_prompt.litellm_params
        )

        updated_prompt_info = (
            request.prompt_info
            if request.prompt_info is not None
            else existing_prompt.prompt_info
        )

        # Ensure we have valid litellm_params
        if updated_litellm_params is None:
            raise HTTPException(status_code=400, detail="litellm_params cannot be None")

        # Create updated prompt spec - cast to satisfy typing
        updated_prompt_db_entry = await prisma_client.db.litellm_prompttable.update(
            where={"prompt_id": prompt_id},
            data={
                "litellm_params": updated_litellm_params.model_dump_json(),
                "prompt_info": updated_prompt_info.model_dump_json(),
            },
        )

        updated_prompt_spec = PromptSpec(**updated_prompt_db_entry.model_dump())

        # Remove the old prompt from memory
        del IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS[prompt_id]
        if prompt_id in IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt:
            del IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt[prompt_id]

        # Initialize the updated prompt
        initialized_prompt = IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
            prompt=updated_prompt_spec, config_file_path=None
        )

        if initialized_prompt is None:
            raise HTTPException(status_code=500, detail="Failed to patch prompt")

        return initialized_prompt

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error patching prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/prompts/test",
    tags=["Prompt Management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def test_prompt(
    request: TestPromptRequest,
    fastapi_request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Test a prompt by rendering it with variables and executing an LLM call.
    
    This endpoint allows testing prompts before saving them to the database.
    The response is always streamed.
    
    👉 [Prompt docs](https://docs.litellm.ai/docs/proxy/prompt_management)
    
    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/prompts/test" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "dotprompt_content": "---\\nmodel: gpt-4o\\ntemperature: 0.7\\n---\\n\\nUser: Hello {{name}}",
            "prompt_variables": {
                "name": "World"
            }
        }'
    ```
    """
    from pydantic import BaseModel

    from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager
    from litellm.integrations.dotprompt.prompt_manager import (
        PromptManager,
        PromptTemplate,
    )
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
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

    try:
        # Parse the dotprompt content and create PromptTemplate
        prompt_manager = PromptManager()
        frontmatter, template_content = prompt_manager._parse_frontmatter(
            content=request.dotprompt_content
        )

        # Create PromptTemplate to leverage existing parameter extraction logic
        template = PromptTemplate(
            content=template_content, metadata=frontmatter, template_id="test_prompt"
        )

        # Extract model from template
        if not template.model:
            raise HTTPException(
                status_code=400, detail="Model is required in dotprompt metadata"
            )

        # Always render the template to extract system messages and other metadata
        variables = request.prompt_variables or {}
        rendered_content = prompt_manager.jinja_env.from_string(
            template_content
        ).render(**variables)

        # Convert rendered content to messages using DotpromptManager's method
        dotprompt_manager = DotpromptManager()
        rendered_messages = dotprompt_manager._convert_to_messages(
            rendered_content=rendered_content
        )

        if not rendered_messages:
            raise HTTPException(
                status_code=400, detail="No messages found in rendered prompt"
            )

        # If conversation history is provided, use it but preserve system messages
        if request.conversation_history:
            # Extract system messages from rendered prompt
            system_messages = [
                msg for msg in rendered_messages if msg.get("role") == "system"
            ]
            # Use conversation history for user/assistant messages
            messages = system_messages + request.conversation_history
        else:
            messages = rendered_messages  # type: ignore[assignment]

        # Use PromptTemplate's optional_params which already extracts all parameters
        optional_params = template.optional_params.copy()

        # Always stream the response
        optional_params["stream"] = True

        # Build request data for chat completion
        data = {
            "model": template.model,
            "messages": messages,
        }
        data.update(optional_params)

        # Use ProxyBaseLLMRequestProcessing to go through all proxy logic
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        result = await base_llm_response_processor.base_process_llm_request(
            request=fastapi_request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acompletion",
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

        if isinstance(result, BaseModel):
            return result.model_dump(exclude_none=True, exclude_unset=True)
        else:
            return result

    except HTTPException as e:
        raise e
    except Exception as e:
        verbose_proxy_logger.exception(f"Error testing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/utils/dotprompt_json_converter",
    tags=["prompts", "utils"],
    dependencies=[Depends(user_api_key_auth)],
)
async def convert_prompt_file_to_json(
    file: UploadFile = File(...),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """
    Convert a .prompt file to JSON format.

    This endpoint accepts a .prompt file upload and returns the equivalent JSON representation
    that can be stored in a database or used programmatically.

    Returns the JSON structure with 'content' and 'metadata' fields.
    """
    global general_settings
    from litellm.integrations.dotprompt.prompt_manager import PromptManager

    # Validate file extension
    if not file.filename or not file.filename.endswith(".prompt"):
        raise HTTPException(status_code=400, detail="File must have .prompt extension")

    temp_file_path = None
    try:
        # Read file content
        file_content = await file.read()

        # Create temporary file
        temp_file_path = Path(tempfile.mkdtemp()) / file.filename
        temp_file_path.write_bytes(file_content)

        # Create a PromptManager instance just for conversion
        prompt_manager = PromptManager()

        # Convert to JSON
        json_data = prompt_manager.prompt_file_to_json(temp_file_path)

        # Extract prompt ID from filename
        prompt_id = temp_file_path.stem

        return {
            "prompt_id": prompt_id,
            "json_data": json_data,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error converting prompt file: {str(e)}"
        )

    finally:
        # Clean up temp file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()
            # Also try to remove the temp directory if it's empty
            try:
                temp_file_path.parent.rmdir()
            except OSError:
                pass  # Directory not empty or other error
