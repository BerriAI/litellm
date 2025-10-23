"""
CRUD ENDPOINTS FOR PROMPTS
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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


router = APIRouter()


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
            return ListPromptsResponse(
                prompts=[
                    IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS[prompt]
                    for prompt in prompts
                    if prompt in IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS
                ]
            )
    # check if user is proxy admin - show all prompts
    if user_api_key_dict.user_role is not None and (
        user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    ):
        return ListPromptsResponse(
            prompts=list(IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS.values())
        )
    else:
        return ListPromptsResponse(prompts=[])


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

    prompt_spec = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(prompt_id)
    if prompt_spec is None:
        raise HTTPException(status_code=400, detail=f"Prompt {prompt_id} not found")

    # Get prompt content from the callback
    prompt_template: Optional[PromptTemplateBase] = None
    try:
        prompt_callback = IN_MEMORY_PROMPT_REGISTRY.get_prompt_callback_by_id(prompt_id)
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
        prompt_spec=prompt_spec,
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
        # Create the prompt spec
        # Check if prompt exists and get current data
        existing_prompt = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(request.prompt_id)
        if existing_prompt is not None:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt with ID {request.prompt_id} already exists",
            )

        # store prompt in db
        prompt_db_entry = await prisma_client.db.litellm_prompttable.create(
            data={
                "prompt_id": request.prompt_id,
                "litellm_params": request.litellm_params.model_dump_json(),
                "prompt_info": (
                    request.prompt_info.model_dump_json()
                    if request.prompt_info
                    else PromptInfo(prompt_type="db").model_dump_json()
                ),
            }
        )

        prompt_spec = PromptSpec(**prompt_db_entry.model_dump())

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
    from datetime import datetime

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
        # Check if prompt exists
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

        # Create updated prompt spec
        updated_prompt_spec = PromptSpec(
            prompt_id=prompt_id,
            litellm_params=request.litellm_params,
            prompt_info=request.prompt_info or PromptInfo(prompt_type="db"),
            created_at=existing_prompt.created_at,
            updated_at=datetime.now(),
        )

        updated_prompt_db_entry = await prisma_client.db.litellm_prompttable.update(
            where={"prompt_id": prompt_id},
            data={
                "litellm_params": updated_prompt_spec.litellm_params.model_dump_json(),
                "prompt_info": updated_prompt_spec.prompt_info.model_dump_json(),
            },
        )

        # Remove the old prompt from memory
        del IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS[prompt_id]
        if prompt_id in IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt:
            del IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt[prompt_id]

        # Initialize the updated prompt
        initialized_prompt = IN_MEMORY_PROMPT_REGISTRY.initialize_prompt(
            prompt=PromptSpec(**updated_prompt_db_entry.model_dump()),
            config_file_path=None,
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
        # Check if prompt exists
        existing_prompt = IN_MEMORY_PROMPT_REGISTRY.get_prompt_by_id(prompt_id)
        if existing_prompt is None:
            raise HTTPException(
                status_code=404, detail=f"Prompt with ID {prompt_id} not found"
            )

        if existing_prompt.prompt_info.prompt_type == "config":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete config prompts.",
            )

        # Delete the prompt from the database
        await prisma_client.db.litellm_prompttable.delete(
            where={"prompt_id": prompt_id}
        )

        # Remove the prompt from memory
        del IN_MEMORY_PROMPT_REGISTRY.IN_MEMORY_PROMPTS[prompt_id]
        if prompt_id in IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt:
            del IN_MEMORY_PROMPT_REGISTRY.prompt_id_to_custom_prompt[prompt_id]

        return {"message": f"Prompt {prompt_id} deleted successfully"}

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

