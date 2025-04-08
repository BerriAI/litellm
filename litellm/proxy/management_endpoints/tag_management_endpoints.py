"""
TAG MANAGEMENT

All /tag management endpoints 

/tag/new   
/tag/info
/tag/update
/tag/delete
/tag/list
"""

import datetime
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.tag_management import (
    TagBase,
    TagDeleteRequest,
    TagInfoRequest,
    TagNewRequest,
    TagUpdateRequest,
)

router = APIRouter()


async def _get_tags_config(prisma_client) -> Dict[str, Any]:
    """Helper function to get tags config from db"""
    try:
        tags_config = await prisma_client.db.litellm_config.find_unique(
            where={"param_name": "tags_config"}
        )
        if tags_config is None:
            return {}
        # Convert from JSON if needed
        if isinstance(tags_config.param_value, str):
            return json.loads(tags_config.param_value)
        return tags_config.param_value or {}
    except Exception:
        return {}


async def _save_tags_config(prisma_client, tags_config: Dict[str, Any]):
    """Helper function to save tags config to db"""
    try:
        verbose_proxy_logger.debug(f"Saving tags config: {tags_config}")
        json_tags_config = json.dumps(tags_config, default=str)
        verbose_proxy_logger.debug(f"JSON tags config: {json_tags_config}")
        await prisma_client.db.litellm_config.upsert(
            where={"param_name": "tags_config"},
            data={
                "create": {
                    "param_name": "tags_config",
                    "param_value": json_tags_config,
                },
                "update": {"param_value": json_tags_config},
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error saving tags config: {str(e)}"
        )


@router.post(
    "/tag/new",
    tags=["tag management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_tag(
    tag: TagNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new tag.

    Parameters:
    - name: str - The name of the tag
    - description: Optional[str] - Description of what this tag represents
    - allowed_llms: List[str] - List of LLM models allowed for this tag
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    try:
        # Get existing tags config
        tags_config = await _get_tags_config(prisma_client)

        # Check if tag already exists
        if tag.name in tags_config:
            raise HTTPException(
                status_code=400, detail=f"Tag {tag.name} already exists"
            )

        # Add new tag
        tags_config[tag.name] = {
            "description": tag.description,
            "allowed_llms": tag.allowed_llms,
            "created_at": str(datetime.datetime.now()),
            "updated_at": str(datetime.datetime.now()),
            "created_by": user_api_key_dict.user_id,
        }

        # Save updated config
        await _save_tags_config(prisma_client, tags_config)

        return {"message": f"Tag {tag.name} created successfully"}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating tag: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tag/update",
    tags=["tag management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_tag(
    tag: TagUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing tag.

    Parameters:
    - name: str - The name of the tag to update
    - description: Optional[str] - Updated description
    - allowed_llms: List[str] - Updated list of allowed LLM models
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Get existing tags config
        tags_config = await _get_tags_config(prisma_client)

        # Check if tag exists
        if tag.name not in tags_config:
            raise HTTPException(status_code=404, detail=f"Tag {tag.name} not found")

        # Update tag
        tags_config[tag.name].update(
            {
                "description": tag.description,
                "allowed_llms": tag.allowed_llms,
                "updated_at": str(datetime.datetime.now()),
                "updated_by": user_api_key_dict.user_id,
            }
        )

        # Save updated config
        await _save_tags_config(prisma_client, tags_config)

        return {"message": f"Tag {tag.name} updated successfully"}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error updating tag: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tag/info",
    tags=["tag management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_tag(
    data: TagInfoRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get information about specific tags.

    Parameters:
    - names: List[str] - List of tag names to get information for
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        tags_config = await _get_tags_config(prisma_client)

        # Filter tags based on requested names
        requested_tags = {name: tags_config.get(name) for name in data.names}

        # Check if any requested tags don't exist
        missing_tags = [name for name in data.names if name not in tags_config]
        if missing_tags:
            raise HTTPException(
                status_code=404, detail=f"Tags not found: {missing_tags}"
            )

        return requested_tags
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/tag/list",
    tags=["tag management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_tags(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all available tags.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        tags_config = await _get_tags_config(prisma_client)
        return tags_config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tag/delete",
    tags=["tag management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_tag(
    data: TagDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a tag.

    Parameters:
    - name: str - The name of the tag to delete
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Get existing tags config
        tags_config = await _get_tags_config(prisma_client)

        # Check if tag exists
        if data.name not in tags_config:
            raise HTTPException(status_code=404, detail=f"Tag {data.name} not found")

        # Delete tag
        del tags_config[data.name]

        # Save updated config
        await _save_tags_config(prisma_client, tags_config)

        return {"message": f"Tag {data.name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
