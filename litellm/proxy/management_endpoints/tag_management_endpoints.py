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
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.tag_management import (
    TagConfig,
    TagDeleteRequest,
    TagInfoRequest,
    TagNewRequest,
    TagUpdateRequest,
)

router = APIRouter()


async def _get_model_names(prisma_client, model_ids: list) -> Dict[str, str]:
    """Helper function to get model names from model IDs"""
    try:
        models = await prisma_client.db.litellm_proxymodeltable.find_many(
            where={"model_id": {"in": model_ids}}
        )
        return {model.model_id: model.model_name for model in models}
    except Exception as e:
        verbose_proxy_logger.error(f"Error getting model names: {str(e)}")
        return {}


async def _get_tags_config(prisma_client) -> Dict[str, TagConfig]:
    """Helper function to get tags config from db"""
    try:
        tags_config = await prisma_client.db.litellm_config.find_unique(
            where={"param_name": "tags_config"}
        )
        if tags_config is None:
            return {}
        # Convert from JSON if needed
        if isinstance(tags_config.param_value, str):
            config_dict = json.loads(tags_config.param_value)
        else:
            config_dict = tags_config.param_value or {}

        # For each tag, get the model names
        for tag_name, tag_config in config_dict.items():
            if isinstance(tag_config, dict) and tag_config.get("models"):
                model_info = await _get_model_names(prisma_client, tag_config["models"])
                tag_config["model_info"] = model_info

        return config_dict
    except Exception:
        return {}


async def _save_tags_config(prisma_client, tags_config: Dict[str, TagConfig]):
    """Helper function to save tags config to db"""
    try:
        verbose_proxy_logger.debug(f"Saving tags config: {tags_config}")
        # Convert TagConfig objects to dictionaries
        tags_config_dict = {}
        for name, tag in tags_config.items():
            if isinstance(tag, TagConfig):
                tag_dict = tag.model_dump()
                # Remove model_info before saving as it will be dynamically generated
                if "model_info" in tag_dict:
                    del tag_dict["model_info"]
                tags_config_dict[name] = tag_dict
            else:
                # If it's already a dict, remove model_info
                tag_copy = tag.copy()
                if "model_info" in tag_copy:
                    del tag_copy["model_info"]
                tags_config_dict[name] = tag_copy

        json_tags_config = json.dumps(tags_config_dict, default=str)
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
    - models: List[str] - List of LLM models allowed for this tag
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
        tags_config[tag.name] = TagConfig(
            name=tag.name,
            description=tag.description,
            models=tag.models,
            created_at=str(datetime.datetime.now()),
            updated_at=str(datetime.datetime.now()),
            created_by=user_api_key_dict.user_id,
        )

        # Save updated config
        await _save_tags_config(
            prisma_client=prisma_client,
            tags_config=tags_config,
        )

        # Update models with new tag
        if tag.models:
            for model_id in tag.models:
                await _add_tag_to_deployment(
                    model_id=model_id,
                    tag=tag.name,
                )

        # Get model names for response
        model_info = await _get_model_names(prisma_client, tag.models or [])
        tags_config[tag.name].model_info = model_info

        return {
            "message": f"Tag {tag.name} created successfully",
            "tag": tags_config[tag.name],
        }
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating tag: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _add_tag_to_deployment(model_id: str, tag: str):
    """Helper function to add tag to deployment"""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    deployment = await prisma_client.db.litellm_proxymodeltable.find_unique(
        where={"model_id": model_id}
    )
    if deployment is None:
        raise HTTPException(status_code=404, detail=f"Deployment {model_id} not found")

    litellm_params = deployment.litellm_params
    if "tags" not in litellm_params:
        litellm_params["tags"] = []
    litellm_params["tags"].append(tag)
    await prisma_client.db.litellm_proxymodeltable.update(
        where={"model_id": model_id},
        data={"litellm_params": safe_dumps(litellm_params)},
    )


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
    - models: List[str] - Updated list of allowed LLM models
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
        tag_config_dict = dict(tags_config[tag.name])
        tag_config_dict.update(
            {
                "description": tag.description,
                "models": tag.models,
                "updated_at": str(datetime.datetime.now()),
                "updated_by": user_api_key_dict.user_id,
            }
        )
        tags_config[tag.name] = TagConfig(**tag_config_dict)

        # Save updated config
        await _save_tags_config(prisma_client, tags_config)

        # Get model names for response
        model_info = await _get_model_names(prisma_client, tag.models or [])
        tags_config[tag.name].model_info = model_info

        return {
            "message": f"Tag {tag.name} updated successfully",
            "tag": tags_config[tag.name],
        }
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
        list_of_tags = list(tags_config.values())
        return list_of_tags
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
