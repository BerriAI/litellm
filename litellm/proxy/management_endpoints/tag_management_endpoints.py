"""
TAG MANAGEMENT

All /tag management endpoints

/tag/new
/tag/info
/tag/update
/tag/delete
/tag/list
"""

import asyncio
import json
from typing import TYPE_CHECKING, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
    compute_tag_metadata_totals,
    get_daily_activity,
)
from litellm.proxy.management_helpers.utils import handle_budget_for_entity
from litellm.types.tag_management import (
    LiteLLM_DailyTagSpendTable,
    TagConfig,
    TagDeleteRequest,
    TagInfoRequest,
    TagNewRequest,
    TagUpdateRequest,
)

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.router import Deployment

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


async def get_deployments_by_model(
    model: str, llm_router: "Router"
) -> List["Deployment"]:
    """
    Get all deployments by model
    """
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    # Check if model id
    deployment = llm_router.get_deployment(model_id=model)
    if deployment is not None:
        return [deployment]

    # Check if model name
    deployments = llm_router.get_model_list(model_name=model)
    if deployments is None:
        return []
    return [
        Deployment(
            model_name=deployment["model_name"],
            litellm_params=LiteLLM_Params(**deployment["litellm_params"]),  # type: ignore
            model_info=ModelInfo(**deployment.get("model_info") or {}),
        )
        for deployment in deployments
    ]


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
    - models: List[str] - List of either 'model_id' or 'model_name' allowed for this tag
    - budget_id: Optional[str] - The id for a budget (tpm/rpm/max budget) for the tag
    
    ### IF NO BUDGET ID - CREATE ONE WITH THESE PARAMS ###
    - max_budget: Optional[float] - Max budget for tag
    - tpm_limit: Optional[int] - Max tpm limit for tag
    - rpm_limit: Optional[int] - Max rpm limit for tag
    - max_parallel_requests: Optional[int] - Max parallel requests for tag
    - soft_budget: Optional[float] - Get a slack alert when this soft budget is reached
    - model_max_budget: Optional[dict] - Max budget for a specific model
    - budget_duration: Optional[str] - Frequency of resetting tag budget
    """
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        llm_router,
        prisma_client,
    )

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )
    if llm_router is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.no_llm_router.value
        )
    try:
        # Check if tag already exists
        existing_tag = await prisma_client.db.litellm_tagtable.find_unique(
            where={"tag_name": tag.name}
        )
        if existing_tag is not None:
            raise HTTPException(
                status_code=400, detail=f"Tag {tag.name} already exists"
            )

        # Handle budget creation/assignment using common helper
        budget_id = await handle_budget_for_entity(
            data=tag,
            existing_budget_id=None,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        )

        # Get model names for model_info
        model_info = await _get_model_names(prisma_client, tag.models or [])

        # Create new tag in database
        new_tag_record = await prisma_client.db.litellm_tagtable.create(
            data={
                "tag_name": tag.name,
                "description": tag.description,
                "models": tag.models or [],
                "model_info": json.dumps(model_info),
                "spend": 0.0,
                "budget_id": budget_id,
                "created_by": user_api_key_dict.user_id,
            }
        )

        # Update models with new tag
        if tag.models:
            tasks = []
            for model in tag.models:
                deployments = await get_deployments_by_model(model, llm_router)
                tasks.extend(
                    [
                        _add_tag_to_deployment(
                            deployment=deployment,
                            tag=tag.name,
                        )
                        for deployment in deployments
                    ]
                )
            await asyncio.gather(*tasks)

        # Build response
        tag_config = TagConfig(
            name=new_tag_record.tag_name,
            description=new_tag_record.description,
            models=new_tag_record.models,
            model_info=model_info,
            created_at=new_tag_record.created_at.isoformat(),
            updated_at=new_tag_record.updated_at.isoformat(),
            created_by=new_tag_record.created_by,
        )

        return {
            "message": f"Tag {tag.name} created successfully",
            "tag": tag_config,
        }
    except Exception as e:
        verbose_proxy_logger.exception(f"Error creating tag: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _add_tag_to_deployment(deployment: "Deployment", tag: str):
    """Helper function to add tag to deployment"""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    litellm_params = deployment.litellm_params
    if "tags" not in litellm_params:
        litellm_params["tags"] = []
    litellm_params["tags"].append(tag)

    try:
        await prisma_client.db.litellm_proxymodeltable.update(
            where={"model_id": deployment.model_info.id},
            data={"litellm_params": safe_dumps(litellm_params)},
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"Error adding tag to deployment: {str(e)}")
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
    - models: List[str] - Updated list of allowed LLM models
    - budget_id: Optional[str] - The id for a budget to associate with the tag
    
    ### BUDGET UPDATE PARAMS ###
    - max_budget: Optional[float] - Max budget for tag
    - tpm_limit: Optional[int] - Max tpm limit for tag
    - rpm_limit: Optional[int] - Max rpm limit for tag
    - max_parallel_requests: Optional[int] - Max parallel requests for tag
    - soft_budget: Optional[float] - Get a slack alert when this soft budget is reached
    - model_max_budget: Optional[dict] - Max budget for a specific model
    - budget_duration: Optional[str] - Frequency of resetting tag budget
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Check if tag exists
        existing_tag = await prisma_client.db.litellm_tagtable.find_unique(
            where={"tag_name": tag.name}
        )
        if existing_tag is None:
            raise HTTPException(status_code=404, detail=f"Tag {tag.name} not found")

        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        # Handle budget updates using common helper
        budget_id = await handle_budget_for_entity(
            data=tag,
            existing_budget_id=existing_tag.budget_id,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        )

        # Get model names for model_info
        model_info = await _get_model_names(prisma_client, tag.models or [])

        # Prepare update data
        update_data = {
            "description": tag.description,
            "models": tag.models or [],
            "model_info": json.dumps(model_info),
        }
        
        # Add budget_id if it changed
        if budget_id != existing_tag.budget_id:
            update_data["budget_id"] = budget_id

        # Update tag in database
        updated_tag_record = await prisma_client.db.litellm_tagtable.update(
            where={"tag_name": tag.name},
            data=update_data,
        )

        # Build response
        tag_config = TagConfig(
            name=updated_tag_record.tag_name,
            description=updated_tag_record.description,
            models=updated_tag_record.models,
            model_info=model_info,
            created_at=updated_tag_record.created_at.isoformat(),
            updated_at=updated_tag_record.updated_at.isoformat(),
            created_by=updated_tag_record.created_by,
        )

        return {
            "message": f"Tag {tag.name} updated successfully",
            "tag": tag_config,
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
        # Query tags from database with budget info
        tag_records = await prisma_client.db.litellm_tagtable.find_many(
            where={"tag_name": {"in": data.names}},
            include={"litellm_budget_table": True},
        )

        # Check if any requested tags don't exist
        found_tag_names = {tag.tag_name for tag in tag_records}
        missing_tags = [name for name in data.names if name not in found_tag_names]
        if missing_tags:
            raise HTTPException(
                status_code=404, detail=f"Tags not found: {missing_tags}"
            )

        # Build response
        requested_tags = {}
        for tag_record in tag_records:
            # Parse model_info from JSON
            model_info = {}
            if tag_record.model_info:
                if isinstance(tag_record.model_info, str):
                    model_info = json.loads(tag_record.model_info)
                else:
                    model_info = tag_record.model_info

            tag_dict = {
                "name": tag_record.tag_name,
                "description": tag_record.description,
                "models": tag_record.models,
                "model_info": model_info,
                "created_at": tag_record.created_at.isoformat(),
                "updated_at": tag_record.updated_at.isoformat(),
                "created_by": tag_record.created_by,
            }

            # Add budget info if available
            if hasattr(tag_record, "litellm_budget_table") and tag_record.litellm_budget_table:
                tag_dict["litellm_budget_table"] = tag_record.litellm_budget_table

            requested_tags[tag_record.tag_name] = tag_dict

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
    List all available tags with their budget information.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        ## QUERY STORED TAGS ##
        tag_records = await prisma_client.db.litellm_tagtable.find_many(
            include={"litellm_budget_table": True}
        )

        stored_tag_names = set()
        list_of_tags = []
        for tag_record in tag_records:
            stored_tag_names.add(tag_record.tag_name)
            # Parse model_info from JSON
            model_info = {}
            if tag_record.model_info:
                if isinstance(tag_record.model_info, str):
                    model_info = json.loads(tag_record.model_info)
                else:
                    model_info = tag_record.model_info

            tag_dict = {
                "name": tag_record.tag_name,
                "description": tag_record.description,
                "models": tag_record.models,
                "model_info": model_info,
                "created_at": tag_record.created_at.isoformat(),
                "updated_at": tag_record.updated_at.isoformat(),
                "created_by": tag_record.created_by,
            }

            # Add budget info if available
            if hasattr(tag_record, "litellm_budget_table") and tag_record.litellm_budget_table:
                tag_dict["litellm_budget_table"] = tag_record.litellm_budget_table

            list_of_tags.append(tag_dict)

        ## QUERY DYNAMIC TAGS ##
        dynamic_tags = await prisma_client.db.litellm_dailytagspend.find_many(
            distinct=["tag"],
        )

        dynamic_tags_list = [
            LiteLLM_DailyTagSpendTable(**dynamic_tag.model_dump())
            for dynamic_tag in dynamic_tags
        ]

        dynamic_tag_config = [
            {
                "name": tag.tag,
                "description": "This is just a spend tag that was passed dynamically in a request. It does not control any LLM models.",
                "models": None,
                "created_at": tag.created_at.isoformat(),
                "updated_at": tag.updated_at.isoformat(),
            }
            for tag in dynamic_tags_list
            if tag.tag not in stored_tag_names
        ]

        return list_of_tags + dynamic_tag_config
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
        # Check if tag exists
        existing_tag = await prisma_client.db.litellm_tagtable.find_unique(
            where={"tag_name": data.name}
        )
        if existing_tag is None:
            raise HTTPException(status_code=404, detail=f"Tag {data.name} not found")

        # Delete tag from database
        await prisma_client.db.litellm_tagtable.delete(where={"tag_name": data.name})

        return {"message": f"Tag {data.name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/tag/daily/activity",
    response_model=SpendAnalyticsPaginatedResponse,
    tags=["tag management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_tag_daily_activity(
    tags: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
):
    """
    Get daily activity for specific tags or all tags.

    Args:
        tags (Optional[str]): Comma-separated list of tags to filter by. If not provided, returns data for all tags.
        start_date (Optional[str]): Start date for the activity period (YYYY-MM-DD).
        end_date (Optional[str]): End date for the activity period (YYYY-MM-DD).
        model (Optional[str]): Filter by model name.
        api_key (Optional[str]): Filter by API key.
        page (int): Page number for pagination.
        page_size (int): Number of items per page.

    Returns:
        SpendAnalyticsPaginatedResponse: Paginated response containing daily activity data.
    """
    from litellm.proxy.proxy_server import prisma_client

    # Convert comma-separated tags string to list if provided
    tag_list = tags.split(",") if tags else None

    return await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailytagspend",
        entity_id_field="tag",
        entity_id=tag_list,
        entity_metadata_field=None,
        start_date=start_date,
        end_date=end_date,
        model=model,
        api_key=api_key,
        page=page,
        page_size=page_size,
        metadata_metrics_func=compute_tag_metadata_totals,
    )
