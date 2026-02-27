"""
Endpoints for /project operations

/project/new
/project/update
/project/delete
/project/info
/project/list
"""

#### PROJECT MANAGEMENT ####

import json
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import _set_object_metadata_field
from litellm.proxy.management_helpers.utils import (
    management_endpoint_wrapper,
)
from litellm.proxy.utils import PrismaClient, handle_exception_on_proxy

router = APIRouter()


async def _check_user_permission_for_project(
    user_api_key_dict: UserAPIKeyAuth,
    team_id: Optional[str],
    prisma_client: PrismaClient,
    require_admin: bool = False,
    team_object: Optional[LiteLLM_TeamTable] = None,
) -> bool:
    """
    Check if user has permission to manage a project.

    Returns True if user is proxy admin or team admin (when team_id provided).
    If require_admin=True, only proxy admins are allowed.

    If team_object is provided, it will be used instead of fetching from DB
    (avoids duplicate DB queries when team was already fetched for validation).
    """
    is_proxy_admin = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN

    if require_admin:
        return is_proxy_admin

    if is_proxy_admin:
        return True

    if not team_id or not user_api_key_dict.user_id:
        return False

    team = team_object
    if team is None:
        team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": team_id}
        )

    if team and team.admins:
        return user_api_key_dict.user_id in team.admins

    return False


async def _validate_team_exists(
    team_id: str,
    prisma_client: PrismaClient,
):
    """Validate that a team exists. Returns the team row."""
    team = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id},
    )

    if team is None:
        raise ProxyException(
            message=f"Team not found, team_id={team_id}",
            type="not_found",
            code=404,
            param="team_id",
        )

    return team


def _check_team_project_limits(
    team_object: LiteLLM_TeamTable,
    data: Union[NewProjectRequest, UpdateProjectRequest],
) -> None:
    """
    Check that project limits respect its parent Team's limits.

    Mirrors _check_org_team_limits() from team_endpoints.py.

    Validates:
    - Project models are a subset of Team models
    - Project max_budget <= Team max_budget
    - Project tpm_limit <= Team tpm_limit
    - Project rpm_limit <= Team rpm_limit
    - Budget values are non-negative
    - soft_budget < max_budget
    """
    # --- Budget non-negativity checks ---
    if data.max_budget is not None and data.max_budget < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"max_budget cannot be negative. Received: {data.max_budget}"
            },
        )
    if data.soft_budget is not None and data.soft_budget < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"soft_budget cannot be negative. Received: {data.soft_budget}"
            },
        )

    # --- soft_budget < max_budget ---
    if data.soft_budget is not None and data.max_budget is not None:
        if data.soft_budget >= data.max_budget:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"soft_budget ({data.soft_budget}) must be strictly lower than max_budget ({data.max_budget})"
                },
            )

    # --- Validate project models are a subset of team models ---
    project_models = getattr(data, "models", None)
    team_models = team_object.models or []
    if project_models and len(team_models) > 0:
        # If team has 'all-proxy-models', skip validation as it allows all models
        if SpecialModelNames.all_proxy_models.value not in team_models:
            for m in project_models:
                if m not in team_models:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"Model '{m}' not in team's allowed models. Team allowed models={team_models}. Team: {team_object.team_id}"
                        },
                    )

    # --- Validate project max_budget <= team max_budget ---
    # Team stores budget fields directly (max_budget, tpm_limit, rpm_limit)
    # unlike Project which uses a separate LiteLLM_BudgetTable relation
    if (
        data.max_budget is not None
        and team_object.max_budget is not None
        and data.max_budget > team_object.max_budget
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Project max_budget ({data.max_budget}) exceeds team's max_budget ({team_object.max_budget}). Team: {team_object.team_id}"
            },
        )

    # --- Validate project tpm_limit <= team tpm_limit ---
    if (
        data.tpm_limit is not None
        and team_object.tpm_limit is not None
        and data.tpm_limit > team_object.tpm_limit
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Project tpm_limit ({data.tpm_limit}) exceeds team's tpm_limit ({team_object.tpm_limit}). Team: {team_object.team_id}"
            },
        )

    # --- Validate project rpm_limit <= team rpm_limit ---
    if (
        data.rpm_limit is not None
        and team_object.rpm_limit is not None
        and data.rpm_limit > team_object.rpm_limit
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Project rpm_limit ({data.rpm_limit}) exceeds team's rpm_limit ({team_object.rpm_limit}). Team: {team_object.team_id}"
            },
        )


async def _create_budget_for_project(
    data: NewProjectRequest,
    user_id: Optional[str],
    litellm_proxy_admin_name: str,
    prisma_client: PrismaClient,
) -> str:
    """Create a budget for the project and return budget_id."""
    budget_params = LiteLLM_BudgetTable.model_fields.keys()
    _json_data = data.json(exclude_none=True)
    _budget_data = {k: v for k, v in _json_data.items() if k in budget_params}
    budget_row = LiteLLM_BudgetTable(**_budget_data)

    new_budget = prisma_client.jsonify_object(budget_row.json(exclude_none=True))

    _budget = await prisma_client.db.litellm_budgettable.create(
        data={
            **new_budget,
            "created_by": user_id or litellm_proxy_admin_name,
            "updated_by": user_id or litellm_proxy_admin_name,
        }
    )

    return _budget.budget_id


async def _set_project_object_permission(
    data: NewProjectRequest,
    prisma_client: Optional[PrismaClient],
) -> Optional[str]:
    """
    Creates the LiteLLM_ObjectPermissionTable record for the project.
    Returns the object_permission_id if created, otherwise None.
    """
    if prisma_client is None:
        return None

    if data.object_permission is not None:
        created_object_permission = (
            await prisma_client.db.litellm_objectpermissiontable.create(
                data=data.object_permission.model_dump(exclude_none=True),
            )
        )
        del data.object_permission
        return created_object_permission.object_permission_id
    return None


def _remove_budget_fields_from_project_data(project_data: dict) -> dict:
    """
    Remove budget fields from project data.
    Budget fields belong to LiteLLM_BudgetTable, not LiteLLM_ProjectTable.
    Keep budget_id as it's a foreign key.

    Following the pattern from organization_endpoints.py
    """
    budget_fields = LiteLLM_BudgetTable.model_fields.keys()
    for field in list(budget_fields):
        if field != "budget_id":  # Keep the foreign key
            project_data.pop(field, None)
    return project_data


@router.post(
    "/project/new",
    tags=["project management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=NewProjectResponse,
)
@management_endpoint_wrapper
async def new_project(
    data: NewProjectRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new project. Projects sit between teams and keys in the hierarchy.

    Only admins or team admins can create projects.

    # Parameters

    - project_alias: *Optional[str]* - The name of the project.
    - description: *Optional[str]* - Description of the project's purpose and use case.
    - team_id: *str* - The team id that this project belongs to. Required.
    - models: *List* - The models the project has access to.
    - budget_id: *Optional[str]* - The id for a budget (tpm/rpm/max budget) for the project.
    ### IF NO BUDGET ID - CREATE ONE WITH THESE PARAMS ###
    - max_budget: *Optional[float]* - Max budget for project
    - tpm_limit: *Optional[int]* - Max tpm limit for project
    - rpm_limit: *Optional[int]* - Max rpm limit for project
    - max_parallel_requests: *Optional[int]* - Max parallel requests for project
    - soft_budget: *Optional[float]* - Get a slack alert when this soft budget is reached. Don't block requests.
    - model_max_budget: *Optional[dict]* - Max budget for a specific model. Example: {"gpt-4": 100.0, "gpt-3.5-turbo": 50.0}
    - model_rpm_limit: *Optional[dict]* - RPM limits per model. Example: {"gpt-4": 1000, "gpt-3.5-turbo": 5000}
    - model_tpm_limit: *Optional[dict]* - TPM limits per model. Example: {"gpt-4": 50000, "gpt-3.5-turbo": 100000}
    - budget_duration: *Optional[str]* - Frequency of reseting project budget
    - metadata: *Optional[dict]* - Metadata for project, store information for project. Example metadata - {"use_case_id": "SNOW-12345", "responsible_ai_id": "RAI-67890"}
    - tags: *Optional[list]* - Tags for the project. Example: ["production", "api"]
    - blocked: *bool* - Flag indicating if the project is blocked or not - will stop all calls from keys with this project_id.
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - project-specific object permission. Example - {"vector_stores": ["vector_store_1", "vector_store_2"]}. IF null or {} then no object permission.

    Example 1: Create new project **without** a budget_id, with model-specific limits

    ```bash
    curl --location 'http://0.0.0.0:4000/project/new' \\
    --header 'Authorization: Bearer sk-1234' \\
    --header 'Content-Type: application/json' \\
    --data '{
        "project_alias": "flight-search-assistant",
        "description": "AI-powered flight search and booking assistant",
        "team_id": "team-123",
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "max_budget": 100,
        "model_rpm_limit": {
            "gpt-4": 1000,
            "gpt-3.5-turbo": 5000
        },
        "model_tpm_limit": {
            "gpt-4": 50000,
            "gpt-3.5-turbo": 100000
        },
        "metadata": {
            "use_case_id": "SNOW-12345",
            "responsible_ai_id": "RAI-67890"
        }
    }'
    ```

    Example 2: Create new project **with** a budget_id

    ```bash
    curl --location 'http://0.0.0.0:4000/project/new' \\
    --header 'Authorization: Bearer sk-1234' \\
    --header 'Content-Type: application/json' \\
    --data '{
        "project_alias": "hotel-recommendations",
        "description": "Personalized hotel recommendation engine",
        "team_id": "team-123",
        "models": ["claude-3-sonnet"],
        "budget_id": "428eeaa8-f3ac-4e85-a8fb-7dc8d7aa8689",
        "metadata": {
            "use_case_id": "SNOW-54321"
        }
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        premium_user,
        prisma_client,
    )

    try:
        if getattr(data, "tags", None) is not None and not premium_user:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only premium users can add tags to projects. "
                    + CommonProxyErrors.not_premium_user.value
                },
            )

        if not premium_user:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Project management is an enterprise feature. "
                    + CommonProxyErrors.not_premium_user.value
                },
            )

        # ADD METADATA FIELDS
        for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
            if getattr(data, field, None) is not None:
                _set_object_metadata_field(
                    object_data=data,
                    field_name=field,
                    value=getattr(data, field),
                )
                delattr(data, field)

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Validate team exists and get team object with budget
        team_object = await _validate_team_exists(
            team_id=data.team_id, prisma_client=prisma_client
        )

        # Validate project limits against team limits
        _check_team_project_limits(
            team_object=LiteLLM_TeamTable(**team_object.model_dump()),
            data=data,
        )

        # Check if user has permission to create projects for this team
        # only team admins can create projects for their team
        has_permission = await _check_user_permission_for_project(
            user_api_key_dict=user_api_key_dict,
            team_id=data.team_id,
            prisma_client=prisma_client,
            team_object=LiteLLM_TeamTable(**team_object.model_dump()),
        )

        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": f"Only admins or team admins can create projects. Your role is {user_api_key_dict.user_role}"
                },
            )

        # Generate project_id if not provided
        if data.project_id is None:
            data.project_id = str(uuid.uuid4())
        else:
            # Check if project_id already exists
            existing_project = await prisma_client.db.litellm_projecttable.find_unique(
                where={"project_id": data.project_id}
            )
            if existing_project is not None:
                raise ProxyException(
                    message=f"Project id = {data.project_id} already exists. Please use a different project id.",
                    type="bad_request",
                    code=400,
                    param="project_id",
                )

        # Create budget if not provided
        if data.budget_id is None:
            data.budget_id = await _create_budget_for_project(
                data=data,
                user_id=user_api_key_dict.user_id,
                litellm_proxy_admin_name=litellm_proxy_admin_name,
                prisma_client=prisma_client,
            )

        ## Handle Object Permission - MCP, Vector Stores etc.
        object_permission_id = await _set_project_object_permission(
            data=data,
            prisma_client=prisma_client,
        )

        # Create project row (following organization_endpoints.py pattern)
        project_row = LiteLLM_ProjectTable(
            **data.json(exclude_none=True),
            object_permission_id=object_permission_id,
            created_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
            updated_by=user_api_key_dict.user_id or litellm_proxy_admin_name,
        )

        for field in LiteLLM_ManagementEndpoint_MetadataFields:
            if getattr(data, field, None) is not None:
                _set_object_metadata_field(
                    object_data=project_row,
                    field_name=field,
                    value=getattr(data, field),
                )

        new_project_row = prisma_client.jsonify_object(
            project_row.json(exclude_none=True)
        )

        # Remove budget fields (following organization_endpoints.py pattern)
        new_project_row = _remove_budget_fields_from_project_data(new_project_row)

        verbose_proxy_logger.info(
            f"new_project_row: {json.dumps(new_project_row, indent=2)}"
        )
        response = await prisma_client.db.litellm_projecttable.create(
            data={
                **new_project_row,  # type: ignore
            },
            include={"litellm_budget_table": True},
        )

        return response
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.project_endpoints.new_project(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.post(
    "/project/update",
    tags=["project management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ProjectTable,
)
@management_endpoint_wrapper
async def update_project(  # noqa: PLR0915
    data: UpdateProjectRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update a project

    Parameters:
    - project_id: *str* - The project id to update. Required.
    - project_alias: *Optional[str]* - Updated name for the project
    - description: *Optional[str]* - Updated description for the project
    - team_id: *Optional[str]* - Updated team_id for the project
    - metadata: *Optional[dict]* - Updated metadata for project
    - models: *Optional[list]* - Updated list of models for the project
    - blocked: *Optional[bool]* - Updated blocked status
    - max_budget: *Optional[float]* - Updated max budget
    - tpm_limit: *Optional[int]* - Updated tpm limit
    - rpm_limit: *Optional[int]* - Updated rpm limit
    - model_rpm_limit: *Optional[dict]* - Updated RPM limits per model
    - model_tpm_limit: *Optional[dict]* - Updated TPM limits per model
    - budget_duration: *Optional[str]* - Updated budget duration
    - tags: *Optional[list]* - Updated list of tags for the project
    - object_permission: Optional[LiteLLM_ObjectPermissionBase] - Updated object permission

    Example:
    ```bash
    curl --location 'http://0.0.0.0:4000/project/update' \\
    --header 'Authorization: Bearer sk-1234' \\
    --header 'Content-Type: application/json' \\
    --data '{
        "project_id": "project-123",
        "description": "Updated flight search system with enhanced capabilities",
        "max_budget": 200,
        "model_rpm_limit": {
            "gpt-4": 2000,
            "gpt-3.5-turbo": 10000
        },
        "metadata": {
            "use_case_id": "SNOW-12345",
            "status": "active"
        }
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        premium_user,
        prisma_client,
    )

    try:
        if getattr(data, "tags", None) is not None and not premium_user:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Only premium users can add tags to projects. "
                    + CommonProxyErrors.not_premium_user.value
                },
            )

        if not premium_user:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Project management is an enterprise feature. "
                    + CommonProxyErrors.not_premium_user.value
                },
            )

        # ADD METADATA FIELDS
        for field in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
            if getattr(data, field, None) is not None:
                _set_object_metadata_field(
                    object_data=data,
                    field_name=field,
                    value=getattr(data, field),
                )
                delattr(data, field)

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        if data.project_id is None:
            raise HTTPException(
                status_code=400,
                detail={"error": "project_id is required"},
            )

        # Fetch existing project
        existing_project = await prisma_client.db.litellm_projecttable.find_unique(
            where={"project_id": data.project_id}
        )

        if existing_project is None:
            raise ProxyException(
                message=f"Project not found, project_id={data.project_id}",
                type="not_found",
                code=404,
                param="project_id",
            )

        # Validate team exists and get team object for limit + permission checks
        team_id_to_check = data.team_id or existing_project.team_id
        team_obj_for_checks = None
        if team_id_to_check is not None:
            team_obj_for_checks = await _validate_team_exists(
                team_id=team_id_to_check, prisma_client=prisma_client
            )

        # Check if user has permission to update this project
        has_permission = await _check_user_permission_for_project(
            user_api_key_dict=user_api_key_dict,
            team_id=existing_project.team_id,
            prisma_client=prisma_client,
            team_object=LiteLLM_TeamTable(**team_obj_for_checks.model_dump())
            if team_obj_for_checks
            else None,
        )

        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail={"error": "Only admins or team admins can update projects"},
            )

        # Validate project limits against team limits
        if team_obj_for_checks is not None:
            _check_team_project_limits(
                team_object=LiteLLM_TeamTable(**team_obj_for_checks.model_dump()),
                data=data,
            )

        # Prepare update data
        update_data = data.json(exclude_none=True, exclude={"project_id"})
        update_data = prisma_client.jsonify_object(update_data)
        update_data["updated_by"] = (
            user_api_key_dict.user_id or litellm_proxy_admin_name
        )

        # Handle budget updates
        budget_fields = LiteLLM_BudgetTable.model_fields.keys()
        budget_updates = {k: v for k, v in update_data.items() if k in budget_fields}

        if budget_updates and existing_project.budget_id:
            # Update existing budget
            await prisma_client.db.litellm_budgettable.update(
                where={"budget_id": existing_project.budget_id},
                data={
                    **budget_updates,
                    "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                },
            )
            # Remove budget fields from project update
            for field in budget_updates.keys():
                update_data.pop(field, None)

        # Handle object permissions
        if "object_permission" in update_data:
            object_permission_data = update_data.pop("object_permission")
            if object_permission_data:
                if existing_project.object_permission_id:
                    # Update existing permission
                    await prisma_client.db.litellm_objectpermissiontable.update(
                        where={
                            "object_permission_id": existing_project.object_permission_id
                        },
                        data=object_permission_data,
                    )
                else:
                    # Create new permission
                    created_permission = (
                        await prisma_client.db.litellm_objectpermissiontable.create(
                            data=object_permission_data,
                        )
                    )
                    update_data[
                        "object_permission_id"
                    ] = created_permission.object_permission_id

        # Handle metadata fields
        for field in LiteLLM_ManagementEndpoint_MetadataFields:
            if field in update_data:
                if update_data.get("metadata") is None:
                    update_data["metadata"] = {}
                update_data["metadata"][field] = update_data.pop(field)

        # Remove budget fields (following organization_endpoints.py pattern)
        update_data = _remove_budget_fields_from_project_data(update_data)

        # Update project
        updated_project = await prisma_client.db.litellm_projecttable.update(
            where={"project_id": data.project_id},
            data=update_data,
            include={"litellm_budget_table": True, "object_permission": True},
        )

        return updated_project
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.project_endpoints.update_project(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.delete(
    "/project/delete",
    tags=["project management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[LiteLLM_ProjectTable],
)
@management_endpoint_wrapper
async def delete_project(
    data: DeleteProjectRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete projects

    Parameters:
    - project_ids: *List[str]* - List of project ids to delete

    Example:
    ```bash
    curl --location --request DELETE 'http://0.0.0.0:4000/project/delete' \\
    --header 'Authorization: Bearer sk-1234' \\
    --header 'Content-Type: application/json' \\
    --data '{
        "project_ids": ["project-123", "project-456"]
    }'
    ```
    """
    from litellm.proxy.proxy_server import premium_user, prisma_client

    try:
        if not premium_user:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Project management is an enterprise feature. "
                    + CommonProxyErrors.not_premium_user.value
                },
            )

        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Check if user is admin (only admins can delete projects)
        has_permission = await _check_user_permission_for_project(
            user_api_key_dict=user_api_key_dict,
            team_id=None,
            prisma_client=prisma_client,
            require_admin=True,
        )

        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail={"error": "Only admins can delete projects"},
            )

        deleted_projects = []

        for project_id in data.project_ids:
            # Check if project exists
            existing_project = await prisma_client.db.litellm_projecttable.find_unique(
                where={"project_id": project_id}
            )

            if existing_project is None:
                raise ProxyException(
                    message=f"Project not found, project_id={project_id}",
                    type="not_found",
                    code=404,
                    param="project_ids",
                )

            # Check if there are any keys associated with this project
            associated_keys = (
                await prisma_client.db.litellm_verificationtoken.find_many(
                    where={"project_id": project_id}
                )
            )

            if len(associated_keys) > 0:
                raise ProxyException(
                    message=f"Cannot delete project {project_id}. {len(associated_keys)} key(s) are associated with it. Please delete or reassign the keys first.",
                    type="bad_request",
                    code=400,
                    param="project_ids",
                )

            # Delete the project
            deleted_project = await prisma_client.db.litellm_projecttable.delete(
                where={"project_id": project_id}
            )

            deleted_projects.append(deleted_project)

        return deleted_projects
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.project_endpoints.delete_project(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/project/info",
    tags=["project management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ProjectTable,
)
async def project_info(
    project_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get information about a specific project

    Parameters:
    - project_id: *str* - The project id to fetch info for

    Example:
    ```bash
    curl --location 'http://0.0.0.0:4000/project/info?project_id=project-123' \\
    --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # Fetch project
        project = await prisma_client.db.litellm_projecttable.find_unique(
            where={"project_id": project_id},
            include={"litellm_budget_table": True, "object_permission": True},
        )

        if project is None:
            raise ProxyException(
                message=f"Project not found, project_id={project_id}",
                type="not_found",
                code=404,
                param="project_id",
            )

        # Check if user has access to this project (admin or team member)
        is_admin = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        is_team_member = False

        if project.team_id and user_api_key_dict.user_id:
            team = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": project.team_id}
            )
            if team:
                is_team_member = (
                    user_api_key_dict.user_id in team.admins
                    or user_api_key_dict.user_id in team.members
                )

        if not (is_admin or is_team_member):
            raise HTTPException(
                status_code=403,
                detail={"error": "You don't have access to this project"},
            )

        return project
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.project_endpoints.project_info(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)


@router.get(
    "/project/list",
    tags=["project management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[LiteLLM_ProjectTable],
)
async def list_projects(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all projects that the user has access to

    Example:
    ```bash
    curl --location 'http://0.0.0.0:4000/project/list' \\
    --header 'Authorization: Bearer sk-1234'
    ```
    """
    from litellm.proxy.proxy_server import prisma_client

    try:
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail={"error": CommonProxyErrors.db_not_connected_error.value},
            )

        # If proxy admin, get all projects
        if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
            projects = await prisma_client.db.litellm_projecttable.find_many(
                include={"litellm_budget_table": True, "object_permission": True}
            )
        else:
            # Get projects for teams the user belongs to
            user_teams = await prisma_client.db.litellm_teamtable.find_many(
                where={
                    "OR": [
                        {"members": {"has": user_api_key_dict.user_id}},
                        {"admins": {"has": user_api_key_dict.user_id}},
                    ]
                }
            )

            team_ids = [team.team_id for team in user_teams]

            projects = await prisma_client.db.litellm_projecttable.find_many(
                where={"team_id": {"in": team_ids}},
                include={"litellm_budget_table": True, "object_permission": True},
            )

        return projects
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.project_endpoints.list_projects(): Exception occured - {}".format(
                str(e)
            )
        )
        raise handle_exception_on_proxy(e)
