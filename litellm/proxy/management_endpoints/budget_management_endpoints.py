"""
BUDGET MANAGEMENT

All /budget management endpoints 

/budget/new   
/budget/info
/budget/update
/budget/delete
/budget/settings
/budget/list
"""

#### BUDGET TABLE MANAGEMENT ####
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
from litellm.proxy.utils import jsonify_object

router = APIRouter()

BudgetLinkedEntity = Dict[str, Optional[str]]


def _record_value(record: Any, field_name: str) -> Any:
    if isinstance(record, dict):
        return record.get(field_name)
    return getattr(record, field_name, None)


def _budget_row_to_dict(budget_row: Any) -> Dict[str, Any]:
    if isinstance(budget_row, dict):
        return {**budget_row}
    if hasattr(budget_row, "model_dump"):
        return budget_row.model_dump()
    return vars(budget_row)


def _append_linked_entity(
    linked_entities_by_budget: DefaultDict[str, List[BudgetLinkedEntity]],
    budget_id: Optional[str],
    entity_type: str,
    entity_id: Optional[str],
    entity_name: Optional[str] = None,
    parent_entity_id: Optional[str] = None,
    parent_entity_name: Optional[str] = None,
) -> None:
    if budget_id is None or entity_id is None:
        return

    entity: BudgetLinkedEntity = {
        "entity_type": entity_type,
        "entity_id": entity_id,
    }
    if entity_name:
        entity["entity_name"] = entity_name
    if parent_entity_id:
        entity["parent_entity_id"] = parent_entity_id
    if parent_entity_name:
        entity["parent_entity_name"] = parent_entity_name
    linked_entities_by_budget[budget_id].append(entity)


def _records_by_id(records: Iterable[Any], id_field: str) -> Dict[str, Any]:
    return {
        record_id: record
        for record in records
        if (record_id := _record_value(record, id_field)) is not None
    }


def _display_name(record: Any, preferred_fields: Iterable[str]) -> Optional[str]:
    for field_name in preferred_fields:
        field_value = _record_value(record, field_name)
        if isinstance(field_value, str) and field_value:
            return field_value
    return None


async def _get_budget_linked_entities(
    prisma_client: Any, budget_ids: List[str]
) -> Dict[str, List[BudgetLinkedEntity]]:
    linked_entities_by_budget: DefaultDict[str, List[BudgetLinkedEntity]] = defaultdict(
        list
    )
    if len(budget_ids) == 0:
        return {}

    budget_id_filter = {"budget_id": {"in": budget_ids}}

    organizations = await prisma_client.db.litellm_organizationtable.find_many(
        where=budget_id_filter,
        select={
            "budget_id": True,
            "organization_id": True,
            "organization_alias": True,
        },
    )
    for organization in organizations:
        _append_linked_entity(
            linked_entities_by_budget=linked_entities_by_budget,
            budget_id=_record_value(organization, "budget_id"),
            entity_type="organization",
            entity_id=_record_value(organization, "organization_id"),
            entity_name=_record_value(organization, "organization_alias"),
        )

    projects = await prisma_client.db.litellm_projecttable.find_many(
        where=budget_id_filter,
        select={
            "budget_id": True,
            "project_id": True,
            "project_alias": True,
            "team_id": True,
        },
    )
    for project in projects:
        _append_linked_entity(
            linked_entities_by_budget=linked_entities_by_budget,
            budget_id=_record_value(project, "budget_id"),
            entity_type="project",
            entity_id=_record_value(project, "project_id"),
            entity_name=_record_value(project, "project_alias"),
            parent_entity_id=_record_value(project, "team_id"),
        )

    keys = await prisma_client.db.litellm_verificationtoken.find_many(
        where=budget_id_filter,
        select={
            "budget_id": True,
            "key_name": True,
            "key_alias": True,
            "user_id": True,
            "team_id": True,
            "project_id": True,
        },
    )
    for key in keys:
        _append_linked_entity(
            linked_entities_by_budget=linked_entities_by_budget,
            budget_id=_record_value(key, "budget_id"),
            entity_type="key",
            entity_id=_display_name(key, ("key_alias", "key_name")) or "virtual_key",
            entity_name=_display_name(key, ("key_alias", "key_name")),
            parent_entity_id=_record_value(key, "project_id")
            or _record_value(key, "team_id")
            or _record_value(key, "user_id"),
        )

    end_users = await prisma_client.db.litellm_endusertable.find_many(
        where=budget_id_filter,
        select={
            "budget_id": True,
            "user_id": True,
            "alias": True,
        },
    )
    for end_user in end_users:
        _append_linked_entity(
            linked_entities_by_budget=linked_entities_by_budget,
            budget_id=_record_value(end_user, "budget_id"),
            entity_type="end_user",
            entity_id=_record_value(end_user, "user_id"),
            entity_name=_record_value(end_user, "alias"),
        )

    tags = await prisma_client.db.litellm_tagtable.find_many(
        where=budget_id_filter,
        select={
            "budget_id": True,
            "tag_name": True,
        },
    )
    for tag in tags:
        _append_linked_entity(
            linked_entities_by_budget=linked_entities_by_budget,
            budget_id=_record_value(tag, "budget_id"),
            entity_type="tag",
            entity_id=_record_value(tag, "tag_name"),
            entity_name=_record_value(tag, "tag_name"),
        )

    team_memberships = await prisma_client.db.litellm_teammembership.find_many(
        where=budget_id_filter,
        select={
            "budget_id": True,
            "user_id": True,
            "team_id": True,
        },
    )
    organization_memberships = (
        await prisma_client.db.litellm_organizationmembership.find_many(
            where=budget_id_filter,
            select={
                "budget_id": True,
                "user_id": True,
                "organization_id": True,
            },
        )
    )

    teams = await prisma_client.db.litellm_teamtable.find_many(
        select={
            "team_id": True,
            "team_alias": True,
            "metadata": True,
        },
    )
    users = await prisma_client.db.litellm_usertable.find_many(
        where={
            "user_id": {
                "in": list(
                    {
                        _record_value(membership, "user_id")
                        for membership in [
                            *team_memberships,
                            *organization_memberships,
                        ]
                        if _record_value(membership, "user_id") is not None
                    }
                )
            }
        },
        select={
            "user_id": True,
            "user_alias": True,
            "user_email": True,
        },
    )

    teams_by_id = _records_by_id(teams, "team_id")
    users_by_id = _records_by_id(users, "user_id")
    organizations_by_id = _records_by_id(organizations, "organization_id")

    organization_member_org_ids = {
        organization_id
        for membership in organization_memberships
        if (organization_id := _record_value(membership, "organization_id")) is not None
    }
    missing_organization_ids = [
        organization_id
        for organization_id in organization_member_org_ids
        if organization_id not in organizations_by_id
    ]
    if missing_organization_ids:
        organization_membership_organizations = (
            await prisma_client.db.litellm_organizationtable.find_many(
                where={"organization_id": {"in": missing_organization_ids}},
                select={
                    "organization_id": True,
                    "organization_alias": True,
                },
            )
        )
        organizations_by_id.update(
            _records_by_id(organization_membership_organizations, "organization_id")
        )

    for membership in team_memberships:
        user_id = _record_value(membership, "user_id")
        team_id = _record_value(membership, "team_id")
        user = users_by_id.get(user_id)
        team = teams_by_id.get(team_id)
        _append_linked_entity(
            linked_entities_by_budget=linked_entities_by_budget,
            budget_id=_record_value(membership, "budget_id"),
            entity_type="team_member",
            entity_id=user_id,
            entity_name=_display_name(user, ("user_alias", "user_email", "user_id")),
            parent_entity_id=team_id,
            parent_entity_name=_display_name(team, ("team_alias", "team_id")),
        )

    for membership in organization_memberships:
        user_id = _record_value(membership, "user_id")
        organization_id = _record_value(membership, "organization_id")
        user = users_by_id.get(user_id)
        organization = organizations_by_id.get(organization_id)
        _append_linked_entity(
            linked_entities_by_budget=linked_entities_by_budget,
            budget_id=_record_value(membership, "budget_id"),
            entity_type="organization_member",
            entity_id=user_id,
            entity_name=_display_name(user, ("user_alias", "user_email", "user_id")),
            parent_entity_id=organization_id,
            parent_entity_name=_display_name(
                organization, ("organization_alias", "organization_id")
            ),
        )

    budget_id_set = set(budget_ids)
    for team in teams:
        metadata = _record_value(team, "metadata") or {}
        team_member_budget_id = metadata.get("team_member_budget_id")
        if team_member_budget_id in budget_id_set:
            _append_linked_entity(
                linked_entities_by_budget=linked_entities_by_budget,
                budget_id=team_member_budget_id,
                entity_type="team_member_default",
                entity_id=_record_value(team, "team_id"),
                entity_name=_display_name(team, ("team_alias", "team_id")),
            )

    return dict(linked_entities_by_budget)


@router.post(
    "/budget/new",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new budget object. Can apply this to teams, orgs, end-users, keys.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
    - budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
    - max_budget: Optional[float] - The max budget for the budget.
    - soft_budget: Optional[float] - The soft budget for the budget.
    - max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
    - tpm_limit: Optional[int] - The tokens per minute limit for the budget.
    - rpm_limit: Optional[int] - The requests per minute limit for the budget.
    - model_max_budget: Optional[dict] - Specify max budget for a given model. Example: {"openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d", "tpm_limit": 100000, "rpm_limit": 100000}}
    - budget_reset_at: Optional[datetime] - Datetime when the initial budget is reset. Default is now.
    """
    from prisma.errors import UniqueViolationError

    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Validate budget values are not negative
    if budget_obj.max_budget is not None and budget_obj.max_budget < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"max_budget cannot be negative. Received: {budget_obj.max_budget}"
            },
        )
    if budget_obj.soft_budget is not None and budget_obj.soft_budget < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"soft_budget cannot be negative. Received: {budget_obj.soft_budget}"
            },
        )

    # Validate model_max_budget if present
    if budget_obj.model_max_budget is not None and len(budget_obj.model_max_budget) > 0:
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            validate_model_max_budget,
        )

        try:
            validate_model_max_budget(budget_obj.model_max_budget)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    # if no budget_reset_at date is set, but a budget_duration is given, then set budget_reset_at initially to the first completed duration interval in future
    if budget_obj.budget_reset_at is None and budget_obj.budget_duration is not None:
        budget_obj.budget_reset_at = get_budget_reset_time(
            budget_duration=budget_obj.budget_duration
        )

    budget_obj_json = budget_obj.model_dump(exclude_none=True)
    budget_obj_jsonified = jsonify_object(budget_obj_json)  # json dump any dictionaries
    try:
        response = await prisma_client.db.litellm_budgettable.create(
            data={
                **budget_obj_jsonified,  # type: ignore
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }  # type: ignore
        )
    except Exception as e:
        if not isinstance(e, UniqueViolationError):
            raise
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Budget with id '{budget_obj.budget_id}' already exists."
            },
        )

    return response


@router.post(
    "/budget/update",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing budget object.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
    - budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
    - max_budget: Optional[float] - The max budget for the budget.
    - soft_budget: Optional[float] - The soft budget for the budget.
    - max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
    - tpm_limit: Optional[int] - The tokens per minute limit for the budget.
    - rpm_limit: Optional[int] - The requests per minute limit for the budget.
    - model_max_budget: Optional[dict] - Specify max budget for a given model. Example: {"openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d", "tpm_limit": 100000, "rpm_limit": 100000}}
    - budget_reset_at: Optional[datetime] - Update the Datetime when the budget was last reset.
    """
    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    if budget_obj.budget_id is None:
        raise HTTPException(status_code=400, detail={"error": "budget_id is required"})

    # Validate budget values are not negative
    if budget_obj.max_budget is not None and budget_obj.max_budget < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"max_budget cannot be negative. Received: {budget_obj.max_budget}"
            },
        )
    if budget_obj.soft_budget is not None and budget_obj.soft_budget < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"soft_budget cannot be negative. Received: {budget_obj.soft_budget}"
            },
        )

    # Validate model_max_budget if present in update
    if budget_obj.model_max_budget is not None and len(budget_obj.model_max_budget) > 0:
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            validate_model_max_budget,
        )

        try:
            validate_model_max_budget(budget_obj.model_max_budget)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    response = await prisma_client.db.litellm_budgettable.update(
        where={"budget_id": budget_obj.budget_id},
        data={
            **budget_obj.model_dump(exclude_unset=True),  # type: ignore
            "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
        },  # type: ignore
    )

    return response


@router.post(
    "/budget/info",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_budget(data: BudgetRequest):
    """
    Get the budget id specific information

    Parameters:
    - budgets: List[str] - The list of budget ids to get information for
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if len(data.budgets) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Specify list of budget id's to query. Passed in={data.budgets}"
            },
        )
    response = await prisma_client.db.litellm_budgettable.find_many(
        where={"budget_id": {"in": data.budgets}},
    )

    return response


@router.get(
    "/budget/settings",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def budget_settings(
    budget_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get list of configurable params + current value for a budget item + description of each field

    Used on Admin UI.

    Query Parameters:
    - budget_id: str - The budget id to get information for
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    ## get budget item from db
    db_budget_row = await prisma_client.db.litellm_budgettable.find_first(
        where={"budget_id": budget_id}
    )

    if db_budget_row is not None:
        db_budget_row_dict = db_budget_row.model_dump(exclude_none=True)
    else:
        db_budget_row_dict = {}

    allowed_args = {
        "max_parallel_requests": {"type": "Integer"},
        "tpm_limit": {"type": "Integer"},
        "rpm_limit": {"type": "Integer"},
        "budget_duration": {"type": "String"},
        "max_budget": {"type": "Float"},
        "soft_budget": {"type": "Float"},
        "model_max_budget": {"type": "Object"},
    }

    return_val = []

    for field_name, field_info in BudgetNewRequest.model_fields.items():
        if field_name in allowed_args:
            _stored_in_db = True

            _response_obj = ConfigList(
                field_name=field_name,
                field_type=allowed_args[field_name]["type"],
                field_description=field_info.description or "",
                field_value=db_budget_row_dict.get(field_name, None),
                stored_in_db=_stored_in_db,
                field_default_value=field_info.default,
            )
            return_val.append(_response_obj)

    return return_val


@router.get(
    "/budget/list",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_budget(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List all the created budgets in proxy db. Used on Admin UI."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    budget_rows = await prisma_client.db.litellm_budgettable.find_many()
    budget_response = [_budget_row_to_dict(budget_row) for budget_row in budget_rows]
    budget_ids = [
        budget["budget_id"]
        for budget in budget_response
        if isinstance(budget.get("budget_id"), str)
    ]
    linked_entities_by_budget = await _get_budget_linked_entities(
        prisma_client=prisma_client,
        budget_ids=budget_ids,
    )

    for budget in budget_response:
        budget_id = budget.get("budget_id")
        budget["linked_entities"] = (
            linked_entities_by_budget.get(budget_id, [])
            if isinstance(budget_id, str)
            else []
        )

    return budget_response


@router.post(
    "/budget/delete",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_budget(
    data: BudgetDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete budget

    Parameters:
    - id: str - The budget id to delete
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "{}, your role={}".format(
                    CommonProxyErrors.not_allowed_access.value,
                    user_api_key_dict.user_role,
                )
            },
        )

    response = await prisma_client.db.litellm_budgettable.delete(
        where={"budget_id": data.id}
    )

    return response
