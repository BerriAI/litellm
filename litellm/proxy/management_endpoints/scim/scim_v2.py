"""
✨ SCIM v2 Endpoints for LiteLLM Proxy using Internal User/Team Management

This is an enterprise feature and requires a premium license.
"""

import uuid
from typing import Any, Dict, List, Optional, Set

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
)

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    NewTeamRequest,
    NewUserRequest,
    TeamMemberAddRequest,
    TeamMemberDeleteRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.proxy.management_endpoints.scim.scim_transformations import (
    ScimTransformations,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    new_team,
    team_member_add,
    team_member_delete,
)
from litellm.proxy.utils import _premium_user_check, handle_exception_on_proxy
from litellm.types.proxy.management_endpoints.scim_v2 import *

scim_router = APIRouter(
    prefix="/scim/v2",
    tags=["✨ SCIM v2 (Enterprise Only)"],
    dependencies=[Depends(_premium_user_check)],
)


# Dependency to set the correct SCIM Content-Type
async def set_scim_content_type(response: Response):
    """Sets the Content-Type header to application/scim+json"""
    # Check if content type is already application/json, only override in that case
    # Avoids overriding for non-JSON responses or already correct types if they were set manually
    response.headers["Content-Type"] = "application/scim+json"


# User Endpoints
@scim_router.get(
    "/Users",
    response_model=SCIMListResponse,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_users(
    startIndex: int = Query(1, ge=1),
    count: int = Query(10, ge=1, le=100),
    filter: Optional[str] = Query(None),
):
    """
    Get a list of users according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        # Parse filter if provided (basic support)
        where_conditions = {}
        if filter:
            # Very basic filter support - only handling userName eq and emails.value eq
            if "userName eq" in filter:
                user_id = filter.split("userName eq ")[1].strip("\"'")
                where_conditions["user_id"] = user_id
            elif "emails.value eq" in filter:
                email = filter.split("emails.value eq ")[1].strip("\"'")
                where_conditions["user_email"] = email

        # Get users from database
        users: List[LiteLLM_UserTable] = (
            await prisma_client.db.litellm_usertable.find_many(
                where=where_conditions,
                skip=(startIndex - 1),
                take=count,
                order={"created_at": "desc"},
            )
        )

        # Get total count for pagination
        total_count = await prisma_client.db.litellm_usertable.count(
            where=where_conditions
        )

        # Convert to SCIM format
        scim_users: List[SCIMUser] = []
        for user in users:
            scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
                user=user
            )
            scim_users.append(scim_user)

        return SCIMListResponse(
            totalResults=total_count,
            startIndex=startIndex,
            itemsPerPage=min(count, len(scim_users)),
            Resources=scim_users,
        )

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.get(
    "/Users/{user_id}",
    response_model=SCIMUser,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_user(
    user_id: str = Path(..., title="User ID"),
):
    """
    Get a single user by ID according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        user = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if not user:
            raise HTTPException(
                status_code=404, detail={"error": f"User not found with ID: {user_id}"}
            )

        # Convert to SCIM format
        scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(user)
        return scim_user

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.post(
    "/Users",
    response_model=SCIMUser,
    status_code=201,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def create_user(
    user: SCIMUser = Body(...),
):
    """
    Create a user according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        verbose_proxy_logger.debug("SCIM CREATE USER request: %s", user)
        # Extract email from SCIM user
        user_email = None
        if user.emails and len(user.emails) > 0:
            user_email = user.emails[0].value

        # Check if user already exists
        existing_user = None
        if user.userName:
            existing_user = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user.userName}
            )

        if existing_user:
            raise HTTPException(
                status_code=409,
                detail={"error": f"User already exists with username: {user.userName}"},
            )

        # Create user in database
        user_id = user.userName or str(uuid.uuid4())
        created_user = await new_user(
            data=NewUserRequest(
                user_id=user_id,
                user_email=user_email,
                user_alias=user.name.givenName,
                teams=[group.value for group in user.groups] if user.groups else None,
                metadata={
                    "scim_metadata": LiteLLM_UserScimMetadata(
                        givenName=user.name.givenName,
                        familyName=user.name.familyName,
                    ).model_dump()
                },
                auto_create_key=False,
            ),
        )
        scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
            user=created_user
        )
        return scim_user
    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.put(
    "/Users/{user_id}",
    response_model=SCIMUser,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def update_user(
    user_id: str = Path(..., title="User ID"),
    user: SCIMUser = Body(...),
):
    """
    Update a user according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})
    try:
        return None
    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.delete(
    "/Users/{user_id}",
    status_code=204,
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_user(
    user_id: str = Path(..., title="User ID"),
):
    """
    Delete a user according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        # Check if user exists
        existing_user = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if not existing_user:
            raise HTTPException(
                status_code=404, detail={"error": f"User not found with ID: {user_id}"}
            )

        # Get teams user belongs to
        teams = []
        if existing_user.teams:
            for team_id in existing_user.teams:
                team = await prisma_client.db.litellm_teamtable.find_unique(
                    where={"team_id": team_id}
                )
                if team:
                    teams.append(team)

        # Remove user from all teams
        for team in teams:
            current_members = team.members or []
            if user_id in current_members:
                new_members = [m for m in current_members if m != user_id]
                await prisma_client.db.litellm_teamtable.update(
                    where={"team_id": team.team_id}, data={"members": new_members}
                )

        # Delete user
        await prisma_client.db.litellm_usertable.delete(where={"user_id": user_id})

        return Response(status_code=204)
    except Exception as e:
        raise handle_exception_on_proxy(e)


def _extract_group_values(value: Any) -> List[str]:
    """Return group ids from a SCIM patch value."""
    group_values: List[str] = []
    if isinstance(value, list):
        for v in value:
            if isinstance(v, dict) and v.get("value"):
                group_values.append(str(v.get("value")))
            elif isinstance(v, str):
                group_values.append(v)
    elif isinstance(value, dict):
        if value.get("value"):
            group_values.append(str(value.get("value")))
    elif isinstance(value, str):
        group_values.append(value)
    return group_values


def _handle_displayname_update(op_type: str, value: Any, update_data: Dict[str, Any]) -> None:
    """Handle displayname updates."""
    if op_type == "remove":
        update_data["user_alias"] = None
    else:
        update_data["user_alias"] = str(value)


def _handle_externalid_update(op_type: str, value: Any, update_data: Dict[str, Any]) -> None:
    """Handle externalid updates."""
    if op_type == "remove":
        update_data["sso_user_id"] = None
    else:
        update_data["sso_user_id"] = str(value)


def _handle_active_update(op_type: str, value: Any, metadata: Dict[str, Any]) -> None:
    """Handle active status updates."""
    if op_type == "remove":
        metadata.pop("scim_active", None)
    else:
        bool_val = value
        if isinstance(value, str):
            bool_val = value.lower() == "true"
        else:
            bool_val = bool(value)
        metadata["scim_active"] = bool_val


def _handle_name_update(path: str, op_type: str, value: Any, scim_metadata: Dict[str, Any]) -> None:
    """Handle name field updates (givenName, familyName)."""
    if path == "name.givenname":
        if op_type == "remove":
            scim_metadata.pop("givenName", None)
        else:
            scim_metadata["givenName"] = str(value)
    elif path == "name.familyname":
        if op_type == "remove":
            scim_metadata.pop("familyName", None)
        else:
            scim_metadata["familyName"] = str(value)


def _handle_group_operations(op_type: str, value: Any, teams_set: Set[str]) -> Optional[Set[str]]:
    """Handle group/team membership operations."""
    group_values = _extract_group_values(value)
    if op_type == "replace":
        return set(group_values)
    elif op_type == "add":
        teams_set.update(group_values)
    elif op_type == "remove":
        for gid in group_values:
            teams_set.discard(gid)
    return None


def _handle_generic_metadata(path: str, op_type: str, value: Any, metadata: Dict[str, Any]) -> None:
    """Handle generic metadata operations for unknown paths."""
    if op_type == "remove":
        metadata.pop(path, None)
    else:
        metadata[path] = value


def _apply_patch_ops(
    existing_user: LiteLLM_UserTable,
    patch_ops: SCIMPatchOp,
) -> tuple[Dict[str, Any], Set[str]]:
    """Apply patch operations and return update data and final team set."""
    update_data: Dict[str, Any] = {}
    metadata = existing_user.metadata or {}
    scim_metadata = metadata.get("scim_metadata", {})

    teams_set: Set[str] = set(existing_user.teams or [])
    replace_team_set: Optional[Set[str]] = None

    for op in patch_ops.Operations:
        path = (op.path or "").lower()
        value = op.value
        op_type = op.op

        if path == "displayname":
            _handle_displayname_update(op_type, value, update_data)
        elif path == "externalid":
            _handle_externalid_update(op_type, value, update_data)
        elif path == "active":
            _handle_active_update(op_type, value, metadata)
        elif path in ("name.givenname", "name.familyname"):
            _handle_name_update(path, op_type, value, scim_metadata)
        elif path.startswith("groups"):
            new_replace_set = _handle_group_operations(op_type, value, teams_set)
            if new_replace_set is not None:
                replace_team_set = new_replace_set
        else:
            _handle_generic_metadata(path, op_type, value, metadata)

    final_team_set = replace_team_set if replace_team_set is not None else teams_set
    metadata["scim_metadata"] = scim_metadata
    update_data["metadata"] = metadata
    return update_data, final_team_set

async def patch_team_membership(
    user_id: str,
    teams_ids_to_add_user_to: List[str],
    teams_ids_to_remove_user_from: List[str],
) -> bool:
    """
    Add or remove user from teams
    """
    for _team_id in teams_ids_to_add_user_to:
            try:
                await team_member_add(
                    data=TeamMemberAddRequest(
                        team_id=_team_id,
                        member=Member(user_id=user_id, role="user"),
                    ),
                    user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
                )
            except Exception as e:
                verbose_proxy_logger.exception(f"Error adding user to team {_team_id}: {e}")

    for _team_id in teams_ids_to_remove_user_from:
        try:
            await team_member_delete(
                data=TeamMemberDeleteRequest(team_id=_team_id, user_id=user_id),
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
            )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error removing user from team {_team_id}: {e}")
    

    return True

@scim_router.patch(
    "/Users/{user_id}",
    response_model=SCIMUser,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def patch_user(
    user_id: str = Path(..., title="User ID"),
    patch_ops: SCIMPatchOp = Body(...),
):
    """
    Patch a user according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    verbose_proxy_logger.debug("SCIM PATCH USER request: %s", patch_ops)

    try:
        # Check if user exists
        existing_user = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if not existing_user:
            raise HTTPException(
                status_code=404, detail={"error": f"User not found with ID: {user_id}"}
            )

        update_data, final_team_set = _apply_patch_ops(
            existing_user=existing_user,
            patch_ops=patch_ops,
        )

        existing_teams = set(existing_user.teams or [])
        added_groups = final_team_set - existing_teams
        removed_groups = existing_teams - final_team_set

        await patch_team_membership(
            user_id=user_id,
            teams_ids_to_add_user_to=list(added_groups),
            teams_ids_to_remove_user_from=list(removed_groups),
        )

        update_data["teams"] = list(final_team_set)

        updated_user = await prisma_client.db.litellm_usertable.update(
            where={"user_id": user_id},
            data=update_data,
        )

        scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(updated_user)

        return scim_user

    except Exception as e:
        raise handle_exception_on_proxy(e)


# Group Endpoints
@scim_router.get(
    "/Groups",
    response_model=SCIMListResponse,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_groups(
    startIndex: int = Query(1, ge=1),
    count: int = Query(10, ge=1, le=100),
    filter: Optional[str] = Query(None),
):
    """
    Get a list of groups according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        # Parse filter if provided (basic support)
        where_conditions = {}
        if filter:
            # Very basic filter support - only handling displayName eq
            if "displayName eq" in filter:
                team_alias = filter.split("displayName eq ")[1].strip("\"'")
                where_conditions["team_alias"] = team_alias

        # Get teams from database
        teams = await prisma_client.db.litellm_teamtable.find_many(
            where=where_conditions,
            skip=(startIndex - 1),
            take=count,
            order={"created_at": "desc"},
        )

        # Get total count for pagination
        total_count = await prisma_client.db.litellm_teamtable.count(
            where=where_conditions
        )

        # Convert to SCIM format
        scim_groups = []
        for team in teams:
            # Get team members
            members = []
            for member_id in team.members or []:
                member = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": member_id}
                )
                if member:
                    display_name = member.user_email or member.user_id
                    members.append(
                        SCIMMember(value=member.user_id, display=display_name)
                    )

            team_alias = getattr(team, "team_alias", team.team_id)
            team_created_at = team.created_at.isoformat() if team.created_at else None
            team_updated_at = team.updated_at.isoformat() if team.updated_at else None

            scim_group = SCIMGroup(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
                id=team.team_id,
                displayName=team_alias,
                members=members,
                meta={
                    "resourceType": "Group",
                    "created": team_created_at,
                    "lastModified": team_updated_at,
                },
            )
            scim_groups.append(scim_group)

        return SCIMListResponse(
            totalResults=total_count,
            startIndex=startIndex,
            itemsPerPage=min(count, len(scim_groups)),
            Resources=scim_groups,
        )

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.get(
    "/Groups/{group_id}",
    response_model=SCIMGroup,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_group(
    group_id: str = Path(..., title="Group ID"),
):
    """
    Get a single group by ID according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": group_id}
        )

        if not team:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Group not found with ID: {group_id}"},
            )

        scim_group = await ScimTransformations.transform_litellm_team_to_scim_group(
            team
        )
        return scim_group

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.post(
    "/Groups",
    response_model=SCIMGroup,
    status_code=201,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def create_group(
    group: SCIMGroup = Body(...),
):
    """
    Create a group according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        # Generate ID if not provided
        team_id = group.id or str(uuid.uuid4())

        # Check if team already exists
        existing_team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": team_id}
        )

        if existing_team:
            raise HTTPException(
                status_code=409,
                detail={"error": f"Group already exists with ID: {team_id}"},
            )

        # Extract members
        members_with_roles: List[Member] = []
        if group.members:
            for member in group.members:
                # Check if user exists
                user = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": member.value}
                )
                if user:
                    members_with_roles.append(Member(user_id=member.value, role="user"))

        # Create team in database
        created_team = await new_team(
            data=NewTeamRequest(
                team_id=team_id,
                team_alias=group.displayName,
                members_with_roles=members_with_roles,
            ),
            http_request=Request(scope={"type": "http", "path": "/scim/v2/Groups"}),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

        scim_group = await ScimTransformations.transform_litellm_team_to_scim_group(
            created_team
        )
        return scim_group
    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.put(
    "/Groups/{group_id}",
    response_model=SCIMGroup,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def update_group(
    group_id: str = Path(..., title="Group ID"),
    group: SCIMGroup = Body(...),
):
    """
    Update a group according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        # Check if team exists
        existing_team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": group_id}
        )

        if not existing_team:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Group not found with ID: {group_id}"},
            )

        # Extract members
        member_ids = []
        if group.members:
            for member in group.members:
                # Check if user exists
                user = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": member.value}
                )
                if user:
                    member_ids.append(member.value)

        # Update team in database
        existing_metadata = existing_team.metadata if existing_team.metadata else {}
        updated_team = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": group_id},
            data={
                "team_alias": group.displayName,
                "members": member_ids,
                "metadata": {**existing_metadata, "scim_data": group.model_dump()},
            },
        )

        # Handle user-team relationships
        current_members = existing_team.members or []

        # Add new members to team
        for member_id in member_ids:
            if member_id not in current_members:
                user = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": member_id}
                )
                if user:
                    current_user_teams = user.teams or []
                    if group_id not in current_user_teams:
                        await prisma_client.db.litellm_usertable.update(
                            where={"user_id": member_id},
                            data={"teams": {"push": group_id}},
                        )

        # Remove former members from team
        for member_id in current_members:
            if member_id not in member_ids:
                user = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": member_id}
                )
                if user:
                    current_user_teams = user.teams or []
                    if group_id in current_user_teams:
                        new_teams = [t for t in current_user_teams if t != group_id]
                        await prisma_client.db.litellm_usertable.update(
                            where={"user_id": member_id}, data={"teams": new_teams}
                        )

        # Get updated members for response
        members = []
        for member_id in member_ids:
            user = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": member_id}
            )
            if user:
                display_name = user.user_email or user.user_id
                members.append(SCIMMember(value=user.user_id, display=display_name))

        team_created_at = (
            updated_team.created_at.isoformat() if updated_team.created_at else None
        )
        team_updated_at = (
            updated_team.updated_at.isoformat() if updated_team.updated_at else None
        )

        return SCIMGroup(
            schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
            id=group_id,
            displayName=updated_team.team_alias or group_id,
            members=members,
            meta={
                "resourceType": "Group",
                "created": team_created_at,
                "lastModified": team_updated_at,
            },
        )

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.delete(
    "/Groups/{group_id}",
    status_code=204,
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_group(
    group_id: str = Path(..., title="Group ID"),
):
    """
    Delete a group according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    try:
        # Check if team exists
        existing_team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": group_id}
        )

        if not existing_team:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Group not found with ID: {group_id}"},
            )

        # For each member, remove this team from their teams list
        for member_id in existing_team.members or []:
            user = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": member_id}
            )
            if user:
                current_teams = user.teams or []
                if group_id in current_teams:
                    new_teams = [t for t in current_teams if t != group_id]
                    await prisma_client.db.litellm_usertable.update(
                        where={"user_id": member_id}, data={"teams": new_teams}
                    )

        # Delete team
        await prisma_client.db.litellm_teamtable.delete(where={"team_id": group_id})

        return Response(status_code=204)

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.patch(
    "/Groups/{group_id}",
    response_model=SCIMGroup,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def patch_group(
    group_id: str = Path(..., title="Group ID"),
    patch_ops: SCIMPatchOp = Body(...),
):
    """
    Patch a group according to SCIM v2 protocol
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})

    verbose_proxy_logger.debug("SCIM PATCH GROUP request: %s", patch_ops)

    try:
        # Check if group exists
        existing_team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": group_id}
        )

        if not existing_team:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Group not found with ID: {group_id}"},
            )
        return None
    except Exception as e:
        raise handle_exception_on_proxy(e)
