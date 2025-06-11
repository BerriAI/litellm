"""
SCIM v2 Endpoints for LiteLLM Proxy using Internal User/Team Management

"""

import uuid
from typing import List, Optional

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
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.proxy.management_endpoints.scim.scim_transformations import (
    ScimTransformations,
)
from litellm.proxy.management_endpoints.team_endpoints import new_team
from litellm.proxy.utils import handle_exception_on_proxy
from litellm.types.proxy.management_endpoints.scim_v2 import *

scim_router = APIRouter(
    prefix="/scim/v2",
    tags=["SCIM v2"],
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
        users: List[
            LiteLLM_UserTable
        ] = await prisma_client.db.litellm_usertable.find_many(
            where=where_conditions,
            skip=(startIndex - 1),
            take=count,
            order={"created_at": "desc"},
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
        # Check if user exists
        existing_user = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )

        if not existing_user:
            raise HTTPException(
                status_code=404, detail={"error": f"User not found with ID: {user_id}"}
            )

        # Extract email from SCIM user
        user_email = None
        if user.emails and len(user.emails) > 0:
            user_email = user.emails[0].value

        # Extract team memberships
        team_ids = []
        if user.groups:
            for group in user.groups:
                # Verify team exists
                team = await prisma_client.db.litellm_teamtable.find_unique(
                    where={"team_id": group.value}
                )
                if team:
                    team_ids.append(group.value)

        # Prepare update data
        update_data = {}
        if user_email:
            update_data["user_email"] = user_email
        if user.name and user.name.givenName:
            update_data["user_alias"] = user.name.givenName
        
        # Update metadata with SCIM information
        existing_metadata = existing_user.metadata or {}
        scim_metadata = LiteLLM_UserScimMetadata(
            givenName=user.name.givenName if user.name else None,
            familyName=user.name.familyName if user.name else None,
        )
        update_data["metadata"] = {
            **existing_metadata, 
            "scim_metadata": scim_metadata.model_dump()
        }
        
        # Update teams if different
        current_teams = set(existing_user.teams or [])
        new_teams = set(team_ids)
        
        if current_teams != new_teams:
            update_data["teams"] = team_ids
            
            # Handle team membership changes
            # Remove user from teams they're no longer in
            for team_id in current_teams - new_teams:
                team = await prisma_client.db.litellm_teamtable.find_unique(
                    where={"team_id": team_id}
                )
                if team:
                    current_members = team.members or []
                    if user_id in current_members:
                        new_members = [m for m in current_members if m != user_id]
                        await prisma_client.db.litellm_teamtable.update(
                            where={"team_id": team_id}, 
                            data={"members": new_members}
                        )
            
            # Add user to new teams
            for team_id in new_teams - current_teams:
                team = await prisma_client.db.litellm_teamtable.find_unique(
                    where={"team_id": team_id}
                )
                if team:
                    current_members = team.members or []
                    if user_id not in current_members:
                        new_members = current_members + [user_id]
                        await prisma_client.db.litellm_teamtable.update(
                            where={"team_id": team_id}, 
                            data={"members": new_members}
                        )

        # Update user
        updated_user = await prisma_client.db.litellm_usertable.update(
            where={"user_id": user_id},
            data=update_data,
        )

        # Convert to SCIM format and return
        scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
            updated_user
        )
        return scim_user
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

        update_data = {}
        existing_metadata = existing_user.metadata or {}
        current_scim_metadata = existing_metadata.get("scim_metadata", {})
        
        for operation in patch_ops.Operations:
            op = operation.op.lower()
            path = operation.path
            value = operation.value

            if op == "replace":
                if path == "active":
                    # Handle user activation/deactivation
                    # For now, we don't have an 'active' field in our user table
                    # but we could store this in metadata
                    update_data["metadata"] = {
                        **existing_metadata,
                        "scim_active": bool(value),
                        "scim_metadata": current_scim_metadata
                    }
                elif path == "displayName":
                    update_data["user_alias"] = str(value)
                elif path == "emails":
                    if isinstance(value, list) and len(value) > 0:
                        email_value = value[0].get("value") if isinstance(value[0], dict) else str(value[0])
                        update_data["user_email"] = email_value
                elif path == "name.givenName":
                    current_scim_metadata["givenName"] = str(value)
                    update_data["metadata"] = {
                        **existing_metadata,
                        "scim_metadata": current_scim_metadata
                    }
                elif path == "name.familyName":
                    current_scim_metadata["familyName"] = str(value)
                    update_data["metadata"] = {
                        **existing_metadata,
                        "scim_metadata": current_scim_metadata
                    }

            elif op == "add":
                if path == "groups" or (path is None and "groups" in str(value)):
                    # Adding user to groups/teams
                    groups_to_add = []
                    if isinstance(value, list):
                        groups_to_add = [g.get("value") if isinstance(g, dict) else str(g) for g in value]
                    elif isinstance(value, dict) and "groups" in value:
                        groups_to_add = [g.get("value") if isinstance(g, dict) else str(g) for g in value["groups"]]
                    
                    current_teams = set(existing_user.teams or [])
                    for group_id in groups_to_add:
                        if group_id not in current_teams:
                            # Add team to user
                            current_teams.add(group_id)
                            # Add user to team
                            team = await prisma_client.db.litellm_teamtable.find_unique(
                                where={"team_id": group_id}
                            )
                            if team:
                                current_members = team.members or []
                                if user_id not in current_members:
                                    await prisma_client.db.litellm_teamtable.update(
                                        where={"team_id": group_id},
                                        data={"members": current_members + [user_id]}
                                    )
                    update_data["teams"] = list(current_teams)

            elif op == "remove":
                if "groups" in (path or ""):
                    # Removing user from groups/teams
                    # Parse path like 'groups[value eq "team-id"]'
                    import re
                    match = re.search(r'groups\[value eq "([^"]+)"\]', path or "")
                    if match:
                        group_id = match.group(1)
                        current_teams = set(existing_user.teams or [])
                        if group_id in current_teams:
                            current_teams.remove(group_id)
                            update_data["teams"] = list(current_teams)
                            # Remove user from team
                            team = await prisma_client.db.litellm_teamtable.find_unique(
                                where={"team_id": group_id}
                            )
                            if team:
                                current_members = team.members or []
                                if user_id in current_members:
                                    await prisma_client.db.litellm_teamtable.update(
                                        where={"team_id": group_id},
                                        data={"members": [m for m in current_members if m != user_id]}
                                    )

        # Apply updates if any
        if update_data:
            updated_user = await prisma_client.db.litellm_usertable.update(
                where={"user_id": user_id},
                data=update_data,
            )
        else:
            updated_user = existing_user

        # Convert to SCIM format and return
        scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
            updated_user
        )
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

        # Extract members and validate they exist
        members_with_roles: List[Member] = []
        if group.members:
            for member in group.members:
                # Try to find user by user_id first, then by email
                user = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": member.value}
                )
                
                # If not found by user_id, try by email
                if not user:
                    user = await prisma_client.db.litellm_usertable.find_unique(
                        where={"user_email": member.value}
                    )
                
                if user:
                    members_with_roles.append(
                        Member(
                            user_id=user.user_id, 
                            user_email=user.user_email,
                            role="user"
                        )
                    )
                else:
                    verbose_proxy_logger.warning(
                        f"SCIM create_group: User not found for member value: {member.value}"
                    )

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

        update_data = {}
        current_members = existing_team.members or []
        
        for operation in patch_ops.Operations:
            op = operation.op.lower()
            path = operation.path
            value = operation.value

            if op == "replace":
                if path == "displayName":
                    update_data["team_alias"] = str(value)
                elif path == "members":
                    # Replace all members
                    new_member_ids = []
                    if isinstance(value, list):
                        for member in value:
                            member_value = member.get("value") if isinstance(member, dict) else str(member)
                            # Try to find user by user_id first, then by email
                            user = await prisma_client.db.litellm_usertable.find_unique(
                                where={"user_id": member_value}
                            )
                            if not user:
                                user = await prisma_client.db.litellm_usertable.find_unique(
                                    where={"user_email": member_value}
                                )
                            if user:
                                new_member_ids.append(user.user_id)
                    
                    # Update team members
                    update_data["members"] = new_member_ids
                    
                    # Handle user-team relationships
                    # Remove team from users no longer in the team
                    for member_id in current_members:
                        if member_id not in new_member_ids:
                            user = await prisma_client.db.litellm_usertable.find_unique(
                                where={"user_id": member_id}
                            )
                            if user:
                                current_user_teams = user.teams or []
                                if group_id in current_user_teams:
                                    new_teams = [t for t in current_user_teams if t != group_id]
                                    await prisma_client.db.litellm_usertable.update(
                                        where={"user_id": member_id}, 
                                        data={"teams": new_teams}
                                    )
                    
                    # Add team to new users
                    for member_id in new_member_ids:
                        if member_id not in current_members:
                            user = await prisma_client.db.litellm_usertable.find_unique(
                                where={"user_id": member_id}
                            )
                            if user:
                                current_user_teams = user.teams or []
                                if group_id not in current_user_teams:
                                    await prisma_client.db.litellm_usertable.update(
                                        where={"user_id": member_id},
                                        data={"teams": current_user_teams + [group_id]}
                                    )

            elif op == "add":
                if path == "members" or (path is None and "members" in str(value)):
                    # Adding members to group
                    members_to_add = []
                    if isinstance(value, list):
                        members_to_add = value
                    elif isinstance(value, dict) and "members" in value:
                        members_to_add = value["members"]
                    
                    for member in members_to_add:
                        member_value = member.get("value") if isinstance(member, dict) else str(member)
                        # Try to find user by user_id first, then by email
                        user = await prisma_client.db.litellm_usertable.find_unique(
                            where={"user_id": member_value}
                        )
                        if not user:
                            user = await prisma_client.db.litellm_usertable.find_unique(
                                where={"user_email": member_value}
                            )
                        if user and user.user_id not in current_members:
                            current_members.append(user.user_id)
                            # Add team to user
                            current_user_teams = user.teams or []
                            if group_id not in current_user_teams:
                                await prisma_client.db.litellm_usertable.update(
                                    where={"user_id": user.user_id},
                                    data={"teams": current_user_teams + [group_id]}
                                )
                    update_data["members"] = current_members

            elif op == "remove":
                if "members" in (path or ""):
                    # Removing members from group
                    # Parse path like 'members[value eq "user-id"]'
                    import re
                    match = re.search(r'members\[value eq "([^"]+)"\]', path or "")
                    if match:
                        member_value = match.group(1)
                        # Try to find user by user_id first, then by email
                        user = await prisma_client.db.litellm_usertable.find_unique(
                            where={"user_id": member_value}
                        )
                        if not user:
                            user = await prisma_client.db.litellm_usertable.find_unique(
                                where={"user_email": member_value}
                            )
                        if user and user.user_id in current_members:
                            current_members.remove(user.user_id)
                            update_data["members"] = current_members
                            # Remove team from user
                            current_user_teams = user.teams or []
                            if group_id in current_user_teams:
                                new_teams = [t for t in current_user_teams if t != group_id]
                                await prisma_client.db.litellm_usertable.update(
                                    where={"user_id": user.user_id}, 
                                    data={"teams": new_teams}
                                )

        # Apply updates if any
        if update_data:
            updated_team = await prisma_client.db.litellm_teamtable.update(
                where={"team_id": group_id},
                data=update_data,
            )
        else:
            updated_team = existing_team

        # Convert to SCIM format and return
        scim_group = await ScimTransformations.transform_litellm_team_to_scim_group(
            updated_team
        )
        return scim_group
    except Exception as e:
        raise handle_exception_on_proxy(e)
