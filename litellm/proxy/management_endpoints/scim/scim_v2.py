"""
✨ SCIM v2 Endpoints for LiteLLM Proxy using Internal User/Team Management

This is an enterprise feature and requires a premium license.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

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
from pydantic import BaseModel
from typing_extensions import TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    NewTeamRequest,
    NewUserRequest,
    NewUserResponse,
    ProxyErrorTypes,
    ProxyException,
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


class UserProvisionerHelpers:
    """Helper methods for user provisioning operations."""

    @staticmethod
    async def handle_existing_user_by_email(
        prisma_client, new_user_request: NewUserRequest
    ) -> Optional[SCIMUser]:
        """
        Check if a user with the given email already exists and update them if found.

        Args:
            prisma_client: Database client
            new_user_request: New user request data

        Returns:
            SCIMUser if user was updated, None if no existing user found
        """
        if not new_user_request.user_email:
            return None

        existing_user = await prisma_client.db.litellm_usertable.find_first(
            where={"user_email": new_user_request.user_email}
        )

        if not existing_user:
            return None

        # Update the user
        updated_user = await prisma_client.db.litellm_usertable.update(
            where={"user_id": existing_user.user_id},
            data={
                "user_id": new_user_request.user_id,
                "user_email": new_user_request.user_email,
                "user_alias": new_user_request.user_alias,
                "teams": new_user_request.teams,
                "metadata": safe_dumps(new_user_request.metadata),
            },
        )

        return await ScimTransformations.transform_litellm_user_to_scim_user(
            updated_user
        )


class ScimUserData(TypedDict):
    """Typed structure for extracted SCIM user data."""

    user_email: Optional[str]
    user_alias: Optional[str]
    sso_user_id: Optional[str]
    teams: List[str]
    given_name: Optional[str]
    family_name: Optional[str]
    active: Optional[bool]


class GroupMemberExtractionResult(BaseModel):
    """Result of extracting and processing group members."""

    existing_member_ids: List[str]
    created_users: List[NewUserResponse]
    all_member_ids: List[str]  # existing + newly created


scim_router = APIRouter(
    prefix="/scim/v2",
    tags=["✨ SCIM v2 (Enterprise Only)"],
    dependencies=[Depends(_premium_user_check)],
)


# Helper functions for common operations
async def _get_prisma_client_or_raise_exception():
    """Check if database is connected and raise HTTPException if not."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})
    return prisma_client


async def _check_user_exists(user_id: str):
    """Check if user exists and return user, raise 404 if not found."""
    prisma_client = await _get_prisma_client_or_raise_exception()

    user = await prisma_client.db.litellm_usertable.find_unique(
        where={"user_id": user_id}
    )

    if not user:
        raise HTTPException(
            status_code=404, detail={"error": f"User not found with ID: {user_id}"}
        )

    return user


async def _check_team_exists(team_id: str):
    """Check if team exists and return team, raise 404 if not found."""
    prisma_client = await _get_prisma_client_or_raise_exception()

    team = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id}
    )

    if not team:
        raise HTTPException(
            status_code=404, detail={"error": f"Group not found with ID: {team_id}"}
        )

    return team


def _extract_scim_user_data(user: SCIMUser) -> ScimUserData:
    """Extract common data from SCIMUser object."""
    user_email = None
    if user.emails and len(user.emails) > 0:
        user_email = user.emails[0].value

    user_alias = None
    if user.name and user.name.givenName:
        user_alias = user.name.givenName

    teams = []
    if user.groups:
        teams = [group.value for group in user.groups]

    return {
        "user_email": user_email,
        "user_alias": user_alias,
        "sso_user_id": user.externalId,
        "teams": teams,
        "given_name": user.name.givenName if user.name else None,
        "family_name": user.name.familyName if user.name else None,
        "active": user.active,
    }


def _build_scim_metadata(
    given_name: Optional[str], family_name: Optional[str], active: Optional[bool] = None
) -> Dict[str, Any]:
    """Build metadata dictionary with SCIM data."""
    metadata: Dict[str, Any] = {
        "scim_metadata": LiteLLM_UserScimMetadata(
            givenName=given_name,
            familyName=family_name,
        ).model_dump()
    }

    if active is not None:
        metadata["scim_active"] = active

    return metadata


async def _extract_group_member_ids(group: SCIMGroup) -> GroupMemberExtractionResult:
    """
    Extract member IDs from SCIMGroup, creating users that don't exist.

    Returns:
        GroupMemberExtractionResult with existing members, created users, and all member IDs
    """
    prisma_client = await _get_prisma_client_or_raise_exception()
    existing_member_ids = []
    created_users = []
    all_member_ids = []

    if group.members:
        for member in group.members:
            user_id = member.value

            # Check if user exists
            user = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user_id}
            )

            if user:
                existing_member_ids.append(user_id)
                all_member_ids.append(user_id)
            else:
                # Create the user if they don't exist using our helper
                created_user = await _create_user_if_not_exists(
                    user_id=user_id, created_via="scim_group_membership"
                )

                if created_user:
                    created_users.append(created_user)
                    all_member_ids.append(user_id)
                # If creation failed, user is skipped (logged in helper)

    return GroupMemberExtractionResult(
        existing_member_ids=existing_member_ids,
        created_users=created_users,
        all_member_ids=all_member_ids,
    )


async def _get_team_members_display(member_ids: List[str]) -> List[SCIMMember]:
    """Get SCIMMember objects with display names for a list of member IDs."""
    prisma_client = await _get_prisma_client_or_raise_exception()
    members: List[SCIMMember] = []

    for member_id in member_ids:
        user = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": member_id}
        )
        if user:
            display_name = user.user_email or user.user_id
            members.append(SCIMMember(value=user.user_id, display=display_name))

    return members


async def _handle_team_membership_changes(
    user_id: str, existing_teams: List[str], new_teams: List[str]
) -> None:
    """Handle adding/removing user from teams based on changes."""
    existing_teams_set = set(existing_teams)
    new_teams_set = set(new_teams)

    teams_to_add = new_teams_set - existing_teams_set
    teams_to_remove = existing_teams_set - new_teams_set

    if teams_to_add or teams_to_remove:
        await patch_team_membership(
            user_id=user_id,
            teams_ids_to_add_user_to=list(teams_to_add),
            teams_ids_to_remove_user_from=list(teams_to_remove),
        )


async def _create_user_if_not_exists(
    user_id: str, created_via: str = "scim_group"
) -> Optional[NewUserResponse]:
    """
    Helper function to create a user if they don't exist.

    Args:
        user_id: The user ID to create
        created_via: Context for where the user was created from

    Returns:
        LiteLLM_UserTable if user was created, None if creation failed
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import new_user

    try:
        # Get default role for new internal users
        default_role: Optional[
            Literal[
                LitellmUserRoles.PROXY_ADMIN,
                LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
                LitellmUserRoles.INTERNAL_USER,
                LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            ]
        ] = LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        if litellm.default_internal_user_params:
            default_role = litellm.default_internal_user_params.get("user_role")

        new_user_request = NewUserRequest(
            user_id=user_id,
            user_email=user_id,  # We don't have email from group membership
            user_alias=None,
            teams=[],  # Teams will be added separately
            metadata={"created_via": created_via},
            auto_create_key=False,
            user_role=default_role,
        )

        created_user = await new_user(data=new_user_request)
        verbose_proxy_logger.info(f"Created user {user_id} via {created_via}")
        return created_user

    except Exception as e:
        verbose_proxy_logger.exception(f"Failed to create user {user_id}: {e}")
        return None


async def _get_team_member_user_ids_from_team(team: LiteLLM_TeamTable) -> List[str]:
    """
    Get the IDs of the members from a team.

    Use one source of truth for the member IDs: team.members_with_roles

    """
    member_user_ids: List[str] = []
    for member in team.members_with_roles or []:
        if hasattr(member, "user_id") and member.user_id is not None:
            member_user_ids.append(member.user_id)
        elif isinstance(member, dict) and "user_id" in member:
            user_id = member.get("user_id")
            if user_id is not None:
                member_user_ids.append(user_id)
    return member_user_ids


# Dependency to set the correct SCIM Content-Type
async def set_scim_content_type(response: Response):
    """Sets the Content-Type header to application/scim+json"""
    # Check if content type is already application/json, only override in that case
    # Avoids overriding for non-JSON responses or already correct types if they were set manually
    response.headers["Content-Type"] = "application/scim+json"


@scim_router.get(
    "/ServiceProviderConfig",
    response_model=SCIMServiceProviderConfig,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_service_provider_config(request: Request):
    """Return SCIM Service Provider Configuration."""
    verbose_proxy_logger.debug(
        "SCIM ServiceProviderConfig request: method=%s url=%s headers=%s",
        request.method,
        request.url,
        dict(request.headers),
    )
    meta = {
        "resourceType": "ServiceProviderConfig",
        "location": str(request.url),
    }
    return SCIMServiceProviderConfig(meta=meta)


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
    verbose_proxy_logger.debug(
        "SCIM GET USERS request: startIndex=%s count=%s filter=%s",
        startIndex,
        count,
        filter,
    )
    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
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
    verbose_proxy_logger.debug("SCIM GET USER request for user_id=%s", user_id)
    try:
        user = await _check_user_exists(user_id)

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
    try:
        verbose_proxy_logger.debug("SCIM CREATE USER request: %s", user.model_dump())
        prisma_client = await _get_prisma_client_or_raise_exception()

        # Extract data from SCIM user
        user_data = _extract_scim_user_data(user)

        # Check if user already exists
        if user.userName:
            existing_user = await prisma_client.db.litellm_usertable.find_unique(
                where={"user_id": user.userName}
            )
            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": f"User already exists with username: {user.userName}"
                    },
                )

        # Create user in database
        user_id = user.userName or str(uuid.uuid4())
        metadata = _build_scim_metadata(
            user_data["given_name"], user_data["family_name"]
        )

        default_role: Optional[
            Literal[
                LitellmUserRoles.PROXY_ADMIN,
                LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
                LitellmUserRoles.INTERNAL_USER,
                LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            ]
        ] = LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        if litellm.default_internal_user_params:
            default_role = litellm.default_internal_user_params.get("user_role")

        new_user_request = NewUserRequest(
            user_id=user_id,
            user_email=user_data["user_email"],
            user_alias=user_data["user_alias"],
            teams=user_data["teams"],
            metadata=metadata,
            auto_create_key=False,
            user_role=default_role,
        )

        # Check if user with email already exists and update if found
        existing_user_scim = await UserProvisionerHelpers.handle_existing_user_by_email(
            prisma_client=prisma_client, new_user_request=new_user_request
        )

        if existing_user_scim:
            return existing_user_scim

        created_user = await new_user(
            data=new_user_request,
        )

        scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(
            user=created_user
        )
        return scim_user
    except (
        HTTPException
    ) as e:  # allow exceptions like SCIMUserAlreadyExists to be raised
        raise e
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
    Update a user according to SCIM v2 protocol (full replacement)
    """
    verbose_proxy_logger.debug(
        "SCIM PUT USER request for user_id=%s: %s",
        user_id,
        user.model_dump(),
    )

    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
        existing_user = await _check_user_exists(user_id)

        # Extract data from SCIM user
        user_data = _extract_scim_user_data(user)

        # Build metadata with SCIM data
        metadata = _build_scim_metadata(
            user_data["given_name"], user_data["family_name"], user_data["active"]
        )

        # Handle team membership changes
        await _handle_team_membership_changes(
            user_id=user_id,
            existing_teams=existing_user.teams or [],
            new_teams=user_data["teams"],
        )

        # Update user with all new data (full replacement)
        update_data = {
            "user_email": user_data["user_email"],
            "user_alias": user_data["user_alias"],
            "sso_user_id": user_data["sso_user_id"],
            "teams": user_data["teams"],
            "metadata": metadata,
        }

        # Serialize metadata to JSON string for Prisma to avoid GraphQL parsing issues
        if "metadata" in update_data and isinstance(update_data["metadata"], dict):
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            update_data["metadata"] = safe_dumps(update_data["metadata"])

        updated_user = await prisma_client.db.litellm_usertable.update(
            where={"user_id": user_id},
            data=update_data,
        )

        # Convert back to SCIM format
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
    verbose_proxy_logger.debug("SCIM DELETE USER request for user_id=%s", user_id)
    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
        existing_user = await _check_user_exists(user_id)

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


def _handle_displayname_update(
    op_type: str, value: Any, update_data: Dict[str, Any]
) -> None:
    """Handle displayname updates."""
    if op_type == "remove":
        update_data["user_alias"] = None
    else:
        update_data["user_alias"] = str(value)


def _handle_externalid_update(
    op_type: str, value: Any, update_data: Dict[str, Any]
) -> None:
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


def _handle_name_update(
    path: str, op_type: str, value: Any, scim_metadata: Dict[str, Any]
) -> None:
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


def _handle_group_operations(
    op_type: str, value: Any, teams_set: Set[str]
) -> Optional[Set[str]]:
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


def _handle_generic_metadata(
    path: str, op_type: str, value: Any, metadata: Dict[str, Any]
) -> None:
    """Handle generic metadata operations for unknown paths."""
    if op_type == "remove":
        metadata.pop(path, None)
    else:
        metadata[path] = value


def _apply_patch_ops(
    existing_user: LiteLLM_UserTable,
    patch_ops: SCIMPatchOp,
) -> Tuple[Dict[str, Any], Set[str]]:
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
    
    Handles duplicate membership gracefully (idempotent operation).
    If a user is already in a team, that's fine - we don't treat it as an error.
    """
    for _team_id in teams_ids_to_add_user_to:
        try:
            await team_member_add(
                data=TeamMemberAddRequest(
                    team_id=_team_id,
                    member=Member(user_id=user_id, role="user"),
                ),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN
                ),
            )
        except ProxyException as e:
            # Handle duplicate membership gracefully - this is idempotent
            if e.type == ProxyErrorTypes.team_member_already_in_team:
                verbose_proxy_logger.debug(
                    f"User {user_id} is already in team {_team_id}, skipping add"
                )
            else:
                verbose_proxy_logger.exception(
                    f"Error adding user to team {_team_id}: {e}"
                )
        except Exception as e:
            verbose_proxy_logger.exception(f"Error adding user to team {_team_id}: {e}")

    for _team_id in teams_ids_to_remove_user_from:
        try:
            await team_member_delete(
                data=TeamMemberDeleteRequest(team_id=_team_id, user_id=user_id),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN
                ),
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error removing user from team {_team_id}: {e}"
            )

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
    verbose_proxy_logger.debug(
        "SCIM PATCH USER request for user_id=%s: %s",
        user_id,
        patch_ops.model_dump(),
    )

    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
        existing_user = await _check_user_exists(user_id)

        update_data, final_team_set = _apply_patch_ops(
            existing_user=existing_user,
            patch_ops=patch_ops,
        )

        # Handle team membership changes
        await _handle_team_membership_changes(
            user_id=user_id,
            existing_teams=existing_user.teams or [],
            new_teams=list(final_team_set),
        )

        update_data["teams"] = list(final_team_set)

        # Serialize metadata to JSON string for Prisma to avoid GraphQL parsing issues
        if "metadata" in update_data and isinstance(update_data["metadata"], dict):
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            update_data["metadata"] = safe_dumps(update_data["metadata"])

        updated_user = await prisma_client.db.litellm_usertable.update(
            where={"user_id": user_id},
            data=update_data,
        )

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
    verbose_proxy_logger.debug(
        "SCIM GET GROUPS request: startIndex=%s count=%s filter=%s",
        startIndex,
        count,
        filter,
    )
    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
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
            # Get team members with display names
            members = await _get_team_members_display(team.members or [])
            verbose_proxy_logger.debug(f"SCIM GET GROUPS members: {members}")
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

        verbose_proxy_logger.debug(f"SCIM GET GROUPS response: {scim_groups}")
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
    verbose_proxy_logger.debug("SCIM GET GROUP request for group_id=%s", group_id)
    try:
        team = await _check_team_exists(group_id)

        scim_group = await ScimTransformations.transform_litellm_team_to_scim_group(
            team
        )
        verbose_proxy_logger.debug(f"SCIM GET GROUP response: {scim_group}")
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
    verbose_proxy_logger.debug(
        "SCIM CREATE GROUP request: %s",
        group.model_dump(),
    )
    try:
        prisma_client = await _get_prisma_client_or_raise_exception()

        # Generate ID if not provided
        team_id = group.id or group.externalId or str(uuid.uuid4())

        # Check if team already exists
        existing_team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": team_id}
        )

        if existing_team:
            raise HTTPException(
                status_code=409,
                detail={"error": f"Group already exists with ID: {team_id}"},
            )

        # Extract and process group members (creating users that don't exist)
        member_result = await _extract_group_member_ids(group)
        members_with_roles = [
            Member(user_id=member_id, role="user")
            for member_id in member_result.all_member_ids
        ]

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
    verbose_proxy_logger.debug(
        "SCIM PUT GROUP request for group_id=%s: %s",
        group_id,
        group.model_dump(),
    )
    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
        existing_team = await _check_team_exists(group_id)

        # Extract and process group members (creating users that don't exist)
        member_result = await _extract_group_member_ids(group)
        verbose_proxy_logger.debug(
            f"SCIM PUT GROUP all_member_ids: {member_result.all_member_ids}"
        )
        verbose_proxy_logger.debug(
            f"SCIM PUT GROUP created_users: {len(member_result.created_users)}"
        )

        # Prepare update data
        existing_metadata = existing_team.metadata if existing_team.metadata else {}
        updated_metadata = {**existing_metadata, "scim_data": group.model_dump()}

        update_data = {
            "team_alias": group.displayName,
            "metadata": safe_dumps(updated_metadata),
        }

        # Update team in database
        updated_team = await prisma_client.db.litellm_teamtable.update(
            where={"team_id": group_id},
            data=update_data,
        )

        # Handle user-team relationship changes
        current_members = set(await _get_team_member_user_ids_from_team(existing_team))
        verbose_proxy_logger.debug(f"SCIM PUT GROUP current_members: {current_members}")
        final_members = set(member_result.all_member_ids)
        verbose_proxy_logger.debug(f"SCIM PUT GROUP final_members: {final_members}")

        await _handle_group_membership_changes(
            group_id=group_id,
            current_members=current_members,
            final_members=final_members,
        )

        # Convert to SCIM format and return
        scim_group = await ScimTransformations.transform_litellm_team_to_scim_group(
            updated_team
        )
        return scim_group

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
    verbose_proxy_logger.debug("SCIM DELETE GROUP request for group_id=%s", group_id)
    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
        existing_team = await _check_team_exists(group_id)

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


async def _process_group_patch_operations(
    patch_ops: SCIMPatchOp, existing_team, prisma_client
) -> Tuple[Dict[str, Any], Set[str]]:
    """Process patch operations for a group and return update data and final members."""
    update_data: Dict[str, Any] = {}

    # Create a fresh copy of existing metadata to avoid Prisma issues
    existing_metadata = existing_team.metadata or {}
    metadata = dict(existing_metadata) if existing_metadata else {}

    # Track member changes
    current_members = set(existing_team.members or [])
    final_members = current_members.copy()

    # Process each patch operation
    for op in patch_ops.Operations:
        path = (op.path or "").lower()
        value = op.value
        op_type = op.op

        if path == "displayname":
            if op_type == "remove":
                update_data["team_alias"] = None
            else:
                update_data["team_alias"] = str(value)
        elif path == "externalid":
            if op_type == "remove":
                metadata.pop("externalId", None)
            else:
                metadata["externalId"] = str(value)
        elif path.startswith("members"):
            # Handle member operations
            member_values = _extract_group_values(value)
            # Create users that don't exist and get all valid member IDs
            valid_members = []
            for member_id in member_values:
                user = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": member_id}
                )
                if user:
                    valid_members.append(member_id)
                else:
                    # Create the user if they don't exist using our helper
                    created_user = await _create_user_if_not_exists(
                        user_id=member_id, created_via="scim_group_patch"
                    )

                    if created_user:
                        valid_members.append(member_id)
                    # If creation failed, user is skipped (logged in helper)

            if op_type == "replace":
                final_members = set(valid_members)
            elif op_type == "add":
                final_members.update(valid_members)
            elif op_type == "remove":
                for member_id in valid_members:
                    final_members.discard(member_id)
        else:
            # Handle other generic metadata
            if op_type == "remove":
                metadata.pop(path, None)
            else:
                metadata[path] = value

    # Include metadata in update data if it exists
    if metadata:
        update_data["metadata"] = metadata

    return update_data, final_members


async def _apply_group_patch_updates(
    group_id: str, update_data: Dict[str, Any], final_members: Set[str], prisma_client
):
    """Apply patch updates to the group in the database."""
    # Serialize metadata if present
    if "metadata" in update_data and isinstance(update_data["metadata"], dict):
        update_data["metadata"] = safe_dumps(update_data["metadata"])

    # Update members list
    update_data["members"] = list(final_members)

    # Update team in database
    updated_team = await prisma_client.db.litellm_teamtable.update(
        where={"team_id": group_id},
        data=update_data,
    )

    return updated_team


async def _handle_group_membership_changes(
    group_id: str, current_members: Set[str], final_members: Set[str]
):
    """Handle adding/removing members from the group."""
    members_to_add = final_members - current_members
    members_to_remove = current_members - final_members

    verbose_proxy_logger.debug(f"members_to_add: {members_to_add}")
    verbose_proxy_logger.debug(f"members_to_remove: {members_to_remove}")

    # Use existing helper functions for team membership changes
    for member_id in members_to_add:
        await patch_team_membership(
            user_id=member_id,
            teams_ids_to_add_user_to=[group_id],
            teams_ids_to_remove_user_from=[],
        )

    for member_id in members_to_remove:
        await patch_team_membership(
            user_id=member_id,
            teams_ids_to_add_user_to=[],
            teams_ids_to_remove_user_from=[group_id],
        )


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
    verbose_proxy_logger.debug(
        "SCIM PATCH GROUP request for group_id=%s: %s",
        group_id,
        patch_ops.model_dump(),
    )

    try:
        prisma_client = await _get_prisma_client_or_raise_exception()
        existing_team = await _check_team_exists(group_id)

        # Process patch operations
        update_data, final_members = await _process_group_patch_operations(
            patch_ops, existing_team, prisma_client
        )

        # Track current members BEFORE update for comparison
        current_members = set(await _get_team_member_user_ids_from_team(existing_team))

        # Apply updates to the database
        updated_team = await _apply_group_patch_updates(
            group_id, update_data, final_members, prisma_client
        )

        # Refresh team data from database to get the latest state after concurrent updates
        # This prevents race conditions when multiple PATCH requests come in simultaneously
        refreshed_team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": group_id}
        )
        if refreshed_team:
            # Re-read current members from refreshed team to account for concurrent updates
            refreshed_current_members = set(
                await _get_team_member_user_ids_from_team(
                    LiteLLM_TeamTable(**refreshed_team.model_dump())
                )
            )
            # Use the refreshed members for comparison
            current_members = refreshed_current_members

        # Handle user-team relationship changes
        await _handle_group_membership_changes(group_id, current_members, final_members)

        # Refresh team one more time to get final state after membership changes
        final_team = await prisma_client.db.litellm_teamtable.find_unique(
            where={"team_id": group_id}
        )
        if final_team:
            updated_team = final_team

        # Convert to SCIM format and return
        scim_group = await ScimTransformations.transform_litellm_team_to_scim_group(
            LiteLLM_TeamTable(**updated_team.model_dump())
        )
        return scim_group

    except Exception as e:
        raise handle_exception_on_proxy(e)
