"""
Tenant-isolation helpers for managed file/batch/vector-store resources.

Returns a Prisma filter and an ownership check that scope managed resources
to the caller's identity: proxy admins see everything, user-keyed callers
see records they created, and service-account keys (no user_id) fall back
to the resource's owning team. Callers with no admin role and no
identifying ids are denied so an empty user_id can never select an
unscoped query.
"""

from typing import Any, Dict, List, Optional

from litellm.proxy._types import (
    UserAPIKeyAuth,
    user_api_key_has_admin_view as _user_has_admin_view,
)


def get_managed_resource_owner_id(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[str]:
    """Identity used to tag (``created_by``) and scope managed resources.

    Falls back to ``end_user_id`` when ``user_id`` is unset. Custom-auth /
    end-user-scoped deployments return only ``end_user_id`` from their auth
    callback (no ``user_id`` / ``team_id``); without this fallback their
    created batches/files are stored ownerless (``created_by=None``) and
    every retrieve/poll is denied. Using ``end_user_id`` as the owner lets
    the same end user read back what they created, while an identity-less
    caller (no user_id, team_id, or end_user_id) stays denied.
    """
    return user_api_key_dict.user_id or user_api_key_dict.end_user_id


def build_list_page(items: List[Any], has_more: bool = False) -> Dict[str, Any]:
    """Build the OpenAI-style paginated list response shape used by managed
    file/batch/vector-store listings. ``first_id`` and ``last_id`` are
    sourced from each item's ``.id`` attribute."""
    return {
        "object": "list",
        "data": items,
        "first_id": items[0].id if items else None,
        "last_id": items[-1].id if items else None,
        "has_more": has_more,
    }


def build_owner_filter(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[Dict[str, Any]]:
    """Return a Prisma `where` fragment that scopes a managed-resource listing
    to records the caller is allowed to see.

    - ``{}`` means no scoping (proxy admins).
    - ``{"created_by": <user_id or end_user_id>}`` for user-keyed and
      end-user-scoped (custom-auth) callers.
    - ``{"team_id": <team_id>}`` for service-account callers
      that have a team but no user_id.
    - ``{"OR": [...]}`` when the caller has both — listing must include
      both their own resources and team-shared ones so it stays consistent
      with ``can_access_resource``.
    - ``None`` means deny: callers MUST skip the query rather than fall
      back to an unscoped fetch.
    """
    if _user_has_admin_view(user_api_key_dict):
        return {}

    # ``user_id`` falls back to ``end_user_id`` for custom-auth / end-user
    # deployments so listings mirror the ownership stamped at create time.
    owner_id = get_managed_resource_owner_id(user_api_key_dict)
    team_id = user_api_key_dict.team_id

    if owner_id is not None and team_id is not None:
        return {
            "OR": [
                {"created_by": owner_id},
                {"team_id": team_id},
            ]
        }

    if owner_id is not None:
        return {"created_by": owner_id}

    if team_id is not None:
        return {"team_id": team_id}

    return None


def can_access_resource(
    user_api_key_dict: UserAPIKeyAuth,
    created_by: Optional[str],
    resource_team_id: Optional[str],
) -> bool:
    """Return True iff the caller may read/modify a managed resource.

    The resource's ``created_by`` and ``team_id`` fields must be non-None
    to match the caller's identity — guarding against the ``None == None``
    bypass that previously let service-account keys read every keyless
    resource.
    """
    if _user_has_admin_view(user_api_key_dict):
        return True

    # ``user_id`` falls back to ``end_user_id`` so custom-auth callers can
    # read resources they created. ``created_by`` must be non-None to match,
    # preserving the deny on the ``None == None`` bypass.
    owner_id = get_managed_resource_owner_id(user_api_key_dict)
    if owner_id is not None and created_by is not None and created_by == owner_id:
        return True

    team_id = user_api_key_dict.team_id
    if (
        team_id is not None
        and resource_team_id is not None
        and resource_team_id == team_id
    ):
        return True

    return False
