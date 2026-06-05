from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .authz.policy_admin import (
    PolicyValidationError,
    make_assignment_rule,
    make_permission_rule,
)
from .authz.policy_store import reset_cache

router = APIRouter(tags=["auth_v2"])


class PermissionRequest(BaseModel):
    role: str
    resource: str
    action: str
    effect: str = "allow"
    domain: str = "*"
    resource_id: Optional[str] = None


class AssignmentRequest(BaseModel):
    subject_type: str
    subject_id: str
    role: str
    # When set (e.g. "team:eng"), the role applies only within that domain.
    domain: Optional[str] = None


def rule_to_row_data(rule: List[str]) -> Dict[str, str]:
    """Convert a casbin rule list (``[ptype, v0, v1, ...]``) to a DB row dict."""
    data: Dict[str, str] = {"ptype": rule[0]}
    for index, value in enumerate(rule[1:]):
        data[f"v{index}"] = value
    return data


def row_to_rule(row: Any) -> List[str]:
    rule = [row.ptype]
    for index in range(6):
        value = getattr(row, f"v{index}", None)
        if value is not None and value != "":
            rule.append(value)
    return rule


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="auth_v2 policy administration requires proxy admin",
        )


def _prisma() -> Any:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="auth_v2 policy administration requires a connected database",
        )
    return prisma_client


async def _add_rule(rule: List[str]) -> None:
    await _prisma().db.litellm_casbinrule.create(data=rule_to_row_data(rule))
    reset_cache()


async def _remove_rule(rule: List[str]) -> int:
    deleted = await _prisma().db.litellm_casbinrule.delete_many(
        where=rule_to_row_data(rule)
    )
    reset_cache()
    return deleted


@router.post("/auth/v2/policy/permission/add")
async def add_permission(
    body: PermissionRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    try:
        rule = make_permission_rule(
            role=body.role,
            resource=body.resource,
            action=body.action,
            effect=body.effect,
            domain=body.domain,
            resource_id=body.resource_id,
        )
    except PolicyValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    await _add_rule(rule)
    return {"added": rule}


@router.post("/auth/v2/policy/permission/remove")
async def remove_permission(
    body: PermissionRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    try:
        rule = make_permission_rule(
            role=body.role,
            resource=body.resource,
            action=body.action,
            effect=body.effect,
            domain=body.domain,
            resource_id=body.resource_id,
        )
    except PolicyValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    removed = await _remove_rule(rule)
    return {"removed": removed, "rule": rule}


@router.post("/auth/v2/policy/assignment/add")
async def add_assignment(
    body: AssignmentRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    try:
        rule = make_assignment_rule(
            body.subject_type, body.subject_id, body.role, body.domain
        )
    except PolicyValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    await _add_rule(rule)
    return {"added": rule}


@router.post("/auth/v2/policy/assignment/remove")
async def remove_assignment(
    body: AssignmentRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    try:
        rule = make_assignment_rule(
            body.subject_type, body.subject_id, body.role, body.domain
        )
    except PolicyValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    removed = await _remove_rule(rule)
    return {"removed": removed, "rule": rule}


@router.get("/auth/v2/policy/list")
async def list_policies(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    _require_admin(user_api_key_dict)
    rows = await _prisma().db.litellm_casbinrule.find_many()
    return {"rules": [row_to_rule(row) for row in rows]}
