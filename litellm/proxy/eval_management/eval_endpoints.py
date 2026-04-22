"""
CRUD endpoints for Evals.
Mirrors litellm/proxy/guardrails/guardrail_endpoints.py
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


# ---------- Request / Response models ----------


class EvalCriterionModel(BaseModel):
    name: str
    weight: int
    description: str
    threshold: Optional[float] = None


class CreateEvalRequest(BaseModel):
    eval_name: str
    criteria: List[EvalCriterionModel]
    judge_model: str
    description: Optional[str] = None
    overall_threshold: Optional[float] = 80.0
    max_iterations: int = 1


class UpdateEvalRequest(BaseModel):
    criteria: Optional[List[EvalCriterionModel]] = None
    judge_model: Optional[str] = None
    description: Optional[str] = None
    overall_threshold: Optional[float] = None
    max_iterations: Optional[int] = None


class AttachEvalRequest(BaseModel):
    eval_id: str
    on_failure: str = "block"  # "block" | "log"
    overall_threshold_override: Optional[float] = None


# ---------- Helpers ----------


def _require_internal_user_or_above(user_api_key_dict: UserAPIKeyAuth) -> None:
    allowed = {
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    }
    if user_api_key_dict.user_role not in allowed:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions to manage evals."
        )


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role not in {
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.INTERNAL_USER,
    }:
        raise HTTPException(
            status_code=403, detail="Admin or internal_user role required."
        )


# ---------- Eval CRUD ----------


@router.post("/litellm_evals/config", tags=["Evals [Beta]"])
async def create_litellm_eval(
    request: CreateEvalRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import add_eval_to_db

    config = {
        **request.model_dump(),
        "criteria": [c.model_dump() for c in request.criteria],
        "created_by": user_api_key_dict.user_id or "",
        "updated_by": user_api_key_dict.user_id or "",
    }
    return await add_eval_to_db(config, prisma_client)


@router.get("/litellm_evals/config", tags=["Evals [Beta]"])
async def list_litellm_evals(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[Dict[str, Any]]:
    from litellm.proxy.proxy_server import prisma_client

    _require_internal_user_or_above(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import list_evals

    return await list_evals(prisma_client)


@router.get("/litellm_evals/config/{eval_id}", tags=["Evals [Beta]"])
async def get_litellm_eval(
    eval_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import prisma_client

    _require_internal_user_or_above(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import get_eval_by_id

    row = await get_eval_by_id(eval_id, prisma_client)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Eval '{eval_id}' not found.")
    return row


@router.put("/litellm_evals/config/{eval_id}", tags=["Evals [Beta]"])
async def update_litellm_eval(
    eval_id: str,
    request: UpdateEvalRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import update_eval_in_db

    update_data = {k: v for k, v in request.model_dump().items() if v is not None}
    if "criteria" in update_data:
        update_data["criteria"] = [
            c if isinstance(c, dict) else c.model_dump()
            for c in update_data["criteria"]
        ]
    update_data["updated_by"] = user_api_key_dict.user_id or ""
    return await update_eval_in_db(eval_id, update_data, prisma_client)


@router.delete("/litellm_evals/config/{eval_id}", tags=["Evals [Beta]"])
async def delete_litellm_eval(
    eval_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import delete_eval_from_db

    return await delete_eval_from_db(eval_id, prisma_client)


# ---------- Agent <-> Eval attachment ----------


@router.post("/litellm_evals/agents/{agent_id}/evals", tags=["Evals [Beta]"])
async def attach_eval_to_agent(
    agent_id: str,
    request: AttachEvalRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import (
        attach_eval_to_agent as _attach,
        get_eval_by_id,
    )

    eval_row = await get_eval_by_id(request.eval_id, prisma_client)
    if eval_row is None:
        raise HTTPException(
            status_code=404, detail=f"Eval '{request.eval_id}' not found."
        )

    return await _attach(
        agent_id=agent_id,
        eval_id=request.eval_id,
        params={
            "eval_name": eval_row["eval_name"],
            "on_failure": request.on_failure,
            "overall_threshold_override": request.overall_threshold_override,
            "created_by": user_api_key_dict.user_id or "",
        },
        prisma_client=prisma_client,
    )


@router.delete(
    "/litellm_evals/agents/{agent_id}/evals/{eval_id}", tags=["Evals [Beta]"]
)
async def detach_eval_from_agent(
    agent_id: str,
    eval_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import (
        detach_eval_from_agent as _detach,
    )

    return await _detach(agent_id, eval_id, prisma_client)


@router.get("/litellm_evals/agents/{agent_id}/evals", tags=["Evals [Beta]"])
async def get_agent_evals(
    agent_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[Dict[str, Any]]:
    from litellm.proxy.proxy_server import prisma_client

    _require_internal_user_or_above(user_api_key_dict)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    from litellm.proxy.eval_management.eval_registry import get_evals_for_agent

    return await get_evals_for_agent(agent_id, prisma_client)
