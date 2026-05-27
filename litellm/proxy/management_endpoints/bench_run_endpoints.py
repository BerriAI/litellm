"""
BENCH RUNS — admin registry of vLLM/SGLang benchmark results.

Backs the grid-ai-onboarding "Bench Runs" dashboard tab. Admin-only CRUD over
LiteLLM_BenchRun: each row is one benchmark run (model + run params + the raw
`vllm bench serve` command and its raw stdout, stored verbatim).

  POST   /bench/run/new      — create a bench run record
  GET    /bench/run/list     — list all bench runs (newest first)
  POST   /bench/run/delete   — delete a bench run by id

All endpoints require PROXY_ADMIN. Grid forwards these with its admin key, so
the user-facing admin gate lives in grid; the PROXY_ADMIN check here is
defense-in-depth (mirrors playground_endpoints._require_admin).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from prisma.errors import RecordNotFoundError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLMPydanticObjectBase,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper

router = APIRouter()

_TAGS = ["bench runs"]
_DEPS = [Depends(user_api_key_auth)]


# ─── request / response models ───────────────────────────────────────────────


class NewBenchRunRequest(LiteLLMPydanticObjectBase):
    model_name: str  # litellm model name (litellm_params.model), required
    deployment_server: Optional[str] = None  # "vllm" | "sglang"
    bench_type: Optional[str] = None  # "random" | "multi-turn"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    max_concurrency: Optional[int] = None
    raw_command: Optional[str] = None
    raw_results: Optional[str] = None
    # Set by grid to the acting user's email so the row is attributed to the
    # real caller rather than the shared admin key.
    created_by: Optional[str] = None


class DeleteBenchRunRequest(LiteLLMPydanticObjectBase):
    bench_run_id: str


class BenchRunResponse(LiteLLMPydanticObjectBase):
    bench_run_id: str
    model_name: str
    deployment_server: Optional[str] = None
    bench_type: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    max_concurrency: Optional[int] = None
    raw_command: Optional[str] = None
    raw_results: Optional[str] = None
    created_at: datetime
    created_by: Optional[str] = None
    updated_at: datetime
    updated_by: Optional[str] = None


# ─── helpers ─────────────────────────────────────────────────────────────────


def _prisma():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(500, "prisma_client not initialized")
    return prisma_client


def _require_admin(key: UserAPIKeyAuth) -> None:
    role = key.user_role
    value = role.value if role and hasattr(role, "value") else role
    if value != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(403, "PROXY_ADMIN required")


# ─── endpoints ────────────────────────────────────────────────────────────────


@router.post(
    "/bench/run/new",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=BenchRunResponse,
)
@management_endpoint_wrapper
async def new_bench_run(
    request: Request,
    data: NewBenchRunRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> BenchRunResponse:
    """Record a benchmark run. Only model_name is required; everything else is
    optional (grid pre-fills what it can parse from the run command)."""
    _require_admin(user_api_key_dict)

    payload = data.model_dump(exclude_unset=True)
    model_name = (payload.get("model_name") or "").strip()
    if not model_name:
        raise HTTPException(400, "model_name is required")
    payload["model_name"] = model_name
    created_by = payload.pop("created_by", None)

    row = await _prisma().db.litellm_benchrun.create(
        data={
            **payload,
            "created_by": created_by,
            "updated_by": created_by,
        }
    )
    verbose_proxy_logger.info(
        f"bench_runs: created {row.bench_run_id} model={model_name} by={created_by}"
    )
    return BenchRunResponse(**row.model_dump())


@router.get(
    "/bench/run/list",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=List[BenchRunResponse],
)
@management_endpoint_wrapper
async def list_bench_runs(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[BenchRunResponse]:
    """All bench runs, newest first. Volume is low (manual entries); grid
    derives filter options and filters client-side."""
    _require_admin(user_api_key_dict)
    rows = await _prisma().db.litellm_benchrun.find_many(order={"created_at": "desc"})
    return [BenchRunResponse(**r.model_dump()) for r in rows]


@router.post(
    "/bench/run/delete",
    tags=_TAGS,
    dependencies=_DEPS,
)
@management_endpoint_wrapper
async def delete_bench_run(
    request: Request,
    data: DeleteBenchRunRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Delete a bench run by id."""
    _require_admin(user_api_key_dict)
    try:
        await _prisma().db.litellm_benchrun.delete(
            where={"bench_run_id": data.bench_run_id}
        )
    except RecordNotFoundError:
        raise HTTPException(404, "bench run not found")
    return {"success": True}
