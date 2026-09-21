from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter()


@router.get(
    "/experimental/online-model-experiments/{experiment_id}/metrics",
    tags=["Online Model Experiments"],
    dependencies=[Depends(user_api_key_auth)],
)
async def online_model_experiment_metrics(
    experiment_id: str,
    start_date: datetime | None = Query(default=None),
    end_date: datetime | None = Query(default=None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict[str, object]:
    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(status_code=403, detail="Admin access is required")
    if not experiment_id:
        raise HTTPException(status_code=400, detail="experiment_id must not be empty")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Prisma Client is not initialized")

    window_start: Final = (start_date or datetime(1970, 1, 1, tzinfo=timezone.utc)).isoformat()
    window_end: Final = (end_date or datetime.now(timezone.utc)).isoformat()
    rows: Final = await prisma_client.db.query_raw(
        """
        SELECT
            metadata->>'online_model_experiment_variant' AS variant,
            COUNT(*)::int AS requests,
            COUNT(DISTINCT NULLIF(metadata->>'online_model_experiment_assignment_key', ''))::int AS subjects,
            COALESCE(SUM(spend), 0)::float AS cost,
            COALESCE(SUM(total_tokens), 0)::int AS total_tokens,
            COALESCE(AVG(request_duration_ms), 0)::float AS latency_ms,
            COUNT(*) FILTER (WHERE COALESCE(status, 'success') NOT IN ('success', ''))::int AS errors
        FROM "LiteLLM_SpendLogs"
        WHERE metadata->>'online_model_experiment_id' = $1
          AND "startTime" >= $2::timestamptz
          AND "startTime" <= $3::timestamptz
        GROUP BY metadata->>'online_model_experiment_variant'
        ORDER BY variant
        """,
        experiment_id,
        window_start,
        window_end,
    )
    return {
        "experiment_id": experiment_id,
        "start_date": window_start,
        "end_date": window_end,
        "variants": [
            {
                "variant": str(row.get("variant") or "unknown"),
                "requests": int(row.get("requests") or 0),
                "subjects": int(row.get("subjects") or 0),
                "cost": float(row.get("cost") or 0),
                "total_tokens": int(row.get("total_tokens") or 0),
                "latency_ms": float(row.get("latency_ms") or 0),
                "errors": int(row.get("errors") or 0),
            }
            for row in rows
        ],
    }
