from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.rust_bridge.collector import normalize_collector_spend_logs

router = APIRouter()

MAX_COLLECTOR_SPEND_LOGS = 1000


def _get_auth_organization_id(user_api_key_dict: UserAPIKeyAuth) -> str | None:
    return getattr(user_api_key_dict, "organization_id", None) or getattr(
        user_api_key_dict, "org_id", None
    )


def _get_auth_key_hash(user_api_key_dict: UserAPIKeyAuth) -> str | None:
    return getattr(user_api_key_dict, "api_key", None) or getattr(
        user_api_key_dict, "token", None
    )


def _get_auth_key_alias(user_api_key_dict: UserAPIKeyAuth) -> str:
    return (
        getattr(user_api_key_dict, "key_alias", None)
        or getattr(user_api_key_dict, "key_name", None)
        or "litellm-relay"
    )


def _get_auth_team_alias(user_api_key_dict: UserAPIKeyAuth) -> str | None:
    return getattr(user_api_key_dict, "team_alias", None) or getattr(
        user_api_key_dict, "team_id", None
    )


def _get_collector_auth_context(user_api_key_dict: UserAPIKeyAuth) -> Dict[str, Any]:
    return {
        "key_hash": _get_auth_key_hash(user_api_key_dict),
        "key_alias": _get_auth_key_alias(user_api_key_dict),
        "team_id": getattr(user_api_key_dict, "team_id", None),
        "team_alias": _get_auth_team_alias(user_api_key_dict),
        "organization_id": _get_auth_organization_id(user_api_key_dict),
        "user_id": getattr(user_api_key_dict, "user_id", None),
    }


@router.post(
    "/collector/spend-logs",
    tags=["Collector"],
    include_in_schema=False,
)
async def ingest_collector_spend_logs(
    payload: Dict[str, List[Dict[str, Any]]],
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Ingest LiteLLM Relay captures into the existing spend-log batcher so they
    appear in the Gateway Logs UI without replaying captured traffic.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Prisma Client is not initialized"},
        )

    logs = payload.get("logs", [])
    if len(logs) == 0:
        return {"enqueued": 0}
    if len(logs) > MAX_COLLECTOR_SPEND_LOGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Collector spend-log ingestion is limited to {MAX_COLLECTOR_SPEND_LOGS} rows"
            },
        )

    try:
        spend_logs = normalize_collector_spend_logs(
            logs=logs,
            auth_context=_get_collector_auth_context(user_api_key_dict),
            now=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(exc)},
        ) from exc

    for spend_log in spend_logs:
        await proxy_logging_obj.db_spend_update_writer._insert_spend_log_to_db(
            payload=spend_log,
            prisma_client=prisma_client,
        )

    return {"enqueued": len(spend_logs)}
