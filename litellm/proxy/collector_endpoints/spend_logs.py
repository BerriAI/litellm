import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from litellm.constants import (
    LITELLM_ASYNCIO_QUEUE_MAXSIZE,
    LITELLM_RELAY_CALL_TYPE,
    MAX_COLLECTOR_SPEND_LOG_BATCH_BYTES,
    MAX_COLLECTOR_SPEND_LOG_BYTES,
    MAX_COLLECTOR_SPEND_LOGS,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class CollectorSpendLogRow(TypedDict, total=False):
    request_id: str
    call_type: str
    api_key: str
    spend: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    startTime: datetime
    endTime: datetime
    model: str
    api_base: str
    custom_llm_provider: str
    user: Optional[str]
    team_id: Optional[str]
    organization_id: Optional[str]
    metadata: dict[str, Any]
    cache_hit: str
    cache_key: str
    request_tags: str
    messages: Any
    response: Any
    proxy_server_request: Any
    status: str


class CollectorSpendLogsIngestResponse(TypedDict):
    enqueued: int


async def _enqueue_collector_spend_logs(
    prisma_client: Any,
    spend_logs: list[CollectorSpendLogRow],
) -> None:
    async with prisma_client._spend_log_transactions_lock:
        queued_spend_logs = len(prisma_client.spend_log_transactions)
        if queued_spend_logs + len(spend_logs) > LITELLM_ASYNCIO_QUEUE_MAXSIZE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Collector spend-log queue is full",
                    "queued": queued_spend_logs,
                    "limit": LITELLM_ASYNCIO_QUEUE_MAXSIZE,
                },
            )
        prisma_client.spend_log_transactions.extend(spend_logs)


class CollectorSpendLogTransformer:
    PASSTHROUGH_FIELDS = {
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "startTime",
        "endTime",
        "completionStartTime",
        "model",
        "model_id",
        "model_group",
        "mcp_namespaced_tool_name",
        "agent_id",
        "api_base",
        "cache_hit",
        "cache_key",
        "end_user",
        "requester_ip_address",
        "messages",
        "response",
        "proxy_server_request",
        "session_id",
        "request_duration_ms",
        "status",
    }

    @staticmethod
    def transform_collector_events_to_spend_logs(
        logs: list[dict[str, Any]],
        user_api_key_dict: Any,
        now: datetime,
    ) -> list[CollectorSpendLogRow]:
        CollectorSpendLogTransformer._validate_raw_batch_size(logs)
        spend_logs = [
            CollectorSpendLogTransformer.transform_collector_event_to_spend_log(
                log=log,
                user_api_key_dict=user_api_key_dict,
                now=now,
            )
            for log in logs
        ]
        CollectorSpendLogTransformer._validate_normalized_batch_size(spend_logs)
        return spend_logs

    @staticmethod
    def transform_collector_event_to_spend_log(
        log: dict[str, Any],
        user_api_key_dict: Any,
        now: datetime,
    ) -> CollectorSpendLogRow:
        collector_request_id = log.get("request_id")
        if (
            not isinstance(collector_request_id, str)
            or not collector_request_id.strip()
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "request_id is required for collector spend-log ingestion"
                },
            )

        key_hash = CollectorSpendLogTransformer._get_auth_key_hash(user_api_key_dict)
        row: CollectorSpendLogRow = {
            "request_id": CollectorSpendLogTransformer._collector_request_id_for(
                key_hash,
                collector_request_id,
            ),
            "call_type": LITELLM_RELAY_CALL_TYPE,
            "api_key": key_hash or LITELLM_RELAY_CALL_TYPE,
            "spend": 0.0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "startTime": now,
            "endTime": now,
            "model": "local-ai-traffic",
            "api_base": "",
            "custom_llm_provider": "",
            "user": getattr(user_api_key_dict, "user_id", None),
            "team_id": getattr(user_api_key_dict, "team_id", None),
            "organization_id": CollectorSpendLogTransformer._get_auth_organization_id(
                user_api_key_dict
            ),
            "metadata": CollectorSpendLogTransformer._normalize_metadata(
                log.get("metadata"),
                user_api_key_dict,
                collector_request_id,
            ),
            "cache_hit": "False",
            "cache_key": "",
            "request_tags": CollectorSpendLogTransformer._normalize_request_tags(
                log.get("request_tags")
            ),
            "messages": {},
            "response": {},
            "proxy_server_request": {},
            "status": "success",
        }

        for key in CollectorSpendLogTransformer.PASSTHROUGH_FIELDS:
            if key in log:
                row[key] = log[key]

        return CollectorSpendLogTransformer._sanitize_json_value(row)

    @staticmethod
    def _get_auth_organization_id(user_api_key_dict: Any) -> Optional[str]:
        return getattr(user_api_key_dict, "organization_id", None) or getattr(
            user_api_key_dict, "org_id", None
        )

    @staticmethod
    def _get_auth_key_hash(user_api_key_dict: Any) -> Optional[str]:
        return getattr(user_api_key_dict, "api_key", None) or getattr(
            user_api_key_dict, "token", None
        )

    @staticmethod
    def _get_auth_key_alias(user_api_key_dict: Any) -> str:
        return (
            getattr(user_api_key_dict, "key_alias", None)
            or getattr(user_api_key_dict, "key_name", None)
            or LITELLM_RELAY_CALL_TYPE
        )

    @staticmethod
    def _get_auth_team_alias(user_api_key_dict: Any) -> Optional[str]:
        return getattr(user_api_key_dict, "team_alias", None) or None

    @staticmethod
    def _collector_request_id_for(
        key_hash: Optional[str], collector_request_id: str
    ) -> str:
        digest = hmac.new(
            (key_hash or LITELLM_RELAY_CALL_TYPE).encode(),
            collector_request_id.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"collector-{digest[:32]}"

    @staticmethod
    def _normalize_metadata(
        metadata: Any,
        user_api_key_dict: Any,
        collector_request_id: str,
    ) -> dict[str, Any]:
        if isinstance(metadata, dict):
            normalized = dict(metadata)
        elif metadata is None:
            normalized = {}
        else:
            normalized = {"relay_raw_metadata": metadata}

        key_hash = CollectorSpendLogTransformer._get_auth_key_hash(user_api_key_dict)
        normalized.update(
            {
                "source": LITELLM_RELAY_CALL_TYPE,
                "collector_request_id": collector_request_id,
                "relay_request_id": collector_request_id,
                "user_api_key": key_hash,
                "user_api_key_alias": CollectorSpendLogTransformer._get_auth_key_alias(
                    user_api_key_dict
                ),
                "user_api_key_user_id": getattr(user_api_key_dict, "user_id", None),
                "user_api_key_team_id": getattr(user_api_key_dict, "team_id", None),
                "user_api_key_team_alias": CollectorSpendLogTransformer._get_auth_team_alias(
                    user_api_key_dict
                ),
                "user_api_key_org_id": CollectorSpendLogTransformer._get_auth_organization_id(
                    user_api_key_dict
                ),
            }
        )
        return normalized

    @staticmethod
    def _normalize_request_tags(value: Any) -> str:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    parsed = [parsed]
            except json.JSONDecodeError:
                parsed = [value] if value.strip() else []
        elif isinstance(value, list):
            parsed = list(value)
        elif value is None:
            parsed = []
        else:
            parsed = [value]

        if LITELLM_RELAY_CALL_TYPE not in parsed:
            parsed.append(LITELLM_RELAY_CALL_TYPE)
        return json.dumps(parsed, separators=(",", ":"))

    @staticmethod
    def _sanitize_json_value(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("\x00", "")
        if not isinstance(value, (dict, list)):
            return value

        sanitized: Any = {} if isinstance(value, dict) else []
        stack = [(value, sanitized)]
        while stack:
            source, target = stack.pop()
            if isinstance(source, dict):
                for key, item in source.items():
                    sanitized_key = str(key).replace("\x00", "")
                    if isinstance(item, str):
                        target[sanitized_key] = item.replace("\x00", "")
                    elif isinstance(item, dict):
                        child: dict[str, Any] = {}
                        target[sanitized_key] = child
                        stack.append((item, child))
                    elif isinstance(item, list):
                        child_list: list[Any] = []
                        target[sanitized_key] = child_list
                        stack.append((item, child_list))
                    else:
                        target[sanitized_key] = item
            else:
                for item in source:
                    if isinstance(item, str):
                        target.append(item.replace("\x00", ""))
                    elif isinstance(item, dict):
                        child = {}
                        target.append(child)
                        stack.append((item, child))
                    elif isinstance(item, list):
                        child_list = []
                        target.append(child_list)
                        stack.append((item, child_list))
                    else:
                        target.append(item)
        return sanitized

    @staticmethod
    def _json_size_bytes(value: Any) -> int:
        return len(json.dumps(value, default=str, separators=(",", ":")).encode())

    @staticmethod
    def _validate_log_size(log: dict[str, Any]) -> int:
        encoded_size = CollectorSpendLogTransformer._json_size_bytes(log)
        if encoded_size > MAX_COLLECTOR_SPEND_LOG_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={
                    "error": f"Collector spend-log entry exceeds {MAX_COLLECTOR_SPEND_LOG_BYTES} bytes"
                },
            )
        return encoded_size

    @staticmethod
    def _validate_batch_size(total_bytes: int) -> None:
        if total_bytes > MAX_COLLECTOR_SPEND_LOG_BATCH_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={
                    "error": f"Collector spend-log batch exceeds {MAX_COLLECTOR_SPEND_LOG_BATCH_BYTES} bytes"
                },
            )

    @staticmethod
    def _validate_raw_batch_size(logs: list[dict[str, Any]]) -> None:
        total_bytes = 0
        for log in logs:
            total_bytes += CollectorSpendLogTransformer._json_size_bytes(log)
            CollectorSpendLogTransformer._validate_batch_size(total_bytes)

    @staticmethod
    def _validate_normalized_batch_size(spend_logs: list[dict[str, Any]]) -> None:
        total_bytes = 0
        for spend_log in spend_logs:
            total_bytes += CollectorSpendLogTransformer._validate_log_size(spend_log)
            CollectorSpendLogTransformer._validate_batch_size(total_bytes)


router = APIRouter(include_in_schema=False)


@router.post(
    "/collector/spend-logs",
    tags=["Collector"],
)
async def ingest_collector_spend_logs(
    payload: dict[str, list[dict[str, Any]]],
    request: Request,
    user_api_key_dict: Any = Depends(user_api_key_auth),
) -> CollectorSpendLogsIngestResponse:
    """
    Ingest LiteLLM Relay captures into the existing spend-log batcher so they
    appear in the Gateway Logs UI without replaying captured traffic.
    """
    prisma_client = getattr(request.app.state, "prisma_client", None)

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Prisma Client is not initialized"},
        )
    if getattr(request.app.state, "proxy_logging_obj", None) is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Proxy logging is not initialized"},
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

    spend_logs = CollectorSpendLogTransformer.transform_collector_events_to_spend_logs(
        logs=logs,
        user_api_key_dict=user_api_key_dict,
        now=datetime.now(timezone.utc),
    )
    await _enqueue_collector_spend_logs(
        prisma_client=prisma_client,
        spend_logs=spend_logs,
    )

    return {"enqueued": len(spend_logs)}
