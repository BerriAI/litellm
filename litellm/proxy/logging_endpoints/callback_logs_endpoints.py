"""
Ingest pre-built logging payloads from external producers and replay them
through LiteLLM's standard success/failure callback fan-out.

This exists for hosts that own a request outside the Python process — e.g. the
`litellm-rust` gateway proxying realtime websockets. Those hosts can't use the
in-process logging object, so they POST a finished `StandardLoggingPayload` here
and Python replays it through the exact same path a normal completion uses:
`Logging.async_success_handler` / `async_failure_handler`. Every registered
callback (spend logs, Langfuse, Datadog, ...) fires unchanged — there is no
spend-logs-specific or callback-specific code here, only the replay.

The endpoint is generic: realtime is the first producer, but the contract is the
self-describing `StandardLoggingPayload`, so completions/responses can use it too.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.proxy.callback_logs_endpoints import (
    CallbackLogFailure,
    CallbackLogRecord,
    CallbackLogsRequest,
    CallbackLogsResponse,
)

# Routes the Python proxy exposes for the Rust data-plane gateway to call into
# (logging today; auth/budgets later). Namespaced under /v1/rust_control_plane so
# they're clearly distinct from the proxy's own control-plane/management routes.
rust_control_plane_router = APIRouter(
    prefix="/v1/rust_control_plane", tags=["rust control plane"]
)


class CallbackLogsReplayer:
    """
    Replays finished logging payloads through LiteLLM's callback fan-out.

    Each helper is small and pure so the replay path is easy to read and test:
    rebuild a `Logging` object from the payload, seed `model_call_details` with
    exactly what the callbacks read, then dispatch to the success/failure
    handler. No spend/callback logic lives here — only the replay.
    """

    @staticmethod
    def _epoch_to_datetime(value: Any) -> datetime:
        """`StandardLoggingPayload` stores startTime/endTime as float epoch seconds."""
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, datetime):
            return value
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _build_logging_obj(payload: dict[str, Any]) -> LiteLLMLogging:
        """
        Reconstruct a `Logging` object from a finished payload and seed
        `model_call_details` with exactly what the success/failure callbacks
        read: the prebuilt `standard_logging_object`, the resolved
        `response_cost`, and the `litellm_params.metadata` keys used for cost
        attribution. Setting `standard_logging_object` up front makes the handler
        skip rebuilding it.
        """
        model = payload.get("model") or ""
        call_type = payload.get("call_type") or "acompletion"
        start_time = CallbackLogsReplayer._epoch_to_datetime(payload.get("startTime"))
        call_id = (
            payload.get("litellm_call_id") or payload.get("id") or str(uuid.uuid4())
        )

        logging_obj = LiteLLMLogging(
            model=model,
            messages=payload.get("messages") or [],
            # A replayed payload is always a *terminal*, fully-aggregated event —
            # the producer (e.g. the rust gateway) already collected the whole
            # session before POSTing. Never mark it streaming: a streaming
            # Logging object makes async_success_handler wait for a
            # complete_streaming_response that will never arrive, so the spend
            # log is never written.
            stream=False,
            call_type=call_type,
            start_time=start_time,
            litellm_call_id=call_id,
            function_id="",
        )

        metadata: dict[str, Any] = payload.get("metadata") or {}
        litellm_metadata: dict[str, Any] = {
            "user_api_key": metadata.get("user_api_key_hash"),
            "user_api_key_alias": metadata.get("user_api_key_alias"),
            "user_api_key_user_id": metadata.get("user_api_key_user_id"),
            "user_api_key_team_id": metadata.get("user_api_key_team_id"),
            "user_api_key_org_id": metadata.get("user_api_key_org_id"),
            "user_api_key_end_user_id": metadata.get("user_api_key_end_user_id"),
            "spend_logs_metadata": metadata.get("spend_logs_metadata"),
        }

        logging_obj.model_call_details.update(
            {
                "model": model,
                "call_type": call_type,
                "custom_llm_provider": payload.get("custom_llm_provider"),
                "response_cost": payload.get("response_cost") or 0.0,
                "standard_logging_object": payload,
                "litellm_params": {"metadata": litellm_metadata},
                "cache_hit": payload.get("cache_hit") or False,
            }
        )
        return logging_obj

    @staticmethod
    def _response_obj_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Minimal response object so usage-derived spend-log fields resolve."""
        return {
            "id": payload.get("id"),
            "usage": {
                "prompt_tokens": payload.get("prompt_tokens", 0),
                "completion_tokens": payload.get("completion_tokens", 0),
                "total_tokens": payload.get("total_tokens", 0),
            },
        }

    async def replay(self, record: CallbackLogRecord) -> None:
        """Replay one record through the matching success/failure handler."""
        payload = record.standard_logging_payload
        verbose_proxy_logger.debug(
            "CallbackLogsReplayer: replaying %s record id=%s model=%s call_type=%s",
            record.status,
            payload.get("id"),
            payload.get("model"),
            payload.get("call_type"),
        )

        logging_obj = self._build_logging_obj(payload)
        start_time = self._epoch_to_datetime(payload.get("startTime"))
        end_time = self._epoch_to_datetime(payload.get("endTime"))

        if record.status == "success":
            await logging_obj.async_success_handler(
                result=self._response_obj_from_payload(payload),
                start_time=start_time,
                end_time=end_time,
            )
        else:
            error_str = record.error or payload.get("error_str") or "replayed failure"
            await logging_obj.async_failure_handler(
                Exception(error_str),
                traceback_exception="",
                start_time=start_time,
                end_time=end_time,
            )

    async def replay_batch(
        self, records: list[CallbackLogRecord]
    ) -> CallbackLogsResponse:
        """Replay a batch; a single bad record never sinks the rest. Each failure
        is reported back with its batch index so the caller can retry/triage it."""
        processed = 0
        failures: list[CallbackLogFailure] = []
        for index, record in enumerate(records):
            try:
                await self.replay(record)
                processed += 1
            except Exception as e:
                failures.append(CallbackLogFailure(index=index, error=str(e)))
                verbose_proxy_logger.exception(
                    "CallbackLogsReplayer: failed to replay record %s: %s",
                    index,
                    str(e),
                )
        verbose_proxy_logger.debug(
            "CallbackLogsReplayer: batch done processed=%s failed=%s",
            processed,
            len(failures),
        )
        return CallbackLogsResponse(
            processed=processed, failed=len(failures), failures=failures
        )


@rust_control_plane_router.post(
    "/logs",
    dependencies=[Depends(user_api_key_auth)],
)
async def ingest_callback_logs(
    body: CallbackLogsRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> CallbackLogsResponse:
    """
    Replay a batch of finished logging payloads through the callback fan-out.

    Admin-only: the payloads write spend logs and trigger every callback, so this
    is a trusted internal route, not a public surface.
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="/v1/rust_control_plane/logs is admin-only (proxy admin key required).",
        )

    return await CallbackLogsReplayer().replay_batch(body.records)
