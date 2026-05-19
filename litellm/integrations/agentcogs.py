# AgentCOGS per-customer margin callback — https://github.com/vaibhav11123/agentcogs

import json
import os
import time
from typing import Any, Optional

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)


def _usage_int(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def _extract_usage(kwargs: dict, response_obj: Any) -> dict:
    usage = kwargs.get("usage")
    if usage is not None:
        return {
            "prompt_tokens": _usage_int(usage, "prompt_tokens"),
            "completion_tokens": _usage_int(usage, "completion_tokens"),
        }
    if (
        isinstance(response_obj, litellm.ModelResponse)
        or isinstance(response_obj, litellm.EmbeddingResponse)
    ) and hasattr(response_obj, "usage"):
        u = response_obj.usage
        if isinstance(u, dict):
            return {
                "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(u.get("completion_tokens", 0) or 0),
            }
        return {
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        }
    return {"prompt_tokens": 0, "completion_tokens": 0}


def _metadata_from_kwargs(kwargs: dict) -> dict:
    meta = kwargs.get("metadata")
    if isinstance(meta, dict):
        return meta
    litellm_params = kwargs.get("litellm_params", {}) or {}
    meta = litellm_params.get("metadata", {})
    return meta if isinstance(meta, dict) else {}


class AgentCOGSLogger(CustomLogger):
    """POST per-completion cost to AgentCOGS /v1/ingest for B2B per-customer margin."""

    def __init__(self) -> None:
        super().__init__()
        self.validate_environment()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_http_handler = HTTPHandler()

    def validate_environment(self) -> None:
        missing_keys = []
        if os.getenv("AGENTCOGS_API_KEY") is None:
            missing_keys.append("AGENTCOGS_API_KEY")
        if os.getenv("AGENTCOGS_WORKSPACE_ID") is None:
            missing_keys.append("AGENTCOGS_WORKSPACE_ID")
        if len(missing_keys) > 0:
            raise Exception("Missing keys={} in environment.".format(missing_keys))

    def _endpoint(self) -> str:
        base = os.getenv("AGENTCOGS_ENDPOINT", "https://api.agentcogs.dev").rstrip("/")
        return f"{base}/v1/ingest"

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(os.environ["AGENTCOGS_API_KEY"]),
            "X-AgentCOGS-SDK-Version": "litellm-callback/0.1.0",
        }

    def _build_event(
        self,
        kwargs: dict,
        response_obj: Any,
        *,
        status: str,
        start_time: Any = None,
        error: Optional[str] = None,
    ) -> Optional[dict]:
        meta = _metadata_from_kwargs(kwargs)
        customer_id = kwargs.get("user") or meta.get("agentcogs_customer_id")
        if not customer_id:
            return None

        model = kwargs.get("model") or "unknown"
        cost = float(kwargs.get("response_cost") or 0)
        usage = _extract_usage(kwargs, response_obj)

        ts_source = start_time or kwargs.get("start_time")
        if ts_source is not None and hasattr(ts_source, "timestamp"):
            ts = int(ts_source.timestamp())
        else:
            ts = int(time.time())

        return {
            "run_id": str(uuid.uuid4()),
            "workspace_id": os.environ["AGENTCOGS_WORKSPACE_ID"],
            "customer_id": str(customer_id),
            "workflow_id": meta.get("agentcogs_workflow_id", "default"),
            "ts": ts,
            "status": status,
            "total_usd": cost,
            "models": {
                model: {
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "usd": cost,
                }
            },
            "node_costs": {},
            "metadata": {"source": "litellm"},
            "error": error,
        }

    async def _async_post(self, event: dict) -> None:
        try:
            response = await self.async_http_handler.post(
                url=self._endpoint(),
                data=json.dumps(event),
                headers=self._headers(),
            )
            response.raise_for_status()
        except Exception as e:
            verbose_logger.debug("AgentCOGS callback error: {}".format(e))

    def _sync_post(self, event: dict) -> None:
        try:
            response = self.sync_http_handler.post(
                url=self._endpoint(),
                data=json.dumps(event),
                headers=self._headers(),
            )
            response.raise_for_status()
        except Exception as e:
            verbose_logger.debug("AgentCOGS callback error: {}".format(e))

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        event = self._build_event(
            kwargs, response_obj, status="completed", start_time=start_time
        )
        if event:
            self._sync_post(event)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        event = self._build_event(
            kwargs, response_obj, status="completed", start_time=start_time
        )
        if event:
            await self._async_post(event)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        err = str(kwargs.get("exception", "error"))[:500]
        event = self._build_event(
            kwargs,
            response_obj,
            status="error",
            start_time=start_time,
            error=err,
        )
        if event:
            self._sync_post(event)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        err = str(kwargs.get("exception", "error"))[:500]
        event = self._build_event(
            kwargs,
            response_obj,
            status="error",
            start_time=start_time,
            error=err,
        )
        if event:
            await self._async_post(event)
