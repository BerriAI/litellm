"""
Tickerr - crowd-sourced outage radar for AI agents.

Reports LLM API failures to https://tickerr.ai so every agent
can see when a provider is down and which model to fall back to.

Usage:
    litellm.callbacks = ["tickerr"]

No API key. No account. Failure-only by default. Success sampling is opt-in.
"""

from __future__ import annotations

import os
import random
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Union

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

_REPORT_URL = "https://tickerr.ai/api/v1/report"
_UA = "litellm-tickerr/1.0"


class TickerrLogger(CustomLogger):
    """
    LiteLLM callback that reports LLM API failures to Tickerr.

    When explicitly enabled via ``litellm.callbacks = ["tickerr"]``,
    anonymous failure metadata is reported. No prompts, responses,
    API keys, or personal data are sent.

    Optional env vars:
        TICKERR_DISABLED      - set to "true" to disable all reporting
        TICKERR_REGION        - e.g. us-east-1
        TICKERR_SAMPLE_RATE   - fraction of successes to report (0.0-1.0, default 0 = off)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.disabled: bool = os.environ.get("TICKERR_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }
        self.region: Optional[str] = os.environ.get("TICKERR_REGION")
        try:
            self.sample_rate: float = min(
                1.0, max(0.0, float(os.environ.get("TICKERR_SAMPLE_RATE", "0")))
            )
        except (ValueError, TypeError):
            self.sample_rate = 0.0

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
    ) -> None:
        self._report(kwargs, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
    ) -> None:
        self._report(kwargs, start_time, end_time)

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
    ) -> None:
        if self.sample_rate > 0 and random.random() < self.sample_rate:
            self._report(kwargs, start_time, end_time, is_success=True)

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
    ) -> None:
        if self.sample_rate > 0 and random.random() < self.sample_rate:
            self._report(kwargs, start_time, end_time, is_success=True)

    def _report(
        self,
        kwargs: Dict[str, Any],
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
        is_success: bool = False,
    ) -> None:
        if self.disabled:
            return

        model: str = kwargs.get("model", "") or ""

        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            latency = round((end_time - start_time).total_seconds() * 1000)
        else:
            latency = round((float(end_time) - float(start_time)) * 1000)  # type: ignore[arg-type]

        payload = {
            k: v
            for k, v in {
                "provider": kwargs.get("litellm_params", {}).get("custom_llm_provider")
                or kwargs.get("custom_llm_provider"),
                "model": model or None,
                "latency_ms": latency,
                "event_type": "success" if is_success else "failure",
                "status_code": getattr(kwargs.get("exception"), "status_code", None),
                "region": self.region,
            }.items()
            if v is not None
        }

        def _send() -> None:
            try:
                client = _get_httpx_client()
                client.post(
                    _REPORT_URL,
                    json=payload,
                    headers={"User-Agent": _UA},
                    timeout=2,
                )
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()
