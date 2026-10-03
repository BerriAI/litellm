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
from collections.abc import Mapping
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,  # noqa: TID251  # internal API needed for non-blocking HTTP
)

_REPORT_URL = "https://tickerr.ai/api/v1/report"
_UA = "litellm-tickerr/1.0"
_MAX_INFLIGHT = threading.Semaphore(10)


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

    def __init__(
        self, **kwargs: object
    ) -> None:  # kwargs-ok: parent signature is untyped  # pyright: ignore[reportAny]
        super().__init__(**kwargs)
        self.disabled: bool = os.environ.get("TICKERR_DISABLED", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self.region: str | None = os.environ.get("TICKERR_REGION")
        try:
            self.sample_rate: float = min(1.0, max(0.0, float(os.environ.get("TICKERR_SAMPLE_RATE", "0"))))
        except (ValueError, TypeError):
            self.sample_rate = 0.0

    def log_failure_event(  # pyright: ignore[reportAny]  # parent is untyped
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> None:
        self._report(kwargs, start_time, end_time)

    async def async_log_failure_event(  # pyright: ignore[reportAny]  # parent is untyped
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> None:
        self._report(kwargs, start_time, end_time)

    def log_success_event(  # pyright: ignore[reportAny]  # parent is untyped
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> None:
        if self.sample_rate > 0 and random.random() < self.sample_rate:
            self._report(kwargs, start_time, end_time, is_success=True)

    async def async_log_success_event(  # pyright: ignore[reportAny]  # parent is untyped
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime | float,
        end_time: datetime | float,
    ) -> None:
        if self.sample_rate > 0 and random.random() < self.sample_rate:
            self._report(kwargs, start_time, end_time, is_success=True)

    def _report(
        self,
        kwargs: Mapping[str, object],
        start_time: datetime | float,
        end_time: datetime | float,
        is_success: bool = False,
    ) -> None:
        if self.disabled:
            return

        model: str = str(kwargs.get("model", "") or "")

        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            latency = round((end_time - start_time).total_seconds() * 1000)
        else:
            latency = round((float(end_time) - float(start_time)) * 1000)

        litellm_params = kwargs.get("litellm_params")
        provider: str | None = None
        if isinstance(litellm_params, Mapping):
            raw = litellm_params.get("custom_llm_provider")
            if isinstance(raw, str):
                provider = raw
        if provider is None:
            raw_fallback = kwargs.get("custom_llm_provider")
            if isinstance(raw_fallback, str):
                provider = raw_fallback

        exception = kwargs.get("exception")
        status_code: int | None = None
        raw_code = getattr(exception, "status_code", None)
        if isinstance(raw_code, int):
            status_code = raw_code

        pairs: tuple[tuple[str, str | int], ...] = (
            *((("provider", provider),) if provider is not None else ()),
            *((("model", model),) if model else ()),
            ("latency_ms", latency),
            ("event_type", "success" if is_success else "failure"),
            *((("status_code", status_code),) if status_code is not None else ()),
            *((("region", self.region),) if self.region is not None else ()),
        )
        payload = dict(pairs)  # mutable-ok: consumed once by httpx.post(json=...)

        def _send() -> None:
            if not _MAX_INFLIGHT.acquire(blocking=False):
                return
            try:
                client = _get_httpx_client()
                client.post(
                    _REPORT_URL,
                    json=payload,
                    headers={"User-Agent": _UA},  # mutable-ok: consumed once by httpx
                    timeout=2,
                )
            except (OSError, ValueError):  # fire-and-forget; network errors are expected
                pass
            finally:
                _MAX_INFLIGHT.release()

        threading.Thread(target=_send, daemon=True).start()
