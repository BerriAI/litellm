"""
Tickerr — outage radar for AI agents.

Reports LLM API failures to https://tickerr.ai so agents can
see how many other agents are hitting the same issue and get
a live routing recommendation (RETRY / RETRY_WITH_DELAY / FALLBACK).

Zero dependencies beyond stdlib. Anonymous. Non-blocking.

Usage:
    litellm.callbacks = ["tickerr"]

No API key required.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Union

from litellm.integrations.custom_logger import CustomLogger

_REPORT_URL = "https://tickerr.ai/api/v1/report"
_UA = "litellm-tickerr/1.0"

# Cap concurrent in-flight reports to avoid thread exhaustion on burst failures
_MAX_INFLIGHT = 5
_inflight = threading.Semaphore(_MAX_INFLIGHT)

# Map litellm custom_llm_provider → Tickerr provider slug
_PROVIDER_MAP: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "vertex_ai": "google",
    "gemini": "google",
    "cohere": "cohere",
    "mistral": "mistral",
    "groq": "groq",
    "together_ai": "together",
    "huggingface": "huggingface",
    "replicate": "replicate",
    "deepinfra": "deepinfra",
    "perplexity": "perplexity",
    "fireworks_ai": "fireworks",
    "openrouter": "openrouter",
    "azure": "azure",
    "bedrock": "aws",
    "ai21": "ai21",
    "cerebras": "cerebras",
    "xai": "xai",
    "deepseek": "deepseek",
    "ollama": "ollama",
    "nlp_cloud": "nlp_cloud",
}

# Only map codes that unambiguously indicate the specific error type.
# 500 is intentionally excluded: it is a generic "Internal Server Error" that
# indicates a crash or bug, not a capacity/overload condition.
_ERROR_TYPE_MAP: Dict[int, str] = {
    429: "rate_limit",
    529: "overloaded",
    503: "overloaded",
    408: "timeout",
    524: "timeout",
    401: "auth",
    403: "auth",
}


def _normalize_provider(model: str, kwargs: Dict[str, Any]) -> str:
    custom = (
        kwargs.get("litellm_params", {}).get("custom_llm_provider")
        or kwargs.get("custom_llm_provider")
        or ""
    )
    if custom:
        return _PROVIDER_MAP.get(custom.lower(), custom.lower())
    if "/" in model:
        prefix = model.split("/")[0].lower()
        return _PROVIDER_MAP.get(prefix, prefix)
    if re.match(r"^claude", model, re.I):
        return "anthropic"
    if re.match(r"^gpt|^o[1-9]", model, re.I):
        return "openai"
    if re.match(r"^gemini", model, re.I):
        return "google"
    if re.match(r"^mistral|^mixtral", model, re.I):
        return "mistral"
    if re.match(r"^llama", model, re.I):
        return "meta"
    if re.match(r"^command", model, re.I):
        return "cohere"
    if re.match(r"^grok", model, re.I):
        return "xai"
    if re.match(r"^deepseek", model, re.I):
        return "deepseek"
    return "unknown"


def _extract_status_code(exception: Optional[BaseException]) -> Optional[int]:
    if exception is None:
        return None
    code = getattr(exception, "status_code", None)
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    return None


def _latency_ms(
    start_time: Union[datetime, float], end_time: Union[datetime, float]
) -> int:
    """Compute elapsed ms whether LiteLLM passes datetime objects or floats."""
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        return round((end_time - start_time).total_seconds() * 1000)
    return round((float(end_time) - float(start_time)) * 1000)  # type: ignore[arg-type]


def _fire_and_forget(payload: Dict[str, Any]) -> None:
    """POST to Tickerr in a daemon thread — never blocks the caller.

    A semaphore caps concurrent in-flight threads so a burst of failures
    (e.g. 100 errors/s) cannot exhaust the thread pool.
    """
    if not _inflight.acquire(blocking=False):
        return  # already at max concurrent reports — drop silently

    def _send() -> None:
        try:
            import json as _json
            import urllib.request

            data = _json.dumps(payload).encode()
            req = urllib.request.Request(
                _REPORT_URL,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": _UA},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception:
            pass  # never crash the caller
        finally:
            _inflight.release()

    t = threading.Thread(target=_send, daemon=True)
    try:
        t.start()
    except Exception:
        # Thread could not be started (e.g. OS thread limit).
        # Release the slot so future reports are not permanently blocked.
        _inflight.release()


class TickerrLogger(CustomLogger):
    """
    LiteLLM built-in callback for Tickerr.

    Activated with:
        litellm.callbacks = ["tickerr"]

    Optional env vars:
        TICKERR_CLIENT_TIER  — "free" | "pro" | "team" | "enterprise" | "api_pay_as_you_go"
        TICKERR_REGION       — e.g. "us-east-1"
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client_tier: Optional[str] = os.environ.get("TICKERR_CLIENT_TIER")
        self.region: Optional[str] = os.environ.get("TICKERR_REGION")

    # ── sync ──────────────────────────────────────────────────────────────────

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
    ) -> None:
        self._report(kwargs, start_time, end_time)

    # ── async ─────────────────────────────────────────────────────────────────

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
    ) -> None:
        self._report(kwargs, start_time, end_time)

    # ── internal ──────────────────────────────────────────────────────────────

    def _report(
        self,
        kwargs: Dict[str, Any],
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
    ) -> None:
        model: str = kwargs.get("model", "") or ""
        exception: Optional[BaseException] = kwargs.get("exception")

        provider = _normalize_provider(model, kwargs)
        status_code = _extract_status_code(exception)
        # Only set error_type for codes we can classify with confidence
        error_type: Optional[str] = (
            _ERROR_TYPE_MAP.get(status_code) if status_code is not None else None
        )

        # Strip provider prefix: "anthropic/claude-3-5-haiku" → "claude-3-5-haiku"
        model_clean = model.split("/", 1)[-1] if "/" in model else model

        payload: Dict[str, Any] = {
            "provider": provider,
            "model": model_clean or None,
            "latency_ms": _latency_ms(start_time, end_time),
        }
        if status_code is not None:
            payload["error_code"] = status_code
        if error_type:
            payload["error_type"] = error_type
        if self.client_tier:
            payload["client_tier"] = self.client_tier
        if self.region:
            payload["region"] = self.region

        _fire_and_forget(payload)
