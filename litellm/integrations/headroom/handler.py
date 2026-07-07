"""Headroom integration for LiteLLM proxy.

Wraps `headroom-ai`'s `HeadroomCallback` with two MindTouch-specific
adjustments needed to deliver real compression on Claude-Code-style traffic:

1.  **`DEFAULT_EXCLUDE_TOOLS` override (module-import time).** Upstream
    Headroom's library hardcodes a frozenset that excludes the very tools
    we want to compress (Bash, Glob, Grep), AND the ones whose verbatim
    content matters for safety (Read, Edit, Write). Our override keeps
    Read/Edit/Write protected (Read: needs `read_lifecycle` to track
    stale/superseded; Edit/Write: records of changes that must round-trip
    verbatim) but removes Bash/Glob/Grep so observation-only outputs do
    compress.

    The patch must run BEFORE any `headroom.compress` import — the
    library's pipeline is a singleton built at first import and freezes
    the exclusion set. We do that at the top of this module, before the
    HeadroomCallback import.

2.  **Aggressive compression flags (per-call).** Upstream's callback
    constructor only exposes `min_tokens`, `model_limit`, `hooks`,
    `api_key`, `api_url`. The flags Phase A bench proved necessary
    (`protect_recent=0`, `compress_user_messages=True`,
    `target_ratio=0.5`) are missing — by default Headroom protects the
    latest user message, which is exactly the tool_result we want to
    shrink. The subclass overrides `_local_compress` to pass them.

Phase F bench numbers on real Claude Code workload (see PR #55 of
expert-sre-agent for receipts):

    Bash (git log/kubectl/find -ls)  53–85% reduction
    Grep (repo search)               64.6%
    Glob (path lists)                93.8%
    Read / Edit / Write              0% (correctly excluded)
    Aggregate                        67.5%
    Latency                          p99 0.6ms per compress() call

Wired in via litellm_settings.callbacks=["mt_headroom_aggressive"] in the
proxy YAML config; the registration logic lives in
`litellm/proxy/common_utils/callback_utils.py`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Compression toggle.
#
# Headroom compression is opt-out, controlled at two levels:
#
#   1. Global default via the ``HEADROOM_COMPRESSION_ENABLED`` env var. Unset
#      or truthy (1/true/yes/on) -> compression on (preserves prior behaviour);
#      falsy (0/false/no/off) -> off cluster-wide.
#   2. Per-request override via the ``x-headroom-compress`` request header,
#      which wins over the env default for that single call. Same truthy/falsy
#      vocabulary. Absent header -> fall back to the env default.
#
# The callback stays registered either way; these flags only govern whether
# ``_local_compress`` actually runs, so compression can be flipped per request
# without redeploying and the ModernBERT pipeline stays pre-warmed.
# ---------------------------------------------------------------------------
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})
_COMPRESS_HEADER = "x-headroom-compress"


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a truthy/falsy string; return ``default`` if unrecognised/None."""
    if value is None:
        return default
    v = value.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    return default


def _compression_enabled_by_env() -> bool:
    """Global default from ``HEADROOM_COMPRESSION_ENABLED`` (default: on)."""
    return _parse_bool(os.environ.get("HEADROOM_COMPRESSION_ENABLED"), default=True)

# ---------------------------------------------------------------------------
# Step 0: surface the headroom.* loggers on the proxy's stdout/stderr.
#
# Why we do this here, not via LITELLM_LOG: LITELLM_LOG=DEBUG only configures
# the ``LiteLLM Proxy`` namespace and its sub-loggers. Third-party loggers
# (``headroom.*``) inherit Python's default WARNING level and have no handlers
# attached, so the ``logger.info("Headroom: X→Y tokens (saved Z, %d%%)")``
# success line and the ``logger.warning("Headroom compression failed: ...")``
# error line never reach pod logs. Operators have no way to verify that
# compression is firing or to debug failures without exec'ing into the pod.
#
# Solution: attach a stderr StreamHandler to the ``headroom`` logger at
# import time and pin its effective level to INFO. We don't touch the root
# logger (would spam every other library's INFO) and we don't propagate
# (avoid duplicate lines if some downstream config later attaches a root
# handler).
# ---------------------------------------------------------------------------
_hr_logger = logging.getLogger("headroom")
if not any(getattr(h, "_mt_headroom_handler", False) for h in _hr_logger.handlers):
    _h = logging.StreamHandler(sys.stderr)
    _h._mt_headroom_handler = True  # type: ignore[attr-defined]
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _h.setLevel(logging.INFO)
    _hr_logger.addHandler(_h)
    _hr_logger.setLevel(logging.INFO)
    _hr_logger.propagate = False

# ---------------------------------------------------------------------------
# Step 1: monkey-patch DEFAULT_EXCLUDE_TOOLS BEFORE any headroom.compress
# import. The pipeline is a singleton; this list is captured at first import.
# ---------------------------------------------------------------------------
import headroom.config as _hr_config  # type: ignore[import-not-found]

_hr_config.DEFAULT_EXCLUDE_TOOLS = frozenset(
    {
        "Read",
        "Edit",
        "Write",
        # Lowercase variants for case-insensitive matching (mirrors upstream).
        "read",
        "edit",
        "write",
    }
)

# ---------------------------------------------------------------------------
# Step 2: subclass HeadroomCallback to pass aggressive flags to compress().
#
# ALSO mix in CustomLogger so LiteLLM's pre-call dispatcher accepts us.
# proxy/utils.py:1486 gates dispatch on
#     isinstance(_callback, CustomLogger)
#         and "async_pre_call_hook" in vars(_callback.__class__)
#         and _callback.__class__.async_pre_call_hook != CustomLogger.async_pre_call_hook
# Upstream HeadroomCallback inherits from object only, so without CustomLogger
# in our MRO the dispatch silently skips us even though we appear in
# ``litellm.callbacks``.
# ---------------------------------------------------------------------------
from headroom.integrations.litellm_callback import (  # noqa: E402  type: ignore[import-not-found]
    HeadroomCallback,
)
from litellm.integrations.custom_logger import CustomLogger  # noqa: E402


class MTHeadroomAggressive(HeadroomCallback, CustomLogger):
    """LiteLLM callback that compresses with Phase A's proven aggressive flags.

    The default `HeadroomCallback._local_compress` calls
    `headroom.compress.compress(...)` with only `model_limit` and `hooks`.
    Override the call to pass `protect_recent=0`,
    `compress_user_messages=True`, `target_ratio=0.5` so the latest
    tool_result becomes a compression target instead of a protected one.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prewarm_pipeline()

    @staticmethod
    def _prewarm_pipeline() -> None:
        """Force ModernBERT weights to load NOW, not on the first request.

        Headroom's text compressor lazy-loads ``answerdotai/ModernBERT-base``
        (~134 weight tensors, ~700MB) the first time it has to compress text.
        Observed cold-start latency on devops-watchtower: ~31 seconds — far
        too long for a request hot path. The same payload measured ~30ms on
        the second call (model in memory, no further loads).

        Drive a synthetic compression at startup so the proxy's first real
        request gets the warm path. The dummy payload is shaped like a
        Bash tool_result (same routing branch as Claude-Code's heaviest
        traffic) so the routing predicate hits the same compressor path.
        """
        try:
            from headroom.compress import compress  # type: ignore[import-not-found]

            dummy_messages = [
                {"role": "user", "content": "warmup"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "warmup_1",
                            "name": "Bash",
                            "input": {"command": "echo warmup"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "warmup_1",
                            # Long enough (>50 words) that the compressor
                            # actually engages instead of short-circuiting.
                            "content": (
                                "warmup line " * 200
                            ),
                        }
                    ],
                },
            ]
            compress(
                messages=dummy_messages,
                model="claude-sonnet-4-5-20250929",
                model_limit=200000,
                hooks=None,
                protect_recent=0,
                compress_user_messages=True,
                target_ratio=0.5,
            )
            logging.getLogger("headroom").info(
                "MTHeadroomAggressive: pipeline pre-warm complete"
            )
        except Exception as e:  # pragma: no cover — pre-warm is best-effort
            logging.getLogger("headroom").warning(
                "MTHeadroomAggressive: pipeline pre-warm failed: %s. "
                "First real request will pay the cold-start latency.",
                e,
            )

    # LiteLLM's pre-call dispatch (litellm/proxy/utils.py:1515) calls this with:
    #   (user_api_key_dict, cache, data, call_type)
    # Defining it satisfies the dispatcher's three gates:
    #   1. isinstance(_callback, CustomLogger) — MRO includes CustomLogger
    #   2. "async_pre_call_hook" in vars(self.__class__) — defined here
    #   3. cls.async_pre_call_hook != CustomLogger.async_pre_call_hook — overridden
    #
    # mt-007: We NO LONGER delegate to HeadroomCallback.async_pre_call_hook
    # because it calls _local_compress SYNCHRONOUSLY, blocking the uvicorn
    # event loop for 5-16s (ModernBERT inference on CPU). This starved the
    # /health/liveliness endpoint → pod restart loops.
    # Instead we implement the dispatch inline and run compress() via
    # asyncio.to_thread() so the event loop stays responsive.
    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        # Gate: only compress chat completions and anthropic messages
        effective_call_type = (
            "acompletion" if call_type == "anthropic_messages" else call_type
        )
        if effective_call_type not in ("completion", "acompletion"):
            return data

        messages = data.get("messages")
        if not messages:
            return data

        logger = logging.getLogger("headroom")

        # Toggle: per-request header overrides the global env default.
        headers = data.get("proxy_server_request", {}).get("headers", {}) or {}
        header_val = headers.get(_COMPRESS_HEADER)
        if header_val is None:
            # Header lookup is case-insensitive; proxy_server_request headers
            # are usually lower-cased already, but don't rely on it.
            for k, v in headers.items():
                if k.lower() == _COMPRESS_HEADER:
                    header_val = v
                    break
        if not _parse_bool(header_val, default=_compression_enabled_by_env()):
            source = "header" if header_val is not None else "env"
            logger.info("Headroom: compression disabled (%s), passing through", source)
            return data

        model = data.get("model", "")

        try:
            result = await asyncio.to_thread(
                self._local_compress, messages, model
            )
        except Exception as e:
            logger.warning("Headroom compression failed (non-fatal): %s", e)
            return data

        if result and result.get("messages"):
            data["messages"] = result["messages"]
            saved = result.get("tokens_saved", 0)
            before = result.get("tokens_before", 0)
            after = result.get("tokens_after", 0)
            pct = int((saved / before) * 100) if before else 0
            logger.info(
                "Headroom: %d→%d tokens (saved %d, %d%%)",
                before, after, saved, pct,
            )

        return data

    def _local_compress(self, messages: list[dict], model: str) -> dict[str, Any] | None:
        from headroom.compress import compress  # type: ignore[import-not-found]

        result = compress(
            messages=messages,
            model=model or "claude-sonnet-4-5-20250929",
            model_limit=self._model_limit,
            hooks=self._hooks,
            protect_recent=0,
            compress_user_messages=True,
            target_ratio=0.5,
        )
        return {
            "messages": result.messages,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
        }
