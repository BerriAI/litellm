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

from typing import Any

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

    # LiteLLM's pre-call dispatch (litellm/proxy/utils.py:1508) only invokes
    # async_pre_call_hook when the method is defined directly on the registered
    # class (`"async_pre_call_hook" in vars(_callback.__class__)`). If we only
    # inherit it from HeadroomCallback, the dispatch skips us silently. So
    # re-declare the hook here as a thin pass-through to super().
    async def async_pre_call_hook(
        self,
        user_api_key: str,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        return await super().async_pre_call_hook(user_api_key, data, call_type)

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
