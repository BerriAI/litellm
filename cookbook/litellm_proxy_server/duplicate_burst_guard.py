"""
Example LiteLLM Proxy callback that blocks short duplicate request bursts.

Register this handler in proxy_config.yaml:

litellm_settings:
  callbacks:
    - duplicate_burst_guard.duplicate_burst_guard

This is an in-memory, single-process example. Use a shared store if your proxy
runs multiple workers or needs duplicate detection across instances.

The request fingerprint uses a whitelist of model-affecting fields and scopes
duplicates by authenticated API-key owner plus request user/session. Adjust
those fields if your deployment needs different duplicate semantics.
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Optional, cast

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

FINGERPRINT_REQUEST_KEYS = (
    "model",
    "system",
    "messages",
    "prompt",
    "input",
    "query",
    "documents",
    "instructions",
    "previous_response_id",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stop",
    "stream",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "seed",
    "modalities",
    "audio",
    "reasoning_effort",
    "thinking",
    "extra_body",
    "vector_store_id",
)
SKIP_CALL_TYPES: frozenset[str] = frozenset({"aimage_edit"})


class DuplicateBurstGuard(CustomLogger):
    def __init__(self, max_calls: int = 2, window_seconds: float = 10.0) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: defaultdict[str, deque[float]] = defaultdict(
            deque
        )  # mutable-ok: sliding window state; use Redis ZSET for multi-worker
        self._last_gc = 0.0  # mutable-ok: local GC watermark for in-process state
        self._lock = asyncio.Lock()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        if call_type in SKIP_CALL_TYPES:
            return data

        fingerprint = self._fingerprint(
            data=data, user_api_key_dict=user_api_key_dict, call_type=call_type
        )
        count = await self._record_and_count(fingerprint)
        if count > self.max_calls:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Duplicate request burst detected",
                    "fingerprint": fingerprint,
                },
            )
        return data

    async def _record_and_count(self, fingerprint: str) -> int:
        now = time.monotonic()
        async with self._lock:
            if now - self._last_gc > self.window_seconds:
                self._prune(now)
                self._last_gc = now

            timestamps = self._calls[fingerprint]
            timestamps.append(now)
            self._prune_key(timestamps, now)
            return len(timestamps)

    def _prune(self, now: float) -> None:
        for fingerprint, timestamps in list(self._calls.items()):
            self._prune_key(timestamps, now)
            if not timestamps:
                self._calls.pop(fingerprint, None)

    def _prune_key(self, timestamps: deque[float], now: float) -> None:
        while timestamps and now - timestamps[0] > self.window_seconds:
            timestamps.popleft()

    def _fingerprint(
        self,
        data: dict[str, object],
        user_api_key_dict: Optional[UserAPIKeyAuth],
        call_type: CallTypesLiteral,
    ) -> str:
        metadata_value = data.get("metadata")
        metadata: Mapping[str, object]
        if isinstance(metadata_value, Mapping):
            metadata = cast(Mapping[str, object], metadata_value)
        else:
            metadata = {}
        key_user_id: object = None
        if user_api_key_dict is not None:
            key_user_id = user_api_key_dict.user_id
        request_user_id: object = (
            data.get("user")
            or metadata.get("user_id")
            or metadata.get("session_id")
            or "anonymous_request"
        )
        request = {key: data[key] for key in FINGERPRINT_REQUEST_KEYS if key in data}
        raw = json.dumps(
            {
                "api_key_identity": key_user_id or "anonymous_key",
                "call_type": call_type,
                "request": request,
                "request_identity": request_user_id,
            },
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


duplicate_burst_guard = DuplicateBurstGuard()
