"""
Example LiteLLM Proxy callback that blocks short duplicate request bursts.

Register this handler in proxy_config.yaml:

litellm_settings:
  callbacks:
    - duplicate_burst_guard.duplicate_burst_guard

This is an in-memory, single-process example. Use a shared store if your proxy
runs multiple workers or needs duplicate detection across instances.
"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Optional, cast

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral


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
        messages_value = data.get("messages")
        system_prompt = ""
        last_user_prompt = ""
        prompt = ""

        if isinstance(messages_value, list):
            messages = cast(list[object], messages_value)
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                message_data = cast(Mapping[str, object], message)
                if message_data.get("role") == "system" and not system_prompt:
                    system_prompt = self._content_text(message_data.get("content"))
                if message_data.get("role") == "user":
                    last_user_prompt = self._content_text(message_data.get("content"))
        else:
            prompt = self._content_text(data.get("prompt"))

        metadata_value = data.get("metadata")
        metadata: Mapping[str, object]
        if isinstance(metadata_value, Mapping):
            metadata = cast(Mapping[str, object], metadata_value)
        else:
            metadata = {}
        user_id: object = (
            (user_api_key_dict.user_id if user_api_key_dict is not None else None)
            or data.get("user")
            or metadata.get("user_id")
            or metadata.get("session_id")
            or "anonymous"
        )
        raw = "|".join(
            (
                str(user_id),
                str(call_type),
                str(data.get("model", "")),
                str(data.get("temperature", "")),
                str(data.get("max_tokens", "")),
                system_prompt,
                last_user_prompt,
                prompt,
            )
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _content_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            content_items = cast(list[object], content)
            return "\n".join(
                text
                for item in content_items
                if isinstance(item, Mapping)
                for text in [cast(Mapping[str, object], item).get("text")]
                if isinstance(text, str)
            )
        return ""


duplicate_burst_guard = DuplicateBurstGuard()
