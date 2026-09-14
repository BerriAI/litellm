from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Protocol


class RustMessages(Protocol):
    def __call__(
        self,
        model: str,
        body: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        has_agentic_hook: bool = False,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAmessages(Protocol):
    def __call__(
        self,
        model: str,
        body: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        has_agentic_hook: bool = False,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError
