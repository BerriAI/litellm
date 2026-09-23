from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException, Request

from litellm.proxy.auth.login_throttle import source_group
from litellm.proxy.auth.network import TrustedProxyConfig, resolve_client_ip


@dataclass(slots=True)
class _Window:
    started: float
    requests: int = 0
    active: int = 0


def _limit(name: str, default: int) -> int:
    value: Final = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class MCPAdmissionLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._client_rpm = _limit("LITELLM_MCP_PUBLIC_RPM", 120)
        self._worker_rpm = _limit("LITELLM_MCP_PUBLIC_WORKER_RPM", 600)
        self._client_active = _limit("LITELLM_MCP_PUBLIC_MAX_IN_FLIGHT", 64)
        self._worker_active = _limit("LITELLM_MCP_PUBLIC_WORKER_MAX_IN_FLIGHT", 128)
        self._max_sources = _limit("LITELLM_MCP_PUBLIC_MAX_SOURCES", 4096)
        self._sources: dict[str, _Window] = {}
        self._worker = _Window(clock())

    @staticmethod
    def _reject(retry_after: int) -> None:
        raise HTTPException(
            status_code=429,
            detail="MCP configuration request limit exceeded; retry later",
            headers={"Retry-After": str(max(1, retry_after))},
        )

    @contextmanager
    def admit(self, source: str) -> Iterator[None]:
        now: Final = self._clock()
        if now - self._worker.started >= 60:
            self._worker = _Window(now, active=self._worker.active)
            self._sources = {
                key: value for key, value in self._sources.items() if value.active or now - value.started < 60
            }
        if self._worker.active >= self._worker_active:
            self._reject(1)
        if self._worker.requests >= self._worker_rpm:
            self._reject(math.ceil(60 - (now - self._worker.started)))
        previous: Final = self._sources.get(source)
        if previous is None and len(self._sources) >= self._max_sources:
            self._reject(math.ceil(60 - (now - self._worker.started)))
        client: Final = (
            previous
            if previous is not None and now - previous.started < 60
            else _Window(now, active=previous.active if previous is not None else 0)
        )
        if client.active >= self._client_active:
            self._reject(1)
        if client.requests >= self._client_rpm:
            self._reject(math.ceil(60 - (now - client.started)))
        self._sources[source] = client
        client.requests += 1
        client.active += 1
        self._worker.requests += 1
        self._worker.active += 1
        try:
            yield
        finally:
            self._sources[source].active -= 1
            self._worker.active -= 1


def admission_source(request: Request) -> str:
    from litellm.proxy.proxy_server import general_settings  # noqa: PLC0415  # runtime proxy configuration

    settings: Final = general_settings or {}
    config: Final = TrustedProxyConfig.model_validate(
        {
            "use_forwarded_for": settings.get("use_x_forwarded_for", False),
            "trusted_proxy_cidrs": settings.get("mcp_trusted_proxy_ranges") or (),
        }
    )
    client_ip, _ = resolve_client_ip(request, config)
    return source_group(client_ip or "unknown")
