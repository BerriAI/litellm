import asyncio
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    requests: int
    failures: int
    fail_open: int
    capture_drops: int
    downstream_seconds: float
    first_byte_seconds: float


class GatewayMetrics:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._requests = 0
        self._failures = 0
        self._fail_open = 0
        self._capture_drops = 0
        self._downstream_seconds = 0.0
        self._first_byte_seconds = 0.0

    async def record_request(self, status: int, downstream_seconds: float, first_byte_seconds: float | None) -> None:
        async with self._lock:
            self._requests += 1
            self._failures += int(status >= 500)
            self._downstream_seconds += downstream_seconds
            self._first_byte_seconds += first_byte_seconds or 0.0

    async def record_fail_open(self) -> None:
        async with self._lock:
            self._fail_open += 1

    async def record_capture_drop(self) -> None:
        async with self._lock:
            self._capture_drops += 1

    async def snapshot(self) -> MetricsSnapshot:
        async with self._lock:
            return MetricsSnapshot(
                requests=self._requests,
                failures=self._failures,
                fail_open=self._fail_open,
                capture_drops=self._capture_drops,
                downstream_seconds=self._downstream_seconds,
                first_byte_seconds=self._first_byte_seconds,
            )

    async def render(self) -> bytes:
        snapshot = await self.snapshot()
        values = (
            ("litellm_codex_gateway_requests_total", snapshot.requests),
            ("litellm_codex_gateway_failures_total", snapshot.failures),
            ("litellm_codex_gateway_fail_open_total", snapshot.fail_open),
            ("litellm_codex_gateway_capture_drops_total", snapshot.capture_drops),
            ("litellm_codex_gateway_downstream_seconds_total", snapshot.downstream_seconds),
            ("litellm_codex_gateway_first_byte_seconds_total", snapshot.first_byte_seconds),
        )
        return "".join(f"{name} {value}\n" for name, value in values).encode()
