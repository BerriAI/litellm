from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx
import pytest

from litellm.proxy.control_plane.dataplanes.client import (
    DataplaneClient,
    DataplaneReply,
    DataplaneUnreachable,
    RetryPolicy,
)
from litellm.proxy.control_plane.dataplanes.credentials import (
    CredentialChain,
    CredentialUnavailable,
    ServiceTokenImpersonation,
)
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from tests.unit.proxy.control_plane.dataplanes.conftest import caller


@dataclass
class FakeTransport:  # mutable-ok: records calls for assertions
    outcomes: Iterator[httpx.Response | Exception]
    calls: list[tuple[str, str, dict[str, str] | None]] = field(default_factory=list)

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> httpx.Response:
        self.calls.append((method, url, headers))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class Clock:  # mutable-ok: records sleeps for assertions
    slept: list[float] = field(default_factory=list)

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)


def _response(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", "https://eu.test/x"))


def _dataplane(token: str | None = "sk-svc") -> Dataplane:
    chain = CredentialChain(() if token is None else (ServiceTokenImpersonation(token),))
    return Dataplane(id="eu", name="eu", url="https://eu.test", credential=chain, credential_kinds=chain.kinds)


def _client(transport: FakeTransport, clock: Clock) -> DataplaneClient:
    return DataplaneClient(
        transport,
        timeout_seconds=1.0,
        retry=RetryPolicy(attempts=3, base_delay_seconds=0.1),
        sleep=clock.sleep,
        jitter=lambda: 1.0,
    )


@pytest.mark.asyncio
async def test_get_injects_the_credential_and_parses_json() -> None:
    transport, clock = FakeTransport(iter([_response(200, {"ok": True})])), Clock()
    reply = await _client(transport, clock).get(_dataplane(), "/user/info", caller("session"))
    assert reply == DataplaneReply(200, {"ok": True}, "service_token")
    method, url, headers = transport.calls[0]
    assert (method, url) == ("GET", "https://eu.test/user/info")
    assert (
        headers is not None
        and headers["Authorization"] == "Bearer sk-svc"
        and headers["X-LiteLLM-Impersonate-User"] == "alice"
    )
    assert clock.slept == []


@pytest.mark.asyncio
async def test_get_retries_transient_failures_with_capped_exponential_backoff() -> None:
    transport, clock = FakeTransport(iter([httpx.ConnectError("down"), _response(503), _response(200, "up")])), Clock()
    reply = await _client(transport, clock).get(_dataplane(), "/health/liveliness", caller("session"))
    assert reply == DataplaneReply(200, "up", "service_token")
    assert clock.slept == [pytest.approx(0.1), pytest.approx(0.2)]


@pytest.mark.asyncio
async def test_get_gives_up_after_the_configured_attempts() -> None:
    transport, clock = FakeTransport(iter([httpx.ConnectTimeout("t")] * 3)), Clock()
    result = await _client(transport, clock).get(_dataplane(), "/x", caller("session"))
    assert result == DataplaneUnreachable("ConnectTimeout: t", attempts=3)
    assert len(transport.calls) == 3


@pytest.mark.asyncio
async def test_post_is_never_retried_and_client_errors_are_replies_not_failures() -> None:
    transport, clock = FakeTransport(iter([_response(502), _response(200)])), Clock()
    result = await _client(transport, clock).post(_dataplane(), "/key/generate", caller("session"), json={"a": 1})
    assert result == DataplaneReply(502, "", "service_token")
    assert len(transport.calls) == 1 and clock.slept == []
    forbidden = FakeTransport(iter([_response(403, {"error": "no"})]))
    denied = await _client(forbidden, clock).get(_dataplane(), "/x", caller("session"))
    assert denied == DataplaneReply(403, {"error": "no"}, "service_token")
    assert len(forbidden.calls) == 1, "a 4xx is the dataplane's answer, not a reason to retry"


@pytest.mark.asyncio
async def test_missing_credential_short_circuits_before_any_request() -> None:
    transport, clock = FakeTransport(iter([])), Clock()
    result = await _client(transport, clock).get(_dataplane(token=None), "/x", caller("session"))
    assert isinstance(result, CredentialUnavailable) and transport.calls == []


def test_retry_delay_is_jittered_within_half_to_full_of_the_capped_backoff() -> None:
    policy = RetryPolicy(attempts=5, base_delay_seconds=0.5, max_delay_seconds=2.0)
    assert policy.delay(0, jitter=0.0) == pytest.approx(0.25) and policy.delay(0, jitter=1.0) == pytest.approx(0.5)
    assert policy.delay(10, jitter=1.0) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_unreachable_detail_never_echoes_the_request_url() -> None:
    leaky = httpx.ConnectError("failed for https://eu.test/key/list?token=sk-secret")
    transport, clock = FakeTransport(iter([leaky])), Clock()
    result = await _client(transport, clock).post(_dataplane(), "/key/list", caller("session"), json={})
    assert isinstance(result, DataplaneUnreachable)
    assert result.detail == "ConnectError" and "sk-secret" not in result.detail
