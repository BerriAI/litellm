from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.proxy.agent_endpoints.kill_switch import (
    build_kill_switch_request,
    fire_kill_switch,
    redact_kill_switch,
    restore_kill_switch,
)
from litellm.types.agents import AgentKillSwitchConfig


@dataclass(frozen=True, slots=True)
class _SentRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    json: Mapping[str, object] | None
    timeout: float


class _RecordingClient:
    def __init__(self, respond: httpx.Response | httpx.HTTPError) -> None:
        self.sent: list[_SentRequest] = []  # mutable-ok: test double records calls
        self.follow_redirects: list[bool] = []  # mutable-ok: test double records calls
        self._respond: Final = respond

    def build_request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object] | None,
        timeout: float,
    ) -> httpx.Request:
        self.sent.append(_SentRequest(method, url, headers, json, timeout))
        return httpx.Request(method, url, headers=dict(headers), json=json)

    async def send(self, request: httpx.Request, *, stream: bool, follow_redirects: bool) -> httpx.Response:
        self.follow_redirects.append(follow_redirects)
        if isinstance(self._respond, httpx.HTTPError):
            raise self._respond
        return self._respond


class _CountingStream(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes, chunks: int) -> None:
        self.pulled: int = 0  # rebind-ok: test double counts reads
        self._chunk: Final = chunk
        self._chunks: Final = chunks

    async def __aiter__(self):
        for _ in range(self._chunks):
            self.pulled += 1  # rebind-ok: test double counts reads
            yield self._chunk


def _config(**overrides: object) -> AgentKillSwitchConfig:
    return AgentKillSwitchConfig.model_validate({"url": "https://ops.example.com/agents/kill", **overrides})


def test_request_carries_endpoint_method_query_params_headers_and_body() -> None:
    request: Final = build_kill_switch_request(
        _config(
            url="https://ops.example.com/kill?env=prod",
            method="PUT",
            query_params={"agent": "billing-bot", "reason": "manual stop"},
            headers={"X-Trace": "abc"},
            body={"action": "stop", "hard": True},
        )
    )

    assert request.method == "PUT"
    assert str(httpx.URL(request.url)) == "https://ops.example.com/kill?env=prod&agent=billing-bot&reason=manual+stop"
    assert dict(request.headers) == {"X-Trace": "abc"}
    assert request.json_body == {"action": "stop", "hard": True}


def test_request_defaults_to_post_with_no_body_and_untouched_url() -> None:
    request: Final = build_kill_switch_request(_config())

    assert (request.method, request.url, dict(request.headers), request.json_body) == (
        "POST",
        "https://ops.example.com/agents/kill",
        {},
        None,
    )


@pytest.mark.parametrize(
    ("auth", "expected_headers"),
    [
        ({"type": "bearer", "token": "tok-123"}, {"Authorization": "Bearer tok-123"}),
        ({"type": "api_key", "api_key": "k-456"}, {"x-api-key": "k-456"}),
        ({"type": "api_key", "header_name": "X-Ops-Key", "api_key": "k-456"}, {"X-Ops-Key": "k-456"}),
        (
            {"type": "basic", "username": "ops", "password": "pw:1"},
            {"Authorization": f"Basic {b64encode(b'ops:pw:1').decode()}"},
        ),
    ],
)
def test_auth_becomes_the_matching_request_header(auth: Mapping[str, object], expected_headers: dict[str, str]) -> None:
    request: Final = build_kill_switch_request(_config(auth=auth))

    assert dict(request.headers) == expected_headers


def test_auth_header_wins_over_a_conflicting_custom_header() -> None:
    request: Final = build_kill_switch_request(
        _config(headers={"Authorization": "stale", "X-Env": "prod"}, auth={"type": "bearer", "token": "fresh"})
    )

    assert dict(request.headers) == {"Authorization": "Bearer fresh", "X-Env": "prod"}


@pytest.mark.parametrize("url", ["ftp://ops.example.com/kill", "/relative/kill", "ops.example.com/kill", ""])
def test_config_rejects_non_http_urls(url: str) -> None:
    with pytest.raises(ValidationError, match="absolute http"):
        _config(url=url)


def test_config_rejects_unknown_auth_type_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _config(auth={"type": "hmac", "secret": "x"})
    with pytest.raises(ValidationError):
        _config(endpoint="https://typo.example.com")


@pytest.mark.parametrize(
    ("auth", "secret_field"),
    [
        ({"type": "bearer", "token": "tok-123"}, "token"),
        ({"type": "api_key", "header_name": "X-K", "api_key": "k-456"}, "api_key"),
        ({"type": "basic", "username": "ops", "password": "pw"}, "password"),
    ],
)
def test_redact_replaces_only_the_secret_and_restore_puts_it_back(auth: dict[str, str], secret_field: str) -> None:
    original: Final = _config(auth=auth)

    redacted: Final = redact_kill_switch(original)
    assert redacted is not None and redacted.auth is not None
    assert redacted.auth.model_dump() == {**auth, secret_field: REDACTED_BY_LITELM_STRING}
    assert original.auth is not None and original.auth.model_dump() == auth, "redact must not mutate its input"

    restored: Final = restore_kill_switch(redacted, original)
    assert restored == original


def test_restore_keeps_a_rotated_secret_and_never_stores_the_marker_itself() -> None:
    rotated: Final = _config(auth={"type": "bearer", "token": "new-token"})
    stored: Final = _config(auth={"type": "bearer", "token": "old-token"})
    assert restore_kill_switch(rotated, stored) == rotated
    assert restore_kill_switch(None, stored) is None

    marker_only: Final = _config(auth={"type": "bearer", "token": REDACTED_BY_LITELM_STRING})
    assert restore_kill_switch(marker_only, None) == _config(auth={"type": "bearer", "token": ""})


def test_restore_does_not_borrow_a_secret_from_a_different_auth_type() -> None:
    incoming: Final = _config(auth={"type": "bearer", "token": REDACTED_BY_LITELM_STRING})
    stored: Final = _config(auth={"type": "api_key", "api_key": "k-456"})

    assert restore_kill_switch(incoming, stored) == _config(auth={"type": "bearer", "token": ""})


def test_redact_passes_through_configs_without_auth() -> None:
    assert redact_kill_switch(None) is None
    plain: Final = _config(headers={"X-Env": "prod"})
    assert redact_kill_switch(plain) is plain


@pytest.mark.asyncio
async def test_fire_sends_exactly_the_built_request_and_reports_the_2xx_reply_without_the_query() -> None:
    client: Final = _RecordingClient(httpx.Response(202, text="stopping"))
    config: Final = _config(
        method="DELETE",
        query_params={"force": "1", "token": "qs-secret"},
        headers={"X-Env": "prod"},
        body={"agent": "billing-bot"},
        auth={"type": "bearer", "token": "tok-123"},
    )

    result: Final = await fire_kill_switch(agent_id="agent-1", config=config, http_client=client, timeout=3.5)

    assert client.sent == [
        _SentRequest(
            method="DELETE",
            url="https://ops.example.com/agents/kill?force=1&token=qs-secret",
            headers={"X-Env": "prod", "Authorization": "Bearer tok-123"},
            json={"agent": "billing-bot"},
            timeout=3.5,
        )
    ]
    assert client.follow_redirects == [False], "a redirecting webhook must not be followed to another host"
    assert result.succeeded is True
    assert result.model_dump() == {
        "agent_id": "agent-1",
        "url": "https://ops.example.com/agents/kill",
        "method": "DELETE",
        "status_code": 202,
        "response_body": "stopping",
        "error": None,
    }


@pytest.mark.asyncio
async def test_fire_reports_a_non_2xx_reply_as_failure_with_the_body() -> None:
    client: Final = _RecordingClient(httpx.Response(503, text="x" * 5000))

    result: Final = await fire_kill_switch(agent_id="agent-1", config=_config(), http_client=client)

    assert result.succeeded is False
    assert result.status_code == 503
    assert result.response_body == "x" * 2000
    assert result.error is None


@pytest.mark.asyncio
async def test_fire_stops_reading_the_body_at_the_cap_instead_of_buffering_the_whole_reply() -> None:
    stream: Final = _CountingStream(b"y" * 500, chunks=100)
    client: Final = _RecordingClient(httpx.Response(200, stream=stream))

    result: Final = await fire_kill_switch(agent_id="agent-1", config=_config(), http_client=client)

    assert result.response_body == "y" * 2000
    assert stream.pulled == 4, f"read {stream.pulled} of 100 chunks for a 2000 char cap"


@pytest.mark.asyncio
async def test_fire_reports_a_transport_error_by_type_without_raising_or_echoing_the_url() -> None:
    client: Final = _RecordingClient(httpx.ConnectError("boom https://ops.example.com/agents/kill?token=qs-secret"))

    result: Final = await fire_kill_switch(
        agent_id="agent-1", config=_config(query_params={"token": "qs-secret"}), http_client=client
    )

    assert result.succeeded is False
    assert (result.status_code, result.response_body) == (None, None)
    assert result.error == "ConnectError"
    assert "qs-secret" not in result.model_dump_json()
