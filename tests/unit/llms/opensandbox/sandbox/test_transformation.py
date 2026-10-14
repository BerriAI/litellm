import json
from collections.abc import Sequence

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.opensandbox.sandbox.transformation import OpenSandboxSandboxConfig

API_BASE = "https://sandbox.test/v1"
PENDING = {"id": "osb_1", "status": {"state": "Pending"}}
RUNNING = {"id": "osb_1", "status": {"state": "Running"}}
EXECD_ENDPOINT = {"endpoint": "execd.local:44772", "headers": {"X-EXECD-ACCESS-TOKEN": "execd-token"}}


def _sandbox_api(created: object, polled: Sequence[object] = ()) -> tuple[AsyncHTTPHandler, list[str]]:
    remaining = list(polled)
    requested: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requested.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            return httpx.Response(200, content=json.dumps(created).encode())
        if "/endpoints/" in request.url.path:
            return httpx.Response(200, content=json.dumps(EXECD_ENDPOINT).encode())
        return httpx.Response(200, content=json.dumps(remaining.pop(0)).encode())

    return AsyncHTTPHandler(transport=httpx.MockTransport(respond)), requested


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("created", "expected_id"),
    [
        pytest.param(
            {**RUNNING, "createdAt": "2026-01-01T00:00:00Z", "metadata": {"team": ["a"]}}, "osb_1", id="text-id"
        ),
        pytest.param({"id": 123, "status": {"state": "Running"}}, "123", id="numeric-id"),
    ],
)
async def test_acreate_sandbox_builds_the_handle_from_a_running_create_response(created: object, expected_id: str):
    client, requested = _sandbox_api(created)

    handle = await OpenSandboxSandboxConfig().acreate_sandbox(api_key="osb_key", api_base=API_BASE, client=client)

    assert (handle.id, handle.provider, handle.domain) == (expected_id, "opensandbox", API_BASE)
    assert type(handle.id) is str
    assert handle._hidden_params == {
        "api_base": API_BASE,
        "api_key": "osb_key",
        "execd_endpoint": "execd.local:44772",
        "execd_headers": {"X-EXECD-ACCESS-TOKEN": "execd-token"},
        "use_server_proxy": False,
    }
    assert requested == ["POST /v1/sandboxes", f"GET /v1/sandboxes/{expected_id}/endpoints/44772"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "created",
    [
        pytest.param(PENDING, id="pending"),
        pytest.param({"id": "osb_1"}, id="status-missing"),
        pytest.param({"id": "osb_1", "status": None}, id="status-null"),
        pytest.param({"id": "osb_1", "status": "Running"}, id="status-text"),
        pytest.param({"id": "osb_1", "status": {"state": None}}, id="state-null"),
    ],
)
async def test_acreate_sandbox_polls_when_the_create_response_does_not_report_running(created: object):
    client, requested = _sandbox_api(created, polled=[RUNNING])

    handle = await OpenSandboxSandboxConfig().acreate_sandbox(api_key="osb_key", api_base=API_BASE, client=client)

    assert handle.id == "osb_1"
    assert requested == ["POST /v1/sandboxes", "GET /v1/sandboxes/osb_1", "GET /v1/sandboxes/osb_1/endpoints/44772"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "not_ready",
    [
        pytest.param(PENDING, id="pending"),
        pytest.param([], id="list"),
        pytest.param("Running", id="text"),
        pytest.param(None, id="null"),
        pytest.param(7, id="number"),
        pytest.param({}, id="empty-object"),
        pytest.param({"status": ["Running"]}, id="status-list"),
        pytest.param({"status": {"state": 5}}, id="state-number"),
    ],
)
async def test_acreate_sandbox_keeps_polling_past_a_status_response_without_a_running_state(not_ready: object):
    client, requested = _sandbox_api(PENDING, polled=[not_ready, RUNNING])

    handle = await OpenSandboxSandboxConfig().acreate_sandbox(
        api_key="osb_key", api_base=API_BASE, poll_interval=0, client=client
    )

    assert handle.id == "osb_1"
    assert requested == [
        "POST /v1/sandboxes",
        "GET /v1/sandboxes/osb_1",
        "GET /v1/sandboxes/osb_1",
        "GET /v1/sandboxes/osb_1/endpoints/44772",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["Failed", "Stopping", "Terminated"])
async def test_acreate_sandbox_raises_when_the_polled_sandbox_reaches_a_terminal_state(state: str):
    client, requested = _sandbox_api(PENDING, polled=[[], {"status": {"state": state}}])

    with pytest.raises(ValueError, match=f"OpenSandbox sandbox osb_1 entered {state}"):
        await OpenSandboxSandboxConfig().acreate_sandbox(
            api_key="osb_key", api_base=API_BASE, poll_interval=0, client=client
        )

    assert requested == ["POST /v1/sandboxes", "GET /v1/sandboxes/osb_1", "GET /v1/sandboxes/osb_1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("created", [["secret-token"], "secret-token", 7, None])
async def test_acreate_sandbox_rejects_a_create_response_that_is_not_an_object_without_echoing_it(created: object):
    client, requested = _sandbox_api(created)

    with pytest.raises(ValidationError) as exc_info:
        await OpenSandboxSandboxConfig().acreate_sandbox(api_key="osb_key", api_base=API_BASE, client=client)

    assert [error["type"] for error in exc_info.value.errors()] == ["dict_type"]
    assert "secret-token" not in str(exc_info.value)
    assert requested == ["POST /v1/sandboxes"]


@pytest.mark.asyncio
async def test_public_acreate_sandbox_raises_the_validation_error_for_a_create_response_that_is_not_an_object():
    client, requested = _sandbox_api([RUNNING])

    with pytest.raises(ValidationError):
        await litellm.acreate_sandbox(provider="opensandbox", api_key="osb_key", api_base=API_BASE, client=client)

    assert requested == ["POST /v1/sandboxes"]


@pytest.mark.asyncio
async def test_acreate_sandbox_requires_a_sandbox_id():
    client, requested = _sandbox_api({"status": {"state": "Running"}})

    with pytest.raises(KeyError, match="id"):
        await OpenSandboxSandboxConfig().acreate_sandbox(api_key="osb_key", api_base=API_BASE, client=client)

    assert requested == ["POST /v1/sandboxes"]
