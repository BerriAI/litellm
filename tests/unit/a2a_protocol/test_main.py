"""Tests for litellm/a2a_protocol/main.py non-streaming send behavior."""

import asyncio
import contextlib
import datetime
import json
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator, Mapping
from typing import Final, TypeAlias

import httpx
import pytest
import respx
from pydantic import BaseModel, ConfigDict

pytest.importorskip("a2a.compat.v0_3.conversions")

from a2a.compat.v0_3 import conversions as _conv
from a2a.compat.v0_3.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.a2a_protocol.main import (
    _send_message,
    _stream_messages,
    aget_agent_card,
    asend_message,
    create_a2a_client,
)
from litellm.a2a_protocol.exceptions import A2AError
from litellm.a2a_protocol.utils import A2ARequestUtils
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.constants import DEFAULT_A2A_AGENT_TIMEOUT
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from types import SimpleNamespace
from uuid import uuid4


def _request() -> SendMessageRequest:
    params = MessageSendParams(
        message={
            "messageId": "m1",
            "role": "user",
            "parts": [{"kind": "text", "text": "hi"}],
        }
    )
    return SendMessageRequest(id="r1", params=params)


def _message_stream_response():
    sr = _conv.pb2_v10.StreamResponse()
    sr.message.message_id = "reply-1"
    sr.message.role = _conv.pb2_v10.Role.ROLE_AGENT
    sr.message.parts.add().text = "hello back"
    return sr


def _status_update_stream_response():
    sr = _conv.pb2_v10.StreamResponse()
    sr.status_update.task_id = "t1"
    sr.status_update.context_id = "c1"
    return sr


class _FakeClient:
    def __init__(self, *events):
        self._events = events

    async def send_message(self, _pb_request, context=None):
        self.context = context
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_send_message_returns_message_result():
    response = await _send_message(_FakeClient(_message_stream_response()), _request())
    result = response.root.result
    assert type(result).__name__ == "Message"
    assert response.root.id == "r1"


@pytest.mark.asyncio
async def test_send_message_rejects_update_event_final_with_runtime_error():
    with pytest.raises(RuntimeError, match="Message or Task"):
        await _send_message(_FakeClient(_status_update_stream_response()), _request())


@pytest.mark.asyncio
async def test_streaming_trace_id_prefers_logging_trace_id():
    """The streaming X-LiteLLM-Trace-Id must use the logging object's trace id (same
    as the non-streaming path), not the JSON-RPC request id, so traces correlate."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from a2a.compat.v0_3.types import (
        MessageSendParams,
        SendStreamingMessageRequest,
    )

    from litellm.a2a_protocol import main as a2a_main
    from litellm.litellm_core_utils.litellm_logging import Logging

    request = SendStreamingMessageRequest(
        id="rpc-1",
        params=MessageSendParams(
            message={
                "messageId": "m1",
                "role": "user",
                "parts": [{"kind": "text", "text": "hi"}],
            }
        ),
    )
    logging_obj = MagicMock(spec=Logging)
    logging_obj.litellm_trace_id = "trace-from-logging"

    captured: dict = {}

    async def _capture(*, base_url, extra_headers=None, streaming=False, **_):
        captured["extra_headers"] = extra_headers
        raise RuntimeError("stop")

    with patch.object(a2a_main, "create_a2a_client", new=AsyncMock(side_effect=_capture)):
        with pytest.raises(RuntimeError, match="stop"):
            async for _ in a2a_main.asend_message_streaming(
                request=request,
                api_base="http://upstream.local",
                litellm_logging_obj=logging_obj,
            ):
                pass

    assert captured["extra_headers"]["X-LiteLLM-Trace-Id"] == "trace-from-logging"


def test_streaming_logging_obj_carries_call_type_into_model_call_details():
    """The streaming logging object is built by hand rather than through
    ``update_environment_variables``, which is the only place ``call_type`` normally
    reaches ``model_call_details``. Callbacks read the call type from there, so
    without this the streamed turn arrives at every logger with no call type at all
    and OTel's GenAI metrics label it ``chat`` instead of ``invoke_agent``."""
    from a2a.compat.v0_3.types import MessageSendParams, SendStreamingMessageRequest

    from litellm.a2a_protocol.main import _build_streaming_logging_obj

    request = SendStreamingMessageRequest(
        id="rpc-call-type",
        params=MessageSendParams(
            message={"messageId": "m1", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}
        ),
    )

    logging_obj = _build_streaming_logging_obj(
        request=request,
        agent_name="some-agent",
        agent_id=None,
        litellm_params=None,
        metadata=None,
        proxy_server_request=None,
        start_time=datetime.datetime(2026, 10, 20, 12, 0, 0),
    )

    assert logging_obj.model_call_details["call_type"] == "asend_message_streaming"


_AGENT_CARD = {
    "protocolVersion": "0.3.0",
    "name": "recording-agent",
    "url": "http://127.0.0.1:9/",
    "preferredTransport": "JSONRPC",
    "version": "1.0.0",
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [],
}

_RPC_REPLY = {
    "jsonrpc": "2.0",
    "id": "reply",
    "result": {
        "messageId": "reply-1",
        "role": "agent",
        "parts": [{"kind": "text", "text": "pong"}],
        "kind": "message",
    },
}

_AGENT_A_HEADERS = {"x-agent-token": "token-for-a", "x-tenant": "tenant-a"}
_AGENT_B_HEADERS = {"x-agent-token": "token-for-b", "x-tenant": "tenant-b"}


_LANGGRAPH_TASK_REPLY = {
    "jsonrpc": "2.0",
    "id": "reply",
    "result": {
        "kind": "task",
        "id": "run-1:task-1",
        "contextId": "thread-1",
        "history": [
            {
                "kind": "message",
                "role": "user",
                "parts": [{"kind": "text", "text": "hi"}],
                "messageId": "m-user",
                "taskId": "run-1:task-1",
                "contextId": "thread-1",
            },
            {
                "kind": "message",
                "role": "agent",
                "parts": [{"kind": "text", "text": "langgraph echo: hi"}],
                "messageId": "m-agent",
                "taskId": "run-1:task-1",
                "contextId": "thread-1",
            },
        ],
        "status": {"state": "completed", "timestamp": "2026-08-24T00:00:00+00:00"},
        "artifacts": [
            {
                "artifactId": "art-1",
                "name": "Assistant Response",
                "parts": [{"kind": "text", "text": "langgraph echo: hi"}],
            }
        ],
    },
}


_LANGGRAPH_STREAM_EVENTS = [
    {
        "jsonrpc": "2.0",
        "id": "reply",
        "result": {
            "kind": "task",
            "id": "run-1:task-1",
            "contextId": "thread-1",
            "history": [
                {
                    "kind": "message",
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hi"}],
                    "messageId": "m-user",
                    "taskId": "run-1:task-1",
                    "contextId": "thread-1",
                }
            ],
            "status": {"state": "submitted"},
        },
    },
    {
        "jsonrpc": "2.0",
        "id": "reply",
        "result": {
            "kind": "status-update",
            "taskId": "run-1:task-1",
            "contextId": "thread-1",
            "status": {
                "state": "working",
                "message": {
                    "kind": "message",
                    "role": "agent",
                    "parts": [{"kind": "text", "text": "langgraph echo: hi"}],
                    "messageId": "m-agent",
                    "taskId": "run-1:task-1",
                    "contextId": "thread-1",
                },
            },
            "final": False,
        },
    },
    {
        "jsonrpc": "2.0",
        "id": "reply",
        "result": {
            "kind": "artifact-update",
            "taskId": "run-1:task-1",
            "contextId": "thread-1",
            "artifact": {
                "artifactId": "art-1",
                "name": "Assistant Response",
                "parts": [{"kind": "text", "text": "langgraph echo: hi"}],
            },
            "lastChunk": True,
        },
    },
    {
        "jsonrpc": "2.0",
        "id": "reply",
        "result": {
            "kind": "status-update",
            "taskId": "run-1:task-1",
            "contextId": "thread-1",
            "status": {"state": "completed"},
            "final": True,
        },
    },
]


_LOWERCASE_BINDING_CARD = {
    "name": "langgraph-agent",
    "version": "1.0.0",
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [],
    "supportedInterfaces": [{"url": "http://127.0.0.1:9/", "protocolBinding": "jsonrpc", "protocolVersion": "1.0"}],
}

_UPPERCASE_BINDING_CARD = {
    "name": "langgraph-agent",
    "version": "1.0.0",
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [],
    "supportedInterfaces": [{"url": "http://127.0.0.1:9/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
}


class _RequestRecorder:
    """Records the headers httpx put on the wire, per outbound request."""

    def __init__(self, card=_AGENT_CARD, rpc_reply=_RPC_REPLY, rpc_stream_events=None):
        self.card = card
        self.rpc_reply = rpc_reply
        self.rpc_stream_events = rpc_stream_events
        self.card_requests = []
        self.card_urls = []
        self.rpc_requests = []
        self.rpc_bodies = []
        self.client = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        if request.method == "GET":
            self.card_requests.append(headers)
            self.card_urls.append(str(request.url))
            return httpx.Response(200, json=self.card)
        self.rpc_requests.append(headers)
        self.rpc_bodies.append(json.loads(request.content))
        if self.rpc_stream_events is not None:
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in self.rpc_stream_events)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        return httpx.Response(200, json=self.rpc_reply)


def _a2a_client_cache_key(timeout: float, provider: str = httpxSpecialProvider.A2AProvider) -> str:
    return "async_httpx_client" + f"timeout_{timeout}" + provider


async def _seed_shared_a2a_client(
    card=_AGENT_CARD,
    rpc_reply=_RPC_REPLY,
    rpc_stream_events=None,
    provider: str = httpxSpecialProvider.A2AProvider,
    recorder: _RequestRecorder | None = None,
) -> _RequestRecorder:
    """Put the one A2A client the cache will hand out behind a mock transport.

    Seeding has to happen on the test's own event loop, because the client cache keys on
    it. The injected client is a real httpx.AsyncClient, so the merge of per-request
    headers over client defaults, which is what these tests are about, stays real.
    """
    active_recorder: Final = recorder or _RequestRecorder(
        card=card, rpc_reply=rpc_reply, rpc_stream_events=rpc_stream_events
    )
    handler = AsyncHTTPHandler(timeout=DEFAULT_A2A_AGENT_TIMEOUT)
    owned_client = handler.client
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(active_recorder))
    await owned_client.aclose()

    litellm.in_memory_llm_clients_cache.set_cache(
        key=_a2a_client_cache_key(DEFAULT_A2A_AGENT_TIMEOUT, provider), value=handler
    )
    seeded = get_async_httpx_client(
        llm_provider=provider,
        params={"timeout": DEFAULT_A2A_AGENT_TIMEOUT},
    )
    assert seeded is handler, "cache key drifted from get_async_httpx_client; these tests would test nothing"

    active_recorder.client = handler.client
    return active_recorder


@pytest.fixture
def isolated_client_cache(monkeypatch):
    cache = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", cache)
    return cache


def _send_request(request_id):
    return SendMessageRequest(
        id=request_id,
        params=MessageSendParams(
            message={"messageId": request_id, "role": "user", "parts": [{"kind": "text", "text": "hi"}]}
        ),
    )


@pytest.mark.asyncio
async def test_extra_headers_never_land_on_the_shared_cached_client(isolated_client_cache):
    """get_async_httpx_client hands back a process-wide shared client, so a caller's
    headers written onto it would outlive the request that supplied them."""
    recorder = await _seed_shared_a2a_client()

    await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=_AGENT_A_HEADERS)

    assert "x-agent-token" not in recorder.client.headers
    assert "x-tenant" not in recorder.client.headers


@pytest.mark.asyncio
async def test_callers_with_different_headers_reuse_one_pooled_client(isolated_client_cache):
    """Headers must not segregate the connection pool. Every A2A caller on one timeout
    shares one cached client, so header sets cannot multiply cached clients."""
    recorder = await _seed_shared_a2a_client()

    client_a = await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=_AGENT_A_HEADERS)
    client_b = await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=_AGENT_B_HEADERS)
    client_none = await create_a2a_client(base_url="http://127.0.0.1:9")

    assert client_a._litellm_httpx_client is recorder.client
    assert client_b._litellm_httpx_client is recorder.client
    assert client_none._litellm_httpx_client is recorder.client

    cached = [key for key in isolated_client_cache.cache_dict if "a2a_provider" in key]
    assert len(cached) == 1, f"expected one pooled A2A client, cached: {cached}"


@pytest.mark.parametrize("order", [("a", "b", "none"), ("b", "none", "a"), ("none", "a", "b")])
@pytest.mark.asyncio
async def test_each_caller_sends_only_its_own_headers(order, isolated_client_cache):
    """Whatever order callers arrive in, each request carries that caller's headers and
    no other caller's, and a caller with no extra_headers sends none."""
    recorder = await _seed_shared_a2a_client()
    headers_by_caller = {"a": _AGENT_A_HEADERS, "b": _AGENT_B_HEADERS, "none": None}

    for caller in order:
        a2a_client = await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=headers_by_caller[caller])
        await _send_message(a2a_client, _send_request(caller))

    received = dict(zip(order, recorder.rpc_requests, strict=True))
    assert received["a"]["x-agent-token"] == "token-for-a"
    assert received["a"]["x-tenant"] == "tenant-a"
    assert received["b"]["x-agent-token"] == "token-for-b"
    assert received["b"]["x-tenant"] == "tenant-b"
    assert "x-agent-token" not in received["none"]
    assert "x-tenant" not in received["none"]


@pytest.mark.asyncio
async def test_streaming_send_carries_only_its_own_caller_headers(isolated_client_cache):
    """The streaming path shares the same pooled client, so it needs the same guard."""
    recorder = await _seed_shared_a2a_client()

    client_a = await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=_AGENT_A_HEADERS, streaming=True)
    await _send_message(client_a, _send_request("a"))

    client_b = await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=_AGENT_B_HEADERS, streaming=True)
    streaming_request = SendStreamingMessageRequest(
        id="b",
        params=MessageSendParams(message={"messageId": "b", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}),
    )
    async for _ in _stream_messages(client_b, streaming_request):
        pass

    received = dict(zip(("a", "b"), recorder.rpc_requests, strict=True))
    assert received["a"]["x-agent-token"] == "token-for-a"
    assert received["b"]["x-agent-token"] == "token-for-b"
    assert received["b"]["x-tenant"] == "tenant-b"


@pytest.mark.asyncio
async def test_lowercase_protocol_binding_card_round_trips_the_langgraph_dialect(isolated_client_cache):
    """LangGraph Platform serves cards with protocolBinding "jsonrpc" and answers in the
    A2A 0.3 JSON dialect ("kind"-discriminated) while declaring protocolVersion "1.0".
    Without binding normalization client creation raises ValueError("no compatible
    transports found."); without the version downgrade the SDK's strict v1 transport
    rejects the reply with 'Message type "lf.a2a.v1.Task" has no field named "kind"'."""
    await _seed_shared_a2a_client(card=_LOWERCASE_BINDING_CARD, rpc_reply=_LANGGRAPH_TASK_REPLY)

    a2a_client = await create_a2a_client(base_url="http://127.0.0.1:9")
    response = await _send_message(a2a_client, _send_request("lc"))

    assert type(response.root.result).__name__ == "Task"
    assert response.root.result.artifacts[0].parts[0].root.text == "langgraph echo: hi"
    interface = a2a_client._litellm_agent_card.supported_interfaces[0]
    assert interface.protocol_binding == "JSONRPC"
    assert interface.protocol_version == "0.3"


@pytest.mark.asyncio
async def test_uppercase_binding_card_without_override_fails_with_actionable_hint(isolated_client_cache):
    await _seed_shared_a2a_client(card=_UPPERCASE_BINDING_CARD, rpc_reply=_LANGGRAPH_TASK_REPLY)

    with pytest.raises(A2AError) as error:
        await asend_message(request=_send_request("uppercase-default"), api_base="http://127.0.0.1:9")

    assert 'has no field named "kind"' in str(error.value)
    assert "a2a_protocol_version" in str(error.value)


@pytest.mark.parametrize("card", [_LOWERCASE_BINDING_CARD, _UPPERCASE_BINDING_CARD])
@pytest.mark.asyncio
async def test_protocol_version_override_round_trips_the_langgraph_dialect(card, isolated_client_cache):
    recorder = await _seed_shared_a2a_client(card=card, rpc_reply=_LANGGRAPH_TASK_REPLY)

    a2a_client = await create_a2a_client(base_url="http://127.0.0.1:9", protocol_version="0.3")
    response = await _send_message(a2a_client, _send_request("override-round-trip"))

    assert type(response.root.result).__name__ == "Task"
    assert response.root.result.artifacts[0].parts[0].root.text == "langgraph echo: hi"
    assert a2a_client._litellm_agent_card.supported_interfaces[0].protocol_version == "0.3"
    assert recorder.rpc_bodies[-1]["method"] == "message/send"


@pytest.mark.parametrize("card", [_LOWERCASE_BINDING_CARD, _UPPERCASE_BINDING_CARD])
@pytest.mark.asyncio
async def test_protocol_version_override_streams_the_langgraph_dialect(card, isolated_client_cache):
    recorder = await _seed_shared_a2a_client(card=card, rpc_stream_events=_LANGGRAPH_STREAM_EVENTS)

    a2a_client = await create_a2a_client(
        base_url="http://127.0.0.1:9",
        streaming=True,
        protocol_version="0.3",
    )
    request = SendStreamingMessageRequest(
        id="reply",
        params=MessageSendParams(
            message={"messageId": "m-user", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}
        ),
    )
    streamed = [chunk async for chunk in _stream_messages(a2a_client, request)]

    artifact_event = next(chunk.root.result for chunk in streamed if chunk.root.result.kind == "artifact-update")
    completed_status = next(
        chunk.root.result for chunk in streamed if chunk.root.result.kind == "status-update" and chunk.root.result.final
    )
    assert artifact_event.artifact.parts[0].root.text == "langgraph echo: hi"
    assert completed_status.status.state.value == "completed"
    assert recorder.rpc_bodies[-1]["method"] == "message/stream"


@pytest.mark.asyncio
async def test_uppercase_binding_card_stream_without_override_fails(isolated_client_cache):
    await _seed_shared_a2a_client(card=_UPPERCASE_BINDING_CARD, rpc_stream_events=_LANGGRAPH_STREAM_EVENTS)

    a2a_client = await create_a2a_client(base_url="http://127.0.0.1:9", streaming=True)
    request = SendStreamingMessageRequest(
        id="reply",
        params=MessageSendParams(
            message={"messageId": "m-user", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}
        ),
    )

    with pytest.raises(Exception, match='has no field named "kind"'):
        async for _ in _stream_messages(a2a_client, request):
            pass


@pytest.mark.asyncio
async def test_asend_message_resolves_float_protocol_version_from_agent_params(isolated_client_cache):
    recorder = await _seed_shared_a2a_client(card=_UPPERCASE_BINDING_CARD, rpc_reply=_LANGGRAPH_TASK_REPLY)

    response = await asend_message(
        request=_send_request("float-protocol-override"),
        api_base="http://127.0.0.1:9",
        litellm_params={"a2a_protocol_version": 0.3},
    )

    assert response.result["artifacts"][0]["parts"][0]["text"] == "langgraph echo: hi"
    assert recorder.rpc_bodies[-1]["method"] == "message/send"


@pytest.mark.parametrize(
    ("litellm_params", "expected_protocol_version"),
    [({"a2a_protocol_version": "0.3"}, "0.3"), (None, None)],
)
@pytest.mark.asyncio
async def test_streaming_passes_agent_protocol_version_to_client(litellm_params, expected_protocol_version):
    from unittest.mock import AsyncMock, patch

    from litellm.a2a_protocol import main as a2a_main

    request = SendStreamingMessageRequest(
        id="stream-protocol-version",
        params=MessageSendParams(
            message={"messageId": "m1", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}
        ),
    )
    captured = {}

    async def _capture(*, base_url, extra_headers=None, streaming=False, protocol_version=None, **_):
        captured["protocol_version"] = protocol_version
        raise RuntimeError("stop")

    with (
        patch.object(a2a_main, "create_a2a_client", new=AsyncMock(side_effect=_capture)),
        pytest.raises(RuntimeError, match="stop"),
    ):
        async for _ in a2a_main.asend_message_streaming(
            request=request,
            api_base="http://upstream.local",
            litellm_params=litellm_params,
        ):
            pass

    assert captured["protocol_version"] == expected_protocol_version


@pytest.mark.asyncio
async def test_agent_card_fetch_carries_the_callers_headers(isolated_client_cache):
    """Agent cards can sit behind the same auth as the agent, so the card fetch must stay
    authenticated once the headers stop living on the client."""
    recorder = await _seed_shared_a2a_client()

    await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=_AGENT_A_HEADERS)

    assert recorder.card_requests, "no agent card request was made"
    assert recorder.card_requests[-1]["x-agent-token"] == "token-for-a"


@pytest.mark.asyncio
async def test_agent_card_path_param_fetches_that_path_with_the_agents_headers(isolated_client_cache):
    """A Microsoft Foundry agent serves its card only at agentCard/v1.0 behind the same Entra bearer
    as the agent, so an agent registered with agent_card_path fetches exactly that path, authenticated,
    instead of probing the well-known paths."""
    recorder = await _seed_shared_a2a_client()

    await asend_message(
        request=_send_request("req-foundry"),
        api_base="http://127.0.0.1:9",
        litellm_params={"agent_card_path": "agentCard/v1.0"},
        agent_extra_headers=_AGENT_A_HEADERS,
    )

    assert recorder.card_urls == ["http://127.0.0.1:9/agentCard/v1.0"]
    assert recorder.card_requests[-1]["x-agent-token"] == "token-for-a"


@pytest.mark.asyncio
async def test_aget_agent_card_carries_the_callers_headers_and_path(isolated_client_cache):
    recorder = await _seed_shared_a2a_client(provider=httpxSpecialProvider.A2A)

    await aget_agent_card(
        base_url="http://127.0.0.1:9", extra_headers=_AGENT_A_HEADERS, relative_card_path="agentCard/v1.0"
    )

    assert recorder.card_urls == ["http://127.0.0.1:9/agentCard/v1.0"]
    assert recorder.card_requests[-1]["x-agent-token"] == "token-for-a"


@pytest.mark.asyncio
async def test_the_pooled_a2a_client_arrives_with_cookie_persistence_disabled(isolated_client_cache):
    """create_a2a_client takes its client from the shared builder rather than building one,
    and the builder is what refuses to persist cookies. This pins the join between those
    two facts, so the A2A path cannot quietly start acquiring a client that keeps a jar.

    test_callers_with_different_headers_reuse_one_pooled_client pins the other half, that
    create_a2a_client hands back exactly this cached client."""
    handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2AProvider,
        params={"timeout": DEFAULT_A2A_AGENT_TIMEOUT},
    )
    request = httpx.Request("GET", "https://agent-a.example.com/")
    handler.client.cookies.extract_cookies(
        httpx.Response(200, headers={"set-cookie": "SESSION=only-agent-a-may-hold-this"}, request=request)
    )

    assert dict(handler.client.cookies) == {}, "the pooled A2A client kept an upstream's cookie"
    await handler.close()


class _UsageRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged = asyncio.Event()
        self.payload = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payload = kwargs["standard_logging_object"]
        self.logged.set()


class _CardServedAtRecorder(_RequestRecorder):
    def __init__(self) -> None:
        super().__init__(rpc_stream_events=_LANGGRAPH_STREAM_EVENTS)
        self.card_served_at: datetime.datetime | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and self.card_served_at is None:
            self.card_served_at = datetime.datetime.now()
        return super().__call__(request)


@pytest.mark.asyncio
async def test_streamed_send_is_timed_from_before_the_agent_card_fetch(
    isolated_client_cache: LLMClientCache, monkeypatch: pytest.MonkeyPatch
):
    from litellm.a2a_protocol import main as a2a_main

    card_recorder: Final = _CardServedAtRecorder()
    await _seed_shared_a2a_client(recorder=card_recorder)
    logger: Final = _UsageRecorder()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    monkeypatch.setattr(litellm, "success_callback", [logger])
    monkeypatch.setattr(litellm, "_async_success_callback", [logger])
    request: Final = SendStreamingMessageRequest(
        id="reply",
        params=MessageSendParams(
            message={"messageId": "m-user", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}
        ),
    )

    _ = [chunk async for chunk in a2a_main.asend_message_streaming(request=request, api_base="http://127.0.0.1:9")]
    await asyncio.wait_for(logger.logged.wait(), timeout=10)

    assert card_recorder.card_served_at is not None
    assert logger.payload["startTime"] <= card_recorder.card_served_at.timestamp()


@pytest.mark.asyncio
async def test_asend_message_counts_usage_off_the_event_loop(monkeypatch):
    from tests.large_text import text
    from tests.unit.litellm_core_utils.event_loop_lag import (
        assert_loop_stayed_free,
        timed_with_loop_lags,
        warm_tokenizer,
    )

    warm_tokenizer("gpt-5.6-luna")
    recorder = _UsageRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])

    reply = _conv.pb2_v10.StreamResponse()
    reply.message.message_id = "reply-1"
    reply.message.role = _conv.pb2_v10.Role.ROLE_AGENT
    reply.message.parts.add().text = text * 100
    request = SendMessageRequest(
        id="r1",
        params=MessageSendParams(
            message={"messageId": "m1", "role": "user", "parts": [{"kind": "text", "text": text * 100}]}
        ),
    )

    response, took, lags = await timed_with_loop_lags(
        lambda: asend_message(a2a_client=_FakeClient(reply), request=request)
    )

    assert response.id == "r1"
    await asyncio.wait_for(recorder.logged.wait(), timeout=10)
    assert recorder.payload["prompt_tokens"] > 100_000
    assert recorder.payload["completion_tokens"] > 100_000
    assert_loop_stayed_free(took, lags)


def test_streaming_logging_obj_keeps_agent_credentials_out_of_logging_params():
    """Callbacks receive the streaming logging object's litellm_params as raw kwargs, so an agent's
    Entra, Databricks, or static credentials must never be copied into it; only pricing keys are."""
    from litellm.a2a_protocol.main import _build_streaming_logging_obj

    request = SendStreamingMessageRequest(
        id="rpc-secrets",
        params=MessageSendParams(
            message={"messageId": "m1", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}
        ),
    )

    logging_obj = _build_streaming_logging_obj(
        request=request,
        agent_name="foundry-agent",
        agent_id="agent-1",
        litellm_params={
            "client_secret": "sp-secret",
            "azure_ad_token": "entra-token",
            "tenant_id": "tenant",
            "databricks_oauth": {"client_secret": "dbx-secret"},
            "api_key": "static-key",
            "cost_per_query": 0.25,
        },
        metadata={"user_api_key": "hashed"},
        proxy_server_request={"url": "http://localhost:4000"},
        start_time=datetime.datetime(2026, 10, 20, 12, 0, 0),
    )

    expected = {
        "cost_per_query": 0.25,
        "metadata": {"user_api_key": "hashed"},
        "proxy_server_request": {"url": "http://localhost:4000"},
    }
    assert logging_obj.litellm_params == expected
    assert logging_obj.optional_params == expected
    assert logging_obj.model_call_details["litellm_params"] == expected


class _LoggedSpend(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_type: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    response_cost: float


class _A2ASpendRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: Final = asyncio.Event()
        self.spend: _LoggedSpend | None = None
        self.agent_id: object = None

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        spend: Final = _LoggedSpend.model_validate(kwargs["standard_logging_object"])
        if spend.call_type not in ("asend_message", "asend_message_streaming"):
            return
        self.spend = spend
        self.agent_id = kwargs.get("agent_id")
        self.logged.set()

    async def logged_spend(self) -> _LoggedSpend:
        await asyncio.wait_for(self.logged.wait(), timeout=10)
        assert self.spend is not None
        return self.spend


@pytest.fixture
def a2a_spend_recorder(monkeypatch: pytest.MonkeyPatch) -> _A2ASpendRecorder:
    recorder: Final = _A2ASpendRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [recorder])
    monkeypatch.setattr(litellm, "_async_success_callback", [recorder])
    return recorder


@pytest.fixture
def respx_reachable_httpx(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


_AgentPrices: TypeAlias = Mapping[str, float]
_ExpectedCost: TypeAlias = Callable[[int, int], float]

_BRIDGE_PROMPT: Final = "What is the capital of France?"
_BRIDGE_REPLY: Final = "Paris is the capital of France."
_LONG_AGENT_REPLY: Final = (
    "Paris is the capital of France. It sits on the Seine, holds about two million people, "
    "and has been the seat of the French government for most of the last thousand years."
)
_PER_TOKEN_PRICES: Final[_AgentPrices] = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
_AGENTCORE_ARN: Final = "arn:aws:bedrock-agentcore:us-west-2:123456789:runtime/trip_planner"
_AGENTCORE_PARAMS: Final[Mapping[str, object]] = {
    "custom_llm_provider": "bedrock",
    "model": f"bedrock/agentcore/{_AGENTCORE_ARN}",
    "api_key": "test-jwt-token",
}
_JSON_RPC_ERROR: Final[Mapping[str, object]] = {
    "jsonrpc": "2.0",
    "id": "agentcore-call",
    "error": {"code": -32603, "message": "agent crashed"},
}


def _per_token_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * 0.001 + completion_tokens * 0.002


_BRIDGE_AGENT_PRICES: Final = (
    pytest.param({"cost_per_query": 0.25}, lambda prompt_tokens, completion_tokens: 0.25, id="cost_per_query"),
    pytest.param(_PER_TOKEN_PRICES, _per_token_cost, id="per_token"),
)


def _bridge_agent_params(prices: _AgentPrices) -> dict[str, object]:
    return {
        "custom_llm_provider": "langgraph",
        "model": "agent",
        "mock_response": _BRIDGE_REPLY,
        **prices,
    }


def _bridge_message_params() -> MessageSendParams:
    return MessageSendParams(
        message={
            "messageId": "m1",
            "role": "user",
            "parts": [{"kind": "text", "text": _BRIDGE_PROMPT}],
        }
    )


def _assert_billed_at_agent_prices(
    spend: _LoggedSpend, recorder: _A2ASpendRecorder, expected_cost: _ExpectedCost, reply: str
) -> None:
    assert spend.prompt_tokens == A2ARequestUtils.count_tokens(_BRIDGE_PROMPT)
    assert spend.completion_tokens == A2ARequestUtils.count_tokens(reply)
    assert spend.response_cost == pytest.approx(expected_cost(spend.prompt_tokens, spend.completion_tokens))
    assert recorder.agent_id == "agent-1"


async def _drain_bridge_stream(litellm_params: Mapping[str, object]) -> None:
    from litellm.a2a_protocol.main import asend_message_streaming

    _ = [
        chunk
        async for chunk in asend_message_streaming(
            request=SendStreamingMessageRequest(id="bridge-stream", params=_bridge_message_params()),
            api_base="http://127.0.0.1:9",
            litellm_params=dict(litellm_params),
            agent_id="agent-1",
            agent_name="trip-planner",
        )
    ]


def _sse(*events: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
    )


def _agentcore_appended_chunks(text: str, chunk_size: int) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "jsonrpc": "2.0",
            "id": "agentcore-call",
            "result": {
                "kind": "artifact-update",
                "taskId": "t1",
                "contextId": "c1",
                "append": start > 0,
                "artifact": {"artifactId": "a1", "parts": [{"kind": "text", "text": text[start : start + chunk_size]}]},
            },
        }
        for start in range(0, len(text), chunk_size)
    )


@pytest.mark.parametrize(("prices", "expected_cost"), _BRIDGE_AGENT_PRICES)
@pytest.mark.asyncio
async def test_completion_bridge_send_bills_the_agents_prices(
    prices: _AgentPrices, expected_cost: _ExpectedCost, a2a_spend_recorder: _A2ASpendRecorder
) -> None:
    await asend_message(
        request=SendMessageRequest(id="bridge-send", params=_bridge_message_params()),
        api_base="http://127.0.0.1:9",
        litellm_params=_bridge_agent_params(prices),
        agent_id="agent-1",
    )

    spend: Final = await a2a_spend_recorder.logged_spend()
    _assert_billed_at_agent_prices(spend, a2a_spend_recorder, expected_cost, _BRIDGE_REPLY)


@pytest.mark.parametrize(("prices", "expected_cost"), _BRIDGE_AGENT_PRICES)
@pytest.mark.asyncio
async def test_completion_bridge_stream_bills_the_agents_prices(
    prices: _AgentPrices, expected_cost: _ExpectedCost, a2a_spend_recorder: _A2ASpendRecorder
) -> None:
    await _drain_bridge_stream(_bridge_agent_params(prices))

    spend: Final = await a2a_spend_recorder.logged_spend()
    _assert_billed_at_agent_prices(spend, a2a_spend_recorder, expected_cost, _BRIDGE_REPLY)
    assert spend.model == "a2a_agent/trip-planner"


@pytest.mark.asyncio
async def test_agentcore_send_bills_a_task_reply_at_the_agents_token_prices(
    a2a_spend_recorder: _A2ASpendRecorder, respx_reachable_httpx: None
) -> None:
    task_reply: Final = {
        "jsonrpc": "2.0",
        "id": "bridge-send",
        "result": {
            "kind": "task",
            "id": "t1",
            "contextId": "c1",
            "status": {"state": "completed"},
            "artifacts": [{"artifactId": "a1", "parts": [{"kind": "text", "text": _LONG_AGENT_REPLY}]}],
        },
    }
    with respx.mock(assert_all_called=True) as router:
        router.post(url__regex=r".*/invocations.*").mock(return_value=httpx.Response(200, json=task_reply))
        await asend_message(
            request=SendMessageRequest(id="bridge-send", params=_bridge_message_params()),
            litellm_params={**_AGENTCORE_PARAMS, **_PER_TOKEN_PRICES},
            agent_id="agent-1",
        )

    spend: Final = await a2a_spend_recorder.logged_spend()
    _assert_billed_at_agent_prices(spend, a2a_spend_recorder, _per_token_cost, _LONG_AGENT_REPLY)


@pytest.mark.parametrize(
    "extra_fields",
    (pytest.param({}, id="result_only"), pytest.param({"error": None}, id="null_error_field")),
)
@pytest.mark.asyncio
async def test_agentcore_stream_bills_every_appended_reply_chunk(
    extra_fields: Mapping[str, object], a2a_spend_recorder: _A2ASpendRecorder, respx_reachable_httpx: None
) -> None:
    events: Final = tuple({**event, **extra_fields} for event in _agentcore_appended_chunks(_LONG_AGENT_REPLY, 40))
    with respx.mock(assert_all_called=True) as router:
        router.post(url__regex=r".*/invocations.*").mock(return_value=_sse(*events))
        await _drain_bridge_stream({**_AGENTCORE_PARAMS, **_PER_TOKEN_PRICES})

    spend: Final = await a2a_spend_recorder.logged_spend()
    _assert_billed_at_agent_prices(spend, a2a_spend_recorder, _per_token_cost, _LONG_AGENT_REPLY)


_PYDANTIC_COMPLETED_TASK: Final[Mapping[str, object]] = {
    "jsonrpc": "2.0",
    "id": "bridge-stream",
    "result": {
        "kind": "task",
        "id": "t1",
        "contextId": "c1",
        "status": {"state": "completed"},
        "artifacts": [{"artifactId": "a1", "parts": [{"kind": "text", "text": _LONG_AGENT_REPLY}]}],
    },
}


@pytest.mark.asyncio
async def test_pydantic_ai_stream_bills_the_whole_chunked_reply(
    a2a_spend_recorder: _A2ASpendRecorder, respx_reachable_httpx: None
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.post("http://127.0.0.1:9/").mock(return_value=httpx.Response(200, json=_PYDANTIC_COMPLETED_TASK))
        await _drain_bridge_stream({"custom_llm_provider": "pydantic_ai_agents", **_PER_TOKEN_PRICES})

    spend: Final = await a2a_spend_recorder.logged_spend()
    assert len(_LONG_AGENT_REPLY) > 50
    _assert_billed_at_agent_prices(spend, a2a_spend_recorder, _per_token_cost, _LONG_AGENT_REPLY)


async def _close_after_first_event(stream: AsyncGenerator[object, None]) -> None:
    _ = await stream.__anext__()
    await stream.aclose()


async def _cancel_reader_after_first_event(stream: AsyncGenerator[object, None]) -> None:
    first_event: Final = asyncio.Event()

    async def _read() -> None:
        async for _ in stream:
            first_event.set()

    reader: Final = asyncio.create_task(_read())
    await first_event.wait()
    reader.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await reader


def _pydantic_bridge_agent_stream() -> AsyncGenerator[object, None]:
    from litellm.a2a_protocol.main import asend_message_streaming

    return asend_message_streaming(
        request=SendStreamingMessageRequest(id="bridge-stream", params=_bridge_message_params()),
        api_base="http://127.0.0.1:9",
        litellm_params={"custom_llm_provider": "pydantic_ai_agents", "cost_per_query": 0.25},
        agent_id="agent-1",
        agent_name="trip-planner",
    )


def _url_agent_stream() -> AsyncGenerator[object, None]:
    from litellm.a2a_protocol.main import asend_message_streaming

    return asend_message_streaming(
        request=SendStreamingMessageRequest(id="reply", params=_bridge_message_params()),
        api_base="http://127.0.0.1:9",
        litellm_params={"cost_per_query": 0.25},
        agent_id="agent-1",
    )


@pytest.mark.parametrize(
    ("open_stream", "billed_model"),
    (
        pytest.param(_pydantic_bridge_agent_stream, "a2a_agent/trip-planner", id="bridge_agent"),
        pytest.param(_url_agent_stream, f"a2a_agent/{_AGENT_CARD['name']}", id="url_agent"),
    ),
)
@pytest.mark.parametrize(
    "drop_stream",
    (
        pytest.param(_close_after_first_event, id="proxy_closes_the_stream"),
        pytest.param(_cancel_reader_after_first_event, id="reader_cancelled_mid_stream"),
    ),
)
@pytest.mark.asyncio
async def test_stream_the_client_drops_still_bills_the_agents_price(
    open_stream: Callable[[], AsyncGenerator[object, None]],
    billed_model: str,
    drop_stream: Callable[[AsyncGenerator[object, None]], Coroutine[object, object, None]],
    a2a_spend_recorder: _A2ASpendRecorder,
    isolated_client_cache: LLMClientCache,
    respx_reachable_httpx: None,
) -> None:
    _ = await _seed_shared_a2a_client(rpc_stream_events=_LANGGRAPH_STREAM_EVENTS)
    with respx.mock(assert_all_called=False) as router:
        router.post("http://127.0.0.1:9/").mock(return_value=httpx.Response(200, json=_PYDANTIC_COMPLETED_TASK))
        await drop_stream(open_stream())

    spend: Final = await a2a_spend_recorder.logged_spend()
    assert spend.response_cost == pytest.approx(0.25)
    assert spend.model == billed_model


class _AgentThatNeverAnswers(_RequestRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.message_sent: Final = asyncio.Event()

    def __call__(self, request: httpx.Request) -> httpx.Response | Coroutine[object, object, httpx.Response]:
        if request.method == "GET":
            return super().__call__(request)
        return self._never_answer()

    async def _never_answer(self) -> httpx.Response:
        self.message_sent.set()
        reply: Final[asyncio.Future[httpx.Response]] = asyncio.get_running_loop().create_future()
        return await reply


@pytest.mark.asyncio
async def test_stream_dropped_before_the_agent_answers_bills_nothing(
    a2a_spend_recorder: _A2ASpendRecorder, isolated_client_cache: LLMClientCache
) -> None:
    agent: Final = _AgentThatNeverAnswers()
    _ = await _seed_shared_a2a_client(recorder=agent)
    stream: Final = _url_agent_stream()
    reader: Final = asyncio.create_task(anext(stream))
    await asyncio.wait_for(agent.message_sent.wait(), timeout=10)
    reader.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await reader

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(a2a_spend_recorder.logged.wait(), timeout=2)


@pytest.mark.asyncio
async def test_completion_bridge_stream_with_an_empty_reply_bills_no_reply_tokens(
    a2a_spend_recorder: _A2ASpendRecorder,
) -> None:
    await _drain_bridge_stream({**_bridge_agent_params(_PER_TOKEN_PRICES), "mock_response": ""})

    spend: Final = await a2a_spend_recorder.logged_spend()
    assert spend.completion_tokens == 0
    assert spend.response_cost == pytest.approx(_per_token_cost(spend.prompt_tokens, 0))


@pytest.mark.parametrize(
    "upstream_reply",
    (
        pytest.param(httpx.Response(200, json=_JSON_RPC_ERROR), id="send"),
        pytest.param(_sse(_JSON_RPC_ERROR), id="stream"),
    ),
)
@pytest.mark.asyncio
async def test_agentcore_error_reply_is_not_billed_at_the_agents_price(
    upstream_reply: httpx.Response, a2a_spend_recorder: _A2ASpendRecorder, respx_reachable_httpx: None
) -> None:
    agent_params: Final = {**_AGENTCORE_PARAMS, "cost_per_query": 0.25}
    with respx.mock(assert_all_called=True) as router:
        router.post(url__regex=r".*/invocations.*").mock(return_value=upstream_reply)
        if upstream_reply.headers["content-type"] == "text/event-stream":
            await _drain_bridge_stream(agent_params)
        else:
            response = await asend_message(
                request=SendMessageRequest(id="agentcore-call", params=_bridge_message_params()),
                litellm_params=dict(agent_params),
                agent_id="agent-1",
            )
            assert response.model_dump(mode="json", exclude_none=True)["error"]["message"] == "agent crashed"

    spend: Final = await a2a_spend_recorder.logged_spend()
    assert spend.response_cost == 0.0


class MockA2AResponse:
    def __init__(self, text: str):
        self._payload = {
            "id": str(uuid4()),
            "jsonrpc": "2.0",
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": text}],
                    "messageId": uuid4().hex,
                }
            },
        }

    def model_dump(self, mode="json", exclude_none=True):
        return self._payload

class MockA2AStreamingChunk(MockA2AResponse):
    def __init__(self, text: str, state: str):
        super().__init__(text=text)
        self._payload["result"]["status"] = {"state": state}

class MockA2AClient:
    def __init__(self):
        self._litellm_agent_card = SimpleNamespace(name="mock-agent", url="http://mock-agent.local")

    async def send_message(self, request, *, context=None):
        from a2a.compat.v0_3.conversions import pb2_v10

        for text in ("hel", "hello"):
            event = pb2_v10.StreamResponse()
            message = event.message
            message.message_id = uuid4().hex
            message.role = pb2_v10.ROLE_AGENT
            message.parts.add().text = text
            yield event

@pytest.fixture
def mock_a2a_client(monkeypatch):
    import litellm.a2a_protocol.main as a2a_main

    async def _fake_create_a2a_client(
        base_url, timeout=60.0, extra_headers=None, streaming=False, relative_card_path=None, **_
    ):
        return MockA2AClient()

    monkeypatch.setattr(a2a_main, "create_a2a_client", _fake_create_a2a_client)

@pytest.mark.asyncio
async def test_a2a_non_streaming(mock_a2a_client):
    """Test non-streaming A2A request."""
    from a2a.compat.v0_3.types import MessageSendParams, SendMessageRequest

    from litellm.a2a_protocol import asend_message

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Say hello in one word"}],
                "messageId": uuid4().hex,
            }
        ),
    )

    response = await asend_message(
        request=request,
        api_base="http://mock",
    )

    assert response is not None
    print(f"\nNon-streaming response: {response}")

@pytest.mark.asyncio
async def test_a2a_streaming(mock_a2a_client):
    """Test streaming A2A request."""
    from a2a.compat.v0_3.types import MessageSendParams, SendStreamingMessageRequest

    from litellm.a2a_protocol import asend_message_streaming

    request = SendStreamingMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Say hello in one word"}],
                "messageId": uuid4().hex,
            }
        ),
    )

    chunks = []
    async for chunk in asend_message_streaming(
        request=request,
        api_base="http://mock",
    ):
        chunks.append(chunk)
        print(f"\nStreaming chunk: {chunk}")

    assert len(chunks) > 0, "Should receive at least one chunk"
    print(f"\nTotal chunks received: {len(chunks)}")
