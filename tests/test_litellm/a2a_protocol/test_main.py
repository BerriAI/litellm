"""Tests for litellm/a2a_protocol/main.py non-streaming send behavior."""

import httpx
import pytest

pytest.importorskip("a2a.compat.v0_3.conversions")

from a2a.compat.v0_3 import conversions as _conv
from a2a.compat.v0_3.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

import litellm
from litellm.a2a_protocol.main import _send_message, _stream_messages, create_a2a_client
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.constants import DEFAULT_A2A_AGENT_TIMEOUT
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)


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

    with patch.object(
        a2a_main, "create_a2a_client", new=AsyncMock(side_effect=_capture)
    ):
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


_LOWERCASE_BINDING_CARD = {
    "name": "langgraph-agent",
    "version": "1.0.0",
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [],
    "supportedInterfaces": [
        {"url": "http://127.0.0.1:9/", "protocolBinding": "jsonrpc", "protocolVersion": "1.0"}
    ],
}


class _RequestRecorder:
    """Records the headers httpx put on the wire, per outbound request."""

    def __init__(self, card=_AGENT_CARD, rpc_reply=_RPC_REPLY):
        self.card = card
        self.rpc_reply = rpc_reply
        self.card_requests = []
        self.rpc_requests = []
        self.client = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        if request.method == "GET":
            self.card_requests.append(headers)
            return httpx.Response(200, json=self.card)
        self.rpc_requests.append(headers)
        return httpx.Response(200, json=self.rpc_reply)


def _a2a_client_cache_key(timeout: float) -> str:
    return "async_httpx_client" + f"timeout_{timeout}" + httpxSpecialProvider.A2AProvider


async def _seed_shared_a2a_client(card=_AGENT_CARD, rpc_reply=_RPC_REPLY) -> _RequestRecorder:
    """Put the one A2A client the cache will hand out behind a mock transport.

    Seeding has to happen on the test's own event loop, because the client cache keys on
    it. The injected client is a real httpx.AsyncClient, so the merge of per-request
    headers over client defaults, which is what these tests are about, stays real.
    """
    recorder = _RequestRecorder(card=card, rpc_reply=rpc_reply)
    handler = AsyncHTTPHandler(timeout=DEFAULT_A2A_AGENT_TIMEOUT)
    owned_client = handler.client
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    await owned_client.aclose()

    litellm.in_memory_llm_clients_cache.set_cache(key=_a2a_client_cache_key(DEFAULT_A2A_AGENT_TIMEOUT), value=handler)
    seeded = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.A2AProvider,
        params={"timeout": DEFAULT_A2A_AGENT_TIMEOUT},
    )
    assert seeded is handler, "cache key drifted from get_async_httpx_client; these tests would test nothing"

    recorder.client = handler.client
    return recorder


@pytest.fixture
def isolated_client_cache():
    previous = getattr(litellm, "in_memory_llm_clients_cache", None)
    litellm.in_memory_llm_clients_cache = LLMClientCache()
    yield litellm.in_memory_llm_clients_cache
    litellm.in_memory_llm_clients_cache = previous


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
async def test_agent_card_fetch_carries_the_callers_headers(isolated_client_cache):
    """Agent cards can sit behind the same auth as the agent, so the card fetch must stay
    authenticated once the headers stop living on the client."""
    recorder = await _seed_shared_a2a_client()

    await create_a2a_client(base_url="http://127.0.0.1:9", extra_headers=_AGENT_A_HEADERS)

    assert recorder.card_requests, "no agent card request was made"
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
