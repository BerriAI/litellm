from litellm.a2a_protocol.providers.config_manager import (
    A2AProviderConfigManager,
)


def test_vertex_agent_engine_routes_to_a2a_provider():
    """Agent Engine should have its own A2A provider."""
    provider = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider="vertex_ai",
        model="agent_engine/123456789",
    )

    assert provider is not None


def test_regular_vertex_model_uses_default_routing():
    """Ordinary Vertex AI models should not use Agent Engine routing."""
    provider = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider="vertex_ai",
        model="gemini-2.5-flash",
    )

    assert provider is None


import pytest


def test_vertex_session_identity_is_stable_and_isolated():
    from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
        session_user_id,
    )

    first = session_user_id("key-hash-A", "conv-1")

    # Same caller and context should produce the same identity.
    assert first == session_user_id("key-hash-A", "conv-1")

    # Different callers must not share identities.
    assert first != session_user_id("key-hash-B", "conv-1")

    # Different conversations must remain separate.
    assert first != session_user_id("key-hash-A", "conv-2")

    # Raw identifiers must not be exposed.
    assert "key-hash-A" not in first
    assert "conv-1" not in first


def test_vertex_session_identity_requires_authenticated_key():
    from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
        session_user_id,
    )

    with pytest.raises(ValueError, match="calling key"):
        session_user_id("", "conv-1")


class FakeVertexSessionBackend:
    """In-memory stand-in for Google Agent Engine session APIs."""

    def __init__(self):
        self.sessions: dict[str, list[str]] = {}
        self.creations = 0

    async def list_sessions(self, *, user_id: str) -> list[str]:
        return list(self.sessions.get(user_id, []))

    async def create_session(self, *, user_id: str) -> str:
        self.creations += 1
        session_id = f"server-generated-{self.creations}"
        self.sessions.setdefault(user_id, []).append(session_id)
        return session_id


@pytest.mark.asyncio
async def test_same_conversation_reuses_vertex_session():
    from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
        resolve_session_id,
    )

    backend = FakeVertexSessionBackend()

    first = await resolve_session_id(
        backend,
        calling_key_hash="key-A",
        context_id="conv-1",
    )

    second = await resolve_session_id(
        backend,
        calling_key_hash="key-A",
        context_id="conv-1",
    )

    assert first == second
    assert backend.creations == 1


@pytest.mark.asyncio
async def test_different_callers_and_contexts_have_separate_sessions():
    from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
        resolve_session_id,
    )

    backend = FakeVertexSessionBackend()

    session_a = await resolve_session_id(backend, calling_key_hash="key-A", context_id="conv-1")
    session_b = await resolve_session_id(backend, calling_key_hash="key-B", context_id="conv-1")
    session_c = await resolve_session_id(backend, calling_key_hash="key-A", context_id="conv-2")

    assert len({session_a, session_b, session_c}) == 3
    assert backend.creations == 3


@pytest.mark.asyncio
async def test_vertex_http_backend_reuses_sessions():
    """Verify Agent Engine session reuse through mocked Google HTTP calls."""
    import json

    import httpx

    from litellm.a2a_protocol.providers.vertex_agent_engine.backend import (
        VertexAgentEngineSessionBackend,
    )
    from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
        resolve_session_id,
    )

    sessions = {}
    methods = []

    def fake_google_api(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.url.path.endswith("/reasoningEngines/123:query")

        payload = json.loads(request.content)
        method = payload["class_method"]
        user_id = payload["input"]["user_id"]
        methods.append(method)

        if method == "async_list_sessions":
            existing = sessions.get(user_id)
            items = [{"id": existing}] if existing else []
            return httpx.Response(200, json={"output": {"sessions": items}})

        if method == "async_create_session":
            session_id = f"vertex-session-{len(sessions) + 1}"
            sessions[user_id] = session_id
            return httpx.Response(200, json={"output": {"id": session_id}})

        return httpx.Response(400, json={"error": "Unknown method"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(fake_google_api)) as client:
        backend = VertexAgentEngineSessionBackend(
            client=client,
            query_url=(
                "https://us-central1-aiplatform.googleapis.com/"
                "v1/projects/demo/locations/us-central1/"
                "reasoningEngines/123:query"
            ),
            auth_headers={"Authorization": "Bearer test-token"},
        )

        first = await resolve_session_id(backend, calling_key_hash="key-A", context_id="conv-1")
        second = await resolve_session_id(backend, calling_key_hash="key-A", context_id="conv-1")
        other = await resolve_session_id(backend, calling_key_hash="key-B", context_id="conv-1")

    assert first == second
    assert first != other
    assert len(sessions) == 2
    assert methods.count("async_create_session") == 2
    assert methods.count("async_list_sessions") == 3


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_vertex_a2a_bridge_forwards_session(monkeypatch, streaming):
    """Both A2A paths must forward the resolved session to completion."""
    from types import SimpleNamespace

    import litellm
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )
    from litellm.a2a_protocol.providers.vertex_agent_engine.config import (
        VertexAgentEngineA2AConfig,
    )

    captured = []

    async def fake_resolve(params, settings):
        return "ctx-1", {
            **settings,
            "user": "isolated-user",
            "user_id": "isolated-user",
            "session_id": "server-session-42",
        }

    async def fake_completion(completion_params):
        captured.append(dict(completion_params))

        if completion_params["stream"]:

            async def chunks():
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="pong"))])

            return chunks()

        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "pong",
                    },
                    "finish_reason": "stop",
                }
            ]
        )

    monkeypatch.setattr(
        VertexAgentEngineA2AConfig,
        "_resolve_session_params",
        staticmethod(fake_resolve),
    )
    monkeypatch.setattr(
        A2ACompletionBridgeHandler,
        "_acompletion",
        staticmethod(fake_completion),
    )

    provider = VertexAgentEngineA2AConfig()

    params = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "ping"}],
            "messageId": "m-1",
            "contextId": "ctx-1",
        }
    }

    settings = {
        "custom_llm_provider": "vertex_ai",
        "model": "agent_engine/123",
        "litellm_a2a_user_api_key_hash": "hashed-key",
    }

    if streaming:
        events = [
            event
            async for event in provider.handle_streaming(
                request_id="req-1",
                params=params,
                litellm_params=settings,
            )
        ]
        assert len(events) == 4
        assert all(event["result"]["contextId"] == "ctx-1" for event in events)
    else:
        response = await provider.handle_non_streaming(
            request_id="req-1",
            params=params,
            litellm_params=settings,
        )
        assert response["result"]["contextId"] == "ctx-1"
        assert response["result"]["parts"][0]["text"] == "pong"

    assert len(captured) == 1

    completion = captured[0]
    assert completion["model"] == "vertex_ai/agent_engine/123"
    assert completion["user"] == "isolated-user"
    assert completion["user_id"] == "isolated-user"
    assert completion["session_id"] == "server-session-42"
    assert completion["stream"] is streaming

    # Internal authentication metadata must not reach completion.
    assert "litellm_a2a_user_api_key_hash" not in completion

    # Shared agent settings must remain unchanged.
    assert "session_id" not in settings
    assert "user_id" not in settings


def test_vertex_session_survives_completion_parameter_mapping():
    """Verify user_id and session_id reach the Vertex Agent Engine payload."""
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )
    from litellm.llms.vertex_ai.agent_engine.transformation import (
        VertexAgentEngineConfig,
    )
    from litellm.utils import get_optional_params

    a2a_params = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Remember my name"}],
            "messageId": "msg-1",
            "contextId": "conversation-1",
        }
    }

    settings = {
        "custom_llm_provider": "vertex_ai",
        "model": "agent_engine/123",
        "user": "scoped-user-123",
        "user_id": "scoped-user-123",
        "session_id": "vertex-generated-session-42",
        "litellm_a2a_user_api_key_hash": "hashed-key",
    }

    # Convert the A2A message into completion parameters.
    completion = A2ACompletionBridgeHandler._build_completion_params(
        params=a2a_params,
        litellm_params=settings,
        api_base=None,
        agent_extra_headers=None,
        stream=False,
    )

    # Simulate LiteLLM's provider parameter mapping.
    optional = get_optional_params(
        model="agent_engine/123",
        custom_llm_provider="vertex_ai",
        user=completion["user"],
        user_id=completion["user_id"],
        session_id=completion["session_id"],
    )

    # Build the payload that would be sent to Vertex Agent Engine.
    payload = VertexAgentEngineConfig().transform_request(
        model="agent_engine/123",
        messages=completion["messages"],
        optional_params=optional,
        litellm_params={},
        headers={},
    )

    # Verify the correct conversation identity and session are forwarded.
    assert payload["input"]["user_id"] == "scoped-user-123"
    assert payload["input"]["session_id"] == "vertex-generated-session-42"
    assert payload["input"]["message"] == "Remember my name"

    # Internal authentication metadata must not leak to completion.
    assert "litellm_a2a_user_api_key_hash" not in completion


@pytest.mark.asyncio
async def test_concurrent_requests_reuse_one_vertex_session():
    """Simultaneous first messages should not create duplicate sessions."""
    import asyncio

    from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
        resolve_session_id,
    )

    class RacingBackend:
        def __init__(self):
            self.sessions = {}
            self.creations = 0

        async def list_sessions(self, *, user_id):
            existing = list(self.sessions.get(user_id, []))

            # Give the second request an opportunity to check
            # the session list before either creates a session.
            await asyncio.sleep(0)
            return existing

        async def create_session(self, *, user_id):
            self.creations += 1
            session_id = f"server-session-{self.creations}"
            self.sessions.setdefault(user_id, []).append(session_id)
            return session_id

    backend = RacingBackend()

    first, second = await asyncio.gather(
        resolve_session_id(
            backend,
            calling_key_hash="same-key",
            context_id="same-conversation",
        ),
        resolve_session_id(
            backend,
            calling_key_hash="same-key",
            context_id="same-conversation",
        ),
    )

    assert first == second
    assert backend.creations == 1


@pytest.mark.parametrize("status_code", [401, 403, 503])
@pytest.mark.asyncio
async def test_vertex_session_lookup_error_does_not_create_session(status_code):
    """A failed lookup must not silently create a replacement session."""
    import httpx

    from litellm.a2a_protocol.providers.vertex_agent_engine.backend import (
        VertexAgentEngineSessionBackend,
    )
    from litellm.a2a_protocol.providers.vertex_agent_engine.session import (
        resolve_session_id,
    )

    requests = []

    def fake_google_api(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code,
            json={"error": {"message": "Session service unavailable"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(fake_google_api)) as client:
        backend = VertexAgentEngineSessionBackend(
            client=client,
            query_url=(
                "https://us-central1-aiplatform.googleapis.com/"
                "v1/projects/demo/locations/us-central1/"
                "reasoningEngines/123:query"
            ),
            auth_headers={"Authorization": "Bearer test-token"},
        )

        with pytest.raises(httpx.HTTPStatusError):
            await resolve_session_id(
                backend,
                calling_key_hash="key-A",
                context_id="conversation-1",
            )

    # Only the failed list request should have been sent.
    # A create_session request would be unsafe here.
    assert len(requests) == 1
    assert b"async_list_sessions" in requests[0].content


@pytest.mark.asyncio
async def test_vertex_resolver_uses_google_session_api(monkeypatch):
    """Test the real resolver with mocked Google HTTP responses."""
    import json

    import httpx

    from litellm.a2a_protocol.providers.vertex_agent_engine.config import (
        VertexAgentEngineA2AConfig,
    )
    from litellm.llms.vertex_ai.agent_engine.transformation import (
        VertexAgentEngineConfig,
    )

    original_client = httpx.AsyncClient
    requests_seen = []
    sessions = {}

    def fake_google_api(request):
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.url.path.endswith("/projects/test-project/locations/us-central1/reasoningEngines/123:query")

        payload = json.loads(request.content)
        requests_seen.append(payload)

        method = payload["class_method"]
        user_id = payload["input"]["user_id"]

        if method == "async_list_sessions":
            existing = sessions.get(user_id)
            return httpx.Response(
                200,
                json={"output": {"sessions": [{"id": existing}] if existing else []}},
            )

        if method == "async_create_session":
            session_id = f"generated-{len(sessions) + 1}"
            sessions[user_id] = session_id
            return httpx.Response(200, json={"output": {"id": session_id}})

        return httpx.Response(400)

    def mock_client(*args, **kwargs):
        return original_client(
            *args,
            transport=httpx.MockTransport(fake_google_api),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    monkeypatch.setattr(
        VertexAgentEngineConfig,
        "_get_auth_headers",
        lambda self, optional_params, litellm_params: {"Authorization": "Bearer test-token"},
    )

    params = {"message": {"contextId": "conversation-1"}}
    settings = {
        "custom_llm_provider": "vertex_ai",
        "model": "agent_engine/123",
        "vertex_project": "test-project",
        "vertex_location": "us-central1",
        "litellm_a2a_user_api_key_hash": "hashed-key",
    }

    first_context, first = await VertexAgentEngineA2AConfig._resolve_session_params(params, settings)
    second_context, second = await VertexAgentEngineA2AConfig._resolve_session_params(params, settings)

    assert first_context == second_context == "conversation-1"
    assert first["session_id"] == second["session_id"]
    assert first["session_id"] == "generated-1"
    assert first["user_id"] == second["user_id"]
    assert first["user"] == first["user_id"]
    assert "session_id" not in settings

    assert [request["class_method"] for request in requests_seen] == [
        "async_list_sessions",
        "async_create_session",
        "async_list_sessions",
    ]
