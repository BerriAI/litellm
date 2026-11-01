"""
Tests for A2A session bridging to Vertex AI Agent Engine.

No real API calls: the Agent Engine sessions endpoint and the access token are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    A2A_USER_API_KEY_HASH_PARAM,
)
from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.a2a_protocol.providers.vertex_agent_engine.config import (
    VertexAgentEngineA2AConfig,
)
from litellm.llms.vertex_ai.agent_engine import a2a as agent_engine_a2a
from litellm.llms.vertex_ai.agent_engine.a2a import (
    A2A_AGENT_ENGINE_USER_ID_PREFIX,
    derive_agent_engine_user_id,
    lookup_latest_session_id,
    merge_a2a_session_into_litellm_params,
)

ENGINE_PATH = "projects/test-project/locations/test-region/reasoningEngines/123"
LITELLM_PARAMS = {
    "custom_llm_provider": "vertex_ai",
    "model": f"agent_engine/{ENGINE_PATH}",
}


def _params(context_id: str | None = "ctx-1") -> dict:
    message: dict = {"role": "user", "parts": [{"kind": "text", "text": "hi"}]}
    if context_id is not None:
        message["contextId"] = context_id
    return {"message": message}


def _mock_sessions_api(sessions: list[dict] | None = None, error: Exception | None = None):
    """Patch the token and the HTTP client used by the session lookup."""
    response = httpx.Response(
        200,
        json={"sessions": sessions or []},
        request=httpx.Request("GET", "https://test-region-aiplatform.googleapis.com"),
    )
    client = MagicMock()
    client.get = AsyncMock(side_effect=error) if error else AsyncMock(return_value=response)
    return (
        patch.object(
            agent_engine_a2a.VertexAgentEngineConfig,
            "get_access_token_async",
            AsyncMock(return_value=("token", "test-project")),
        ),
        patch.object(agent_engine_a2a, "get_async_httpx_client", return_value=client),
        client,
    )


class TestDeriveUserId:
    def test_same_key_and_context_is_stable(self):
        assert derive_agent_engine_user_id("ctx-1", "hash-a") == derive_agent_engine_user_id("ctx-1", "hash-a")

    def test_same_key_different_context_is_isolated(self):
        assert derive_agent_engine_user_id("ctx-1", "hash-a") != derive_agent_engine_user_id("ctx-2", "hash-a")

    def test_different_keys_same_context_are_isolated(self):
        """Two keys reusing a contextId must not share an Agent Engine session."""
        assert derive_agent_engine_user_id("ctx-1", "hash-a") != derive_agent_engine_user_id("ctx-1", "hash-b")

    def test_principal_and_context_are_not_sent_verbatim(self):
        user_id = derive_agent_engine_user_id("ctx-1", "hash-a")
        assert user_id is not None
        assert user_id.startswith(A2A_AGENT_ENGINE_USER_ID_PREFIX)
        assert "hash-a" not in user_id
        assert "ctx-1" not in user_id

    @pytest.mark.parametrize("context_id,principal", [(None, "hash-a"), ("", "hash-a"), ("ctx-1", None), ("ctx-1", "")])
    def test_missing_part_returns_none(self, context_id, principal):
        assert derive_agent_engine_user_id(context_id, principal) is None


class TestLookupLatestSessionId:
    @pytest.mark.asyncio
    async def test_returns_latest_session_of_user(self):
        user_id = derive_agent_engine_user_id("ctx-1", "hash-a")
        sessions = [
            {"name": f"{ENGINE_PATH}/sessions/111", "userId": user_id, "updateTime": "2026-01-01T00:00:00Z"},
            {"name": f"{ENGINE_PATH}/sessions/222", "userId": user_id, "updateTime": "2026-01-02T00:00:00Z"},
        ]
        token_patch, client_patch, client = _mock_sessions_api(sessions)
        with token_patch, client_patch:
            assert await lookup_latest_session_id(LITELLM_PARAMS, user_id) == "222"

        url = client.get.call_args.args[0]
        assert url == f"https://test-region-aiplatform.googleapis.com/v1beta1/{ENGINE_PATH}/sessions"
        assert client.get.call_args.kwargs["params"]["filter"] == f'user_id="{user_id}"'

    @pytest.mark.asyncio
    async def test_ignores_sessions_of_other_users(self):
        user_id = derive_agent_engine_user_id("ctx-1", "hash-a")
        other = derive_agent_engine_user_id("ctx-1", "hash-b")
        sessions = [{"name": f"{ENGINE_PATH}/sessions/999", "userId": other, "updateTime": "2026-01-03T00:00:00Z"}]
        token_patch, client_patch, _ = _mock_sessions_api(sessions)
        with token_patch, client_patch:
            assert await lookup_latest_session_id(LITELLM_PARAMS, user_id) is None

    @pytest.mark.asyncio
    async def test_lookup_error_returns_none(self):
        token_patch, client_patch, _ = _mock_sessions_api(error=RuntimeError("boom"))
        with token_patch, client_patch:
            assert await lookup_latest_session_id(LITELLM_PARAMS, "litellm-a2a-x") is None

    @pytest.mark.asyncio
    async def test_engine_id_only_model_uses_vertex_project_and_location(self):
        params = {
            "custom_llm_provider": "vertex_ai",
            "model": "agent_engine/456",
            "vertex_project": "other-project",
            "vertex_location": "other-region",
        }
        token_patch, client_patch, client = _mock_sessions_api([])
        with token_patch, client_patch:
            await lookup_latest_session_id(params, "litellm-a2a-x")
        assert client.get.call_args.args[0] == (
            "https://other-region-aiplatform.googleapis.com/v1beta1/"
            "projects/other-project/locations/other-region/reasoningEngines/456/sessions"
        )

    @pytest.mark.asyncio
    async def test_non_agent_engine_model_skips_lookup(self):
        token_patch, client_patch, client = _mock_sessions_api([])
        with token_patch, client_patch:
            result = await lookup_latest_session_id(
                {"custom_llm_provider": "vertex_ai", "model": "gemini-pro"}, "litellm-a2a-x"
            )
        assert result is None
        client.get.assert_not_called()


class TestMergeSessionIntoLitellmParams:
    @pytest.mark.asyncio
    async def test_first_turn_sets_user_id_only(self):
        token_patch, client_patch, _ = _mock_sessions_api([])
        with token_patch, client_patch:
            merged = await merge_a2a_session_into_litellm_params(LITELLM_PARAMS, _params(), "hash-a")
        assert merged["user_id"] == derive_agent_engine_user_id("ctx-1", "hash-a")
        assert "session_id" not in merged

    @pytest.mark.asyncio
    async def test_next_turn_reuses_existing_session(self):
        user_id = derive_agent_engine_user_id("ctx-1", "hash-a")
        sessions = [{"name": f"{ENGINE_PATH}/sessions/777", "userId": user_id, "updateTime": "2026-01-01T00:00:00Z"}]
        token_patch, client_patch, _ = _mock_sessions_api(sessions)
        with token_patch, client_patch:
            merged = await merge_a2a_session_into_litellm_params(LITELLM_PARAMS, _params(), "hash-a")
        assert merged["user_id"] == user_id
        assert merged["session_id"] == "777"

    @pytest.mark.asyncio
    async def test_lookup_failure_falls_back_to_new_session(self):
        token_patch, client_patch, _ = _mock_sessions_api(error=RuntimeError("boom"))
        with token_patch, client_patch:
            merged = await merge_a2a_session_into_litellm_params(LITELLM_PARAMS, _params(), "hash-a")
        assert "session_id" not in merged

    @pytest.mark.asyncio
    @pytest.mark.parametrize("context_id,principal", [(None, "hash-a"), ("ctx-1", None)])
    async def test_without_context_or_principal_is_noop(self, context_id, principal):
        token_patch, client_patch, client = _mock_sessions_api([])
        with token_patch, client_patch:
            merged = await merge_a2a_session_into_litellm_params(LITELLM_PARAMS, _params(context_id), principal)
        assert merged == LITELLM_PARAMS
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_user_id_is_respected(self):
        token_patch, client_patch, client = _mock_sessions_api([])
        with token_patch, client_patch:
            merged = await merge_a2a_session_into_litellm_params(
                {**LITELLM_PARAMS, "user_id": "configured-user"}, _params(), "hash-a"
            )
        assert merged["user_id"] == "configured-user"
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_input_params_are_not_mutated(self):
        original = dict(LITELLM_PARAMS)
        token_patch, client_patch, _ = _mock_sessions_api([])
        with token_patch, client_patch:
            await merge_a2a_session_into_litellm_params(LITELLM_PARAMS, _params(), "hash-a")
        assert LITELLM_PARAMS == original

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sessions", [[], None])
    async def test_session_backed_call_bypasses_response_cache(self, sessions):
        user_id = derive_agent_engine_user_id("ctx-1", "hash-a")
        existing = [{"name": f"{ENGINE_PATH}/sessions/777", "userId": user_id, "updateTime": "2026-01-01T00:00:00Z"}]
        token_patch, client_patch, _ = _mock_sessions_api(existing if sessions is None else sessions)
        with token_patch, client_patch:
            merged = await merge_a2a_session_into_litellm_params(LITELLM_PARAMS, _params(), "hash-a")
        assert merged["cache"] == {"no-cache": True, "no-store": True}

    @pytest.mark.asyncio
    async def test_noop_path_leaves_cache_settings_alone(self):
        token_patch, client_patch, _ = _mock_sessions_api([])
        with token_patch, client_patch:
            merged = await merge_a2a_session_into_litellm_params(LITELLM_PARAMS, _params(None), "hash-a")
        assert "cache" not in merged


class TestVertexAgentEngineA2AConfig:
    def test_registered_for_agent_engine_models(self):
        cfg = A2AProviderConfigManager.get_provider_config(
            custom_llm_provider="vertex_ai", model=f"agent_engine/{ENGINE_PATH}"
        )
        assert isinstance(cfg, VertexAgentEngineA2AConfig)

    def test_not_registered_for_other_vertex_models(self):
        assert A2AProviderConfigManager.get_provider_config(custom_llm_provider="vertex_ai", model="gemini-pro") is None

    @pytest.mark.asyncio
    async def test_non_streaming_forwards_session_to_bridge(self):
        litellm_params = {**LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: "hash-a"}
        with (
            patch(
                "litellm.a2a_protocol.providers.vertex_agent_engine.config.merge_a2a_session_into_litellm_params",
                AsyncMock(return_value={**litellm_params, "user_id": "u", "session_id": "s"}),
            ) as merge,
            patch(
                "litellm.a2a_protocol.providers.vertex_agent_engine.config.A2ACompletionBridgeHandler.handle_non_streaming",
                AsyncMock(return_value={"ok": True}),
            ) as bridge,
        ):
            result = await VertexAgentEngineA2AConfig().handle_non_streaming(
                request_id="r1", params=_params(), litellm_params=litellm_params
            )
        assert result == {"ok": True}
        merge.assert_awaited_once()
        assert merge.call_args.args[2] == "hash-a"
        kwargs = bridge.call_args.kwargs
        assert kwargs["litellm_params"]["session_id"] == "s"
        assert kwargs["_skip_a2a_provider_routing"] is True

    @pytest.mark.asyncio
    async def test_streaming_forwards_session_to_bridge(self):
        litellm_params = {**LITELLM_PARAMS, A2A_USER_API_KEY_HASH_PARAM: "hash-a"}

        async def fake_stream(**kwargs):
            yield {"chunk": 1}
            yield {"chunk": 2}

        bridge = MagicMock(side_effect=fake_stream)
        with (
            patch(
                "litellm.a2a_protocol.providers.vertex_agent_engine.config.merge_a2a_session_into_litellm_params",
                AsyncMock(return_value={**litellm_params, "user_id": "u", "session_id": "s"}),
            ),
            patch(
                "litellm.a2a_protocol.providers.vertex_agent_engine.config.A2ACompletionBridgeHandler.handle_streaming",
                bridge,
            ),
        ):
            chunks = [
                c
                async for c in VertexAgentEngineA2AConfig().handle_streaming(
                    request_id="r1", params=_params(), litellm_params=litellm_params
                )
            ]
        assert chunks == [{"chunk": 1}, {"chunk": 2}]
        kwargs = bridge.call_args.kwargs
        assert kwargs["litellm_params"]["session_id"] == "s"
        assert kwargs["_skip_a2a_provider_routing"] is True

    @pytest.mark.asyncio
    async def test_missing_litellm_params_raises(self):
        with pytest.raises(ValueError, match="litellm_params is required"):
            await VertexAgentEngineA2AConfig().handle_non_streaming(request_id="r1", params=_params())
