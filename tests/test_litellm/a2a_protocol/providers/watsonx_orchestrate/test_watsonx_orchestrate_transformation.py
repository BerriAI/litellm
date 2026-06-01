import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.a2a_protocol.providers.watsonx_orchestrate.handler import (
    WatsonxOrchestrateHandler,
)
from litellm.a2a_protocol.providers.watsonx_orchestrate.transformation import (
    WatsonxOrchestrateTransformation,
)


class _JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _ShortTtlTokenClient:
    def __init__(self):
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        return _JsonResponse({"access_token": f"token-{self.calls}", "expires_in": 30})


class _SSELines:
    def __init__(self, lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class TestWatsonxOrchestrateTransformation:
    def test_get_api_base_url(self):
        url = WatsonxOrchestrateTransformation.get_api_base_url(
            "https://cpd.example.com/",
            "1769134113217795",
        )
        assert (
            url == "https://cpd.example.com/orchestrate/cpd/instances/1769134113217795"
        )

    def test_extract_text_from_a2a_params(self):
        params = {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "text", "text": "Hello"},
                    {"kind": "text", "text": "world"},
                ],
            }
        }
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
            == "Hello world"
        )

    def test_extract_text_from_a2a_params_ignores_non_text_parts_with_text(self):
        params = {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "data", "text": "metadata label", "data": {}},
                    {"kind": "file", "text": "file label", "file": {}},
                    {"kind": "text", "text": "Hello"},
                    {"text": "legacy"},
                    {"kind": "", "text": "empty-kind"},
                ],
            }
        }
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
            == "Hello legacy empty-kind"
        )

    def test_build_wxo_run_body_with_thread(self):
        body = WatsonxOrchestrateTransformation.build_wxo_run_body(
            wxo_agent_id="agent-uuid",
            text="Hi",
            thread_id="thread-1",
        )
        assert body["agent_id"] == "agent-uuid"
        assert body["thread_id"] == "thread-1"
        assert body["message"]["content"][0]["response_type"] == "conversational_search"
        assert body["message"]["content"][0]["text"] == "Hi"

    @pytest.mark.parametrize(
        "result,expected",
        [
            (
                {
                    "last_message": {
                        "content": [{"type": "text", "text": "from last_message"}]
                    }
                },
                "from last_message",
            ),
            (
                {
                    "result": {
                        "data": {
                            "message": {"content": [{"text": "from nested result"}]}
                        }
                    }
                },
                "from nested result",
            ),
            ({"results": "raw string"}, "raw string"),
        ],
    )
    def test_extract_text_from_wxo_result(self, result, expected):
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_wxo_result(result)
            == expected
        )

    def test_build_a2a_message_response(self):
        out = WatsonxOrchestrateTransformation.build_a2a_message_response(
            "req-1", "answer"
        )
        assert out["jsonrpc"] == "2.0"
        assert out["id"] == "req-1"
        assert out["result"]["kind"] == "message"
        assert out["result"]["parts"][0]["text"] == "answer"

    def test_extract_text_from_a2a_message_response(self):
        envelope = WatsonxOrchestrateTransformation.build_a2a_message_response(
            "req-1", "answer"
        )
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_message_response(
                envelope
            )
            == "answer"
        )
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_message_response(
                {"result": {}}
            )
            == ""
        )


def test_cp4d_token_ttl_from_absolute_expiration():
    wall = 1_750_000_000.0
    assert (
        WatsonxOrchestrateHandler._cp4d_token_ttl_seconds(1_750_003_600, wall) == 3600
    )
    assert WatsonxOrchestrateHandler._cp4d_token_ttl_seconds(1_749_999_000, wall) == 0


@pytest.mark.asyncio
async def test_accumulate_wxo_sse_text_ignores_non_dict_json_events():
    response = _SSELines(
        [
            "data: null",
            "data: true",
            'data: {"results": "streamed text"}',
        ]
    )
    assert await WatsonxOrchestrateHandler._accumulate_wxo_sse_text(response) == (
        "streamed text"
    )


@pytest.mark.asyncio
async def test_short_lived_tokens_are_not_served_from_cache():
    client = _ShortTtlTokenClient()
    token_1 = await WatsonxOrchestrateHandler._get_bearer_token(
        cp4d_host="https://cpd.example.com",
        auth_mode="ibm_cloud",
        api_key="short-ttl-cache-key",
        client=client,
    )
    token_2 = await WatsonxOrchestrateHandler._get_bearer_token(
        cp4d_host="https://cpd.example.com",
        auth_mode="ibm_cloud",
        api_key="short-ttl-cache-key",
        client=client,
    )
    assert token_1 == "token-1"
    assert token_2 == "token-2"
    assert client.calls == 2


def test_config_manager_returns_wxo_provider():
    config = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider="watsonx_orchestrate"
    )
    assert config is not None
    assert config.__class__.__name__ == "WatsonxOrchestrateA2AConfig"
