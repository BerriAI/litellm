import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.a2a_protocol.providers.watsonx_orchestrate.transformation import (
    WatsonxOrchestrateTransformation,
)


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


def test_config_manager_returns_wxo_provider():
    config = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider="watsonx_orchestrate"
    )
    assert config is not None
    assert config.__class__.__name__ == "WatsonxOrchestrateA2AConfig"
