"""
Unit tests for the iFlytek Spark configuration.

iFlytek Spark is an OpenAI-compatible provider; these tests validate the
provider routing, default API base, reverse endpoint detection and the
IFlytekConfig param mapping.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

import litellm
from litellm import completion, get_llm_provider
from litellm.llms.iflytek.chat.transformation import IFlytekConfig

IFLYTEK_API_BASE = "https://spark-api-open.xf-yun.com/v1"


class TestIFlytekRouting:
    def test_provider_prefix_routing(self):
        model, provider, dynamic_api_key, api_base = get_llm_provider(
            model="iflytek/4.0Ultra", api_key="fake-iflytek-key"
        )
        assert model == "4.0Ultra"
        assert provider == "iflytek"
        assert api_base == IFLYTEK_API_BASE

    def test_reverse_endpoint_detection(self):
        _, provider, _, api_base = get_llm_provider(model="generalv3.5", api_base=IFLYTEK_API_BASE)
        assert provider == "iflytek"
        assert api_base == IFLYTEK_API_BASE

    def test_api_base_env_override(self, monkeypatch):
        monkeypatch.setenv("IFLYTEK_API_BASE", "https://custom.spark.example/v1")
        _, provider, _, api_base = get_llm_provider(model="iflytek/lite", api_key="fake-iflytek-key")
        assert provider == "iflytek"
        assert api_base == "https://custom.spark.example/v1"


class TestIFlytekConfig:
    def test_map_max_completion_tokens(self):
        mapped = IFlytekConfig().map_openai_params(
            non_default_params={"max_completion_tokens": 50, "temperature": 0.3},
            optional_params={},
            model="4.0Ultra",
            drop_params=False,
        )
        assert mapped["max_tokens"] == 50
        assert mapped["temperature"] == 0.3
        assert "max_completion_tokens" not in mapped

    @pytest.mark.respx()
    def test_iflytek_completion_uses_default_base(self, respx_mock):
        """
        With no api_base passed, the request must go to the iFlytek Spark default
        endpoint with the provided key. This fails if the default-base routing or
        the openai-compatible wiring is broken.
        """
        litellm.disable_aiohttp_transport = True

        api_key = "fake-iflytek-key"
        route = respx_mock.post(f"{IFLYTEK_API_BASE}/chat/completions").respond(
            json={
                "id": "chatcmpl-spark-1",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "4.0Ultra",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "你好,来自 LiteLLM"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 6, "total_tokens": 15},
            },
            status_code=200,
        )

        response = completion(
            model="iflytek/4.0Ultra",
            messages=[{"role": "user", "content": "say hi"}],
            api_key=api_key,
        )

        assert route.called
        sent = route.calls.last.request
        assert sent.headers["authorization"] == f"Bearer {api_key}"
        assert response.choices[0].message.content == "你好,来自 LiteLLM"
