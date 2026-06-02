"""
Tests for Inception (Mercury) provider integration
"""

import json
import os
from unittest import mock

import httpx
import pytest

import litellm
from litellm.llms.inception.chat.transformation import InceptionChatConfig
from litellm.llms.inception.completion.transformation import (
    InceptionTextCompletionConfig,
)


def test_inception_config_initialization():
    config = InceptionChatConfig()
    assert config.custom_llm_provider == "inception"


def test_inception_chat_supports_diffusion_params():
    """The chat config must expose Inception's diffusion-LLM request controls"""
    params = InceptionChatConfig().get_supported_openai_params("mercury-2")
    for p in (
        "reasoning_effort",
        "reasoning_summary",
        "reasoning_summary_wait",
        "diffusing",
        "realtime",
        "tools",
        "tool_choice",
        "response_format",
    ):
        assert p in params, f"{p} should be a supported chat param"


def test_inception_chat_sends_diffusion_params_in_body():
    """reasoning_effort (incl. `instant`) and the diffusion flags reach the request body"""

    captured = {}

    def fake_send(self, request, **kwargs):
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "id": "c-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "mercury-2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                }
            ).encode(),
        )

    with mock.patch("httpx.Client.send", new=fake_send):
        litellm.completion(
            model="inception/mercury-2",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-x",
            reasoning_effort="instant",
            reasoning_summary=True,
            reasoning_summary_wait=True,
            diffusing=True,
            realtime=True,
            max_completion_tokens=128,
        )

    body = captured["body"]
    assert body["reasoning_effort"] == "instant"
    assert body["reasoning_summary"] is True
    assert body["reasoning_summary_wait"] is True
    assert body["diffusing"] is True
    assert body["realtime"] is True
    assert body["max_tokens"] == 128  # max_completion_tokens mapped to max_tokens


def test_inception_chat_response_surfaces_reasoning_and_usage():
    """reasoning_summary / warning survive, and reasoning_tokens maps to usage details"""

    def fake_send(self, request, **kwargs):
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "id": "c-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "mercury-2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                        "reasoning_tokens": 4,
                        "cached_input_tokens": 3,
                    },
                    "reasoning_summary": {
                        "content": "step by step",
                        "status": "complete",
                    },
                    "warning": "heads up",
                }
            ).encode(),
        )

    with mock.patch("httpx.Client.send", new=fake_send):
        r = litellm.completion(
            model="inception/mercury-2",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-x",
        )

    assert r.reasoning_summary == {"content": "step by step", "status": "complete"}
    assert r.warning == "heads up"
    assert r.usage.completion_tokens_details.reasoning_tokens == 4
    assert r.usage.model_extra.get("cached_input_tokens") == 3


def test_inception_fim_supported_params_match_schema():
    """FIM exposes the OpenAI subset of Inception's FIMCompletionRequest only"""
    params = InceptionTextCompletionConfig().get_supported_openai_params("mercury-edit-2")
    for p in ("suffix", "top_p", "frequency_penalty", "presence_penalty", "stop"):
        assert p in params
    # Chat-only sampling controls are not part of Inception's FIM schema
    for p in ("temperature", "seed", "logprobs", "n", "user"):
        assert p not in params


def test_inception_fim_extra_body_forwards_vllm_params():
    """top_k / repetition_penalty are reachable via extra_body (not OpenAI params)"""

    captured = {}

    def fake_send(self, request, **kwargs):
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "id": "f-1",
                    "object": "text_completion",
                    "created": 1,
                    "model": "mercury-edit-2",
                    "choices": [
                        {
                            "text": "x",
                            "index": 0,
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            ).encode(),
        )

    with mock.patch("httpx.Client.send", new=fake_send):
        litellm.text_completion(
            model="text-completion-inception/mercury-edit-2",
            prompt="def f(",
            suffix=")",
            api_key="sk-x",
            top_p=0.9,
            extra_body={"top_k": 40, "repetition_penalty": 1.1},
        )

    body = captured["body"]
    assert body["top_p"] == 0.9
    assert body["top_k"] == 40
    assert body["repetition_penalty"] == 1.1


def test_inception_get_openai_compatible_provider_info():
    config = InceptionChatConfig()

    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch.object(litellm, "inception_key", None):
            api_base, api_key = config._get_openai_compatible_provider_info(None, None)
            assert api_base == "https://api.inceptionlabs.ai/v1"
            assert api_key is None

    with mock.patch.dict(
        os.environ,
        {
            "INCEPTION_API_KEY": "test-key",
            "INCEPTION_API_BASE": "https://custom.inceptionlabs.ai/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.inceptionlabs.ai/v1"
        assert api_key == "test-key"

    with mock.patch.dict(
        os.environ,
        {
            "INCEPTION_API_KEY": "env-key",
            "INCEPTION_API_BASE": "https://env.inceptionlabs.ai/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://param.inceptionlabs.ai/v1", "param-key"
        )
        assert api_base == "https://param.inceptionlabs.ai/v1"
        assert api_key == "param-key"


def test_inception_key_module_attr_fallback():
    """litellm.inception_key is used when no param/env key is provided"""
    config = InceptionChatConfig()
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch.object(litellm, "inception_key", "module-attr-key"):
            _, api_key = config._get_openai_compatible_provider_info(None, None)
            assert api_key == "module-attr-key"


def test_get_llm_provider_inception():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, _ = get_llm_provider("inception/mercury-2")
    assert model == "mercury-2"
    assert provider == "inception"

    model, provider, _, api_base = get_llm_provider("mercury-2", api_base="https://api.inceptionlabs.ai/v1")
    assert model == "mercury-2"
    assert provider == "inception"
    assert api_base == "https://api.inceptionlabs.ai/v1"


def test_inception_in_provider_lists():
    assert "inception" in litellm.openai_compatible_providers
    assert "inception" in litellm.provider_list
    assert "https://api.inceptionlabs.ai/v1" in litellm.openai_compatible_endpoints


def test_inception_model_configuration():
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.inception_models = set()
    litellm.add_known_models()

    info = get_model_info("inception/mercury-2")
    assert info.get("litellm_provider") == "inception"
    assert info.get("mode") == "chat"
    assert info.get("max_input_tokens") == 128000
    assert info.get("input_cost_per_token") == 2.5e-07
    assert info.get("output_cost_per_token") == 7.5e-07
    assert info.get("cache_read_input_token_cost") == 2.5e-08
    assert info.get("supports_function_calling") is True
    assert info.get("supports_tool_choice") is True
    assert info.get("supports_response_schema") is True


def test_inception_model_list_populated():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.inception_models = set()
    litellm.add_known_models()

    assert "inception/mercury-2" in litellm.inception_models
    for model in litellm.inception_models:
        assert model.startswith("inception/")


def test_inception_completion_targets_inception_endpoint():
    """
    End-to-end: a completion routed through the inception provider must hit
    Inception's base URL and path, send a Bearer token, and strip the
    `inception/` prefix from the model name.
    """

    captured = {}

    def fake_send(self, request, **kwargs):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "mercury-2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 1,
                        "total_tokens": 6,
                    },
                }
            ).encode(),
        )

    with mock.patch("httpx.Client.send", new=fake_send):
        response = litellm.completion(
            model="inception/mercury-2",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test-fake-123",
        )

    assert captured["url"] == "https://api.inceptionlabs.ai/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test-fake-123"
    assert captured["body"]["model"] == "mercury-2"
    assert response.choices[0].message.content == "hi"


@pytest.mark.asyncio
async def test_inception_completion_call():
    """Live smoke test (requires INCEPTION_API_KEY)"""
    if not os.getenv("INCEPTION_API_KEY"):
        pytest.skip("INCEPTION_API_KEY not set")

    response = await litellm.acompletion(
        model="inception/mercury-2",
        messages=[{"role": "user", "content": "Hello, this is a test"}],
        max_tokens=10,
    )
    assert response.choices[0].message.content
    assert response.model
    assert response.usage


def test_inception_fim_supports_suffix_param():
    """The FIM config must keep `suffix` (otherwise FIM requests lose context)"""
    config = InceptionTextCompletionConfig()
    assert "suffix" in config.get_supported_openai_params("mercury-edit-2")

    mapped = config.map_openai_params(
        non_default_params={"suffix": "\n    return x", "max_completion_tokens": 50},
        optional_params={},
        model="mercury-edit-2",
        drop_params=False,
    )
    assert mapped["suffix"] == "\n    return x"
    assert mapped["max_tokens"] == 50


def test_text_completion_inception_in_provider_lists():
    from litellm.types.utils import LlmProviders

    assert LlmProviders.TEXT_COMPLETION_INCEPTION == "text-completion-inception"
    assert "text-completion-inception" in litellm.provider_list


def test_inception_fim_model_configuration():
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.text_completion_inception_models = set()
    litellm.add_known_models()

    assert "text-completion-inception/mercury-edit-2" in litellm.text_completion_inception_models
    info = get_model_info("text-completion-inception/mercury-edit-2")
    assert info.get("litellm_provider") == "text-completion-inception"
    assert info.get("mode") == "completion"
    assert info.get("max_input_tokens") == 32000


def test_inception_fim_targets_fim_endpoint():
    """
    End-to-end: a FIM request must hit `/v1/fim/completions` (NOT
    `/v1/completions`), carry the `suffix`, and parse the standard `text` field.
    """

    captured = {}

    def fake_send(self, request, **kwargs):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "id": "fim-1",
                    "object": "text_completion",
                    "created": 1,
                    "model": "mercury-edit-2",
                    "choices": [
                        {
                            "text": "a + b",
                            "index": 0,
                            "finish_reason": "stop",
                            "logprobs": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                }
            ).encode(),
        )

    with mock.patch("httpx.Client.send", new=fake_send):
        response = litellm.text_completion(
            model="text-completion-inception/mercury-edit-2",
            prompt="def add(a, b):\n    return ",
            suffix="\n",
            api_key="sk-fim-fake",
            max_tokens=20,
        )

    assert captured["url"] == "https://api.inceptionlabs.ai/v1/fim/completions"
    assert captured["auth"] == "Bearer sk-fim-fake"
    assert captured["body"]["model"] == "mercury-edit-2"
    assert captured["body"]["suffix"] == "\n"
    assert "prompt" in captured["body"]
    assert response.choices[0].text == "a + b"
