"""
Tests for Inception (Mercury) chat provider integration
"""

import json
import os
from unittest import mock

import httpx

import litellm
from litellm.llms.inception.chat.transformation import InceptionChatConfig


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


def test_inception_does_not_leak_key_to_caller_api_base():
    """
    The server-managed Inception key must not be forwarded to a caller-supplied
    api_base. It is only resolved for the default/server base, or when the
    caller also supplies their own key.
    """
    config = InceptionChatConfig()
    with mock.patch.dict(
        os.environ, {"INCEPTION_API_KEY": "server-secret"}, clear=True
    ):
        with mock.patch.object(litellm, "inception_key", "module-secret"):
            # caller overrides api_base without a key -> server key withheld
            api_base, api_key = config._get_openai_compatible_provider_info(
                "https://attacker.example/v1", None
            )
            assert api_base == "https://attacker.example/v1"
            assert api_key is None

            # caller overrides api_base AND supplies their own key -> used as-is
            _, api_key = config._get_openai_compatible_provider_info(
                "https://attacker.example/v1", "caller-key"
            )
            assert api_key == "caller-key"

            # default/server base -> server-managed key resolved
            _, api_key = config._get_openai_compatible_provider_info(None, None)
            assert api_key == "module-secret"


def test_get_llm_provider_inception():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, _ = get_llm_provider("inception/mercury-2")
    assert model == "mercury-2"
    assert provider == "inception"

    model, provider, _, api_base = get_llm_provider(
        "mercury-2", api_base="https://api.inceptionlabs.ai/v1"
    )
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
    Inception's base URL and path, send a Bearer token, strip the
    `inception/` prefix from the model name, and forward tool_choice.
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with mock.patch("httpx.Client.send", new=fake_send):
        response = litellm.completion(
            model="inception/mercury-2",
            messages=[{"role": "user", "content": "hello"}],
            api_key="sk-test-fake-123",
            tools=tools,
            tool_choice="auto",
        )

    assert captured["url"] == "https://api.inceptionlabs.ai/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test-fake-123"
    assert captured["body"]["model"] == "mercury-2"
    assert captured["body"]["tool_choice"] == "auto"
    assert response.choices[0].message.content == "hi"
