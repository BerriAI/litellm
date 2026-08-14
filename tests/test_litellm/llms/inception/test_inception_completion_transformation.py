"""
Tests for Inception (Mercury) fill-in-the-middle (FIM) provider integration
"""

import json
import os
from unittest import mock

import httpx
import pytest

import litellm
from litellm.llms.inception.completion.transformation import (
    InceptionTextCompletionConfig,
)


def _fim_response_bytes():
    return json.dumps(
        {
            "id": "fim-1",
            "object": "text_completion",
            "created": 1,
            "model": "mercury-edit-2",
            "choices": [
                {"text": "a + b", "index": 0, "finish_reason": "stop", "logprobs": None}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    ).encode()


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


def test_inception_fim_supported_params_match_schema():
    """FIM exposes the OpenAI subset of Inception's FIMCompletionRequest only"""
    params = InceptionTextCompletionConfig().get_supported_openai_params(
        "mercury-edit-2"
    )
    for p in ("suffix", "top_p", "frequency_penalty", "presence_penalty", "stop"):
        assert p in params
    # Chat-only sampling controls are not part of Inception's FIM schema
    for p in ("temperature", "seed", "logprobs", "n", "user"):
        assert p not in params


def test_text_completion_inception_in_provider_lists():
    from litellm.types.utils import LlmProviders

    assert LlmProviders.TEXT_COMPLETION_INCEPTION == "text-completion-inception"
    assert "text-completion-inception" in litellm.provider_list


def test_inception_get_supported_openai_params_dispatch():
    """litellm.get_supported_openai_params routes the FIM provider to our config"""
    params = litellm.get_supported_openai_params(
        model="mercury-edit-2", custom_llm_provider="text-completion-inception"
    )
    assert "suffix" in params
    assert "temperature" not in params


@pytest.mark.parametrize("provider", ["inception", "text-completion-inception"])
def test_inception_validate_environment(provider):
    model = (
        "inception/mercury-2"
        if provider == "inception"
        else "text-completion-inception/mercury-edit-2"
    )

    with mock.patch.dict(os.environ, {}, clear=True):
        result = litellm.validate_environment(model)
        assert result["keys_in_environment"] is False
        assert "INCEPTION_API_KEY" in result["missing_keys"]

    with mock.patch.dict(os.environ, {"INCEPTION_API_KEY": "sk-x"}, clear=True):
        result = litellm.validate_environment(model)
        assert result["keys_in_environment"] is True


def test_inception_completion_endpoint_returns_chat_object():
    """
    Calling chat `completion()` with the FIM provider converts the text
    completion result into a chat-shaped ModelResponse.
    """

    def fake_send(self, request, **kwargs):
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=_fim_response_bytes(),
        )

    with mock.patch("httpx.Client.send", new=fake_send):
        r = litellm.completion(
            model="text-completion-inception/mercury-edit-2",
            messages=[{"role": "user", "content": "def add(a, b): return "}],
            api_key="sk-x",
        )

    assert r.choices[0].message.content == "a + b"


@pytest.mark.asyncio
async def test_inception_fim_async():
    """async FIM path (acompletion) hits Inception's /v1/fim/completions"""

    captured = {}

    async def fake_asend(self, request, **kwargs):
        captured["url"] = str(request.url)
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=_fim_response_bytes(),
        )

    with mock.patch("httpx.AsyncClient.send", new=fake_asend):
        r = await litellm.atext_completion(
            model="text-completion-inception/mercury-edit-2",
            prompt="def add(a, b): return ",
            suffix="\n",
            api_key="sk-x",
            max_tokens=10,
        )

    assert captured["url"] == "https://api.inceptionlabs.ai/v1/fim/completions"
    assert r.choices[0].text == "a + b"


def test_inception_fim_model_configuration():
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.text_completion_inception_models = set()
    litellm.add_known_models()

    assert (
        "text-completion-inception/mercury-edit-2"
        in litellm.text_completion_inception_models
    )
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


def test_inception_fim_does_not_leak_global_api_key():
    """
    Regression: the global litellm.api_key (commonly an OpenAI key) must not be
    forwarded to Inception. Only an Inception-specific key (param,
    litellm.inception_key, or INCEPTION_API_KEY) may be sent to the Inception base.
    """

    captured = {}

    def fake_send(self, request, **kwargs):
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            status_code=200,
            request=request,
            headers={"content-type": "application/json"},
            content=_fim_response_bytes(),
        )

    with mock.patch.dict(
        os.environ, {"INCEPTION_API_KEY": "sk-inception-correct"}, clear=True
    ):
        with mock.patch.object(litellm, "inception_key", None):
            with mock.patch.object(litellm, "api_key", "sk-global-should-not-leak"):
                with mock.patch("httpx.Client.send", new=fake_send):
                    litellm.text_completion(
                        model="text-completion-inception/mercury-edit-2",
                        prompt="def add(a, b): return ",
                        max_tokens=10,
                    )

    assert captured["auth"] == "Bearer sk-inception-correct"


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
