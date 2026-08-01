"""
Unit tests for Triton request transformation.

Covers the parameter handling for the Triton `/generate` endpoint:
- Standard sampling params (temperature, top_p, presence_penalty,
  frequency_penalty) are forwarded instead of being dropped / rejected.
- `max_completion_tokens` is normalized to Triton's `max_tokens`.
- Provider-specific passthrough params (e.g. `chat_template_kwargs`, `top_k`)
  still reach the request body.

Related issue: https://github.com/BerriAI/litellm/issues/31092
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.triton.completion.transformation import (
    TritonConfig,
    TritonGenerateConfig,
)
from litellm.utils import get_optional_params


def test_get_supported_openai_params_includes_sampling_params():
    """Triton should advertise support for the common sampling params."""
    supported = TritonConfig().get_supported_openai_params(model="any-triton-model")
    for param in [
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ]:
        assert param in supported


def test_map_openai_params_forwards_sampling_params():
    """Sampling params should be mapped 1:1 into optional_params."""
    non_default_params = {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "frequency_penalty": 0.5,
    }
    optional_params = TritonConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="any-triton-model",
        drop_params=False,
    )
    assert optional_params == non_default_params


def test_map_openai_params_normalizes_max_completion_tokens():
    """`max_completion_tokens` should be normalized to Triton's `max_tokens`."""
    optional_params = TritonConfig().map_openai_params(
        non_default_params={"max_completion_tokens": 42},
        optional_params={},
        model="any-triton-model",
        drop_params=False,
    )
    assert optional_params == {"max_tokens": 42}


def test_get_optional_params_does_not_drop_sampling_params():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/31092

    Standard sampling params used to raise `UnsupportedParamsError` (or be
    silently dropped with `drop_params=True`) because Triton only advertised
    `max_tokens`. They should now flow through to optional_params.
    """
    optional_params = get_optional_params(
        model="any-triton-model",
        custom_llm_provider="triton",
        max_tokens=100,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        frequency_penalty=0.5,
    )
    assert optional_params["temperature"] == 0.7
    assert optional_params["top_p"] == 0.8
    assert optional_params["presence_penalty"] == 1.5
    assert optional_params["frequency_penalty"] == 0.5
    assert optional_params["max_tokens"] == 100


def test_generate_transform_request_includes_sampling_params():
    """The `/generate` request body should carry the sampling params."""
    optional_params = {
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "frequency_penalty": 0.5,
    }
    body = TritonGenerateConfig().transform_request(
        model="any-triton-model",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={"api_base": "http://localhost:8000/generate"},
        headers={},
    )
    parameters = body["parameters"]
    assert parameters["max_tokens"] == 100
    assert parameters["temperature"] == 0.7
    assert parameters["top_p"] == 0.8
    assert parameters["presence_penalty"] == 1.5
    assert parameters["frequency_penalty"] == 0.5


def test_generate_transform_request_forwards_chat_template_kwargs():
    """
    Provider-specific passthrough params (e.g. `chat_template_kwargs` to control
    `enable_thinking`, or `top_k`) must reach the Triton request body so the
    backend can act on them. See issue #31092.
    """
    optional_params = {
        "max_tokens": 100,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = TritonGenerateConfig().transform_request(
        model="any-triton-model",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={"api_base": "http://localhost:8000/generate"},
        headers={},
    )
    parameters = body["parameters"]
    assert parameters["top_k"] == 20
    assert parameters["chat_template_kwargs"] == {"enable_thinking": False}


def test_completion_triton_generate_forwards_sampling_params_end_to_end():
    """
    End-to-end check (mocked HTTP): sampling params provided to
    `litellm.completion` reach the Triton `/generate` request body.
    """
    mock_response = MagicMock()
    mock_response.json = lambda: {"text_output": "hello"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        litellm.completion(
            model="triton/llama-3-8b-instruct",
            messages=[{"role": "user", "content": "who are u?"}],
            max_tokens=10,
            temperature=0.3,
            top_p=0.9,
            presence_penalty=0.5,
            frequency_penalty=0.2,
            api_base="http://localhost:8000/generate",
        )

        mock_post.assert_called_once()
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        parameters = request_data["parameters"]
        assert parameters["max_tokens"] == 10
        assert parameters["temperature"] == 0.3
        assert parameters["top_p"] == 0.9
        assert parameters["presence_penalty"] == 0.5
        assert parameters["frequency_penalty"] == 0.2
