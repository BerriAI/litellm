"""Unit tests for Triton /generate request transformation.

These live under tests/test_litellm/ (rather than tests/llm_translation/) so CI
actually runs them and reports coverage for the transformation module.
"""

import json

from litellm.llms.triton.completion.transformation import TritonGenerateConfig


def test_triton_generate_request_serializes_dict_params():
    """Triton's `parameters` field only accepts int/bool/string values.

    Nested dict params (e.g. chat_template_kwargs) must be JSON-encoded rather
    than forwarded as raw objects, otherwise Triton rejects the request.
    """
    config = TritonGenerateConfig()
    data_for_triton = config.transform_request(
        model="triton/qwen3.6-27b",
        messages=[{"role": "user", "content": "test?"}],
        optional_params={
            "max_tokens": 10,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        litellm_params={},
        headers={},
    )

    assert data_for_triton["parameters"]["max_tokens"] == 10
    assert isinstance(data_for_triton["parameters"]["max_tokens"], int)
    assert data_for_triton["parameters"]["temperature"] == 0.7
    assert isinstance(data_for_triton["parameters"]["chat_template_kwargs"], str)
    assert json.loads(data_for_triton["parameters"]["chat_template_kwargs"]) == {"enable_thinking": False}


def test_triton_generate_request_serializes_tuple_params():
    """Tuples must be JSON-encoded too (Triton only accepts int/bool/string)."""
    config = TritonGenerateConfig()
    data_for_triton = config.transform_request(
        model="triton/qwen3.6-27b",
        messages=[{"role": "user", "content": "test?"}],
        optional_params={"max_tokens": 10, "stop_sequences": ("foo", "bar")},
        litellm_params={},
        headers={},
    )

    assert isinstance(data_for_triton["parameters"]["stop_sequences"], str)
    assert json.loads(data_for_triton["parameters"]["stop_sequences"]) == ["foo", "bar"]


def test_triton_generate_request_max_tokens_always_int():
    """max_tokens must stay an int in the payload.

    The explicit int() cast must not be silently overwritten by the loop that
    copies the remaining optional_params into `parameters`.
    """
    config = TritonGenerateConfig()
    data_for_triton = config.transform_request(
        model="triton/llama-3-8b-instruct",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 10},
        litellm_params={},
        headers={},
    )

    assert data_for_triton["parameters"]["max_tokens"] == 10
    assert isinstance(data_for_triton["parameters"]["max_tokens"], int), (
        "max_tokens must be int; the param loop must not overwrite the int() cast"
    )


def test_triton_generate_request_max_completion_tokens_fallback():
    """max_completion_tokens is used when max_tokens is absent, and is not
    forwarded as a separate Triton parameter."""
    config = TritonGenerateConfig()
    data_for_triton = config.transform_request(
        model="triton/llama-3-8b-instruct",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_completion_tokens": 20},
        litellm_params={},
        headers={},
    )

    assert data_for_triton["parameters"]["max_tokens"] == 20
    assert isinstance(data_for_triton["parameters"]["max_tokens"], int)
    assert "max_completion_tokens" not in data_for_triton["parameters"]
