"""
Unit tests for Triton /generate and /infer request transformation.

Regression tests for the chat_template_kwargs leak: chat_template_kwargs is a
LiteLLM-level param consumed while rendering the prompt; TritonSamplingParams
has no such argument, so forwarding it to the server made requests fail.
"""

from unittest.mock import patch

from litellm.llms.triton.completion.transformation import (
    TritonGenerateConfig,
    TritonInferConfig,
)


def test_generate_chat_template_kwargs_consumed_not_forwarded():
    """chat_template_kwargs must reach prompt_factory and never the parameters dict."""
    config = TritonGenerateConfig()
    with patch(
        "litellm.llms.triton.completion.transformation.prompt_factory",
        return_value="rendered-prompt",
    ) as mock_pf:
        data = config.transform_request(
            model="triton/qwen3",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={
                "max_tokens": 10,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            litellm_params={},
            headers={},
        )

    mock_pf.assert_called_once()
    assert mock_pf.call_args.kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert data["text_input"] == "rendered-prompt"
    # must not leak into the Triton parameters payload
    assert "chat_template_kwargs" not in data["parameters"]
    assert data["parameters"]["max_tokens"] == 10


def test_generate_no_chat_template_kwargs_defaults_to_empty_mapping():
    """Absent chat_template_kwargs must pass an empty mapping to prompt_factory."""
    config = TritonGenerateConfig()
    with patch(
        "litellm.llms.triton.completion.transformation.prompt_factory",
        return_value="p",
    ) as mock_pf:
        data = config.transform_request(
            model="triton/llama-3-8b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"max_tokens": 5},
            litellm_params={},
            headers={},
        )

    assert mock_pf.call_args.kwargs["chat_template_kwargs"] == {}
    assert "chat_template_kwargs" not in data["parameters"]


def test_generate_none_chat_template_kwargs_normalized_to_empty_mapping():
    """chat_template_kwargs=None must be normalized, not passed as None."""
    config = TritonGenerateConfig()
    with patch(
        "litellm.llms.triton.completion.transformation.prompt_factory",
        return_value="p",
    ) as mock_pf:
        config.transform_request(
            model="triton/llama-3-8b-instruct",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"max_tokens": 5, "chat_template_kwargs": None},
            litellm_params={},
            headers={},
        )

    assert mock_pf.call_args.kwargs["chat_template_kwargs"] == {}


def test_infer_chat_template_kwargs_excluded_from_inputs():
    """/infer inputs must skip stream, max_retries, and chat_template_kwargs."""
    config = TritonInferConfig()
    data = config.transform_request(
        model="triton/custom-model",
        messages=[{"role": "user", "content": "text in"}],
        optional_params={
            "temperature": 0.5,
            "stream": False,
            "max_retries": 3,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        litellm_params={},
        headers={},
    )

    input_names = [i["name"] for i in data["inputs"]]
    assert "chat_template_kwargs" not in input_names
    assert "stream" not in input_names
    assert "max_retries" not in input_names
    assert "temperature" in input_names
