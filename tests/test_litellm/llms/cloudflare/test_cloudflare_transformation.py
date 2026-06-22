import httpx
import pytest

import litellm
from litellm.llms.cloudflare.chat.transformation import (
    CloudflareChatConfig,
    CloudflareChatResponseIterator,
)
from litellm.types.utils import ModelResponse


def test_get_complete_url_encodes_model_path_segment():
    config = CloudflareChatConfig()

    assert (
        config.get_complete_url(
            api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
            api_key="cf-key",
            model="@cf/meta/llama?x=1#frag",
            optional_params={},
            litellm_params={},
        )
        == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/%40cf/meta/llama%3Fx%3D1%23frag"
    )

    with pytest.raises(ValueError, match="dot path segment"):
        config.get_complete_url(
            api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
            api_key="cf-key",
            model="../../accounts/other",
            optional_params={},
            litellm_params={},
        )


def _transform(payload):
    return CloudflareChatConfig().transform_response(
        model="@cf/google/gemma-4-26b-a4b-it",
        raw_response=httpx.Response(200, json=payload),
        model_response=ModelResponse(),
        logging_obj=None,
        request_data={},
        messages=[{"role": "user", "content": "Capital of France?"}],
        optional_params={},
        litellm_params={},
        encoding=litellm.encoding,
    )


def test_transform_response_reasoning_model_reads_choices_and_usage():
    # Reasoning models leave result.response empty and return the answer in
    # choices[].message.content, with chain-of-thought in message.reasoning.
    resp = _transform(
        {
            "result": {
                "choices": [
                    {
                        "message": {
                            "content": "The capital of France is Paris.",
                            "reasoning": "The user asks for the capital of France.",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 43,
                    "total_tokens": 63,
                },
            }
        }
    )
    message = resp.choices[0].message
    assert message.content == "The capital of France is Paris."
    assert message.reasoning_content == "The user asks for the capital of France."
    assert resp.usage.prompt_tokens == 20
    assert resp.usage.completion_tokens == 43
    assert resp.usage.total_tokens == 63


def test_transform_response_legacy_response_field_still_works():
    resp = _transform(
        {
            "result": {
                "response": "The capital of France is Paris.",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            }
        }
    )
    assert resp.choices[0].message.content == "The capital of France is Paris."
    assert resp.usage.completion_tokens == 8


def test_transform_response_response_text_field_still_works():
    # Newer Workers AI models (e.g. Nemotron) return the answer under
    # "response_text" rather than the legacy "response" key, and without an
    # OpenAI-style choices block. Content must fall through to "response_text".
    resp = _transform(
        {
            "result": {
                "response_text": "The capital of France is Paris.",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            }
        }
    )
    assert resp.choices[0].message.content == "The capital of France is Paris."
    assert resp.usage.completion_tokens == 8


def test_transform_response_missing_usage_is_estimated_not_zero():
    # No usage block: the content must survive and tokens must be estimated
    # (non-zero), not reported as 0.
    resp = _transform(
        {
            "result": {
                "choices": [{"message": {"content": "The capital of France is Paris."}}]
            }
        }
    )
    assert resp.choices[0].message.content == "The capital of France is Paris."
    assert resp.usage.completion_tokens > 0
    assert (
        resp.usage.total_tokens
        == resp.usage.prompt_tokens + resp.usage.completion_tokens
    )


def test_transform_response_respects_provider_zero_completion_tokens():
    # A genuine 0 from the provider must be kept (not re-estimated), and the
    # provider total must stay consistent with prompt + completion.
    resp = _transform(
        {
            "result": {
                "choices": [{"message": {"content": ""}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 0,
                    "total_tokens": 20,
                },
            }
        }
    )
    assert resp.usage.prompt_tokens == 20
    assert resp.usage.completion_tokens == 0
    assert resp.usage.total_tokens == 20


def test_chunk_parser_openai_style_delta_and_total_default():
    it = CloudflareChatResponseIterator(streaming_response=iter([]), sync_stream=True)

    content_chunk = it.chunk_parser(
        {
            "choices": [{"delta": {"content": " Paris"}, "index": 0}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
        }
    )
    assert content_chunk["text"] == " Paris"
    # total_tokens defaults to prompt + completion when the provider omits it.
    assert content_chunk["usage"]["total_tokens"] == 50

    reasoning_chunk = it.chunk_parser(
        {"choices": [{"delta": {"reasoning": "Thinking"}, "index": 0}]}
    )
    assert reasoning_chunk["text"] == ""
    assert reasoning_chunk["provider_specific_fields"] == {
        "reasoning_content": "Thinking"
    }


def test_chunk_parser_legacy_response_field_still_works():
    it = CloudflareChatResponseIterator(streaming_response=iter([]), sync_stream=True)
    chunk = it.chunk_parser({"response": "Paris", "index": 0})
    assert chunk["text"] == "Paris"
