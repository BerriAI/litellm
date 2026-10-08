"""
Token-ID prompts must reach the native text completion handler for every
provider in `openai_text_completion_compatible_providers` (e.g. hosted_vllm),
not just the handful named in `text_completion`. Covers issue #45253.
"""

from unittest.mock import patch

import pytest

import litellm
from litellm import text_completion

TOKEN_IDS = [151644, 872, 198, 14990, 151645]


def _fake_completion_response(model: str):
    return litellm.TextCompletionResponse(
        id="chatcmpl-test",
        created=0,
        model=model,
        object="text_completion",
        choices=[
            {
                "text": "ok",
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    )


@pytest.mark.parametrize(
    "model",
    [
        "hosted_vllm/Qwen/Qwen3-8B",
        "together_ai/Qwen/Qwen3-8B",
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct",
    ],
)
def test_text_completion_token_ids_openai_compatible_providers(model):
    captured = {}

    def fake_complete(ctx):
        captured["messages"] = ctx.messages
        captured["custom_llm_provider"] = ctx.custom_llm_provider
        return _fake_completion_response(model)

    with patch("litellm.main._complete_text_completion_openai", side_effect=fake_complete):
        text_completion(
            model=model,
            prompt=TOKEN_IDS,
            api_base="http://127.0.0.1:18000/v1",
            api_key="EMPTY",
            max_tokens=4,
        )

    assert captured["messages"] == [{"role": "user", "content": TOKEN_IDS}]


def test_text_completion_token_ids_unsupported_provider_raises():
    with pytest.raises(Exception, match="Unmapped prompt format"):
        text_completion(
            model="anthropic/claude-3-5-sonnet-20241022",
            prompt=TOKEN_IDS,
            api_key="sk-ant-fake",
            max_tokens=4,
        )
