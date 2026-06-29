"""
Token usage on synthetic guardrail-blocked responses for the OpenAI-format
proxy endpoints (/v1/chat/completions and /v1/completions).

The ModifyResponseException handlers in proxy_server.py previously hardcoded
zero usage; _blocked_response_usage computes real counts (input from the
request, output from the block message). These tests mock litellm.token_counter
so they assert the wiring/arithmetic without invoking a real tokenizer.
"""

from unittest.mock import patch

from litellm.proxy.proxy_server import _blocked_response_usage

BLOCK_MESSAGE = "The response was blocked by Rubrik Agent Cloud (Reference ID: abc)"


def test_chat_usage_counts_messages_tools_and_block_message():
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "f"}}]

    def fake_counter(*, model, messages=None, text=None, tools=None):
        # Distinct, deterministic values for the input vs output calls.
        return 40 if messages is not None else 8

    with patch("litellm.token_counter", side_effect=fake_counter) as mock_counter:
        usage = _blocked_response_usage(
            "gpt-4o", BLOCK_MESSAGE, messages=messages, tools=tools
        )

    assert usage.prompt_tokens == 40
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 48
    # Tools must be forwarded into the prompt-token count.
    mock_counter.assert_any_call(model="gpt-4o", messages=messages, tools=tools)


def test_text_completion_usage_counts_prompt_and_block_message():
    def fake_counter(*, model, text=None, messages=None, tools=None):
        return 5 if text == "my prompt" else 3

    with patch("litellm.token_counter", side_effect=fake_counter):
        usage = _blocked_response_usage("gpt-4o", BLOCK_MESSAGE, prompt="my prompt")

    assert usage.prompt_tokens == 5
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 8


def test_usage_falls_back_to_zero_on_error():
    with patch("litellm.token_counter", side_effect=RuntimeError("boom")):
        usage = _blocked_response_usage(
            "gpt-4o", BLOCK_MESSAGE, messages=[{"role": "user", "content": "hi"}]
        )

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
