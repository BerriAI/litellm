import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.types.utils import HiddenParams


def test_hidden_params_response_ms():
    hidden_params = HiddenParams()
    setattr(hidden_params, "_response_ms", 100)
    hidden_params_dict = hidden_params.model_dump()
    assert hidden_params_dict.get("_response_ms") == 100


def test_chat_completion_delta_tool_call():
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    tool = ChatCompletionDeltaToolCall(
        id="call_m87w",
        function=Function(
            arguments='{"location": "San Francisco", "unit": "imperial"}',
            name="get_current_weather",
        ),
        type="function",
        index=0,
    )

    assert "function" in tool


def test_empty_choices():
    from litellm.types.utils import Choices

    Choices()


def test_usage_dump():
    from litellm.types.utils import (
        CompletionTokensDetailsWrapper,
        PromptTokensDetailsWrapper,
        Usage,
    )

    current_usage = Usage(
        completion_tokens=37,
        prompt_tokens=7,
        total_tokens=44,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=None,
            text_tokens=7,
            image_tokens=None,
            web_search_requests=1,
        ),
        web_search_requests=None,
    )

    assert current_usage.prompt_tokens_details.web_search_requests == 1

    new_usage = Usage(**current_usage.model_dump())
    assert new_usage.prompt_tokens_details.web_search_requests == 1


def test_usage_server_tool_use_dict_is_coerced_and_round_trips():
    from litellm.types.utils import ServerToolUse, Usage

    current_usage = Usage(
        completion_tokens=1,
        prompt_tokens=1,
        total_tokens=2,
        server_tool_use={"web_search_requests": 1},
    )

    assert isinstance(current_usage.server_tool_use, ServerToolUse)
    assert current_usage.server_tool_use.web_search_requests == 1

    new_usage = Usage(**current_usage.model_dump())
    assert isinstance(new_usage.server_tool_use, ServerToolUse)
    assert new_usage.server_tool_use.web_search_requests == 1


def test_usage_converts_server_tool_use_dict():
    from litellm.types.utils import ServerToolUse, Usage

    usage = Usage(
        completion_tokens=2,
        prompt_tokens=1,
        total_tokens=3,
        server_tool_use={"web_search_requests": 4, "tool_search_requests": 1},
    )

    assert isinstance(usage.server_tool_use, ServerToolUse)
    assert usage.server_tool_use.web_search_requests == 4
    assert usage.server_tool_use["web_search_requests"] == 4
    assert usage.server_tool_use.tool_search_requests == 1
    with pytest.raises(KeyError):
        usage.server_tool_use["unknown_metric"]

    round_trip = Usage(**usage.model_dump())
    assert isinstance(round_trip.server_tool_use, ServerToolUse)
    assert round_trip.server_tool_use.web_search_requests == 4
    assert round_trip.server_tool_use["web_search_requests"] == 4
    assert round_trip.server_tool_use.tool_search_requests == 1


def test_usage_completion_tokens_details_text_tokens():
    from litellm.types.utils import Usage

    # Test data from the reported issue
    usage_data = {
        "completion_tokens": 77,
        "prompt_tokens": 11937,
        "total_tokens": 12014,
        "completion_tokens_details": {
            "accepted_prediction_tokens": None,
            "audio_tokens": None,
            "reasoning_tokens": 65,
            "rejected_prediction_tokens": None,
            "text_tokens": 12,
        },
        "prompt_tokens_details": {
            "audio_tokens": None,
            "cached_tokens": None,
            "text_tokens": 11937,
            "image_tokens": None,
        },
    }

    # Create Usage object
    u = Usage(**usage_data)

    # Verify the object has the text_tokens field
    assert hasattr(u.completion_tokens_details, "text_tokens")
    assert u.completion_tokens_details.text_tokens == 12

    # Get model_dump output
    dump_result = u.model_dump()

    # Verify text_tokens is present in the model_dump output
    assert "completion_tokens_details" in dump_result
    assert "text_tokens" in dump_result["completion_tokens_details"]
    assert dump_result["completion_tokens_details"]["text_tokens"] == 12

    # Verify the full completion_tokens_details structure
    expected_completion_details = {
        "accepted_prediction_tokens": None,
        "audio_tokens": None,
        "reasoning_tokens": 65,
        "rejected_prediction_tokens": None,
        "text_tokens": 12,
        "image_tokens": None,
        "video_tokens": None,
    }
    assert dump_result["completion_tokens_details"] == expected_completion_details

    # Verify round-trip serialization works
    new_usage = Usage(**dump_result)
    assert new_usage.completion_tokens_details.text_tokens == 12


def test_chat_completion_token_logprob_null_top_logprobs():
    """
    Test that ChatCompletionTokenLogprob normalizes null top_logprobs to [].

    Some providers return null for top_logprobs when logprobs=true but
    top_logprobs is unset. The OpenAI spec requires top_logprobs to be an array.

    Regression test for https://github.com/BerriAI/litellm/issues/21932
    """
    from litellm.types.utils import ChatCompletionTokenLogprob

    logprob = ChatCompletionTokenLogprob(
        token="Hello",
        bytes=[72, 101, 108, 108, 111],
        logprob=-0.31725305,
        top_logprobs=None,
    )
    assert logprob.top_logprobs == []
    assert isinstance(logprob.top_logprobs, list)


def test_chat_completion_token_logprob_valid_top_logprobs():
    """
    Test that ChatCompletionTokenLogprob still accepts valid top_logprobs arrays.
    """
    from litellm.types.utils import ChatCompletionTokenLogprob, TopLogprob

    logprob = ChatCompletionTokenLogprob(
        token="Hello",
        bytes=[72, 101, 108, 108, 111],
        logprob=-0.31725305,
        top_logprobs=[
            TopLogprob(
                token="Hello", logprob=-0.31725305, bytes=[72, 101, 108, 108, 111]
            ),
            TopLogprob(token="Hi", logprob=-1.3190403, bytes=[72, 105]),
        ],
    )
    assert len(logprob.top_logprobs) == 2
    assert logprob.top_logprobs[0].token == "Hello"


def test_choice_logprobs_with_null_top_logprobs():
    """
    Test that ChoiceLogprobs correctly parses content tokens that have
    null top_logprobs (the full nested parsing path).

    Regression test for https://github.com/BerriAI/litellm/issues/21932
    """
    from litellm.types.utils import ChoiceLogprobs

    logprobs_dict = {
        "content": [
            {
                "token": "Sil",
                "bytes": [83, 105, 108],
                "logprob": -2.1518118381500244,
                "top_logprobs": None,
            },
            {
                "token": "ent",
                "bytes": [101, 110, 116],
                "logprob": -0.13957086205482483,
                "top_logprobs": None,
            },
        ]
    }

    result = ChoiceLogprobs(**logprobs_dict)
    assert result.content is not None
    assert len(result.content) == 2
    for token_logprob in result.content:
        assert token_logprob.top_logprobs == []
        assert isinstance(token_logprob.top_logprobs, list)


def test_chat_completion_token_logprob_invalid_top_logprobs_rejected():
    """
    Test that invalid (non-list, non-null) top_logprobs values are still
    rejected by Pydantic validation. The validator only normalizes null,
    it does not coerce other invalid types.
    """
    from pydantic import ValidationError

    from litellm.types.utils import ChatCompletionTokenLogprob

    with pytest.raises(ValidationError):
        ChatCompletionTokenLogprob(
            token="Hello",
            bytes=[72, 101, 108, 108, 111],
            logprob=-0.31725305,
            top_logprobs="invalid_string",
        )


# ---------------------------------------------------------------------------
# native_finish_reason in provider_specific_fields
# ---------------------------------------------------------------------------


class TestNativeFinishReason:
    """Choices exposes the raw provider finish_reason in provider_specific_fields
    when it differs from the mapped OpenAI-compatible value."""

    def test_provider_reason_exposed_when_mapped(self):
        from litellm.types.utils import Choices

        choice = Choices(finish_reason="end_turn")
        assert choice.finish_reason == "stop"
        assert choice.provider_specific_fields["native_finish_reason"] == "end_turn"

    def test_provider_reason_not_set_when_already_openai(self):
        from litellm.types.utils import Choices

        choice = Choices(finish_reason="stop")
        assert choice.finish_reason == "stop"
        assert not hasattr(choice, "provider_specific_fields")

    def test_provider_reason_merged_with_existing_fields(self):
        from litellm.types.utils import Choices

        choice = Choices(
            finish_reason="max_tokens",
            provider_specific_fields={"citations": [{"url": "http://example.com"}]},
        )
        assert choice.finish_reason == "length"
        assert choice.provider_specific_fields["native_finish_reason"] == "max_tokens"
        assert choice.provider_specific_fields["citations"] == [
            {"url": "http://example.com"}
        ]

    def test_gemini_safety_reason_exposed(self):
        from litellm.types.utils import Choices

        choice = Choices(finish_reason="SAFETY")
        assert choice.finish_reason == "content_filter"
        assert choice.provider_specific_fields["native_finish_reason"] == "SAFETY"

    def test_anthropic_tool_use_reason_exposed(self):
        from litellm.types.utils import Choices

        choice = Choices(finish_reason="tool_use")
        assert choice.finish_reason == "tool_calls"
        assert choice.provider_specific_fields["native_finish_reason"] == "tool_use"

    def test_max_tokens_reason_exposed(self):
        from litellm.types.utils import Choices

        choice = Choices(finish_reason="MAX_TOKENS")
        assert choice.finish_reason == "length"
        assert choice.provider_specific_fields["native_finish_reason"] == "MAX_TOKENS"


def test_parallel_request_limiter_internal_fields_in_all_litellm_params():
    """
    Regression test: internal fields written by parallel_request_limiter_v3 must
    be in all_litellm_params so they are stripped before forwarding to upstream
    providers.  If missing, they are sent as extra body parameters and providers
    like OpenAI reject the request with a 400 invalid_request_error.
    """
    from litellm.types.utils import all_litellm_params

    internal_fields = [
        "_litellm_rate_limit_descriptors",
        "_litellm_tpm_reserved_tokens",
        "_litellm_tpm_reserved_model",
        "_litellm_tpm_reserved_scopes",
        "_litellm_tpm_reservation_released",
    ]
    for field in internal_fields:
        assert field in all_litellm_params, (
            f"{field!r} is not in all_litellm_params. "
            "It will be forwarded to upstream providers and cause 400 errors."
        )


def test_delta_maps_reasoning_to_reasoning_content():
    """
    Test that Delta maps 'reasoning' field to 'reasoning_content'.

    Providers like Cerebras and Groq return delta.reasoning for gpt-oss models,
    but LiteLLM expects delta.reasoning_content.
    """
    from litellm.types.utils import Delta

    # When provider sends 'reasoning' (e.g., Cerebras gpt-oss streaming)
    delta = Delta(content=None, role="assistant", reasoning="thinking step by step")
    assert delta.reasoning_content == "thinking step by step"
    assert not hasattr(
        delta, "reasoning"
    ), "reasoning should not leak as an extra attribute"

    # When provider sends 'reasoning_content' directly (e.g., NIM), it still works
    delta2 = Delta(content="hello", reasoning_content="direct reasoning")
    assert delta2.reasoning_content == "direct reasoning"

    # When both are present, reasoning_content takes precedence
    delta3 = Delta(reasoning_content="from_rc", reasoning="from_r")
    assert delta3.reasoning_content == "from_rc"

    # When neither is present, reasoning_content is not set (OpenAI spec)
    delta4 = Delta(content="hello")
    assert not hasattr(delta4, "reasoning_content")


def test_message_accepts_thinking_block_with_null_signature():
    """Open-source reasoning models (DeepSeek-R1, Qwen, etc.) emit thinking blocks
    without an Anthropic-style signature. Message must accept signature=None so the
    success-logging handler can build the StandardLoggingObject instead of silently
    dropping the log record. Regression for LIT-4007.
    """
    from litellm.types.utils import Choices, Message

    thinking_blocks = [
        {"type": "thinking", "thinking": "step by step reasoning", "signature": None}
    ]

    message = Message(
        content="the answer is 4", role="assistant", thinking_blocks=thinking_blocks
    )
    assert message.thinking_blocks is not None
    assert message.thinking_blocks[0]["signature"] is None
    assert message.thinking_blocks[0]["thinking"] == "step by step reasoning"

    validated = Message.model_validate(
        {
            "role": "assistant",
            "content": "the answer is 4",
            "thinking_blocks": thinking_blocks,
        }
    )
    dumped = validated.model_dump()
    assert dumped["thinking_blocks"][0]["signature"] is None
    assert dumped["thinking_blocks"][0]["thinking"] == "step by step reasoning"

    choice = Choices.model_validate(
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "the answer is 4",
                "thinking_blocks": thinking_blocks,
            },
        }
    )
    assert choice.message.thinking_blocks is not None
    assert choice.message.thinking_blocks[0]["signature"] is None


def test_delta_serialization_contract():
    """
    Lock the exact per-chunk serialization shape that the streaming path emits.

    Delta is built once per streaming chunk and serialized via
    ModelResponseStream.model_dump(), which defaults to exclude_unset=True.
    The construction therefore has to mark content/role/function_call/
    tool_calls/audio as "set" (so they survive exclude_unset) while keeping
    OpenAI-omitted fields (reasoning_content, thinking_blocks, reasoning_items,
    images, annotations) absent unless explicitly provided. This guards that
    contract for both the default dump and the exclude_unset dump.
    """
    from litellm.types.utils import Delta

    base_keys = {"content", "role", "function_call", "tool_calls", "audio"}

    # Plain content delta: only the OpenAI-compatible keys appear, nothing extra
    delta = Delta(content="hi", role="assistant")
    assert set(delta.model_dump(exclude_unset=True).keys()) == base_keys
    assert set(delta.model_dump().keys()) == base_keys | {"provider_specific_fields"}
    assert delta.model_dump(exclude_unset=True) == {
        "content": "hi",
        "role": "assistant",
        "function_call": None,
        "tool_calls": None,
        "audio": None,
    }

    # Empty delta still emits the base keys (used for the trailing chunk)
    assert set(Delta().model_dump(exclude_unset=True).keys()) == base_keys

    # model_fields_set is part of the contract. The legacy setattr-then-delattr
    # path marked content/role/function_call/tool_calls/audio/images/annotations
    # as set (pydantic's __delattr__ does not clear __pydantic_fields_set__), so
    # images/annotations remain in model_fields_set even though they are omitted
    # from the dump when absent. Lock that exact set so a pydantic change to
    # fields_set handling fails here rather than silently shifting the contract.
    expected_fields_set = base_keys | {"images", "annotations"}
    assert Delta(content="hi", role="assistant").model_fields_set == expected_fields_set
    assert Delta().model_fields_set == expected_fields_set
    assert (
        Delta(
            content="x",
            images=[{"type": "image_url", "image_url": {"url": "http://x"}}],
        ).model_fields_set
        == expected_fields_set
    )

    # Optional fields only show up when provided
    for kwargs, expected_extra in [
        ({"reasoning_content": "t"}, "reasoning_content"),
        (
            {
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "a", "signature": "s"}
                ]
            },
            "thinking_blocks",
        ),
        ({"reasoning_items": []}, "reasoning_items"),
        (
            {"images": [{"type": "image_url", "image_url": {"url": "http://x"}}]},
            "images",
        ),
        (
            {
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "start_index": 0,
                            "end_index": 1,
                            "title": "t",
                            "url": "u",
                        },
                    }
                ]
            },
            "annotations",
        ),
    ]:
        present = Delta(content="x", **kwargs)
        assert expected_extra in present.model_dump(exclude_unset=True)
        absent = Delta(content="x")
        assert expected_extra not in absent.model_dump(exclude_unset=True)
        assert not hasattr(absent, expected_extra)

    # tool_calls dicts are coerced and back-filled with index/type
    tc_delta = Delta(
        tool_calls=[{"id": "1", "function": {"name": "f", "arguments": "{}"}}]
    )
    dumped = tc_delta.model_dump(exclude_unset=True)["tool_calls"]
    assert dumped == [
        {
            "id": "1",
            "function": {"arguments": "{}", "name": "f"},
            "type": "function",
            "index": 0,
        }
    ]

    # Extra provider params survive (extra='allow') and, because super().__init__
    # populates them before the base keys are appended, order ahead of "content".
    extra_delta = Delta(content="x", custom_field="v")
    extra_dump = extra_delta.model_dump(exclude_unset=True)
    keys = list(extra_dump.keys())
    assert extra_dump["custom_field"] == "v"
    assert keys.index("custom_field") < keys.index("content")
