"""
Regression tests for model_dump_with_preserved_fields.

This function serializes ModelResponse / ModelResponseStream objects to dicts
while preserving 3 specific None fields for OpenAI API compatibility:
  - choices[*].message.content  (null when tool_calls present)
  - choices[*].message.role     (always present)
  - choices[*].delta.content    (null in streaming chunks)
"""

from litellm.proxy.utils import model_dump_with_preserved_fields
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


def test_message_content_null_preserved_with_tool_calls():
    """content: null must be kept when tool_calls are present (issue #6677)."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                ),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    msg = result["choices"][0]["message"]
    assert msg["content"] is None
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"


def test_message_role_always_preserved():
    """role must always appear in the serialized message."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Hello", role="assistant"),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"


def test_delta_content_null_preserved():
    """delta.content: null must be preserved in streaming choices."""
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content=None, role="assistant"),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    delta = result["choices"][0]["delta"]
    assert delta["content"] is None
    assert delta["role"] == "assistant"


def test_delta_empty_preserves_content_null():
    """Default Delta() should still have content: null in output."""
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    delta = result["choices"][0]["delta"]
    assert "content" in delta
    assert delta["content"] is None


def test_none_fields_stripped_from_message():
    """function_call, tool_calls, audio etc. should be omitted when None."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Hello", role="assistant"),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    msg = result["choices"][0]["message"]
    assert "function_call" not in msg
    assert "tool_calls" not in msg
    assert "audio" not in msg


def test_none_fields_stripped_from_top_level():
    """system_fingerprint=None should be omitted from top-level."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Hello", role="assistant"),
            )
        ],
        system_fingerprint=None,
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    assert "system_fingerprint" not in result


def test_multiple_choices_independent():
    """Mixed content/null across multiple choices must be handled independently."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Hello", role="assistant"),
            ),
            Choices(
                finish_reason="tool_calls",
                index=1,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_456",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                        }
                    ],
                ),
            ),
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    assert result["choices"][0]["message"]["content"] == "Hello"
    assert result["choices"][1]["message"]["content"] is None
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][1]["message"]["role"] == "assistant"


def test_content_empty_string_not_stripped():
    """Empty string '' is not None and must be kept as-is."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="", role="assistant"),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    assert result["choices"][0]["message"]["content"] == ""


def test_multiple_tool_calls():
    """Parallel tool calls scenario from issue #6677."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"NYC"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": '{"tz":"EST"}',
                            },
                        },
                    ],
                ),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)
    msg = result["choices"][0]["message"]
    assert msg["content"] is None
    assert len(msg["tool_calls"]) == 2
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert msg["tool_calls"][1]["function"]["name"] == "get_time"


def test_full_output_structure_non_streaming():
    """
    Snapshot test: verify the complete dict shape for a non-streaming response.

    Catches any field that behaves differently between exclude_none=False (old)
    and exclude_none=True (new) that we didn't account for.
    """
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="Hello!", role="assistant"),
            )
        ],
        model="gpt-4.1",
        system_fingerprint="fp_abc123",
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)

    # Top-level keys
    assert set(result.keys()) == {
        "id",
        "choices",
        "created",
        "model",
        "object",
        "system_fingerprint",
        "usage",
    }
    assert result["object"] == "chat.completion"
    assert result["model"] == "gpt-4.1"
    assert result["system_fingerprint"] == "fp_abc123"
    assert isinstance(result["id"], str)
    assert isinstance(result["created"], int)

    # Choice structure
    choice = result["choices"][0]
    assert set(choice.keys()) == {"finish_reason", "index", "message"}
    assert choice["finish_reason"] == "stop"
    assert choice["index"] == 0

    # Message structure — only content and role, nothing else
    msg = choice["message"]
    assert set(msg.keys()) == {"content", "role"}
    assert msg["content"] == "Hello!"
    assert msg["role"] == "assistant"

    # Usage structure
    usage = result["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def test_full_output_structure_tool_calls():
    """
    Snapshot test: verify complete dict shape for a tool_calls response.

    The critical case — content must be null (not absent), tool_calls must
    be fully serialized, and no extra None fields should leak through.
    """
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC"}',
                            },
                        }
                    ],
                ),
            )
        ],
        model="gpt-4.1",
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)

    msg = result["choices"][0]["message"]
    # Must have exactly content, role, and tool_calls — no function_call, audio, etc.
    assert set(msg.keys()) == {"content", "role", "tool_calls"}
    assert msg["content"] is None
    assert msg["role"] == "assistant"

    tc = msg["tool_calls"][0]
    assert set(tc.keys()) == {"id", "type", "function"}
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "get_weather"


def test_full_output_structure_streaming():
    """
    Snapshot test: verify complete dict shape for a streaming chunk.

    Delta content must be null (not absent), and no extra None fields
    from Delta's dynamic attributes should leak through.
    """
    response = ModelResponseStream(
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content=None, role="assistant"),
            )
        ],
    )
    result = model_dump_with_preserved_fields(response, exclude_unset=True)

    assert result["object"] == "chat.completion.chunk"

    choice = result["choices"][0]
    # finish_reason is None so it gets stripped by exclude_none=True
    assert "finish_reason" not in choice or choice["finish_reason"] is None
    assert choice["index"] == 0

    delta = choice["delta"]
    # Only content and role — no tool_calls, function_call, audio, etc.
    assert set(delta.keys()) == {"content", "role"}
    assert delta["content"] is None
    assert delta["role"] == "assistant"


def test_delta_dynamic_attributes_in_model_dump():
    """
    Verifies Delta's dynamically-set content/role appear in model_dump().

    Delta sets content and role via self.content / self.role (not as declared
    Pydantic fields), so this is a regression guard ensuring they survive
    model_dump(exclude_none=True).
    """
    delta = Delta(content="hello", role="assistant")
    dump = delta.model_dump(exclude_none=True)
    assert dump.get("content") == "hello"
    assert dump.get("role") == "assistant"

    # Also verify None content is excluded by exclude_none=True
    delta_none = Delta(content=None, role="assistant")
    dump_none = delta_none.model_dump(exclude_none=True)
    # content=None should be excluded
    assert "content" not in dump_none
    # role=None should also be excluded
    delta_no_role = Delta(content=None, role=None)
    dump_no_role = delta_no_role.model_dump(exclude_none=True)
    assert "role" not in dump_no_role


def test_preserve_fields_param_backward_compat():
    """preserve_fields parameter is accepted (deprecated) without error."""
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                ),
            )
        ],
    )
    result_default = model_dump_with_preserved_fields(response, exclude_unset=True)
    result_explicit = model_dump_with_preserved_fields(
        response,
        preserve_fields=[
            "choices.*.message.content",
            "choices.*.message.role",
            "choices.*.delta.content",
        ],
        exclude_unset=True,
    )
    assert result_default == result_explicit
    assert result_default["choices"][0]["message"]["content"] is None
    assert result_default["choices"][0]["message"]["role"] == "assistant"
