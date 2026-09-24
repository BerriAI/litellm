"""
Unit tests for keeping system messages at the front of the bridged chat request.

``instructions`` becomes a leading system message and the Responses input can carry
a system message of its own, so an Anthropic ``/v1/messages`` conversation bridged
to chat completions could come out as system, user, system. Chat templates that
require the system message to come first reject that with
"System message must be at the beginning".

See: https://github.com/BerriAI/litellm/issues/40693
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _roles(input, responses_api_request):
    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input, responses_api_request=responses_api_request
    )
    return [message.get("role") for message in messages], messages


def test_system_message_after_user_content_is_hoisted():
    roles, messages = _roles(
        [
            {"role": "user", "content": "first question"},
            {"role": "system", "content": "mid"},
            {"role": "user", "content": "second question"},
        ],
        {"instructions": "lead"},
    )

    assert roles == ["system", "user", "user"]
    assert "system" not in roles[1:]


def test_both_system_prompts_survive_the_merge():
    _, messages = _roles(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": "mid"},
        ],
        {"instructions": "lead"},
    )

    assert messages[0]["content"] == "lead\n\nmid"


def test_part_list_content_collapses_to_its_text():
    """A system message given as parts still contributes its text to the merged prompt."""
    _, messages = _roles(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": [{"type": "text", "text": "B"}]},
        ],
        {"instructions": "A"},
    )

    assert messages[0]["content"] == "A\n\nB"


def test_an_already_leading_system_message_is_left_alone():
    """The common case must not be rewritten."""
    roles, messages = _roles([{"role": "user", "content": "q"}], {"instructions": "lead"})

    assert roles == ["system", "user"]
    assert messages[0]["content"] == "lead"


def test_a_conversation_without_a_system_message_is_unchanged():
    roles, _ = _roles([{"role": "user", "content": "q"}], {})

    assert roles == ["user"]


def test_bare_strings_in_a_part_list_survive_the_merge():
    """Upstream normalization leaves plain strings in a content list, so filtering the
    list down to dicts silently dropped part of the prompt."""
    _, messages = _roles(
        [
            {"role": "user", "content": "q"},
            {"role": "system", "content": ["B", {"type": "text", "text": "C"}]},
        ],
        {"instructions": "A"},
    )

    assert messages[0]["content"] == "A\n\nB\n\nC"
