"""
Tests for the Interactions -> Responses input bridge.

Covers both conversation shapes: the step form Google's spec uses today and the older
role-tagged turns it replaced.
"""

from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)
from litellm.types.interactions import ModelOutputStep, TextContent, Turn, UserInputStep

transform_input = LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input


def _roles(messages) -> list[str]:
    return [message["role"] for message in messages]


def _texts(messages) -> list[str]:
    return [part["text"] for message in messages for part in message["content"]]


class TestStepInput:
    """A conversation sent as steps, the shape Google's current spec requires."""

    def test_step_types_decide_the_role(self):
        messages = transform_input(
            [
                {"type": "user_input", "content": [{"type": "text", "text": "My name is Alice."}]},
                {"type": "model_output", "content": [{"type": "text", "text": "Hello Alice."}]},
                {"type": "user_input", "content": [{"type": "text", "text": "What is my name?"}]},
            ]
        )

        assert _roles(messages) == ["user", "assistant", "user"]
        assert _texts(messages) == ["My name is Alice.", "Hello Alice.", "What is my name?"]

    def test_typed_steps_behave_like_their_dicts(self):
        messages = transform_input(
            [
                UserInputStep(content=[TextContent(type="text", text="My name is Alice.")]),
                ModelOutputStep(content=[TextContent(type="text", text="Hello Alice.")]),
            ]
        )

        assert _roles(messages) == ["user", "assistant"]
        assert _texts(messages) == ["My name is Alice.", "Hello Alice."]


class TestLegacyTurnInput:
    """The role-tagged turns Google dropped still have to keep working."""

    def test_model_turns_become_assistant_messages(self):
        messages = transform_input(
            [
                {"role": "user", "content": [{"type": "text", "text": "My name is Alice."}]},
                {"role": "model", "content": [{"type": "text", "text": "Hello Alice."}]},
            ]
        )

        assert _roles(messages) == ["user", "assistant"]

    def test_typed_turns_behave_like_their_dicts(self):
        messages = transform_input(
            [
                Turn(role="user", content=[TextContent(type="text", text="My name is Alice.")]),
                Turn(role="model", content=[TextContent(type="text", text="Hello Alice.")]),
            ]
        )

        assert _roles(messages) == ["user", "assistant"]
        assert _texts(messages) == ["My name is Alice.", "Hello Alice."]

    def test_a_turn_without_a_role_is_the_user_speaking(self):
        messages = transform_input([{"content": [{"type": "text", "text": "hi"}]}])

        assert _roles(messages) == ["user"]

    def test_string_content_is_wrapped_as_text(self):
        messages = transform_input([Turn(role="user", content="hi")])

        assert _roles(messages) == ["user"]
        assert _texts(messages) == ["hi"]


class TestOtherInputShapes:
    def test_a_bare_string_passes_straight_through(self):
        assert transform_input("Hello") == "Hello"

    def test_a_single_content_object_becomes_one_user_message(self):
        messages = transform_input({"type": "text", "text": "Hello"})

        assert _roles(messages) == ["user"]
        assert _texts(messages) == ["Hello"]
