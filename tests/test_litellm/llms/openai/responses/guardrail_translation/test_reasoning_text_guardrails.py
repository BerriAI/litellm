"""Reasoning content parts must stay visible to output guardrails.

Before `reasoning_text` existed, a reasoning item's parts were emitted as
`output_text`, so the guardrail handler's `isinstance(..., OutputText)` checks
saw them. Introducing `OutputReasoningText` would have silently removed
reasoning text from guardrail inspection and rewriting, which fails open: the
guardrail keeps reporting success while no longer reading the content.
"""

from litellm.llms.openai.responses.guardrail_translation.handler import (
    GUARDRAIL_TEXT_CONTENT_PARTS,
    OpenAIResponsesHandler,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    OutputReasoningText,
    OutputText,
)


def _reasoning_item(text: str) -> GenericResponseOutputItem:
    return GenericResponseOutputItem(
        type="reasoning",
        id="rs_1",
        status="completed",
        role="assistant",
        content=[OutputReasoningText(type="reasoning_text", text=text)],
    )


def _message_item(text: str) -> GenericResponseOutputItem:
    return GenericResponseOutputItem(
        type="message",
        id="msg_1",
        status="completed",
        role="assistant",
        content=[OutputText(type="output_text", text=text, annotations=None)],
    )


def test_both_content_part_types_are_inspected():
    assert OutputText in GUARDRAIL_TEXT_CONTENT_PARTS
    assert OutputReasoningText in GUARDRAIL_TEXT_CONTENT_PARTS


def test_reasoning_text_is_extracted_for_checking():
    handler = OpenAIResponsesHandler()
    texts: list[str] = []
    images: list[str] = []
    mappings: list[tuple[int, int]] = []

    handler._extract_output_text_and_images(
        _reasoning_item("secret chain of thought"), 0, texts, images, mappings
    )

    assert texts == ["secret chain of thought"]
    assert mappings == [(0, 0)]


def test_message_text_is_still_extracted():
    handler = OpenAIResponsesHandler()
    texts: list[str] = []
    images: list[str] = []
    mappings: list[tuple[int, int]] = []

    handler._extract_output_text_and_images(
        _message_item("the answer"), 0, texts, images, mappings
    )

    assert texts == ["the answer"]
