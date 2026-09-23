"""
Unit tests for the OpenAI chat completions guardrail translation handler's
attachment extraction: image-only and file-only turns must still reach
apply_guardrail so a guardrail can scan or refuse them.
"""

from typing import Any, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from litellm.types.utils import GenericGuardrailAPIInputs


class RecordingGuardrail(CustomGuardrail):
    """Records the inputs each apply_guardrail call was handed."""

    def __init__(self, guardrail_name: str = "recording"):
        super().__init__(guardrail_name=guardrail_name)
        self.calls = 0
        self.inputs: GenericGuardrailAPIInputs | None = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.calls += 1
        self.inputs = inputs.copy()
        return inputs


class ScanningGuardrail(RecordingGuardrail):
    """A guardrail that opted into attachment scanning, like Bedrock."""

    scans_attachments = True


class TestOpenAIChatHandlerAttachments:
    _PNG_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="

    @pytest.mark.asyncio
    async def test_image_only_message_invokes_guardrail_with_images(self):
        """An image-only turn used to skip apply_guardrail entirely."""
        guardrail = ScanningGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": self._PNG_DATA_URI}}],
                }
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1, "image-only turn never reached apply_guardrail"
        assert guardrail.inputs is not None
        assert guardrail.inputs["images"] == [self._PNG_DATA_URI]

    @pytest.mark.asyncio
    async def test_file_part_reaches_guardrail_as_files(self):
        guardrail = ScanningGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "summarize this document"},
                        {
                            "type": "file",
                            "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AAAA"},
                        },
                    ],
                }
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["files"] == ["data:application/pdf;base64,AAAA"]

    @pytest.mark.asyncio
    async def test_file_only_message_invokes_guardrail(self):
        """A turn carrying only a provider file id still must reach the guardrail."""
        guardrail = ScanningGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "file", "file": {"file_id": "file_abc"}}],
                }
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1, "file-only turn never reached apply_guardrail"
        assert guardrail.inputs is not None
        assert guardrail.inputs["files"] == ["file_abc"]

    @pytest.mark.asyncio
    async def test_image_url_part_with_only_a_file_id_reaches_guardrail_as_images(self):
        """A file-backed image_url has no inline url; it still must surface so the
        guardrail can refuse it instead of dropping it past the scan."""
        guardrail = ScanningGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"file_id": "file_abc"}}],
                }
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["images"] == ["file_abc"]


class TestOpenAIChatHandlerAttachmentsDefaultScope:
    """Guardrails that did not opt into attachment scanning see base behavior."""

    _PNG_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="

    @pytest.mark.asyncio
    async def test_image_only_turn_never_calls_apply_guardrail(self):
        guardrail = RecordingGuardrail()
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": self._PNG_DATA_URI}}],
            }
        ]
        data = {"model": "gpt-4o", "messages": messages}

        result = await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 0, "non-scanning guardrail fired on an image-only turn"
        assert result["messages"] == messages

    @pytest.mark.asyncio
    async def test_image_url_file_id_yields_no_images_entry(self):
        guardrail = RecordingGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe it"},
                        {"type": "image_url", "image_url": {"file_id": "file_abc"}},
                    ],
                }
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["describe it"]
        assert "images" not in guardrail.inputs
        assert "files" not in guardrail.inputs

    @pytest.mark.asyncio
    async def test_text_plus_file_part_sends_texts_only(self):
        guardrail = RecordingGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "summarize this document"},
                        {"type": "file", "file": {"file_id": "file_abc"}},
                    ],
                }
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["summarize this document"]
        assert "files" not in guardrail.inputs

    @pytest.mark.asyncio
    async def test_custom_guardrail_default_scans_attachments_is_false(self):
        assert CustomGuardrail.scans_attachments is False
