"""
Unit tests for the Bedrock Converse passthrough guardrail handler's attachment
extraction: image and document blocks in converse content must reach
apply_guardrail so a guardrail can scan or refuse them.
"""

from collections.abc import Mapping
from typing import Any, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.bedrock.passthrough.guardrail_translation.handler import (
    BedrockPassthroughGuardrailHandler,
)
from litellm.types.utils import GenericGuardrailAPIInputs


class InputRecordingGuardrail(CustomGuardrail):
    """Records the inputs each apply_guardrail call was handed."""

    def __init__(self, guardrail_name: str = "recording"):
        super().__init__(guardrail_name=guardrail_name)
        self.calls = 0
        self.inputs: GenericGuardrailAPIInputs | None = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.calls += 1
        self.inputs = inputs.copy()
        return inputs


class ScanningGuardrail(InputRecordingGuardrail):
    """A guardrail that opted into attachment scanning, like Bedrock."""

    scans_attachments = True


class TestBedrockConverseHandlerAttachments:
    @staticmethod
    def _data(body: Mapping[str, object]) -> dict[str, object]:
        return {
            "endpoint": "/bedrock/model/us.amazon.nova-lite-v1:0/converse",
            "model": "us.amazon.nova-lite-v1:0",
            "data": body,
        }

    @pytest.mark.asyncio
    async def test_image_block_reaches_guardrail_as_data_uri(self):
        """Converse image blocks used to be dropped before the guardrail ran."""
        guardrail = ScanningGuardrail()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "what does this show?"},
                        {"image": {"format": "gif", "source": {"bytes": "R0lGODlhAQABAAAAACw="}}},
                    ],
                }
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["images"] == ["data:image/gif;base64,R0lGODlhAQABAAAAACw="]

    @pytest.mark.asyncio
    async def test_document_block_reaches_guardrail_as_files(self):
        """Converse document blocks used to be dropped before the guardrail ran."""
        guardrail = ScanningGuardrail()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "summarize this"},
                        {"document": {"format": "pdf", "name": "a", "source": {"bytes": "QUFBQQ=="}}},
                    ],
                }
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["files"] == ["data:application/pdf;base64,QUFBQQ=="]

    @pytest.mark.asyncio
    async def test_document_only_turn_invokes_guardrail(self):
        """A turn whose only content is a document still must reach the guardrail."""
        guardrail = ScanningGuardrail()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"document": {"format": "pdf", "name": "a", "source": {"bytes": "QUFBQQ=="}}},
                    ],
                }
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1, "document-only turn never reached apply_guardrail"
        assert guardrail.inputs is not None
        assert guardrail.inputs["files"] == ["data:application/pdf;base64,QUFBQQ=="]

    @pytest.mark.asyncio
    async def test_document_inside_tool_result_reaches_guardrail_as_files(self):
        guardrail = ScanningGuardrail()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "use this"},
                        {
                            "toolResult": {
                                "toolUseId": "tu_1",
                                "content": [
                                    {"document": {"format": "pdf", "name": "a", "source": {"bytes": "QkJCQg=="}}}
                                ],
                            }
                        },
                    ],
                }
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["files"] == ["data:application/pdf;base64,QkJCQg=="]

    @pytest.mark.asyncio
    async def test_s3_backed_document_yields_its_uri(self):
        guardrail = ScanningGuardrail()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "document": {
                                "format": "pdf",
                                "name": "a",
                                "source": {"s3Location": {"uri": "s3://bucket/a.pdf"}},
                            }
                        },
                    ],
                }
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["files"] == ["s3://bucket/a.pdf"]


class TestBedrockConverseAttachmentsDefaultScope:
    """Guardrails that did not opt into attachment scanning see base behavior."""

    @staticmethod
    def _data(body: Mapping[str, object]) -> dict[str, object]:
        return {
            "endpoint": "/bedrock/model/us.amazon.nova-lite-v1:0/converse",
            "model": "us.amazon.nova-lite-v1:0",
            "data": body,
        }

    @pytest.mark.asyncio
    async def test_document_only_turn_never_calls_apply_guardrail(self):
        guardrail = InputRecordingGuardrail()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"document": {"format": "pdf", "name": "a", "source": {"bytes": "QUFBQQ=="}}},
                    ],
                }
            ]
        }

        result = await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 0, "non-scanning guardrail fired on a document-only turn"
        assert result["data"] == body

    @pytest.mark.asyncio
    async def test_text_plus_document_sends_texts_only(self):
        guardrail = InputRecordingGuardrail()
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "summarize this"},
                        {"document": {"format": "pdf", "name": "a", "source": {"bytes": "QUFBQQ=="}}},
                    ],
                }
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["summarize this"]
        assert "files" not in guardrail.inputs
        assert "images" not in guardrail.inputs


class TestScopedOutToolResultAttachments:
    """A toolUse/toolResult block scoped out by skip_tool_message_in_guardrail
    still contributes its document refs so the guardrail can refuse them; its
    text and inline images are never scanned."""

    @staticmethod
    def _data(body: Mapping[str, object]) -> dict[str, object]:
        return {
            "endpoint": "/bedrock/model/us.amazon.nova-lite-v1:0/converse",
            "model": "us.amazon.nova-lite-v1:0",
            "data": body,
        }

    @pytest.mark.asyncio
    async def test_skipped_tool_result_document_reaches_files_not_texts(self):
        guardrail = ScanningGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        body = {
            "messages": [
                {"role": "user", "content": [{"text": "hello"}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "tu_1",
                                "content": [
                                    {"document": {"format": "pdf", "name": "a", "source": {"bytes": "QkJCQg=="}}}
                                ],
                            }
                        },
                    ],
                },
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["hello"]
        assert guardrail.inputs["files"] == ["data:application/pdf;base64,QkJCQg=="]

    @pytest.mark.asyncio
    async def test_skipped_tool_result_with_only_text_and_inline_png_never_calls_guardrail(self):
        guardrail = ScanningGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "tu_1",
                                "content": [
                                    {"text": "tool said hi"},
                                    {"image": {"format": "png", "source": {"bytes": "iVBORw0K"}}},
                                ],
                            }
                        },
                    ],
                },
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 0

    @pytest.mark.asyncio
    async def test_non_attachment_guardrail_keeps_base_shape_on_skipped_tool_result(self):
        guardrail = InputRecordingGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        body = {
            "messages": [
                {"role": "user", "content": [{"text": "hello"}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "tu_1",
                                "content": [
                                    {"document": {"format": "pdf", "name": "a", "source": {"bytes": "QkJCQg=="}}}
                                ],
                            }
                        },
                    ],
                },
            ]
        }

        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data=self._data(body), guardrail_to_apply=guardrail
        )

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["hello"]
        assert "files" not in guardrail.inputs
