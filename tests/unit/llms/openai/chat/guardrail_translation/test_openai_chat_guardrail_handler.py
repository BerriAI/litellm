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

    @pytest.mark.asyncio
    async def test_spec_mock_guardrail_never_calls_apply_guardrail_on_attachment_only_turn(self):
        from unittest.mock import MagicMock

        guardrail = MagicMock(spec=CustomGuardrail)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._PNG_DATA_URI}},
                    {"type": "file", "file": {"file_id": "file_abc"}},
                ],
            }
        ]
        data = {"model": "gpt-4o", "messages": messages}

        result = await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        guardrail.apply_guardrail.assert_not_called()
        assert result["messages"] == messages

    @pytest.mark.asyncio
    async def test_empty_tools_and_empty_model_are_absent_from_inputs(self):
        guardrail = RecordingGuardrail()
        data = {
            "model": "",
            "messages": [{"role": "user", "content": "hi there"}],
            "tools": [],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert "tools" not in guardrail.inputs
        assert "model" not in guardrail.inputs

    @pytest.mark.asyncio
    async def test_empty_image_url_string_is_forwarded_as_images(self):
        guardrail = ScanningGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": ""}]}],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["images"] == [""]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "shell",
        [
            {"type": "video_url"},
            {"type": "input_audio"},
            {"type": "file"},
            {"type": "video_url", "video_url": {}},
            {"type": "input_audio", "input_audio": ""},
        ],
    )
    async def test_bare_attachment_shells_yield_no_files_entry(self, shell):
        """A payload-less attachment part carries nothing the guardrail can refuse or scan."""
        guardrail = ScanningGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, shell]}],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["look"]
        assert "files" not in guardrail.inputs
        assert "images" not in guardrail.inputs


class TestOpenAIChatHandlerToolsAndModelForwarding:
    @pytest.mark.asyncio
    async def test_tools_dict_is_forwarded_verbatim_to_inputs(self):
        guardrail = ScanningGuardrail()
        tools = {"name": "lookup", "parameters": {"query": "str"}}
        data = {
            "model": "gpt-4o",
            "tools": tools,
            "messages": [{"role": "user", "content": "hi"}],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["tools"] == tools

    @pytest.mark.asyncio
    async def test_model_list_is_forwarded_verbatim_to_inputs(self):
        guardrail = ScanningGuardrail()
        data = {
            "model": ["gpt-4o"],
            "messages": [{"role": "user", "content": "hi"}],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["model"] == ["gpt-4o"]


class _BaseSignatureExtractInputsHandler(OpenAIChatCompletionsHandler):
    """A subclass written against the pre-attachment-scanning _extract_inputs signature."""

    def _extract_inputs(
        self,
        message,
        msg_idx,
        texts_to_check,
        images_to_check,
        tool_calls_to_check,
        text_task_mappings,
        tool_call_task_mappings,
        skip_system_message=False,
        skip_tool_message=False,
        scan_only_tool_results=False,
    ) -> None:
        return super()._extract_inputs(
            message=message,
            msg_idx=msg_idx,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            tool_calls_to_check=tool_calls_to_check,
            text_task_mappings=text_task_mappings,
            tool_call_task_mappings=tool_call_task_mappings,
            skip_system_message=skip_system_message,
            skip_tool_message=skip_tool_message,
            scan_only_tool_results=scan_only_tool_results,
        )


class _HeadSignatureRecordingExtractInputsHandler(OpenAIChatCompletionsHandler):
    seen_scan_attachments: bool | None = None
    seen_files_to_check_is_not_none: bool | None = None

    def _extract_inputs(
        self,
        message,
        msg_idx,
        texts_to_check,
        images_to_check,
        tool_calls_to_check,
        text_task_mappings,
        tool_call_task_mappings,
        skip_system_message=False,
        skip_tool_message=False,
        scan_only_tool_results=False,
        scan_attachments=False,
        files_to_check=None,
    ) -> None:
        _HeadSignatureRecordingExtractInputsHandler.seen_scan_attachments = scan_attachments
        _HeadSignatureRecordingExtractInputsHandler.seen_files_to_check_is_not_none = files_to_check is not None
        return super()._extract_inputs(
            message=message,
            msg_idx=msg_idx,
            texts_to_check=texts_to_check,
            images_to_check=images_to_check,
            tool_calls_to_check=tool_calls_to_check,
            text_task_mappings=text_task_mappings,
            tool_call_task_mappings=tool_call_task_mappings,
            skip_system_message=skip_system_message,
            skip_tool_message=skip_tool_message,
            scan_only_tool_results=scan_only_tool_results,
            scan_attachments=scan_attachments,
            files_to_check=files_to_check,
        )


class TestExtractInputsBaseSignatureCompatibility:
    @pytest.mark.asyncio
    async def test_legacy_signature_subclass_runs_for_plain_guardrail(self):
        """A non-attachment guardrail must not force files_to_check onto old subclasses."""
        guardrail = RecordingGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "summarize this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="},
                        },
                    ],
                }
            ],
        }

        await _BaseSignatureExtractInputsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["summarize this"]

    @pytest.mark.asyncio
    async def test_attachment_guardrail_still_passes_the_new_kwargs(self):
        """With scans_attachments on, the head signature receives scan_attachments and files."""
        _HeadSignatureRecordingExtractInputsHandler.seen_scan_attachments = None
        _HeadSignatureRecordingExtractInputsHandler.seen_files_to_check_is_not_none = None
        guardrail = ScanningGuardrail()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "summarize this"},
                        {
                            "type": "file",
                            "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AAAA"},
                        },
                    ],
                }
            ],
        }

        await _HeadSignatureRecordingExtractInputsHandler().process_input_messages(
            data=data, guardrail_to_apply=guardrail
        )

        assert _HeadSignatureRecordingExtractInputsHandler.seen_scan_attachments is True
        assert _HeadSignatureRecordingExtractInputsHandler.seen_files_to_check_is_not_none is True
        assert guardrail.inputs is not None
        assert guardrail.inputs["files"]


class TestScopedOutMessageAttachments:
    """A message scoped out by a skip flag still contributes its unscannable
    attachment refs so the guardrail can refuse them; its text and inline images
    are never scanned."""

    _PNG_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    _PDF_FILE_PART = {
        "type": "file",
        "file": {"filename": "q.pdf", "file_data": "data:application/pdf;base64,AAAA"},
    }

    @staticmethod
    def _tool_message(content):
        return {"role": "tool", "tool_call_id": "call_1", "content": content}

    @pytest.mark.asyncio
    async def test_skipped_tool_message_still_contributes_file_refs(self):
        guardrail = ScanningGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hello"},
                self._tool_message([{"type": "text", "text": "tool said hi"}, self._PDF_FILE_PART]),
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["hello"]
        assert guardrail.inputs["files"] == ["data:application/pdf;base64,AAAA"]

    @pytest.mark.asyncio
    async def test_skipped_tool_message_with_only_text_and_inline_png_never_calls_guardrail(self):
        guardrail = ScanningGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        data = {
            "model": "gpt-4o",
            "messages": [
                self._tool_message(
                    [
                        {"type": "text", "text": "tool said hi"},
                        {"type": "image_url", "image_url": {"url": self._PNG_DATA_URI}},
                    ]
                ),
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 0

    @pytest.mark.asyncio
    async def test_non_attachment_guardrail_keeps_base_shape_on_skipped_tool_file(self):
        guardrail = RecordingGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hello"},
                self._tool_message([{"type": "text", "text": "tool said hi"}, self._PDF_FILE_PART]),
            ],
        }

        await OpenAIChatCompletionsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["hello"]
        assert "files" not in guardrail.inputs


class _BaseSignatureNotRunReasonHandler(OpenAIChatCompletionsHandler):
    def _not_run_reason(self, messages):
        return "no scannable content"


class TestScansAttachmentsWithBaseSignatureOnTextOnly:
    """A scans_attachments guardrail must still give base-signature subclasses the
    base call shape when the request carries no attachment parts."""

    @pytest.mark.asyncio
    async def test_extract_inputs_base_signature_on_text_only(self):
        guardrail = ScanningGuardrail()
        data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "just text"}]}

        await _BaseSignatureExtractInputsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.calls == 1
        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["just text"]
        assert "files" not in guardrail.inputs

    @pytest.mark.asyncio
    async def test_not_run_reason_base_signature_on_text_only(self):
        guardrail = ScanningGuardrail()
        data = {"model": "gpt-4o", "messages": [{"role": "user", "content": []}]}

        await _BaseSignatureNotRunReasonHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)


class _InPlaceToolCallMutatingGuardrail(CustomGuardrail):
    """Mutates the tool_calls list handed to apply_guardrail in place and returns
    a payload without a tool_calls key."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        if inputs.get("tool_calls"):
            tool_call = inputs["tool_calls"][0]
            assert isinstance(tool_call, dict)
            function = tool_call.get("function")
            assert isinstance(function, dict)
            inputs["tool_calls"][0] = {
                **tool_call,
                "function": {**function, "arguments": "MASKED_ARGS"},
            }
        return GenericGuardrailAPIInputs(texts=list(inputs.get("texts") or []))


class TestOutputToolCallAliasing:
    @pytest.mark.asyncio
    async def test_in_place_tool_call_mutation_survives_when_guardrail_returns_no_tool_calls(self):
        import litellm

        response = litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "calling the tool",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "original"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        )

        result = await OpenAIChatCompletionsHandler().process_output_response(
            response=response, guardrail_to_apply=_InPlaceToolCallMutatingGuardrail()
        )

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls[0]["function"]["arguments"] == "MASKED_ARGS"


def _two_tool_call_response():
    import litellm

    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "calling tools",
                    "tool_calls": [
                        {"id": "call_A", "type": "function", "function": {"name": "first", "arguments": "args_A"}},
                        {"id": "call_B", "type": "function", "function": {"name": "second", "arguments": "args_B"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )


class _PostCallToolCallGuardrail(CustomGuardrail):
    """Returns a scripted tool_calls payload, optionally replacing inputs['tool_calls'] first."""

    def __init__(self, *, returned_tool_calls=None, replace_inputs_with=None):
        super().__init__()
        self._returned = returned_tool_calls
        self._replace_with = replace_inputs_with

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Any | None = None,
    ) -> GenericGuardrailAPIInputs:
        if self._replace_with is not None:
            inputs["tool_calls"] = self._replace_with
        payload: GenericGuardrailAPIInputs = GenericGuardrailAPIInputs(texts=list(inputs.get("texts") or []))
        if self._returned is not None:
            payload["tool_calls"] = self._returned
        return payload


class TestPostCallToolCallFallback:
    @pytest.mark.asyncio
    async def test_same_length_returned_tool_calls_are_applied(self):
        masked = [{"id": "call_A", "type": "function", "function": {"name": "first", "arguments": "MASKED"}}]

        result = await OpenAIChatCompletionsHandler().process_output_response(
            response=_two_tool_call_response(),
            guardrail_to_apply=_PostCallToolCallGuardrail(
                returned_tool_calls=masked
                + [{"id": "call_B", "type": "function", "function": {"name": "second", "arguments": "args_B"}}]
            ),
        )

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls[0]["function"]["arguments"] == "MASKED"

    @pytest.mark.asyncio
    async def test_shorter_returned_tool_calls_fall_back_to_originals(self):
        returned = [{"id": "call_B", "type": "function", "function": {"name": "second", "arguments": "args_B"}}]

        result = await OpenAIChatCompletionsHandler().process_output_response(
            response=_two_tool_call_response(),
            guardrail_to_apply=_PostCallToolCallGuardrail(returned_tool_calls=returned),
        )

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls[0]["id"] == "call_A"
        assert tool_calls[0]["function"]["arguments"] == "args_A"
        assert tool_calls[1]["id"] == "call_B"
        assert tool_calls[1]["function"]["arguments"] == "args_B"

    @pytest.mark.asyncio
    async def test_non_list_tool_calls_mutation_leaves_response_unchanged(self):
        result = await OpenAIChatCompletionsHandler().process_output_response(
            response=_two_tool_call_response(),
            guardrail_to_apply=_PostCallToolCallGuardrail(replace_inputs_with={"some": "object"}),
        )

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls[0]["id"] == "call_A"
        assert tool_calls[1]["id"] == "call_B"

    @pytest.mark.asyncio
    async def test_no_tool_calls_returned_leaves_response_unchanged(self):
        result = await OpenAIChatCompletionsHandler().process_output_response(
            response=_two_tool_call_response(),
            guardrail_to_apply=_PostCallToolCallGuardrail(),
        )

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls[0]["id"] == "call_A"
        assert tool_calls[1]["id"] == "call_B"
