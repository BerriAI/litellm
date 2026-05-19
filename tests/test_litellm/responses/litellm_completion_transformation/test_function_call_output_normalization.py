"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_function_call_output_list_input_text_is_converted_to_tool_string_content():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "hello"},
                {"type": "input_text", "text": " world"},
            ],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "hello world"


def test_function_call_output_string_passthrough():
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"ok":true}',
        }
    )
    assert len(out) == 1
    assert out[0]["content"] == '{"ok":true}'


def test_function_call_output_input_file_is_preserved_as_file_block():
    """A PDF returned from a tool call (input_file part inside
    function_call_output.output) must survive the lowering to a chat-completions
    tool message as a structured file content block. Previously the file part
    was silently dropped and only the input_text sibling reached the model."""
    pdf_data_url = "data:application/pdf;base64,JVBERi0xLjQKJfb=="

    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_pdf",
            "output": [
                {"type": "input_text", "text": "Here is the PDF."},
                {
                    "type": "input_file",
                    "file_data": pdf_data_url,
                    "filename": "test.pdf",
                },
            ],
        }
    )

    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_pdf"

    content = msg["content"]
    assert isinstance(content, list), (
        "expected structured content list when a file part is present, "
        f"got {type(content).__name__}: {content!r}"
    )

    text_blocks = [b for b in content if b.get("type") == "text"]
    file_blocks = [b for b in content if b.get("type") == "file"]

    assert any(b.get("text") == "Here is the PDF." for b in text_blocks)
    assert len(file_blocks) == 1
    assert file_blocks[0]["file"]["file_data"] == pdf_data_url


def test_function_call_output_input_file_with_file_id_is_preserved():
    """Same as above but for the file_id form (no inline bytes)."""
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_pdf_id",
            "output": [
                {"type": "input_text", "text": "See attached."},
                {
                    "type": "input_file",
                    "file_id": "file-abc123",
                    "filename": "report.pdf",
                },
            ],
        }
    )

    assert len(out) == 1
    content = out[0]["content"]
    assert isinstance(content, list)

    file_blocks = [b for b in content if b.get("type") == "file"]
    assert len(file_blocks) == 1
    assert file_blocks[0]["file"]["file_id"] == "file-abc123"


def test_function_call_output_input_file_without_data_is_skipped():
    """An input_file part with no file_id, file_url, or file_data carries no
    payload — appending it as ``{"type": "file", "file": {}}`` would activate
    the structured-content path and emit a hollow block that downstream
    adapters (Gemini, Bedrock) reject or silently ignore. Skip it instead."""
    out = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output={
            "type": "function_call_output",
            "call_id": "call_empty_file",
            "output": [
                {"type": "input_text", "text": "Just text here."},
                {"type": "input_file", "filename": "missing.pdf"},
            ],
        }
    )

    assert len(out) == 1
    # No usable file payload -> fall back to the plain-string text path.
    assert out[0]["content"] == "Just text here."
