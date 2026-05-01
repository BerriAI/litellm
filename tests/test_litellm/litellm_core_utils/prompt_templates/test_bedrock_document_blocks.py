"""
Tests for Bedrock Converse API document block handling in factory.py.
Covers: basic document conversion, citations, context, tool results, and format mapping.
"""

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    BedrockConverseMessagesProcessor,
    _bedrock_converse_messages_pt,
    _convert_to_bedrock_tool_call_result,
)

_process_document_message = BedrockConverseMessagesProcessor._process_document_message


def _make_doc_message(extra_fields: dict = {}) -> list:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "dGVzdA==",
                    },
                    **extra_fields,
                },
                {"type": "text", "text": "What is the title?"},
            ],
        }
    ]


def test_document_block_basic():
    """Document block produces correct format, source.bytes, and name."""
    result = _bedrock_converse_messages_pt(
        _make_doc_message(), "anthropic.claude-sonnet-4-6", "bedrock"
    )
    content = result[0]["content"]
    doc_parts = [b for b in content if "document" in b]
    assert len(doc_parts) == 1
    doc = doc_parts[0]["document"]
    assert doc["format"] == "pdf"
    assert doc["source"]["bytes"] == "dGVzdA=="
    assert doc["name"].startswith("Document_")


def test_document_block_not_dropped_with_text():
    """Mixed document + text preserves both blocks (reproduces the silent-drop bug)."""
    result = _bedrock_converse_messages_pt(
        _make_doc_message(), "anthropic.claude-sonnet-4-6", "bedrock"
    )
    content = result[0]["content"]
    types = [list(b.keys())[0] for b in content]
    assert "document" in types
    assert "text" in types


def test_document_block_citations_forwarded():
    """citations.enabled=True is forwarded to BedrockCitationsConfig on the DocumentBlock."""
    result = _bedrock_converse_messages_pt(
        _make_doc_message({"citations": {"enabled": True}}),
        "anthropic.claude-sonnet-4-6",
        "bedrock",
    )
    content = result[0]["content"]
    doc = next(b["document"] for b in content if "document" in b)
    assert doc.get("citations") == {"enabled": True}


def test_document_block_citations_not_set_when_disabled():
    """citations field is absent when citations.enabled is False or not set."""
    result = _bedrock_converse_messages_pt(
        _make_doc_message({"citations": {"enabled": False}}),
        "anthropic.claude-sonnet-4-6",
        "bedrock",
    )
    content = result[0]["content"]
    doc = next(b["document"] for b in content if "document" in b)
    assert "citations" not in doc


def test_document_block_context_forwarded():
    """context field is forwarded to the Bedrock DocumentBlock."""
    result = _bedrock_converse_messages_pt(
        _make_doc_message({"context": "Q4 financial report"}),
        "anthropic.claude-sonnet-4-6",
        "bedrock",
    )
    content = result[0]["content"]
    doc = next(b["document"] for b in content if "document" in b)
    assert doc.get("context") == "Q4 financial report"


def test_document_block_deterministic_name():
    """Same document data always produces the same deterministic name."""
    element = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "dGVzdA=="},
    }
    block1 = _process_document_message(element)
    block2 = _process_document_message(element)
    assert block1["document"]["name"] == block2["document"]["name"]


@pytest.mark.parametrize(
    "media_type,expected_format",
    [
        ("application/pdf", "pdf"),
        ("text/csv", "csv"),
        ("text/html", "html"),
        ("text/plain", "txt"),
        ("text/markdown", "md"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    ],
)
def test_document_format_mapping(media_type, expected_format):
    """Various MIME types map to the correct Bedrock document format."""
    block = _process_document_message(
        {
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": "dGVzdA=="},
        }
    )
    assert block["document"]["format"] == expected_format


def test_document_in_tool_result():
    """Document block in tool result content produces correct BedrockToolResultContentBlock."""
    message = {
        "role": "tool",
        "tool_call_id": "tool_123",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "dGVzdA==",
                },
            }
        ],
    }
    result = _convert_to_bedrock_tool_call_result(message)
    tool_result_content = result["toolResult"]["content"]
    doc_blocks = [b for b in tool_result_content if "document" in b]
    assert len(doc_blocks) == 1
    assert doc_blocks[0]["document"]["format"] == "pdf"


def test_non_base64_source_raises():
    """Non-base64 source type raises BadRequestError."""
    with pytest.raises(litellm.BadRequestError, match="base64"):
        _process_document_message(
            {
                "type": "document",
                "source": {"type": "url", "url": "https://example.com/doc.pdf"},
            }
        )
