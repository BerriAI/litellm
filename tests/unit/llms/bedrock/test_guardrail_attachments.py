import base64

import pytest

from litellm.llms.bedrock.guardrail_attachments import find_request_attachments
from litellm.types.utils import CallTypes

PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake-png").decode()
JPEG_B64 = base64.b64encode(b"\xff\xd8\xfffake-jpeg").decode()
OVERSIZE_PNG_B64 = base64.b64encode(b"\x89PNG" + b"\0" * (4 * 1024 * 1024)).decode()
PDF_B64 = base64.b64encode(b"%PDF-1.4 fake").decode()


def _png_item(encoded: str = PNG_B64) -> dict:
    return {"image": {"format": "png", "source": {"bytes": encoded}}}


def _jpeg_item() -> dict:
    return {"image": {"format": "jpeg", "source": {"bytes": JPEG_B64}}}


def _chat(*blocks: object, role: str = "user") -> dict:
    return {"messages": [{"role": role, "content": list(blocks)}]}


def _converse(*blocks: object, endpoint: str = "model/haiku/converse", provider: str = "bedrock") -> dict:
    return {
        "endpoint": endpoint,
        "custom_llm_provider": provider,
        "data": {"messages": [{"role": "user", "content": list(blocks)}]},
    }


TEXT = {"type": "text", "text": "hello"}


@pytest.mark.parametrize(
    "data, call_type, images, unscannable",
    [
        pytest.param(_chat(TEXT), CallTypes.acompletion.value, [], [], id="chat-text-only"),
        pytest.param(
            {"messages": [{"role": "user", "content": "hi"}]},
            CallTypes.acompletion.value,
            [],
            [],
            id="chat-string-content",
        ),
        pytest.param(
            _chat(TEXT, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}),
            CallTypes.acompletion.value,
            [_png_item()],
            [],
            id="chat-png-data-uri-after-text",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": f"data:image/jpeg;base64,{JPEG_B64}"}),
            CallTypes.completion.value,
            [_jpeg_item()],
            [],
            id="chat-jpeg-string-url",
        ),
        pytest.param(
            _chat(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}},
                TEXT,
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{JPEG_B64}"}},
            ),
            CallTypes.acompletion.value,
            [_png_item(), _jpeg_item()],
            [],
            id="chat-two-images",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}),
            CallTypes.acompletion.value,
            [],
            ["image_url (remote URL or file id)"],
            id="chat-remote-image",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{PNG_B64}"}}),
            CallTypes.acompletion.value,
            [],
            ["image_url (image/gif)"],
            id="chat-gif",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": {"url": "data:image/png;base64,***not-base64***"}}),
            CallTypes.acompletion.value,
            [],
            ["image_url (invalid base64)"],
            id="chat-malformed-base64",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{OVERSIZE_PNG_B64}"}}),
            CallTypes.acompletion.value,
            [],
            ["image_url (over 4 MB)"],
            id="chat-image-over-4mb",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}),
            CallTypes.acompletion.value,
            [],
            ["image_url (invalid base64)"],
            id="chat-empty-base64",
        ),
        pytest.param(
            _chat(TEXT, {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{PDF_B64}"}}),
            CallTypes.acompletion.value,
            [],
            ["file"],
            id="chat-pdf-file",
        ),
        pytest.param(
            _chat(
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}},
            ),
            CallTypes.acompletion.value,
            [],
            ["input_audio", "video_url"],
            id="chat-audio-and-video",
        ),
        pytest.param(
            _chat(
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64}},
            ),
            CallTypes.anthropic_messages.value,
            [_png_item()],
            ["document"],
            id="anthropic-image-and-document",
        ),
        pytest.param(
            _chat({"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}),
            CallTypes.anthropic_messages.value,
            [],
            ["image (url or file source)"],
            id="anthropic-url-image",
        ),
        pytest.param(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "hi"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"},
                            {"type": "input_file", "file_data": f"data:application/pdf;base64,{PDF_B64}"},
                        ],
                    }
                ]
            },
            CallTypes.aresponses.value,
            [_png_item()],
            ["input_file"],
            id="responses-image-and-file",
        ),
        pytest.param({"input": "just text"}, CallTypes.responses.value, [], [], id="responses-string-input"),
        pytest.param(
            _converse({"text": "hi"}, _png_item(), {"document": {"format": "pdf", "source": {"bytes": PDF_B64}}}),
            CallTypes.allm_passthrough_route.value,
            [_png_item()],
            ["document"],
            id="converse-image-and-document",
        ),
        pytest.param(
            _converse({"image": {"format": "png", "source": {"s3Location": {"uri": "s3://b/k.png"}}}}),
            CallTypes.allm_passthrough_route.value,
            [],
            ["image (no inline bytes)"],
            id="converse-s3-image",
        ),
        pytest.param(
            _converse(
                {"video": {"format": "mp4", "source": {"bytes": PDF_B64}}}, endpoint="model/haiku/converse-stream"
            ),
            CallTypes.allm_passthrough_route.value,
            [],
            ["video"],
            id="converse-stream-video",
        ),
        pytest.param(
            _converse(_png_item(), endpoint="model/haiku/invoke"),
            CallTypes.allm_passthrough_route.value,
            [],
            [],
            id="bedrock-invoke-not-converse",
        ),
        pytest.param(
            _converse(_png_item(), provider="vertex_ai"),
            CallTypes.allm_passthrough_route.value,
            [],
            [],
            id="non-bedrock-passthrough",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": {"url": f"data:image/PNG;BASE64,{PNG_B64}"}}),
            CallTypes.acompletion.value,
            [_png_item()],
            [],
            id="chat-uppercase-data-uri",
        ),
        pytest.param(
            _chat(
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
                        {"type": "document", "source": {"type": "url", "url": "https://example.com/r.pdf"}},
                    ],
                }
            ),
            CallTypes.anthropic_messages.value,
            [_png_item()],
            ["document"],
            id="anthropic-tool-result-image-and-document",
        ),
        pytest.param(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [
                            {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"},
                            {"type": "input_file", "file_id": "file-1"},
                        ],
                    },
                    {"type": "function_call_output", "call_id": "call_2", "output": "plain text"},
                ]
            },
            CallTypes.aresponses.value,
            [_png_item()],
            ["input_file"],
            id="responses-function-call-output-image-and-file",
        ),
        pytest.param(
            {
                "input": [
                    {
                        "type": "computer_call_output",
                        "call_id": "call_1",
                        "output": {"type": "computer_screenshot", "image_url": f"data:image/png;base64,{PNG_B64}"},
                    },
                    {
                        "type": "computer_call_output",
                        "call_id": "call_2",
                        "output": {"type": "computer_screenshot", "file_id": "file-1"},
                    },
                ]
            },
            CallTypes.responses.value,
            [_png_item()],
            ["computer_screenshot (remote URL or file id)"],
            id="responses-computer-screenshot",
        ),
        pytest.param(
            _converse(
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "content": [{"document": {"format": "pdf", "source": {"bytes": PDF_B64}}}, _png_item()],
                    }
                }
            ),
            CallTypes.allm_passthrough_route.value,
            [_png_item()],
            ["document"],
            id="converse-tool-result-document-and-image",
        ),
        pytest.param(
            _chat({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}),
            CallTypes.call_mcp_tool.value,
            [],
            [],
            id="mcp-call-type-ignored",
        ),
    ],
)
def test_find_request_attachments(data, call_type, images, unscannable):
    found = find_request_attachments(data, call_type, skip_tool_messages=False, latest_user_message_only=False)

    assert list(found.images) == images
    assert list(found.unscannable) == unscannable


def test_whitespace_in_base64_is_stripped_before_sending():
    wrapped = PNG_B64[:8] + "\n" + PNG_B64[8:]
    data = _chat({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{wrapped}"}})

    found = find_request_attachments(data, CallTypes.acompletion.value, False, False)

    assert list(found.images) == [_png_item()]


def test_tool_messages_skipped_only_when_asked():
    data = {
        "messages": [
            {"role": "user", "content": [TEXT]},
            {"role": "tool", "content": [{"type": "file", "file": {"file_id": "file-1"}}]},
        ]
    }

    scanned = find_request_attachments(data, CallTypes.acompletion.value, False, False)
    skipped = find_request_attachments(data, CallTypes.acompletion.value, True, False)

    assert scanned.unscannable == ("file",)
    assert skipped.unscannable == ()


def test_scan_only_tool_results_keeps_only_tool_output():
    data = _chat(
        {"type": "document", "source": {"type": "url", "url": "https://example.com/user.pdf"}},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "document", "source": {}}]},
    )

    everything = find_request_attachments(data, CallTypes.anthropic_messages.value, False, False)
    tool_only = find_request_attachments(data, CallTypes.anthropic_messages.value, False, False, True)
    skip_tool = find_request_attachments(data, CallTypes.anthropic_messages.value, True, False)

    assert everything.unscannable == ("document", "document")
    assert tool_only.unscannable == ("document",)
    assert skip_tool.unscannable == ("document",)
    assert find_request_attachments(data, CallTypes.anthropic_messages.value, True, False, True).unscannable == ()


def test_long_mime_is_truncated_in_label():
    data = _chat({"type": "image_url", "image_url": {"url": f"data:image/{'x' * 500};base64,{PNG_B64}"}})

    found = find_request_attachments(data, CallTypes.acompletion.value, False, False)

    assert found.unscannable == (f"image_url (image/{'x' * 34})",)


def test_latest_user_message_only():
    data = {
        "messages": [
            {"role": "user", "content": [{"type": "file", "file": {"file_id": "old"}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/png;base64,{PNG_B64}"}]},
        ]
    }

    found = find_request_attachments(data, CallTypes.acompletion.value, False, True)

    assert list(found.images) == [_png_item()]
    assert found.unscannable == ()
