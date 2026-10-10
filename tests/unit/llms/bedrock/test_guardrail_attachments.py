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
            _chat(
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{JPEG_B64}"},
                {"type": "input_file", "file_id": "file-1"},
            ),
            CallTypes.acompletion.value,
            [_png_item(), _jpeg_item()],
            ["input_file"],
            id="chat-anthropic-and-responses-shaped-parts",
        ),
        pytest.param(
            {"input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]},
            CallTypes.aresponses.value,
            [],
            ["image_url (remote URL or file id)"],
            id="responses-chat-shaped-image-url",
        ),
        pytest.param(
            _converse({"audio": {"format": "mp3", "source": {"bytes": PDF_B64}}}),
            CallTypes.allm_passthrough_route.value,
            [],
            ["audio"],
            id="converse-audio",
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


def test_non_string_role_and_type_do_not_raise():
    data = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": f"data:image/png;base64,{PNG_B64}"}]},
            {"role": ["tool"], "type": {"x": 1}, "content": [{"type": "file", "file": {"file_id": "f"}}]},
            {"role": {"r": "tool"}, "type": ["function_call_output"], "content": [{"type": "file", "file": {}}]},
        ]
    }

    found = find_request_attachments(data, CallTypes.acompletion.value, True, False)

    assert list(found.images) == [_png_item()]
    assert found.unscannable == ("file", "file")


def test_converse_null_document_is_not_an_attachment():
    data = _converse(_png_item(), {"document": None}, {"video": None, "audio": None}, {"audio": {"format": "mp3"}})

    found = find_request_attachments(data, CallTypes.allm_passthrough_route.value, False, False)

    assert list(found.images) == [_png_item()]
    assert found.unscannable == ("audio",)


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "hi"}}, id="text"
        ),
        pytest.param(
            {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "hi"}]}},
            id="content",
        ),
        pytest.param({"type": "document", "source": {"type": "content", "content": "hi"}}, id="content-string"),
    ],
)
@pytest.mark.parametrize(
    "call_type", [CallTypes.anthropic_messages.value, CallTypes.acompletion.value], ids=["messages", "chat"]
)
def test_text_source_document_is_not_an_attachment(block, call_type):
    pdf = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64}}

    found = find_request_attachments(_chat(TEXT, block, pdf), call_type, False, False)

    assert found.images == ()
    assert found.unscannable == ("document",)
    assert found.document_texts == ("hi",)


def test_unpadded_and_url_safe_base64_are_sent_as_standard_base64():
    raw = b"\x89PNG\r\n\x1a\n\xfb\xff\xfe-fake"
    standard = base64.b64encode(raw).decode()
    unpadded = standard.rstrip("=")
    url_safe = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    data = _chat(
        *(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            for encoded in (unpadded, url_safe)
        )
    )

    found = find_request_attachments(data, CallTypes.acompletion.value, False, False)

    assert unpadded != standard
    assert set("-_") & set(url_safe)
    assert list(found.images) == [_png_item(standard), _png_item(standard)]
    assert found.unscannable == ()


def test_oversize_base64_is_refused_by_length_before_decoding():
    encoded = "!" * (len(OVERSIZE_PNG_B64) + 4)
    data = _chat({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})

    found = find_request_attachments(data, CallTypes.acompletion.value, False, False)

    assert found.unscannable == ("image_url (over 4 MB)",)


@pytest.mark.parametrize(
    "call_type", [CallTypes.anthropic_messages.value, CallTypes.acompletion.value], ids=["messages", "chat"]
)
def test_images_inside_a_content_source_document_are_scanned(call_type):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}
    pdf = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64}}
    block = {"type": "document", "source": {"type": "content", "content": [TEXT, image, pdf]}}

    found = find_request_attachments(_chat(block), call_type, False, False)

    assert list(found.images) == [_png_item()]
    assert found.unscannable == ("document",)
    assert found.document_texts == ("hello",)


def test_document_images_inside_a_tool_result_follow_the_tool_scope():
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}
    document = {"type": "document", "source": {"type": "content", "content": [image]}}
    data = _chat({"type": "tool_result", "tool_use_id": "t1", "content": [document]})

    scanned = find_request_attachments(data, CallTypes.anthropic_messages.value, False, False)
    skipped = find_request_attachments(data, CallTypes.anthropic_messages.value, True, False)

    assert list(scanned.images) == [_png_item()]
    assert skipped.images == ()


def test_document_title_and_context_are_scanned_as_text():
    text_doc = {
        "type": "document",
        "title": "record",
        "context": "SSN 123-45-6789",
        "source": {"type": "text", "media_type": "text/plain", "data": "body"},
    }
    pdf = {
        "type": "document",
        "context": "note",
        "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64},
    }

    found = find_request_attachments(_chat(text_doc, pdf), CallTypes.anthropic_messages.value, False, False)

    assert found.document_texts == ("record\nSSN 123-45-6789\nbody", "note")
    assert found.unscannable == ("document",)


def _content_document(*inner: object) -> dict:
    return {"type": "document", "source": {"type": "content", "content": list(inner)}}


def test_nested_documents_are_scanned_and_deep_nesting_is_refused():
    inner_text = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "inner"}}
    nested = _content_document(inner_text, _content_document({"type": "text", "text": "deeper"}))
    too_deep = _content_document(_content_document(_content_document(_content_document(TEXT))))

    found = find_request_attachments(_chat(nested, too_deep), CallTypes.anthropic_messages.value, False, False)

    assert found.document_texts == ("inner", "deeper")
    assert found.unscannable == ("document (nested too deep)",)


def _png_bytes(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02"


def _jpeg_bytes(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03"
    return b"\xff\xd8" + app0 + b"\xff" + sof0


def _image_url(mime: str, raw: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}


@pytest.mark.parametrize(
    "mime, raw, sent_format",
    [
        pytest.param("image/png", _jpeg_bytes(256, 256), "jpeg", id="jpeg-labeled-png"),
        pytest.param("image/jpeg", _png_bytes(256, 256), "png", id="png-labeled-jpeg"),
        pytest.param("image/png", _png_bytes(8000, 8000), "png", id="png-at-pixel-limit"),
        pytest.param("image/jpeg", _jpeg_bytes(8000, 10), "jpeg", id="jpeg-at-pixel-limit"),
    ],
)
def test_image_format_is_taken_from_the_bytes(mime, raw, sent_format):
    found = find_request_attachments(_chat(_image_url(mime, raw)), CallTypes.acompletion.value, False, False)

    assert [item["image"]["format"] for item in found.images] == [sent_format]
    assert found.unscannable == ()


@pytest.mark.parametrize(
    "mime, raw, label",
    [
        pytest.param("image/png", _png_bytes(8001, 10), "image_url (over 8000 pixels)", id="png-too-wide"),
        pytest.param("image/png", _png_bytes(10, 9000), "image_url (over 8000 pixels)", id="png-too-tall"),
        pytest.param("image/jpeg", _jpeg_bytes(9000, 10), "image_url (over 8000 pixels)", id="jpeg-too-wide"),
        pytest.param("image/jpeg", _jpeg_bytes(10, 8001), "image_url (over 8000 pixels)", id="jpeg-too-tall"),
        pytest.param("image/png", b"GIF89a\x01\x00\x01\x00", "image_url (not PNG or JPEG data)", id="gif-labeled-png"),
    ],
)
def test_images_bedrock_cannot_scan_are_unscannable(mime, raw, label):
    data = _chat(_image_url("image/png", _png_bytes(16, 16)), _image_url(mime, raw))

    found = find_request_attachments(data, CallTypes.acompletion.value, False, False)

    assert len(found.images) == 1
    assert found.unscannable == (label,)


def _converse_document(source: dict, **fields: str) -> dict:
    return {"document": {"format": "txt", "name": "notes", **fields, "source": source}}


@pytest.mark.parametrize(
    "block, document_texts",
    [
        pytest.param(_converse_document({"text": "SSN 123-45-6789"}), ("notes\nSSN 123-45-6789",), id="text"),
        pytest.param(
            _converse_document({"text": "body"}, context="SSN 123-45-6789"),
            ("notes\nSSN 123-45-6789\nbody",),
            id="text-with-context",
        ),
        pytest.param(
            _converse_document({"content": [{"text": "first"}, {"text": "SSN 123-45-6789"}, {"text": "last"}]}),
            ("notes\nfirst\nSSN 123-45-6789\nlast",),
            id="content",
        ),
        pytest.param(_converse_document({"text": ""}, name=""), (), id="empty-text"),
    ],
)
def test_converse_text_source_document_is_scanned_as_text(block, document_texts):
    found = find_request_attachments(
        _converse({"text": "hi"}, block), CallTypes.allm_passthrough_route.value, False, False
    )

    assert found.document_texts == document_texts
    assert found.images == ()
    assert found.unscannable == ()


@pytest.mark.parametrize(
    "source",
    [
        pytest.param({"bytes": PDF_B64}, id="bytes"),
        pytest.param({"s3Location": {"uri": "s3://b/k.txt"}}, id="s3"),
        pytest.param({"text": "hi", "bytes": PDF_B64}, id="text-and-bytes"),
        pytest.param({"content": [{"text": "hi"}, {"image": {"format": "png"}}]}, id="content-with-non-text"),
        pytest.param({"content": [{"text": "hi"}, "raw"]}, id="content-with-string"),
        pytest.param({"content": "hi"}, id="content-not-a-list"),
        pytest.param({}, id="no-source-fields"),
    ],
)
def test_converse_document_without_text_source_stays_unscannable(source):
    found = find_request_attachments(
        _converse(_converse_document(source)), CallTypes.allm_passthrough_route.value, False, False
    )

    assert found.document_texts == ()
    assert found.unscannable == ("document",)


def test_converse_text_document_in_tool_result_follows_the_tool_scope():
    data = _converse({"toolResult": {"toolUseId": "t1", "content": [_converse_document({"text": "SSN 123-45-6789"})]}})

    scanned = find_request_attachments(data, CallTypes.allm_passthrough_route.value, False, False)
    skipped = find_request_attachments(data, CallTypes.allm_passthrough_route.value, True, False)

    assert scanned.document_texts == ("notes\nSSN 123-45-6789",)
    assert skipped.document_texts == ()


def test_converse_guard_content_image_is_scanned():
    data = _converse(
        {"guardContent": {"image": _png_item()["image"]}},
        {"guardContent": {"image": {"format": "png", "source": {"s3Location": {"uri": "s3://b/k.png"}}}}},
        {"guardContent": {"text": {"text": "hi"}}},
    )

    found = find_request_attachments(data, CallTypes.allm_passthrough_route.value, False, False)

    assert list(found.images) == [_png_item()]
    assert found.unscannable == ("image (no inline bytes)",)
