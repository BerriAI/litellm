import base64

import pytest

from litellm.proxy.guardrails.guardrail_hooks.akto.akto_attachments import (
    Attachment,
    RequestAttachments,
    request_attachments,
    without_attachment_content,
)

PDF_B64 = base64.b64encode(b"%PDF-1.7 card 4111").decode()


PNG_B64 = base64.b64encode(b"\x89PNG screenshot").decode()


def test_request_attachments_reads_every_shape_in_every_message():
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}],
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "check these"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/remote.png"}},
                    {
                        "type": "file",
                        "file": {"file_data": f"data:application/pdf;base64,{PDF_B64}", "filename": "c.pdf"},
                    },
                    {"type": "file", "file": {"file_id": "file-123"}},
                    {
                        "type": "document",
                        "title": "notes.txt",
                        "source": {"type": "text", "media_type": "text/plain", "data": "hi"},
                    },
                    {"type": "document", "source": {"type": "url", "url": "https://example.com/spec.pdf"}},
                    {
                        "type": "tool_result",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}
                        ],
                    },
                ],
            },
        ]
    }

    assert request_attachments(request_data) == RequestAttachments(
        attachments=(
            Attachment("attachment-0.png", "image", content=PNG_B64),
            Attachment("remote.png", "image", url="https://example.com/remote.png"),
            Attachment("c.pdf", "file", content=PDF_B64),
            Attachment("notes.txt", "file", content=base64.b64encode(b"hi").decode()),
            Attachment("spec.pdf", "file", url="https://example.com/spec.pdf"),
            Attachment("attachment-7.png", "image", content=PNG_B64),
        ),
        unsendable_count=1,
    ), "only the file_id reference has nothing to send"


def test_request_attachments_reads_responses_api_input():
    request_data = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_data": f"data:application/pdf;base64,{PDF_B64}", "filename": "r.pdf"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"},
                ],
            }
        ]
    }

    assert request_attachments(request_data) == RequestAttachments(
        attachments=(
            Attachment("r.pdf", "file", content=PDF_B64),
            Attachment("attachment-1.png", "image", content=PNG_B64),
        ),
        unsendable_count=0,
    )


def test_request_attachments_names_files_by_their_type():
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "title": "Q3 report",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64},
                    },
                    {"type": "file", "file": {"file_data": PDF_B64, "filename": "../../etc/raw.pdf"}},
                    {"type": "input_audio", "input_audio": {"data": f"{PDF_B64[:8]}\n{PDF_B64[8:]}", "format": "wav"}},
                    {"type": "image_url", "image_url": "https://example.com/plain.png"},
                    {"type": "file", "file": {"file_data": "not base64!", "filename": "bad.pdf"}},
                    {"type": "file", "file": "not a file block"},
                    {"type": "document", "source": {"type": "file", "file_id": "file_011"}},
                ],
            }
        ]
    }

    assert request_attachments(request_data) == RequestAttachments(
        attachments=(
            Attachment("Q3 report.pdf", "file", content=PDF_B64),
            Attachment("raw.pdf", "file", content=PDF_B64),
            Attachment("attachment-2.wav", "audio", content=PDF_B64),
            Attachment("plain.png", "image", url="https://example.com/plain.png"),
        ),
        unsendable_count=2,
    ), "names get an extension from the media type; raw and line-wrapped base64 are sent; invalid base64 is not"


def test_a_malformed_attachment_url_is_named_by_position():
    request_data = {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "https://[::1/x.png"}]}]
    }

    [attachment] = request_attachments(request_data).attachments
    assert (attachment.filename, attachment.url) == ("attachment-0", "https://[::1/x.png")


def test_a_url_attachment_is_named_by_its_decoded_path():
    request_data = {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "https://x.io/My%20Doc.png"}]}]
    }

    [attachment] = request_attachments(request_data).attachments
    assert attachment.filename == "My Doc.png"


PADDED_B64 = base64.b64encode(b"%PDF-1.7 card").decode()


@pytest.mark.parametrize(
    ("block", "content"),
    [
        ({"type": "image_url", "image_url": f"DATA:image/png;base64,{PNG_B64}"}, PNG_B64),
        ({"type": "input_audio", "input_audio": {"data": PADDED_B64.rstrip("=")}}, PADDED_B64),
        ({"type": "input_audio", "input_audio": {"data": PADDED_B64[:-1]}}, PADDED_B64),
        ({"type": "image_url", "image_url": f"  data:image/png;BASE64,{PNG_B64}"}, PNG_B64),
        ({"type": "input_audio", "input_audio": {"data": base64.urlsafe_b64encode(b"\xfb\xff").decode()}}, "+/8="),
        ({"type": "image_url", "image_url": "data:text/plain,card%204111"}, base64.b64encode(b"card 4111").decode()),
    ],
)
def test_attachment_bytes_are_sent_as_standard_base64(block, content):
    request_data = {"messages": [{"role": "user", "content": [block]}]}

    [attachment] = request_attachments(request_data).attachments
    assert attachment.content == content


def test_audio_without_data_counts_as_unsendable():
    request_data = {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]}

    assert request_attachments(request_data) == RequestAttachments(attachments=(), unsendable_count=1)


def test_responses_api_tool_outputs_are_checked_and_stripped():
    image = {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"}
    request_data = {"input": [{"type": "function_call_output", "call_id": "c1", "output": [image]}]}

    [attachment] = request_attachments(request_data).attachments
    assert attachment.content == PNG_B64
    [item] = without_attachment_content(request_data["input"])
    assert item["output"] == ({"type": "input_image"},)


@pytest.mark.parametrize("output", [1, {"a": 1}, "text"])
def test_an_unexpected_output_field_does_not_hide_a_messages_attachments(output):
    image = {"type": "image_url", "image_url": f"data:image/png;base64,{PNG_B64}"}
    request_data = {"messages": [{"role": "user", "content": [image], "output": output}]}

    assert [a.content for a in request_attachments(request_data).attachments] == [PNG_B64]


def test_a_document_of_text_blocks_is_sent_as_a_text_file():
    source = {"type": "content", "content": [{"type": "text", "text": "card"}, {"type": "text", "text": "4111"}]}
    document = {"type": "document", "title": "notes", "source": source}
    request_data = {"messages": [{"role": "user", "content": [document]}]}

    assert request_attachments(request_data).attachments == (
        Attachment("notes.txt", "file", content=base64.b64encode(b"card\n4111").decode()),
    )


def test_an_uppercase_remote_url_is_sent_as_a_url():
    request_data = {
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": " HTTPS://x.io/a.png "}]}]
    }

    [attachment] = request_attachments(request_data).attachments
    assert attachment.url == "HTTPS://x.io/a.png"


def test_a_document_of_one_text_string_is_sent_as_a_text_file():
    document = {"type": "document", "source": {"type": "content", "content": "card 4111"}}
    request_data = {"messages": [{"role": "user", "content": [document]}]}

    [attachment] = request_attachments(request_data).attachments
    assert attachment.content == base64.b64encode(b"card 4111").decode()


def test_images_inside_a_document_of_blocks_are_checked_too():
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}
    document = {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "a"}, image]}}
    request_data = {"messages": [{"role": "user", "content": [document]}]}

    assert [a.content for a in request_attachments(request_data).attachments] == [
        base64.b64encode(b"a").decode(),
        PNG_B64,
    ]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image_url", "image_url": "data:image/png;base64,"},
        {"type": "document", "source": {"type": "content", "content": []}},
    ],
)
def test_attachments_with_nothing_inside_are_unsendable(block):
    request_data = {"messages": [{"role": "user", "content": [block]}]}

    assert request_attachments(request_data) == RequestAttachments(attachments=(), unsendable_count=1)


def test_a_data_uri_without_a_media_type_gets_no_extension():
    image = {"type": "image_url", "image_url": f"data:;base64,{PNG_B64}"}
    request_data = {"messages": [{"role": "user", "content": [image]}]}

    [attachment] = request_attachments(request_data).attachments
    assert attachment.filename == "attachment-0"


def test_images_in_a_document_inside_a_tool_result_are_checked():
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}}
    document = {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "hi"}, image]}}
    tool_result = {"type": "tool_result", "tool_use_id": "t1", "content": [document]}
    request_data = {"messages": [{"role": "user", "content": [tool_result]}]}

    assert [a.content for a in request_attachments(request_data).attachments] == [
        base64.b64encode(b"hi").decode(),
        PNG_B64,
    ]


@pytest.mark.parametrize("video_url", [{"url": f"data:video/mp4;base64,{PNG_B64}"}, f"data:video/mp4;base64,{PNG_B64}"])
def test_a_video_is_sent_as_a_file_and_kept_out_of_the_text_check(video_url):
    request_data = {"messages": [{"role": "user", "content": [{"type": "video_url", "video_url": video_url}]}]}

    assert request_attachments(request_data).attachments == (Attachment("attachment-0.mp4", "file", content=PNG_B64),)
    [message] = without_attachment_content(request_data["messages"])
    assert message["content"] == ({"type": "video_url"},)


@pytest.mark.parametrize(
    "block",
    [
        {"type": "document", "source": {"type": "text", "data": "a\ud800"}},
        {"type": "document", "source": {"type": "content", "content": "a\ud800"}},
        {"type": "image_url", "image_url": "data:text/plain,a\ud800"},
    ],
)
def test_text_that_isnt_valid_utf8_is_still_sent(block):
    request_data = {"messages": [{"role": "user", "content": [block]}]}

    [attachment] = request_attachments(request_data).attachments
    assert base64.b64decode(attachment.content or "") == "a\ud800".encode(errors="surrogatepass")
