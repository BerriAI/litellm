"""Bedrock Converse document blocks, video blocks and citations on `/v1/chat/completions`.

An OpenAI `file` part (`file.file_data` as a base64 data URI) on a Converse deployment must become a
Converse `document` block: `format` from the MIME type (`pdf`, `csv`, ...), `source.bytes` holding the
base64 payload, and a deterministic `name` of `DocumentPDFmessages_<16 hex>_<format>` derived from a
SHA-256 of the payload and its length (`BedrockImageProcessor._create_bedrock_block`,
`litellm/litellm_core_utils/prompt_templates/factory.py:4214`). Bedrock rejects a request with two
documents of the same name, so a repeated file gets a `_2` suffix, and a user turn holding only documents
gets a `.` text block, since Converse refuses a document without text (`factory.py:4766`). A base64 video
data URI becomes a `video` block whose `format` uses Bedrock's spelling (`3gp` is `three_gp`).

On the way back, Converse `citationsContent` blocks (Nova grounding, web search) are turned into the cited
text plus OpenAI `url_citation` annotations whose offsets are shifted by the length of earlier cited
blocks (`AmazonConverseConfig._transform_citations_to_annotations`,
`litellm/llms/bedrock/chat/converse_transformation.py:2846`). The Converse transformation is due for the
Rust port, and every one of these shapes is checked by Bedrock or read by the caller literally.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock3_support import unique_text
from integration.providers.dec1_bedrock_support import (
    CLAUDE_SONNET_45,
    NOVA_PRO,
    bedrock,
    body_of,
    converse_reply,
)
from pydantic import JsonValue

PDF_BASE64: Final = base64.b64encode(b"%PDF-1.4 synthetic dec1 bedrock3 document").decode()
CSV_BASE64: Final = base64.b64encode(b"city,temp\nOslo,3\n").decode()
VIDEO_BASE64: Final = base64.b64encode(b"\x00\x00\x00\x18ftypmp42synthetic").decode()


def _document_name(payload: str, document_format: str) -> str:
    """The name `_create_bedrock_block` derives: sha256(first 64 KiB of the whitespace-free payload + its length)."""
    normalized: Final = "".join(payload.split()).encode()
    hasher: Final = hashlib.sha256()
    hasher.update(normalized[: 64 * 1024])
    hasher.update(str(len(normalized)).encode())
    return f"DocumentPDFmessages_{hasher.hexdigest()[:16]}_{document_format}"


def _file_part(mime: str, payload: str) -> dict[str, JsonValue]:
    return {"type": "file", "file": {"file_data": f"data:{mime};base64,{payload}"}}


def _ok(_: Request) -> Reply:
    return converse_reply()


def _user_blocks(rig_body: dict[str, JsonValue], index: int = 0) -> list[JsonValue]:
    messages: Final = rig_body["messages"]
    assert isinstance(messages, list)
    message: Final = messages[index]
    assert isinstance(message, dict)
    content: Final = message["content"]
    assert isinstance(content, list)
    return content


# ------------------------------------------------------------------ Documents


def test_a_pdf_file_part_becomes_a_converse_document_block_with_the_hashed_name(gateway: Gateway) -> None:
    """A user turn with text `summarize` and a `file` part `data:application/pdf;base64,<payload>`.

    Bedrock must see the text block and then `{"document": {"source": {"bytes": <payload>}, "format": "pdf",
    "name": "DocumentPDFmessages_<sha256 prefix>_pdf"}}`, the name computed from the payload as documented.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "summarize"}, _file_part("application/pdf", PDF_BASE64)],
                    }
                ]
            }
        )
        assert response.status_code == 200, response.text
        blocks: Final = _user_blocks(body_of(rig.one_post()))
        assert blocks[0] == {"text": "summarize"}, blocks
        assert blocks[1] == {
            "document": {"source": {"bytes": PDF_BASE64}, "format": "pdf", "name": _document_name(PDF_BASE64, "pdf")}
        }, blocks


def test_a_csv_file_part_is_sent_with_the_csv_document_format(gateway: Gateway) -> None:
    """A `file` part with MIME `text/csv`.

    `text/*` counts as a document, and `_get_document_format` maps it to Converse format `csv`, so the
    document block must carry `format: "csv"` and a name ending in `_csv`.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "read"}, _file_part("text/csv", CSV_BASE64)]}
                ]
            }
        )
        assert response.status_code == 200, response.text
        document: Final = _user_blocks(body_of(rig.one_post()))[1]["document"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert document["format"] == "csv", document  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert document["name"] == _document_name(CSV_BASE64, "csv"), document  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_a_document_only_user_turn_gets_a_dot_text_block_after_the_document(gateway: Gateway) -> None:
    """A user turn with only a PDF `file` part and no text.

    Converse refuses a document without text, so `_with_text_when_document_only` must append
    `{"text": "."}` after the document block: the turn is `[document, {"text": "."}]`.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat_body(
            {"messages": [{"role": "user", "content": [_file_part("application/pdf", PDF_BASE64)]}]}
        )
        assert response.status_code == 200, response.text
        blocks: Final = _user_blocks(body_of(rig.one_post()))
        assert len(blocks) == 2, blocks
        assert "document" in blocks[0], blocks  # pyright: ignore[reportOperatorIssue]
        assert blocks[1] == {"text": "."}, blocks


def test_the_same_document_in_two_turns_gets_a_suffixed_second_name(gateway: Gateway) -> None:
    """The same PDF is attached in the first user turn and again in the third.

    Bedrock rejects duplicate document names, so the first keeps `DocumentPDFmessages_<hash>_pdf` (prompt
    cache prefixes stay stable) and the second must be renamed `<that name>_2`.
    """
    name: Final = _document_name(PDF_BASE64, "pdf")
    with bedrock(gateway, _ok, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "first"}, _file_part("application/pdf", PDF_BASE64)],
                    },
                    {"role": "assistant", "content": "noted"},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "again"}, _file_part("application/pdf", PDF_BASE64)],
                    },
                ]
            }
        )
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert _user_blocks(sent, 0)[1]["document"]["name"] == name  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert _user_blocks(sent, 2)[1]["document"]["name"] == f"{name}_2"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_an_unsupported_document_mime_type_never_reaches_bedrock(gateway: Gateway) -> None:
    """A `file` part with MIME `application/x-dec1-unknown`, which has no Converse document format.

    `_get_document_format` raises `No supported extensions for MIME type`, so the caller must get a 4xx
    or 5xx error and Bedrock must receive no POST.
    UNVERIFIED: the exact status (400 expected from the bad-request mapping, possibly 500).
    """
    with bedrock(gateway, _ok, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "x"},
                            _file_part("application/x-dec1-unknown", PDF_BASE64),
                        ],
                    }
                ]
            }
        )
        assert response.status_code >= 400, response.text
        assert rig.posts() == ()


# ------------------------------------------------------------------ Video


def test_a_base64_mp4_becomes_a_converse_video_block(gateway: Gateway) -> None:
    """A Nova Pro deployment gets an `image_url` part `data:video/mp4;base64,<payload>`.

    Video MIME types are not documents, and `mp4` is a supported video format, so Bedrock must see
    `{"video": {"source": {"bytes": <payload>}, "format": "mp4"}}`.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{NOVA_PRO}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{VIDEO_BASE64}"}},
                        ],
                    }
                ]
            }
        )
        assert response.status_code == 200, response.text
        assert _user_blocks(body_of(rig.one_post()))[1] == {
            "video": {"source": {"bytes": VIDEO_BASE64}, "format": "mp4"}
        }


def test_a_3gp_video_is_sent_with_bedrocks_three_gp_format_name(gateway: Gateway) -> None:
    """A `data:video/3gp;base64,...` part on Nova Pro.

    Bedrock spells that format `three_gp` (`_CONVERSE_VIDEO_FORMAT_NAMES`, `factory.py:2425`), so the video
    block must carry `format: "three_gp"`, never `3gp`.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{NOVA_PRO}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": f"data:video/3gp;base64,{VIDEO_BASE64}"}},
                        ],
                    }
                ]
            }
        )
        assert response.status_code == 200, response.text
        assert _user_blocks(body_of(rig.one_post()))[1]["video"]["format"] == "three_gp"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_an_unsupported_video_format_never_reaches_bedrock(gateway: Gateway) -> None:
    """A `data:video/avi;base64,...` part; `avi` is not in Converse's video or image list.

    `validate_format` raises `Unsupported image format: avi`, so the caller gets an error status and Bedrock
    receives no POST. UNVERIFIED: the exact status (400 expected).
    """
    with bedrock(gateway, _ok, model=f"bedrock/{NOVA_PRO}") as rig:
        response: Final = rig.chat_body(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": f"data:video/avi;base64,{VIDEO_BASE64}"}},
                        ],
                    }
                ]
            }
        )
        assert response.status_code >= 400, response.text
        assert rig.posts() == ()


# ------------------------------------------------------------------ Citations


def _citations_block(text: str, title: str, url: str, start: int, end: int) -> dict[str, JsonValue]:
    return {
        "citationsContent": {
            "content": [{"text": text}],
            "citations": [
                {"title": title, "source": url, "location": {"searchResultLocation": {"start": start, "end": end}}}
            ],
        }
    }


def test_a_citations_only_answer_becomes_content_with_url_citation_annotations(gateway: Gateway) -> None:
    """Nova answers only one `citationsContent` block: text `Oslo is cold.` citing `Weather` at
    `https://example.com/oslo`, `searchResultLocation` 0 to 4.

    The caller must get `content: "Oslo is cold."` and `annotations == [{"type": "url_citation",
    "url_citation": {"start_index": 0, "end_index": 4, "title": "Weather", "url": "https://example.com/oslo"}}]`.
    """
    with bedrock(
        gateway,
        lambda _: converse_reply(
            content=[_citations_block("Oslo is cold.", "Weather", "https://example.com/oslo", 0, 4)]
        ),
        model=f"bedrock/{NOVA_PRO}",
    ) as rig:
        response: Final = rig.chat(unique_text())
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "Oslo is cold."
        assert message["annotations"] == [
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": 0,
                    "end_index": 4,
                    "title": "Weather",
                    "url": "https://example.com/oslo",
                },
            }
        ], message


def test_annotation_offsets_of_a_second_citations_block_are_shifted_by_the_first_text(gateway: Gateway) -> None:
    """Two `citationsContent` blocks: `First. ` (7 chars, citing 0..6) then `Second.` (citing 0..7).

    The content must be `First. Second.` and the second annotation must start at 7 and end at 14, since
    offsets are relative to the concatenated cited text.
    """
    with bedrock(
        gateway,
        lambda _: converse_reply(
            content=[
                _citations_block("First. ", "A", "https://example.com/a", 0, 6),
                _citations_block("Second.", "B", "https://example.com/b", 0, 7),
            ]
        ),
        model=f"bedrock/{NOVA_PRO}",
    ) as rig:
        response: Final = rig.chat(unique_text())
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "First. Second."
        spans: Final = [
            (item["url_citation"]["start_index"], item["url_citation"]["end_index"], item["url_citation"]["url"])
            for item in message["annotations"]
        ]
        assert spans == [(0, 6, "https://example.com/a"), (7, 14, "https://example.com/b")], spans


def test_citations_beside_real_text_keep_the_text_and_add_no_annotations(gateway: Gateway) -> None:
    """Nova answers a text block `Plain answer` and a `citationsContent` block.

    The text has letters, so the cited text is not stitched in: `content` stays `Plain answer` and no
    `annotations` key is set (`converse_transformation.py:3080`), because the offsets would not line up.
    """
    with bedrock(
        gateway,
        lambda _: converse_reply(
            content=[{"text": "Plain answer"}, _citations_block("Cited.", "C", "https://example.com/c", 0, 6)]
        ),
        model=f"bedrock/{NOVA_PRO}",
    ) as rig:
        response: Final = rig.chat(unique_text())
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "Plain answer"
        assert not message.get("annotations"), message


def test_punctuation_only_text_gets_the_cited_text_prepended_with_annotations(gateway: Gateway) -> None:
    """Nova answers `citationsContent` `Rain today` (citing 0..4) and a text block holding only `.`.

    Bedrock sometimes emits the cited sentence in the citation and only punctuation as text; the proxy
    must return `content: "Rain today."` with the annotation at 0..4.
    """
    with bedrock(
        gateway,
        lambda _: converse_reply(
            content=[_citations_block("Rain today", "R", "https://example.com/r", 0, 4), {"text": "."}]
        ),
        model=f"bedrock/{NOVA_PRO}",
    ) as rig:
        response: Final = rig.chat(unique_text())
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "Rain today."
        assert message["annotations"][0]["url_citation"]["start_index"] == 0
        assert message["annotations"][0]["url_citation"]["end_index"] == 4


def test_a_citation_without_a_search_result_location_is_dropped_from_annotations(gateway: Gateway) -> None:
    """One `citationsContent` block whose only citation has `location: {"documentChar": {...}}`.

    Only `searchResultLocation` spans become annotations, so the caller gets the cited text as content and
    no annotations at all.
    """
    block: Final[dict[str, JsonValue]] = {
        "citationsContent": {
            "content": [{"text": "From the doc."}],
            "citations": [
                {
                    "title": "D",
                    "source": "doc",
                    "location": {"documentChar": {"documentIndex": 0, "start": 0, "end": 4}},
                }
            ],
        }
    }
    with bedrock(gateway, lambda _: converse_reply(content=[block]), model=f"bedrock/{NOVA_PRO}") as rig:
        response: Final = rig.chat(unique_text())
        assert response.status_code == 200, response.text
        message: Final = response.json()["choices"][0]["message"]
        assert message["content"] == "From the doc."
        assert not message.get("annotations"), message
