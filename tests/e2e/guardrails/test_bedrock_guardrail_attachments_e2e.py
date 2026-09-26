"""Live e2e for LIT-6046: the Bedrock guardrail must scan or fail closed on
attachment-shaped request content (images, files, document blocks) on every
endpoint family, not just plain text.

The suite registers a per-request bedrock guardrail (pre_call) and a during_call
one, both pointed at BEDROCK_GUARDRAIL_IDENTIFIER/BEDROCK_GUARDRAIL_VERSION like
the rest of the guardrails suite. The converse passthrough tests register their
guardrail default_on because the passthrough route does not honor the
per-request `guardrails` selector: a bogus name returns 200 instead of a
guardrail-not-found error, so only default_on reaches the INPUT scan there.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from typing import Final, cast

import pytest
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from guardrails_client import (
    BedrockGuardrailParamsBody,
    GuardrailMode,
    GuardrailsClient,
    poll_until_blocked_stream,
)
from lifecycle import ResourceManager
from models import (
    AnthropicDocumentBlock,
    AnthropicDocumentSource,
    AnthropicMessagesBody,
    ChatBody,
    ChatMessage,
    ContentPart,
    ConverseBody,
    ConverseMessage,
    FileContentBody,
    FileContentPart,
    GuardrailRunRecord,
    ImageContentPart,
    ImageUrl,
    ResponsesApiBody,
    ResponsesInputFilePart,
    ResponsesInputImagePart,
    ResponsesInputMessage,
    ResponsesInputTextPart,
    SpendLogRow,
    TextContentPart,
)
from pydantic import JsonValue

pytestmark = pytest.mark.e2e

OPENAI_MODEL: Final = os.environ.get("E2E_BEDROCK_ATTACHMENTS_OPENAI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL: Final = os.environ.get("E2E_BEDROCK_ATTACHMENTS_ANTHROPIC_MODEL", "gemini-2.5-flash")
CONVERSE_MODEL: Final = os.environ.get("E2E_BEDROCK_ATTACHMENTS_CONVERSE_MODEL", "us.amazon.nova-lite-v1:0")

_SSN_TEXT: Final = "my social security number is 123-45-6789"


def _png_b64() -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x40\x80\xff")
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


_GIF_B64: Final = base64.b64encode(
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00"
    b",\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
).decode()


def _pdf_b64() -> str:
    stream = f"BT /F1 12 Tf 72 720 Td ({_SSN_TEXT}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return base64.b64encode(bytes(out)).decode()


def _register_bedrock(
    client: GuardrailsClient,
    resources: ResourceManager,
    name: str,
    *,
    mode: GuardrailMode = "pre_call",
    default_on: bool = False,
) -> str:
    guardrail_id = client.register(
        name,
        BedrockGuardrailParamsBody(
            mode=mode,
            default_on=default_on,
            guardrailIdentifier=os.environ["BEDROCK_GUARDRAIL_IDENTIFIER"],
            guardrailVersion=os.environ["BEDROCK_GUARDRAIL_VERSION"],
            aws_region_name=os.environ.get("BEDROCK_GUARDRAIL_REGION", "us-east-1"),
        ),
    )
    resources.defer(lambda: client.delete_guardrail(guardrail_id))
    return name


def _assert_guardrail_block(response: StreamingResponse, case: str) -> None:
    assert response.status_code == 400, (
        f"{case}: expected 400 guardrail block, got {response.status_code}: {response.body[:300]}"
    )
    assert "Violated guardrail policy" in response.body, (
        f"{case}: 400 body should name the guardrail verdict; got: {response.body[:300]}"
    )


def _has_pre_call_record(rows: list[SpendLogRow]) -> bool:
    records = (rows[0].metadata.guardrail_information if rows and rows[0].metadata else None) or []
    return any(record.guardrail_mode == "pre_call" for record in records)


def _images_scanned_total(record: GuardrailRunRecord) -> int:
    response = cast("dict[str, JsonValue]", record.guardrail_response)
    coverage: Final = response.get("guardrailCoverage")
    images: Final = coverage.get("images") if isinstance(coverage, dict) else None
    total: Final = images.get("total") if isinstance(images, dict) else None
    return total if isinstance(total, int) else 0


class TestBedrockGuardrailAttachments:
    def _chat(
        self, client: GuardrailsClient, key: str, content: list[ContentPart], guardrails: list[str]
    ) -> StreamingResponse:
        return client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(key),
            json=ChatBody(
                model=OPENAI_MODEL,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=16,
                guardrails=guardrails,
            ),
        )

    @pytest.mark.covers("guardrail.bedrock.attachments.image_only.blocks", exercised_on=["chat_completions"])
    def test_chat_completions_image_only_gif_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}")
        response = poll_until_blocked_stream(
            lambda: self._chat(
                client,
                scoped_key,
                [ImageContentPart(image_url=ImageUrl(url=f"data:image/gif;base64,{_GIF_B64}"))],
                [name],
            )
        )
        _assert_guardrail_block(response, "image-only gif message")

    @pytest.mark.covers("guardrail.bedrock.attachments.image_only.scans", exercised_on=["chat_completions"])
    def test_chat_completions_image_only_png_scanned(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}")
        response = self._chat(
            client,
            scoped_key,
            [ImageContentPart(image_url=ImageUrl(url=f"data:image/png;base64,{_png_b64()}"))],
            [name],
        )
        assert response.ok, (
            f"png image-only message should pass the scan, got {response.status_code}: {response.body[:300]}"
        )
        parsed: Final = cast("dict[str, object]", json.loads(response.body))
        request_id: Final = parsed.get("id")
        assert isinstance(request_id, str) and request_id, (
            f"200 body must carry a response id for the spend-log lookup; got: {response.body[:300]}"
        )
        rows = client.proxy.poll_logs_for_request_id(request_id, predicate=_has_pre_call_record)
        records = (rows[0].metadata.guardrail_information if rows[0].metadata else None) or []
        images_covered = [
            total
            for record in records
            if record.guardrail_mode == "pre_call"
            for total in [_images_scanned_total(record)]
            if total >= 1
        ]
        assert images_covered, f"expected a pre_call scan covering the png; got records: {records}"

    @pytest.mark.covers("guardrail.bedrock.attachments.openai_file.blocks", exercised_on=["chat_completions"])
    def test_chat_completions_file_part_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}")
        response = poll_until_blocked_stream(
            lambda: self._chat(
                client,
                scoped_key,
                [
                    TextContentPart(text=f"summarize this document {unique_marker()}"),
                    FileContentPart(
                        file=FileContentBody(filename="ssn.pdf", file_data=f"data:application/pdf;base64,{_pdf_b64()}")
                    ),
                ],
                [name],
            )
        )
        _assert_guardrail_block(response, "chat file part")

    @pytest.mark.covers("guardrail.bedrock.attachments.anthropic_document.blocks", exercised_on=["messages"])
    def test_messages_document_block_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}")
        response = poll_until_blocked_stream(
            lambda: client.proxy.transport.send(
                "/v1/messages",
                headers=client.proxy.transport.bearer(scoped_key),
                json=AnthropicMessagesBody(
                    model=ANTHROPIC_MODEL,
                    max_tokens=16,
                    guardrails=[name],
                    messages=[
                        ChatMessage(
                            role="user",
                            content=[
                                AnthropicDocumentBlock(
                                    source=AnthropicDocumentSource(media_type="application/pdf", data=_pdf_b64())
                                ),
                                TextContentPart(text=f"summarize {unique_marker()}"),
                            ],
                        )
                    ],
                ),
            )
        )
        _assert_guardrail_block(response, "anthropic document block")

    @pytest.mark.covers("guardrail.bedrock.attachments.responses_file.blocks", exercised_on=["responses"])
    def test_responses_input_file_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}")
        response = poll_until_blocked_stream(
            lambda: client.proxy.transport.send(
                "/v1/responses",
                headers=client.proxy.transport.bearer(scoped_key),
                json=ResponsesApiBody(
                    model=OPENAI_MODEL,
                    guardrails=[name],
                    input=[
                        ResponsesInputMessage(
                            role="user",
                            content=[
                                ResponsesInputTextPart(text=f"summarize {unique_marker()}"),
                                ResponsesInputFilePart(
                                    filename="ssn.pdf", file_data=f"data:application/pdf;base64,{_pdf_b64()}"
                                ),
                            ],
                        )
                    ],
                ),
            )
        )
        _assert_guardrail_block(response, "responses input_file")

    @pytest.mark.covers("guardrail.bedrock.attachments.responses_image_only.blocks", exercised_on=["responses"])
    def test_responses_input_image_only_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}")
        response = poll_until_blocked_stream(
            lambda: client.proxy.transport.send(
                "/v1/responses",
                headers=client.proxy.transport.bearer(scoped_key),
                json=ResponsesApiBody(
                    model=OPENAI_MODEL,
                    guardrails=[name],
                    input=[
                        ResponsesInputMessage(
                            role="user",
                            content=[ResponsesInputImagePart(image_url=f"data:image/gif;base64,{_GIF_B64}")],
                        )
                    ],
                ),
            )
        )
        _assert_guardrail_block(response, "responses input_image only")

    @pytest.mark.covers("guardrail.bedrock.attachments.converse_document.blocks", exercised_on=["converse"])
    def test_converse_document_block_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}", default_on=True)
        response = poll_until_blocked_stream(
            lambda: client.proxy.transport.send(
                f"/bedrock/model/{CONVERSE_MODEL}/converse",
                headers=client.proxy.transport.bearer(scoped_key),
                json=ConverseBody(
                    messages=[
                        ConverseMessage(
                            role="user",
                            content=[
                                {"document": {"format": "pdf", "name": "ssn", "source": {"bytes": _pdf_b64()}}},
                                {"text": f"summarize {unique_marker()}"},
                            ],
                        )
                    ]
                ),
            )
        )
        _assert_guardrail_block(response, "converse document block")

    @pytest.mark.covers("guardrail.bedrock.attachments.converse_image.blocks", exercised_on=["converse"])
    def test_converse_image_block_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        _register_bedrock(client, resources, f"e2e-bedrock-att-{unique_marker()}", default_on=True)
        response = poll_until_blocked_stream(
            lambda: client.proxy.transport.send(
                f"/bedrock/model/{CONVERSE_MODEL}/converse",
                headers=client.proxy.transport.bearer(scoped_key),
                json=ConverseBody(
                    messages=[
                        ConverseMessage(
                            role="user",
                            content=[
                                {"image": {"format": "gif", "source": {"bytes": _GIF_B64}}},
                                {"text": f"describe {unique_marker()}"},
                            ],
                        )
                    ]
                ),
            )
        )
        _assert_guardrail_block(response, "converse image block")

    @pytest.mark.covers("guardrail.bedrock.attachments.during_call_file.blocks", exercised_on=["chat_completions"])
    def test_during_call_file_part_blocks(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = _register_bedrock(client, resources, f"e2e-bedrock-during-{unique_marker()}", mode="during_call")
        response = poll_until_blocked_stream(
            lambda: self._chat(
                client,
                scoped_key,
                [
                    TextContentPart(text=f"summarize this document {unique_marker()}"),
                    FileContentPart(
                        file=FileContentBody(filename="ssn.pdf", file_data=f"data:application/pdf;base64,{_pdf_b64()}")
                    ),
                ],
                [name],
            )
        )
        _assert_guardrail_block(response, "during_call file part")
