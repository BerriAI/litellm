import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import Final

from integration._support.wire import Reply
from pydantic import JsonValue

PDF_MEDIA_TYPE: Final = "application/pdf"
COUNT_TOKENS_TARGET: Final = "/v1/messages/count_tokens"
COUNT_REFUSED: Final = Reply(
    status=400,
    body=json.dumps(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "count_tokens is not supported"}}
    ).encode(),
)


@dataclass(frozen=True, slots=True)
class Page:
    width: int = 612
    height: int = 792
    text: str | None = None


LETTER: Final = Page()
NARROW: Final = Page(width=100, height=1000)
_RENDERED_TOKENS: Final = MappingProxyType({(612, 792): 1534, (100, 1000): 328})


def rendered_tokens(pages: Sequence[Page]) -> int:
    return sum(_RENDERED_TOKENS[(page.width, page.height)] for page in pages)


def _escaped(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(page: Page) -> bytes:
    stream: Final = f"BT /F1 12 Tf 72 {page.height - 72} Td ({_escaped(page.text or '')}) Tj ET".encode()
    return b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"


def _page_object(page: Page, contents_id: int | None) -> bytes:
    contents: Final = f" /Contents {contents_id} 0 R" if contents_id is not None else ""
    return (
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page.width} {page.height}]"
        f" /Resources << /Font << /F1 1 0 R >> >>{contents} >>"
    ).encode()


def _page_bodies(pages: Sequence[Page], starts: Sequence[int]) -> Iterator[bytes]:
    for page, start in zip(pages, starts):
        if page.text is None:
            yield _page_object(page, None)
        else:
            yield _content_stream(page)
            yield _page_object(page, start)


def pdf_bytes(pages: Sequence[Page]) -> bytes:
    starts: Final = tuple(accumulate((1 if page.text is None else 2 for page in pages), initial=3))
    page_ids: Final = tuple(start + (0 if page.text is None else 1) for start, page in zip(starts, pages))
    kids: Final = " ".join(f"{identity} 0 R" for identity in page_ids)
    bodies: Final = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        *_page_bodies(pages, starts),
        b"<< /Type /Catalog /Pages 2 0 R >>",
    )
    header: Final = b"%PDF-1.4\n"
    objects: Final = tuple(
        f"{number} 0 obj\n".encode() + body + b"\nendobj\n" for number, body in enumerate(bodies, start=1)
    )
    offsets: Final = tuple(accumulate((len(chunk) for chunk in objects), initial=len(header)))
    entries: Final = b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[:-1])
    trailer: Final = (
        f"xref\n0 {len(bodies) + 1}\n".encode()
        + b"0000000000 65535 f \n"
        + entries
        + f"trailer\n<< /Size {len(bodies) + 1} /Root {starts[-1]} 0 R >>\nstartxref\n{offsets[-1]}\n%%EOF\n".encode()
    )
    return header + b"".join(objects) + trailer


def encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def data_url(media_type: str, raw: bytes) -> str:
    return f"data:{media_type};base64,{encoded(raw)}"


def pdf_data_url(pages: Sequence[Page]) -> str:
    return data_url(PDF_MEDIA_TYPE, pdf_bytes(pages))


def base64_source(data: JsonValue, media_type: JsonValue = PDF_MEDIA_TYPE) -> dict[str, JsonValue]:
    return {"type": "base64", "media_type": media_type, "data": data}


def document(source: Mapping[str, JsonValue], **fields: JsonValue) -> dict[str, JsonValue]:
    return {"type": "document", "source": dict(source), **fields}


def pdf_document(pages: Sequence[Page], **fields: JsonValue) -> dict[str, JsonValue]:
    return document(base64_source(encoded(pdf_bytes(pages))), **fields)


def text_document(text: str) -> dict[str, JsonValue]:
    return document({"type": "text", "media_type": "text/plain", "data": text})


def chat_file(file_data: JsonValue, filename: JsonValue = "document.pdf") -> dict[str, JsonValue]:
    return {"type": "file", "file": {"filename": filename, "file_data": file_data}}


def responses_input_file(file_data: JsonValue, filename: JsonValue = "document.pdf") -> dict[str, JsonValue]:
    return {"type": "input_file", "filename": filename, "file_data": file_data}


def messages_body(model: str, blocks: Sequence[JsonValue], text: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [*blocks, {"type": "text", "text": text}]}],
        **fields,
    }


def chat_body(model: str, parts: Sequence[JsonValue], text: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [*parts, {"type": "text", "text": text}]}],
        **fields,
    }


def responses_body(model: str, items: Sequence[JsonValue], text: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_output_tokens": 64,
        "input": [{"role": "user", "content": [*items, {"type": "input_text", "text": text}]}],
        **fields,
    }
