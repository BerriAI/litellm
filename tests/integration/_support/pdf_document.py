import base64
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Final

from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class Page:
    width: int = 612
    height: int = 792
    text: str | None = None


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


def pdf_document(pages: Sequence[Page]) -> dict[str, JsonValue]:
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes(pages)).decode(),
        },
    }
