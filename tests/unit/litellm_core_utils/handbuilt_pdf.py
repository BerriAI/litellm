import base64
from itertools import accumulate, chain
from typing import Final


def pdf_base64(
    pages: tuple[str, ...],
    width: int = 612,
    height: int = 792,
    xobjects: tuple[bytes, ...] = (),
    drawing: bytes = b"",
    page_resources: str | None = None,
) -> str:
    """`xobjects` are numbered from 4 + 2 * len(pages), every page lists the first one as its /X0 XObject, and every
    page's content stream ends with `drawing`. `page_resources` replaces every page's /Resources dictionary."""
    first_xobject: Final = 4 + 2 * len(pages)
    page_xobjects: Final = f" /XObject << /X0 {first_xobject} 0 R >>" if xobjects else ""
    resources: Final = page_resources if page_resources is not None else f"<< /Font << /F1 3 0 R >>{page_xobjects} >>"

    def page_objects(index: int, text: str) -> tuple[bytes, bytes]:
        escaped: Final = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream: Final = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1") + drawing
        content: Final = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        page: Final = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources {resources} /Contents {4 + 2 * index} 0 R >>"
        ).encode()
        return content, page

    kids: Final = " ".join(f"{5 + 2 * index} 0 R" for index in range(len(pages)))
    bodies: Final = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Count {len(pages)} /Kids [ {kids} ] >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        *chain.from_iterable(page_objects(index, text) for index, text in enumerate(pages)),
        *xobjects,
    )
    header: Final = b"%PDF-1.4\n"
    objects: Final = tuple(
        f"{number} 0 obj\n".encode() + body + b"\nendobj\n" for number, body in enumerate(bodies, start=1)
    )
    offsets: Final = accumulate((len(header), *(len(obj) for obj in objects[:-1])))
    xref: Final = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode() + b"".join(
        f"{offset:010d} 00000 n \n".encode() for offset in offsets
    )
    trailer: Final = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{len(header) + sum(len(obj) for obj in objects)}\n%%EOF\n"
    ).encode()
    return base64.b64encode(header + b"".join(objects) + xref + trailer).decode()
