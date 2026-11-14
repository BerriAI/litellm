"""PDF pages and model names that once took seconds to minutes to count, kept here so a slowdown shows on CodSpeed.

Each input is smaller than the one that exposed the slowdown, since the old cost grew with the same input size."""

import base64
from itertools import accumulate, chain
from typing import Final

import pytest

from litellm.litellm_core_utils.token_counter import token_counter

TEXT_LINE: Final = "Revenue grew eleven percent while churn fell to two percent."


def _pdf_base64(text: str, page_resources: str, drawing: bytes, xobjects: tuple[bytes, ...] = ()) -> str:
    escaped: Final = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream: Final = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1") + drawing
    bodies: Final = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [ 5 0 R ] >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources {page_resources} /Contents 4 0 R >>".encode(),
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


def _form(resources_number: int, drawing: bytes) -> bytes:
    return (
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources {resources_number} 0 R "
        f"/Length {len(drawing)} >>\nstream\n".encode()
        + drawing
        + b"\nendstream"
    )


def _count_pdf(pdf_base64: str) -> int:
    document: Final = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}}
    return token_counter(model="gpt-4o", messages=[{"role": "user", "content": [document]}])


def _page_of_drawing_operators() -> str:
    return _pdf_base64("", "<< >>", b" q Q" * 50_000)


def _page_of_forms_sharing_resources() -> str:
    form_count: Final = 300
    shared_resources: Final = (
        "<< /XObject << " + " ".join(f"/F{index} {8 + index} 0 R" for index in range(form_count)) + " >> >>"
    ).encode()
    forms: Final = tuple(_form(7, b"/F0 Do") for _ in range(form_count))
    return _pdf_base64(
        TEXT_LINE,
        "<< /Font << /F1 3 0 R >> /XObject << /X0 6 0 R >> >>",
        b" q 10 0 0 10 0 0 cm /X0 Do Q",
        (forms[0], shared_resources, *forms),
    )


def _page_of_parents_sharing_a_child_form() -> str:
    parent_count: Final = 100
    first: Final = 6
    child_number: Final = first + 2 * parent_count
    child_drawing: Final = b" ".join((b"/D Do",) * 4000)
    child: Final = (
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Length {len(child_drawing)} >>\nstream\n".encode()
        + child_drawing
        + b"\nendstream"
    )
    parents: Final = tuple(
        chain.from_iterable(
            (_form(first + 2 * index + 1, b"/C Do"), f"<< /XObject << /C {child_number} 0 R >> >>".encode())
            for index in range(parent_count)
        )
    )
    page_resources: Final = (
        "<< /Font << /F1 3 0 R >> /XObject << "
        + " ".join(f"/P{index} {first + 2 * index} 0 R" for index in range(parent_count))
        + " >> >>"
    )
    page_drawing: Final = b" " + b" ".join(f"/P{index} Do".encode() for index in range(parent_count))
    return _pdf_base64(TEXT_LINE, page_resources, page_drawing, (*parents, child))


PAGE_OF_DRAWING_OPERATORS: Final = _page_of_drawing_operators()
PAGE_OF_FORMS_SHARING_RESOURCES: Final = _page_of_forms_sharing_resources()
PAGE_OF_PARENTS_SHARING_A_CHILD_FORM: Final = _page_of_parents_sharing_a_child_form()
BLANK_PAGE_COUNT: Final = _count_pdf(_pdf_base64("", "<< >>", b""))
TEXT_PAGE_COUNT: Final = _count_pdf(_pdf_base64(TEXT_LINE, "<< /Font << /F1 3 0 R >> >>", b""))
LONG_UNKNOWN_MODEL_NAME: Final = "unknown-" + "a-" * 5_000
NAME_SAMPLE: Final = "Ceci est un texte en français: éèàç. " * 8
UNKNOWN_MODEL_NAME_COUNT: Final = token_counter(model="my-private-model", text=NAME_SAMPLE)


@pytest.mark.benchmark
def test_pdf_page_of_drawing_operators_and_no_image() -> None:
    assert _count_pdf(PAGE_OF_DRAWING_OPERATORS) == BLANK_PAGE_COUNT


@pytest.mark.benchmark
def test_pdf_page_of_forms_sharing_one_resources_dictionary() -> None:
    assert _count_pdf(PAGE_OF_FORMS_SHARING_RESOURCES) == TEXT_PAGE_COUNT


@pytest.mark.benchmark
def test_pdf_page_of_parent_forms_sharing_one_child_form() -> None:
    assert _count_pdf(PAGE_OF_PARENTS_SHARING_A_CHILD_FORM) > TEXT_PAGE_COUNT


@pytest.mark.benchmark
def test_token_counter_long_unknown_model_name() -> None:
    assert token_counter(model=LONG_UNKNOWN_MODEL_NAME, text=NAME_SAMPLE) == UNKNOWN_MODEL_NAME_COUNT
