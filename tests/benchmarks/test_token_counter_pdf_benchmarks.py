"""PDF counting workloads retained for CodSpeed."""

import base64
from functools import reduce
from itertools import accumulate, chain
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.litellm_core_utils.token_counter import (
    _PDF_EMPTY_RESOURCES,
    PDF_TEXT_OBJECT,
    _pdf_bounded_text_samples,
    _PdfStreamMetrics,
    _price_pdf_pages,
    token_counter,
)


class _PdfBenchmarkForm:
    def get_data(self) -> bytes:
        return b"q Q\n"

    def get(self, key: str) -> str | None:
        return "/Form" if key == "/Subtype" else None

    def items(self) -> tuple[tuple[str, object], ...]:
        return ()


def _pdf_page(content: bytes, with_form: bool = False) -> SimpleNamespace:
    xobjects: Final = {"/Fm1": _PdfBenchmarkForm()} if with_form else {}
    resources: Final = {"/XObject": xobjects}
    return SimpleNamespace(
        mediabox=SimpleNamespace(width=612.0, height=792.0),
        get_contents=lambda: SimpleNamespace(get_data=lambda: content),
        extract_text=lambda: "",
        get=lambda key: resources if key == "/Resources" else None,
    )


def _pdf_doubling_form_metrics(depth: int, leaf_content: bytes) -> _PdfStreamMetrics:
    leaf_metrics: Final = _PdfStreamMetrics(
        content=leaf_content,
        content_bytes=len(leaf_content),
        operand_bytes=0,
        resources=_PDF_EMPTY_RESOURCES,
        forms=(),
    )
    return reduce(
        lambda child, _: _PdfStreamMetrics(
            content=b"/Fm Do\n/Fm Do\n",
            content_bytes=0,
            operand_bytes=0,
            resources=_PDF_EMPTY_RESOURCES,
            forms=(child, child),
        ),
        range(depth),
        leaf_metrics,
    )


@pytest.mark.benchmark
def test_unterminated_pdf_literal_string_scan():  # test-quality-ok: CodSpeed performance benchmark
    page: Final = _pdf_page(b"(" * (4 * 1024 * 1024))

    _price_pdf_pages((page,), len)


@pytest.mark.benchmark
def test_pdf_string_operand_scan_after_escaped_parentheses():  # test-quality-ok: CodSpeed performance benchmark
    path_operator: Final = b"10 10 m 20 20 l S\n"
    padding_repetitions: Final = 64 * 1024 // len(path_operator)
    padding_bytes: Final = padding_repetitions * len(path_operator)
    padding: Final = path_operator * padding_repetitions + b" " * (64 * 1024 - padding_bytes)
    page: Final = _pdf_page(padding + b"%" + b"\\(" * 8_192 + path_operator * 1_000)

    _price_pdf_pages((page,), len)


@pytest.mark.benchmark
def test_pdf_invocation_scan_ignores_unterminated_comments():  # test-quality-ok: CodSpeed performance benchmark
    page: Final = _pdf_page(b"/a %" * (4 * 1024 * 1024 // 4), with_form=True)

    priced: Final = _price_pdf_pages((page,), len)

    assert priced > 0


@pytest.mark.benchmark
def test_pdf_inline_image_scan_repeated_bi_without_id():  # test-quality-ok: CodSpeed performance benchmark
    page: Final = _pdf_page(b"BI " * 1_000_000)

    _price_pdf_pages((page,), len)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "content",
    [
        b"BT " * 1_000_000,
        b"BT (" * 1_000_000,
        b"BT " + b"(ET) " * 1_000_000,
    ],
    ids=["repeated-bt", "repeated-open-literal", "repeated-et-literal"],
)
def test_pdf_text_object_scan(content: bytes):  # test-quality-ok: CodSpeed performance benchmark
    tuple(PDF_TEXT_OBJECT.finditer(content))


@pytest.mark.benchmark
def test_pdf_16m_text_object_stream_is_capped():  # test-quality-ok: CodSpeed performance benchmark
    text_object: Final = b"BT (x) Tj ET\n"
    repetitions: Final = 16 * 1024 * 1024 // len(text_object) + 1
    page: Final = _pdf_page(text_object * repetitions)

    _price_pdf_pages((page,), len)


@pytest.mark.benchmark
def test_pdf_textless_doubling_form_tree_skips_subtrees():  # test-quality-ok: CodSpeed performance benchmark
    metrics: Final = _pdf_doubling_form_metrics(30, b"q Q\n")

    _pdf_bounded_text_samples(metrics)


@pytest.mark.benchmark
def test_pdf_text_leaf_doubling_form_tree_stays_within_sample_cap():  # test-quality-ok: CodSpeed performance benchmark
    metrics: Final = _pdf_doubling_form_metrics(30, b"BT (x) Tj ET\n")

    _pdf_bounded_text_samples(metrics)


@pytest.mark.benchmark
@pytest.mark.parametrize("repetitions", [100_000, 1_000_000], ids=["100k", "1m"])
def test_pdf_inline_image_scan_no_ei(repetitions: int):  # test-quality-ok: CodSpeed performance benchmark
    page: Final = _pdf_page(b"BI /W 1 " + b" ID x" * repetitions)

    _price_pdf_pages((page,), len)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "comments",
    [
        b"%comment\r\n" * 23,
        b"% a   \r\n" * 40,
    ],
    ids=["23-crlf-comments", "40-crlf-comments-with-trailing-spaces"],
)
def test_pdf_invocation_scan_with_crlf_comments(comments: bytes):  # test-quality-ok: CodSpeed performance benchmark
    page: Final = _pdf_page(b"BT /F1 " + comments + b"12 Tf (hello) Tj ET", with_form=True)

    priced: Final = _price_pdf_pages((page,), len)

    assert priced > 0


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
    document: Final = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_base64},
    }
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
