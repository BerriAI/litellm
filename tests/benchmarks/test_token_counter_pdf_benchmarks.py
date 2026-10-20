from types import SimpleNamespace
from typing import Final

import pytest

from litellm.litellm_core_utils.token_counter import _price_pdf_pages


def _pdf_page(content: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        mediabox=SimpleNamespace(width=612.0, height=792.0),
        get_contents=lambda: SimpleNamespace(get_data=lambda: content),
        extract_text=lambda: "",
        get=lambda key: {"/XObject": {}} if key == "/Resources" else None,
    )


@pytest.mark.benchmark
def test_unterminated_pdf_literal_string_scan():
    page: Final = _pdf_page(b"(" * (4 * 1024 * 1024))

    _price_pdf_pages((page,), len)


@pytest.mark.benchmark
def test_pdf_string_operand_scan_after_escaped_parentheses():
    path_operator: Final = b"10 10 m 20 20 l S\n"
    padding_repetitions: Final = 64 * 1024 // len(path_operator)
    padding_bytes: Final = padding_repetitions * len(path_operator)
    padding: Final = path_operator * padding_repetitions + b" " * (64 * 1024 - padding_bytes)
    page: Final = _pdf_page(padding + b"%" + b"\\(" * 8_192 + path_operator * 1_000)

    _price_pdf_pages((page,), len)


@pytest.mark.benchmark
def test_pdf_invocation_scan_ignores_unterminated_comments():
    page: Final = _pdf_page(b"/a %" * (4 * 1024 * 1024 // 4))

    _price_pdf_pages((page,), len)
