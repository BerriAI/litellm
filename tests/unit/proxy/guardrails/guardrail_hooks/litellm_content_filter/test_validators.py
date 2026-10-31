from itertools import product
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.validators import (
    passes_iban_mod97,
    passes_luhn,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("4111 1111 1111 1111", True),
        ("4111-1111-1111-1112", False),
        ("٤" + "١" * 15, True),
        ("4111 1111 1111 111x", False),
        ("", False),
    ],
)
def test_passes_luhn(value: str, expected: bool):
    assert passes_luhn(value) is expected


@pytest.mark.parametrize(
    "valid_number",
    ["4111111111111111", "4012888888881881", "5555555555554444", "378282246310005", "6011000990139424"],
)
def test_passes_luhn_rejects_every_single_digit_change(valid_number: str):
    single_digit_changes: Final = tuple(
        valid_number[:position] + replacement + valid_number[position + 1 :]
        for position, replacement in product(range(len(valid_number)), "0123456789")
    )

    assert passes_luhn(valid_number)
    assert not any(passes_luhn(changed) for changed in single_digit_changes if changed != valid_number)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("GB82 WEST 1234 5698 7654 32", True),
        ("gb82west12345698765432", True),
        ("GB82WEST12345698765433", False),
        ("GB82", False),
        ("GB82WEST1234569876543é", False),
        ("DE89370400440532013000EUR", True),
        ("DE88370400440532013000EUR", False),
    ],
)
def test_passes_iban_mod97(value: str, expected: bool):
    assert passes_iban_mod97(value) is expected
