from typing import Final

import pytest

from litellm.llms.microsoft_365_copilot.common_utils import extract_caller_assertion
from litellm.types.proxy.litellm_pre_call_utils import SecretFields


@pytest.mark.parametrize(
    ("authorization", "expected"),
    [
        ("bEaReR header.payload.signature", "header.payload.signature"),
        ("Bearer header..signature", None),
        ("Bearer not-a-jwt", None),
        ("Basic header.payload.signature", None),
        ("Bearer header.payload.signature extra", None),
    ],
)
def test_extract_caller_assertion_requires_a_three_part_bearer_jwt(
    authorization: str,
    expected: str | None,
) -> None:
    secret_fields: Final[SecretFields] = SecretFields(raw_headers={"aUtHoRiZaTiOn": authorization})

    assertion: Final = extract_caller_assertion(secret_fields)

    assert assertion == expected
