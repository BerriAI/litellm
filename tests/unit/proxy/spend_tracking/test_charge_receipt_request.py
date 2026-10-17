from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.charge_receipt_capture import ChargeReceiptRuntime
from litellm.proxy.spend_tracking.charge_receipt_request import required_receipt_key
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptSettings, ChargeReceiptStore


async def no_database(query: str, *parameters: object) -> Sequence[Mapping[str, object]]:
    pytest.fail("Receipt key selection must not access a database")


@pytest.mark.parametrize("representation", ("raw", "bearer", "lowercase-bearer", "hashed", "other"))
def test_receipt_policy_uses_the_authenticated_virtual_key_identity(representation: str) -> None:
    raw: Final = "sk-receipt-policy-unit-only"
    original: Final = UserAPIKeyAuth(api_key=raw)
    assert original.api_key is not None
    assert original.token == original.api_key
    inputs: Final = {
        "raw": raw,
        "bearer": "Bearer " + raw,
        "lowercase-bearer": "bearer " + raw,
        "hashed": original.api_key,
        "other": "sk-another-policy-unit-only",
    }
    runtime: Final = ChargeReceiptRuntime(
        ChargeReceiptSettings(key_hashes=frozenset((original.api_key,))),
        ChargeReceiptStore(no_database),
    )
    authenticated: Final = UserAPIKeyAuth(api_key=inputs[representation])
    assert required_receipt_key(authenticated, runtime) == (None if representation == "other" else original.api_key)
    assert required_receipt_key(authenticated, None) is None
