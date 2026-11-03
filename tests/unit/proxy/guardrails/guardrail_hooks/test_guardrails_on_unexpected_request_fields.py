"""Guardrails on built-in pass-through routes read client JSON that request setup hasn't normalized."""

from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation
from litellm.proxy.guardrails.guardrail_hooks.noma.noma_v2 import NomaV2Guardrail

NOT_AN_OBJECT: Final = (None, 0, False, "", [], "text", 7)


@pytest.mark.parametrize("value", NOT_AN_OBJECT)
def test_generic_guardrail_reads_metadata_past_a_litellm_metadata_that_is_not_an_object(value: object) -> None:
    guardrail: Final = GenericGuardrailAPI(api_base="http://vendor.invalid", guardrail_name="generic")

    metadata: Final = guardrail._extract_user_api_key_metadata(
        {"metadata": {"user_api_key_hash": "hash"}, "litellm_metadata": value}
    )

    assert metadata["user_api_key_hash"] == "hash"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", NOT_AN_OBJECT)
async def test_noma_v2_scans_a_request_whose_litellm_metadata_is_not_an_object(value: object) -> None:
    guardrail: Final = NomaV2Guardrail(api_key="key", api_base="http://vendor.invalid", guardrail_name="noma")
    verdict: Final = httpx.Response(
        200, json={"action": "NONE"}, request=httpx.Request("POST", "http://vendor.invalid")
    )

    with patch.object(guardrail.async_handler, "post", new=AsyncMock(return_value=verdict)):
        result: Final = await guardrail.apply_guardrail(
            inputs={"texts": ["hello"]},
            request_data={"metadata": {}, "litellm_metadata": value},
            input_type="request",
        )

    assert result["texts"] == ["hello"]


@pytest.mark.asyncio
@pytest.mark.parametrize("token_arrays", [[1, 2, 3], [[1, 2], [3, 4]]])
async def test_legacy_lakera_has_no_text_to_scan_in_token_array_embeddings(token_arrays: list[object]) -> None:
    guardrail: Final = lakeraAI_Moderation(api_key="key", api_base="http://vendor.invalid")
    vendor_error: Final = httpx.HTTPStatusError(
        "vendor called",
        request=httpx.Request("POST", "http://vendor.invalid"),
        response=httpx.Response(500, text="vendor called"),
    )

    with patch.object(guardrail.async_handler, "post", new=AsyncMock(side_effect=vendor_error)):
        await guardrail._check(
            data={"input": token_arrays},
            user_api_key_dict=UserAPIKeyAuth(api_key="key"),
            call_type="pass_through_endpoint",
        )


@pytest.mark.asyncio
async def test_legacy_lakera_still_scans_string_inputs() -> None:
    guardrail: Final = lakeraAI_Moderation(api_key="key", api_base="http://vendor.invalid")
    vendor_error: Final = httpx.HTTPStatusError(
        "vendor called",
        request=httpx.Request("POST", "http://vendor.invalid"),
        response=httpx.Response(500, text="vendor called"),
    )

    with (
        patch.object(guardrail.async_handler, "post", new=AsyncMock(side_effect=vendor_error)),
        pytest.raises(Exception, match="vendor called"),
    ):
        await guardrail._check(
            data={"input": ["hello"]},
            user_api_key_dict=UserAPIKeyAuth(api_key="key"),
            call_type="pass_through_endpoint",
        )
