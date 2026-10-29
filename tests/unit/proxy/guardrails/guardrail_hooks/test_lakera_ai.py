from collections.abc import Mapping
from itertools import chain
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.caching import DualCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation
from litellm.types.guardrails import GuardrailEventHooks

_LAKERA_REQUEST_BODY: Final = TypeAdapter(dict[str, object])


class _Metadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    applied_guardrails: tuple[str, ...] = ()


def _applied_guardrails(data: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        chain.from_iterable(
            _Metadata.model_validate(data.get(bucket) or {}).applied_guardrails
            for bucket in ("metadata", "litellm_metadata")
        )
    )


def _lakera(transport: httpx.MockTransport) -> lakeraAI_Moderation:
    return lakeraAI_Moderation(
        guardrail_name="lakera-v1",
        event_hook=GuardrailEventHooks.pre_call,
        default_on=True,
        api_key="lk-test",
        api_base="https://lakera.test",
        async_handler=AsyncHTTPHandler(transport=transport),
    )


def _refuse_any_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected Lakera request with body {request.content!r}")


def _clean_scan_of(expected_input: str) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        assert _LAKERA_REQUEST_BODY.validate_json(request.content)["input"] == expected_input
        return httpx.Response(200, json={"results": [{"flagged": False}]})

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
async def test_request_without_text_lakera_reads_skips_the_vendor_and_is_not_listed() -> None:
    guardrail: Final = _lakera(httpx.MockTransport(_refuse_any_request))
    data: Final[dict[str, object]] = {
        "model": "m",
        "state": "paid with sk-live-1234",
        "fields": {"total": {"type": "number"}},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-9876"), cache=DualCache(), data=data, call_type="aextraction"
    )

    assert _applied_guardrails(data) == ()


@pytest.mark.asyncio
async def test_clean_scan_of_string_input_lists_the_guardrail() -> None:
    guardrail: Final = _lakera(_clean_scan_of("is this a refund request?"))
    data: Final[dict[str, object]] = {
        "model": "m",
        "input": "is this a refund request?",
        "questions": [{"type": "predicate", "instructions": "Is it a refund request?"}],
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-9876"), cache=DualCache(), data=data, call_type="adecisions"
    )

    assert _applied_guardrails(data) == ("lakera-v1",)
