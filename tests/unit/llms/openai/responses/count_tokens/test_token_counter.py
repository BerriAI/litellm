import json
import operator
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.openai.responses.count_tokens.token_counter import OpenAITokenCounter
from litellm.types.utils import TokenCountResponse

COUNT_URL: Final = "https://api.openai.com/v1/responses/input_tokens"


async def _count(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> TokenCountResponse | None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock:
        respx.post(COUNT_URL).mock(return_value=response)
        return await OpenAITokenCounter().count_tokens(
            model_to_use="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"api_key": "sk-test"}},
            request_model="openai/gpt-4o",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_total"),
    [
        ({"input_tokens": 15}, 15),
        ({"input_tokens": "15"}, 15),
        ({"input_tokens": 15.0}, 15),
        ({"input_tokens": True}, 1),
        ({"input_tokens": -3}, -3),
        ({}, 0),
        ({"object": "response.input_tokens", "error": {"message": "429 rate limit"}}, 0),
        ({"input_tokens": 7, "object": "response.input_tokens", "nested": {"a": [1, None]}}, 7),
    ],
)
async def test_count_reports_upstream_input_tokens_and_keeps_the_raw_payload(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object], expected_total: int
) -> None:
    result: Final = await _count(monkeypatch, httpx.Response(200, json=payload))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": expected_total,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "original_response": payload,
            "error": False,
            "error_message": None,
            "status_code": None,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejected_count",
    [
        None,
        15.5,
        "Request timed out",
        "429 Too Many Requests",
        [15],
        [*range(403), "Request timed out"],
        {"value": 429},
    ],
)
async def test_non_integer_input_tokens_becomes_an_error_response_carrying_the_validation_message(
    monkeypatch: pytest.MonkeyPatch, rejected_count: object
) -> None:
    with pytest.raises(ValidationError) as rejection:
        TokenCountResponse.model_validate(
            {
                "total_tokens": rejected_count,
                "request_model": "openai/gpt-4o",
                "model_used": "gpt-4o",
                "tokenizer_type": "openai_api",
            }
        )

    result: Final = await _count(monkeypatch, httpx.Response(200, json={"input_tokens": rejected_count}))

    assert result is not None
    assert [error["loc"] for error in rejection.value.errors()] == [("total_tokens",)]
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "original_response": None,
            "error": True,
            "error_message": str(rejection.value),
            "status_code": 500,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [[*({"input_tokens": position} for position in range(403)), "429"], "input_tokens", 15, 0, False],
)
async def test_count_payload_that_is_not_an_object_becomes_an_error_response_naming_the_failed_lookup(
    monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    with pytest.raises(AttributeError) as failed_lookup:
        operator.attrgetter("get")(payload)

    result: Final = await _count(monkeypatch, httpx.Response(200, json=payload))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "original_response": None,
            "error": True,
            "error_message": str(failed_lookup.value),
            "status_code": 500,
        }
    )


@pytest.mark.asyncio
async def test_null_count_payload_falls_back_to_local_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _count(monkeypatch, httpx.Response(200, content=b"null")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_status", "upstream_text"),
    [(429, "rate limit exceeded"), (403, "forbidden"), (408, "Request timed out")],
)
async def test_upstream_error_status_becomes_an_error_response_with_that_status_and_text(
    monkeypatch: pytest.MonkeyPatch, upstream_status: int, upstream_text: str
) -> None:
    result: Final = await _count(monkeypatch, httpx.Response(upstream_status, text=upstream_text))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "original_response": None,
            "error": True,
            "error_message": upstream_text,
            "status_code": upstream_status,
        }
    )


@pytest.mark.asyncio
async def test_undecodable_count_body_becomes_a_500_error_response_carrying_the_decoder_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(json.JSONDecodeError) as undecodable:
        json.loads("not json")

    result: Final = await _count(monkeypatch, httpx.Response(200, text="not json"))

    assert result is not None
    assert repr(result.model_dump()) == repr(
        {
            "total_tokens": 0,
            "request_model": "openai/gpt-4o",
            "model_used": "gpt-4o",
            "tokenizer_type": "openai_api",
            "original_response": None,
            "error": True,
            "error_message": f"CountTokens processing error: {undecodable.value}",
            "status_code": 500,
        }
    )
