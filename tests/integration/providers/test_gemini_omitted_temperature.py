"""A Gemini 3+ request that omits temperature reaches the provider without one: streamed on each endpoint, and in
a burst of mixed-endpoint requests spread across the proxy's workers."""

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.provider import SharedProvider
from integration._support.wire import Reply, Request
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.gemini import (
    GEMINI_3_8_FLASH_TEST_CASE as CHAT_COMPLETIONS_TEST_CASE,
)
from integration.translation.messages.bases.gemini import GEMINI_3_8_FLASH_TEST_CASE as MESSAGES_TEST_CASE
from integration.translation.responses.bases.gemini import GEMINI_3_8_FLASH_TEST_CASE as RESPONSES_TEST_CASE
from pydantic import JsonValue

GENERATE_TARGET: Final = "/models/gemini-3.8-flash:generateContent"
STREAM_TARGET: Final = "/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
ENDPOINT_CASES: Final = (CHAT_COMPLETIONS_TEST_CASE, MESSAGES_TEST_CASE, RESPONSES_TEST_CASE)
BURST: Final = ENDPOINT_CASES * 4
BURST_ANSWER: Final = "I need to inspect `shop/validation.py` next."
STREAM_USAGE: Final = {
    "promptTokenCount": 10,
    "candidatesTokenCount": 2,
    "totalTokenCount": 12,
    "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
}


def _frame(payload: Mapping[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _streamed_reply() -> Reply:
    opening: Final = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello."}]}, "index": 0}],
        "modelVersion": "gemini-3.8-flash",
        "responseId": "stream-gemini-3-8-flash",
    }
    closing: Final = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": STREAM_USAGE,
        "modelVersion": "gemini-3.8-flash",
        "responseId": "stream-gemini-3-8-flash",
    }
    return Reply(content_type="text/event-stream", chunks=(_frame(opening), _frame(closing)))


def _generation_config(request: Request) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(request.body)["generationConfig"])


def _expected_generation_config(case: TranslationTestCase) -> dict[str, JsonValue]:
    expected: Final = object_value(case.expected_provider_request["generationConfig"])
    assert "temperature" not in expected, expected
    return expected


@pytest.mark.parametrize("case", ENDPOINT_CASES, ids=lambda case: case.litellm_endpoint)
def test_streamed_request_without_temperature_forwards_none(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    provider.expect(_streamed_reply())
    response: Final = gateway.request("POST", case.litellm_endpoint, {**case.litellm_request, "stream": True})
    assert response.status_code == 200, response.text
    assert "Hello." in response.text, response.text
    received: Final = provider.received()
    assert [(request.method, request.target) for request in received] == [("POST", STREAM_TARGET)]
    assert _generation_config(received[0]) == _expected_generation_config(case)


def test_concurrent_mixed_endpoint_burst_without_temperature_forwards_none(
    gateway: Gateway, provider: SharedProvider
) -> None:
    reply_body: Final = json.dumps(CHAT_COMPLETIONS_TEST_CASE.mock_provider_response).encode()
    provider.expect(*(Reply(body=reply_body) for _ in BURST))
    with ThreadPoolExecutor(max_workers=len(BURST)) as pool:
        responses: Final = tuple(
            pool.map(lambda case: gateway.request("POST", case.litellm_endpoint, case.litellm_request), BURST)
        )
    assert [response.status_code for response in responses] == [200] * len(BURST), [r.text for r in responses]
    assert all(BURST_ANSWER in response.text for response in responses), [r.text for r in responses]
    received: Final = provider.received()
    assert [(request.method, request.target) for request in received] == [("POST", GENERATE_TARGET)] * len(BURST)
    expected_configs: Final = [_expected_generation_config(CHAT_COMPLETIONS_TEST_CASE)] * len(BURST)
    assert sorted(map(json.dumps, (_generation_config(r) for r in received))) == sorted(
        map(json.dumps, expected_configs)
    )
