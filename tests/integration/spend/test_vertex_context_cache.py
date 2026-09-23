import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse

_CACHE_TOKENS: Final = 2500
_GEMINI_MODEL: Final = "gemini-2.5-pro"


@pytest.mark.covers("quota_management.spend_tracking.vertex_context_cache.creation_and_read_tokens_charged")
@pytest.mark.timeout(180)
def test_gemini_context_cache_creation_is_billed_once_at_its_own_rate(gateway: Gateway) -> None:
    scenario_id: Final = f"sc-{uuid.uuid4().hex}"
    handle: Final = register_scenario(
        scenario_id,
        RoutedResponse(
            content_type="application/x-routed",
            routes={
                f"GET /models/{_GEMINI_MODEL}:cachedContents": JsonResponse(
                    content_type="application/json", body={"cachedContents": []}
                ),
                f"POST /models/{_GEMINI_MODEL}:cachedContents": JsonResponse(
                    content_type="application/json",
                    body={
                        "name": "cachedContents/created-cache",
                        "model": f"models/{_GEMINI_MODEL}",
                        "createTime": "2026-01-01T00:00:00Z",
                        "expireTime": "2026-01-01T01:00:00Z",
                        "usageMetadata": {"totalTokenCount": _CACHE_TOKENS},
                    },
                ),
                f"POST /models/{_GEMINI_MODEL}:generateContent": JsonResponse(
                    content_type="application/json",
                    body={
                        "candidates": [
                            {
                                "content": {"role": "model", "parts": [{"text": "cached answer"}]},
                                "finishReason": "STOP",
                                "index": 0,
                            }
                        ],
                        "modelVersion": _GEMINI_MODEL,
                        "usageMetadata": {
                            "promptTokenCount": 2520,
                            "cachedContentTokenCount": _CACHE_TOKENS,
                            "candidatesTokenCount": 10,
                            "totalTokenCount": 2530,
                        },
                    },
                ),
            },
        ),
    )
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model=f"gemini/{_GEMINI_MODEL}",
            api_key="sk-scripted-provider",
            api_base=handle.api_base(),
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            cache_read_input_token_cost=0.0004,
            cache_creation_input_token_cost=0.0005,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "cacheable prefix " * 4000,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {"role": "user", "content": "answer from the cache"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        usage: Final = object_value(body["usage"])
        assert usage["prompt_tokens"] == 2520 + _CACHE_TOKENS, body
        assert usage["cache_creation_input_tokens"] == _CACHE_TOKENS, body
        expected_spend: Final = (
            20 * 0.001 + _CACHE_TOKENS * 0.0004 + _CACHE_TOKENS * 0.0005 + 10 * 0.002
        )
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected_spend)
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (body["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["prompt_tokens"] == 2520 + _CACHE_TOKENS, rows
        assert float(rows[0]["spend"]) == pytest.approx(expected_spend), rows
