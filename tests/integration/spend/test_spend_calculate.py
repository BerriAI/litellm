import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value, string_value


@pytest.mark.covers("quota_management.spend_tracking.spend_calculate.rejects_unpriced_model")
def test_spend_calculate_rejects_unpriced_model_with_400(gateway: Gateway) -> None:
    model: Final = f"openrouter/integration-unpriced-{uuid.uuid4().hex}"
    response: Final = gateway.request(
        "POST",
        "/spend/calculate",
        {"model": model, "messages": [{"role": "user", "content": "price this request"}]},
    )
    assert response.status_code == 400, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.text)["error"])
    assert error["type"] == "invalid_request_error", response.text
    assert error["param"] == "model", response.text
    assert model in string_value(error["message"]), response.text


GEMINI_LIVE_PREVIEW_MODELS: Final = (
    "gemini-live-2.5-flash-preview-native-audio-09-2025",
    "gemini/gemini-live-2.5-flash-preview-native-audio-09-2025",
)


@pytest.mark.parametrize("model", GEMINI_LIVE_PREVIEW_MODELS)
@pytest.mark.covers("quota_management.spend_tracking.spend_calculate.live_preview_cached_tokens_cost_fresh_rate")
def test_live_preview_entry_charges_cached_tokens_at_the_fresh_rate(gateway: Gateway, model: str) -> None:
    def cost_with_cached_tokens(cached_tokens: int) -> float:
        response: Final = gateway.request(
            "POST",
            "/spend/calculate",
            {
                "completion_response": {
                    "id": "chatcmpl-live-preview",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "live preview answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 101_000,
                        "completion_tokens": 0,
                        "total_tokens": 101_000,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                }
            },
        )
        assert response.status_code == 200, response.text
        cost: Final = object_value(JSON_OBJECT.validate_json(response.text))["cost"]
        assert isinstance(cost, int | float)
        return float(cost)

    cached_cost: Final = cost_with_cached_tokens(100_000)
    fresh_cost: Final = cost_with_cached_tokens(0)
    assert fresh_cost > 0, fresh_cost
    assert cached_cost == pytest.approx(fresh_cost), (
        f"the entry publishes no cached rate, so 100k cached tokens must bill like fresh ones: {cached_cost} vs {fresh_cost}"
    )
