import json
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "fal-ai/moondream3-preview/query"
_PROMPT: Final = "what is in this image?"
_COST_MAP_PATH: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_COST_MAP: Final = TypeAdapter(dict[str, dict[str, object]])


def _catalog_cost(key: str, field: str) -> float:
    cost_map: Final = _COST_MAP.validate_json(_COST_MAP_PATH.read_bytes())
    cost_value: Final = cost_map[key][field]
    assert isinstance(cost_value, (int, float))
    return float(cost_value)


def _approx(value: float) -> object:
    return pytest.approx(value, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]  # pytest lacks typed approx stubs


@pytest.mark.covers("other.provider_wire.fal_ai.moondream3_chat_query_wire_and_token_pricing")
def test_fal_moondream3_chat_sends_prompt_image_and_reasoning(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        assert request.target == f"/{_MODEL}"
        assert request.headers["content-type"] == "application/json"
        assert _JSON_OBJECT.validate_json(request.body) == {
            "prompt": _PROMPT,
            "image_url": "https://example.com/pic.png",
            "reasoning": False,
            "temperature": 0.2,
        }
        return Reply(
            body=json.dumps(
                {
                    "output": "a red circle on a blue background",
                    "reasoning": "inspected the shapes",
                    "finish_reason": "stop",
                    "usage_info": {
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "prefill_time_ms": 1.0,
                        "decode_time_ms": 2.0,
                        "ttft_ms": 1.5,
                    },
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fal_ai/{_MODEL}", api_base=wire.url, api_key="synthetic-fal-key")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
                        ],
                    }
                ],
                "reasoning_effort": "none",
                "temperature": 0.2,
            },
        )
        assert response.status_code == 200, response.text
        payload: Final = _JSON_OBJECT.validate_json(response.content)
        assert payload["choices"] == [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "a red circle on a blue background",
                    "reasoning_content": "inspected the shapes",
                },
            }
        ]
        assert payload["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        cost: Final = float(response.headers["x-litellm-response-cost"])
        assert cost == _approx(
            11 * _catalog_cost(f"fal_ai/{_MODEL}", "input_cost_per_token")
            + 7 * _catalog_cost(f"fal_ai/{_MODEL}", "output_cost_per_token")
        )
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", f"/{_MODEL}")]


@pytest.mark.covers("other.provider_wire.fal_ai.chat_non_string_reasoning_effort_rejected_before_wire")
def test_fal_moondream3_chat_rejects_non_string_reasoning_effort_before_the_wire(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"provider must not be reached: {request.method} {request.target}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fal_ai/{_MODEL}", api_base=wire.url, api_key="synthetic-fal-key")
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
                        ],
                    }
                ],
                "reasoning_effort": {"level": "low"},
            },
        )
        assert response.status_code == 400, response.text
        assert "reasoning_effort" in response.text
        assert wire.drain() == ()
