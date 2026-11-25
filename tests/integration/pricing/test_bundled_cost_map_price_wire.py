import json
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.openai_wire import answering_model_discovery, posted_targets
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gpt-4o-mini"
_API_KEY: Final = "synthetic-openai-key"
_PROMPT_TOKENS: Final = 1000
_COMPLETION_TOKENS: Final = 500
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PACKAGED_PRICES: Final = Path(__file__).resolve().parents[3] / "litellm" / "model_prices_and_context_window_backup.json"
_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl-bundled-price",
        "object": "chat.completion",
        "created": 1,
        "model": _BACKEND,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "bundled price control"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": _PROMPT_TOKENS,
            "completion_tokens": _COMPLETION_TOKENS,
            "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        },
    }
).encode()


def _peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    return Reply(body=_REPLY)


def test_a_deployment_with_no_configured_price_is_charged_at_the_packaged_cost_map_rates(gateway: Gateway) -> None:
    packaged: Final = _JSON_OBJECT.validate_json(_PACKAGED_PRICES.read_bytes())[_BACKEND]
    assert isinstance(packaged, dict)
    input_rate: Final = packaged["input_cost_per_token"]
    output_rate: Final = packaged["output_cost_per_token"]
    assert isinstance(input_rate, float) and isinstance(output_rate, float)
    assert input_rate > 0 and output_rate > 0
    with wire_server(answering_model_discovery(_peer)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/{_BACKEND}", api_base=f"{wire.url}/v1", api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "synthetic priced request"}]},
        )
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(
            _PROMPT_TOKENS * input_rate + _COMPLETION_TOKENS * output_rate, rel=1e-9
        )
        assert posted_targets(wire) == ("/v1/chat/completions",)
