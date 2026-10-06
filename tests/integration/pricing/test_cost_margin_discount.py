import json
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from pydantic import BaseModel, JsonValue


class CostEstimate(BaseModel):
    model: str
    cost_per_request: float
    input_cost_per_request: float
    output_cost_per_request: float
    margin_cost_per_request: float


def _settings_row() -> dict[str, JsonValue]:
    rows: Final = read_rows('SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s', ("litellm_settings",))
    assert len(rows) == 1, rows
    value: Final = rows[0]["param_value"]
    return object_value(json.loads(value) if isinstance(value, str) else value)


def _patch(gateway: Gateway, path: str, body: Mapping[str, JsonValue], message: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("PATCH", path, body)
    assert response.status_code == 200, response.text
    parsed: Final = object_value(response.json())
    assert parsed == {"message": message, "status": "success", "values": dict(body)}, response.text
    return parsed


def _chat_cost(gateway: Gateway, model: str, key: str) -> tuple[str, float]:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": f"margin {uuid.uuid4().hex}"}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    request_id: Final = string_value(object_value(response.json())["id"])
    return request_id, float(response.headers["x-litellm-response-cost"])


def _logged_spend(request_id: str) -> float:
    rows: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return float(str(rows[0]["spend"]))


def _estimated(gateway: Gateway, model: str) -> CostEstimate:
    response: Final = gateway.request(
        "POST", "/cost/estimate", {"model": model, "input_tokens": 20, "output_tokens": 20}
    )
    assert response.status_code == 200, response.text
    return CostEstimate.model_validate(response.json())


def test_discount_and_margin_reprice_a_request_and_survive_rejection(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        original_discount: Final = gateway.get("/config/cost_discount_config")["values"]
        original_margin: Final = gateway.get("/config/cost_margin_config")["values"]

        def restore() -> None:
            _patch(
                gateway,
                "/config/cost_discount_config",
                original_discount,
                "Cost discount configuration updated successfully",
            )
            _patch(
                gateway,
                "/config/cost_margin_config",
                original_margin,
                "Cost margin configuration updated successfully",
            )

        scenario.cleanups.callback(restore)
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])

        # Phase A: 5% openai discount, openai margin of 10% + $0.001 fixed, 5% global margin
        _patch(
            gateway,
            "/config/cost_discount_config",
            {"openai": 0.05},
            "Cost discount configuration updated successfully",
        )
        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": {"percentage": 0.1, "fixed_amount": 0.001}, "global": 0.05},
            "Cost margin configuration updated successfully",
        )
        assert gateway.get("/config/cost_discount_config")["values"] == {"openai": 0.05}
        assert gateway.get("/config/cost_margin_config")["values"] == {
            "openai": {"percentage": 0.1, "fixed_amount": 0.001},
            "global": 0.05,
        }
        persisted: Final = _settings_row()
        assert persisted["cost_discount_config"] == {"openai": 0.05}, persisted
        assert persisted["cost_margin_config"] == {
            "openai": {"percentage": 0.1, "fixed_amount": 0.001},
            "global": 0.05,
        }, persisted

        expected: Final = 0.06 * 0.95 * 1.1 + 0.001
        request_id, header_cost = _chat_cost(gateway, model, key)
        assert header_cost == pytest.approx(expected)
        assert _logged_spend(request_id) == pytest.approx(expected)
        estimate: Final = _estimated(gateway, model)
        assert estimate.model == model
        assert estimate.cost_per_request == pytest.approx(expected), estimate
        # the estimate reports raw token shares; the 5% discount is the gap between them and the
        # discounted subtotal the margin was applied to: 0.06 - 0.057 = 0.06 * 0.05
        assert estimate.input_cost_per_request == pytest.approx(0.02), estimate
        assert estimate.output_cost_per_request == pytest.approx(0.04), estimate
        discounted_subtotal: Final = estimate.cost_per_request - estimate.margin_cost_per_request
        assert (
            estimate.input_cost_per_request + estimate.output_cost_per_request - discounted_subtotal
            == pytest.approx(0.06 * 0.05)
        ), estimate
        assert estimate.margin_cost_per_request == pytest.approx(0.06 * 0.95 * 0.1 + 0.001), estimate

        # Phase B: no discount, 100% global margin doubles the base cost
        _patch(gateway, "/config/cost_discount_config", {}, "Cost discount configuration updated successfully")
        _patch(
            gateway,
            "/config/cost_margin_config",
            {"global": 1},
            "Cost margin configuration updated successfully",
        )
        request_id, header_cost = _chat_cost(gateway, model, key)
        assert header_cost == pytest.approx(0.12)
        assert _logged_spend(request_id) == pytest.approx(0.12)

        # Rejected writes leave the GET surface and the DB row untouched
        margin_response: Final = gateway.request("PATCH", "/config/cost_margin_config", {"not_a_provider": 0.1})
        assert margin_response.status_code == 400, margin_response.text
        discount_response: Final = gateway.request("PATCH", "/config/cost_discount_config", {"openai": 1.5})
        assert discount_response.status_code == 400, discount_response.text
        global_discount: Final = gateway.request("PATCH", "/config/cost_discount_config", {"global": 0.1})
        assert global_discount.status_code == 400, global_discount.text
        assert gateway.get("/config/cost_discount_config")["values"] == {}
        assert gateway.get("/config/cost_margin_config")["values"] == {"global": 1}
        persisted = _settings_row()
        assert persisted["cost_discount_config"] == {}, persisted
        assert persisted["cost_margin_config"] == {"global": 1}, persisted
