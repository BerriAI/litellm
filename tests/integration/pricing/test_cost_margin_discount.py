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


def _chat_cost(gateway: Gateway, model: str, key: str, expected: float) -> str:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": f"margin {uuid.uuid4().hex}"}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    request_id: Final = string_value(object_value(response.json())["id"])
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected), response.text
    return request_id


def _assert_logged_spends(expectations: tuple[tuple[str, float], ...]) -> None:
    placeholders: Final = ", ".join("%s" for _ in expectations)
    request_ids: Final = tuple(request_id for request_id, _ in expectations)
    rows: Final = eventually(
        lambda: read_rows(
            f'SELECT request_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id IN ({placeholders})', request_ids
        ),
        lambda values: len(values) == len(expectations),
        seconds=40,
    )
    actual: Final = {string_value(row["request_id"]): float(str(row["spend"])) for row in rows}
    assert actual == pytest.approx(dict(expectations)), rows


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

        _patch(gateway, "/config/cost_discount_config", {}, "Cost discount configuration updated successfully")
        _patch(
            gateway,
            "/config/cost_margin_config",
            {},
            "Cost margin configuration updated successfully",
        )
        base_request_id: Final = _chat_cost(gateway, model, key, 0.06)

        _patch(
            gateway,
            "/config/cost_discount_config",
            {"openai": 0.05},
            "Cost discount configuration updated successfully",
        )
        discounted_request_id: Final = _chat_cost(gateway, model, key, 0.057)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": 0.05},
            "Cost margin configuration updated successfully",
        )
        bare_margin_request_id: Final = _chat_cost(gateway, model, key, 0.06 * 0.95 * 1.05)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": {"fixed_amount": 0.001}},
            "Cost margin configuration updated successfully",
        )
        fixed_margin_request_id: Final = _chat_cost(gateway, model, key, 0.057 + 0.001)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": {"percentage": 0.1, "fixed_amount": 0.001}, "global": 0.05},
            "Cost margin configuration updated successfully",
        )
        provider_margin_request_id: Final = _chat_cost(gateway, model, key, 0.057 * 1.1 + 0.001)

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

        expected: Final = 0.057 * 1.1 + 0.001
        estimate: Final = _estimated(gateway, model)
        assert estimate.model == model
        assert estimate.cost_per_request == pytest.approx(expected), estimate
        assert estimate.input_cost_per_request == pytest.approx(0.02), estimate
        assert estimate.output_cost_per_request == pytest.approx(0.04), estimate
        discounted_subtotal: Final = estimate.cost_per_request - estimate.margin_cost_per_request
        assert (
            estimate.input_cost_per_request + estimate.output_cost_per_request - discounted_subtotal
            == pytest.approx(0.06 * 0.05)
        ), estimate
        assert estimate.margin_cost_per_request == pytest.approx(0.06 * 0.95 * 0.1 + 0.001), estimate

        _patch(gateway, "/config/cost_discount_config", {}, "Cost discount configuration updated successfully")
        no_discount_request_id: Final = _chat_cost(gateway, model, key, 0.06 * 1.1 + 0.001)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"global": 1},
            "Cost margin configuration updated successfully",
        )
        global_margin_request_id: Final = _chat_cost(gateway, model, key, 0.12)

        _assert_logged_spends(
            (
                (base_request_id, 0.06),
                (discounted_request_id, 0.057),
                (bare_margin_request_id, 0.06 * 0.95 * 1.05),
                (fixed_margin_request_id, 0.057 + 0.001),
                (provider_margin_request_id, 0.057 * 1.1 + 0.001),
                (no_discount_request_id, 0.06 * 1.1 + 0.001),
                (global_margin_request_id, 0.12),
            )
        )

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
