import json
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
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


class ChatChunk(BaseModel):
    id: str
    object: str


def _observed(upstream: httpx.Client) -> list[JsonValue]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = object_value(response.json())["requests"]
    assert isinstance(requests, list), response.text
    return requests


def _upstream_chat(content: str, streamed: bool) -> dict[str, JsonValue]:
    return {
        "path": "/v1/chat/completions",
        "authorization": "Bearer integration-provider-key",
        "body": {
            "model": "gpt-4o-mini",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": content}],
            **({"stream": True, "stream_options": {"include_usage": True}} if streamed else {}),
        },
        "method": "POST",
        "api_key": "",
    }


def _chat_cost(gateway: Gateway, upstream: httpx.Client, model: str, key: str, expected: float) -> str:
    content: Final = f"margin {uuid.uuid4().hex}"
    _observed(upstream)
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": content}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    request_id: Final = string_value(object_value(response.json())["id"])
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected), response.text
    assert _observed(upstream) == [_upstream_chat(content, streamed=False)], response.text
    return request_id


def _streamed_chat(gateway: Gateway, upstream: httpx.Client, model: str, key: str) -> str:
    content: Final = f"margin stream {uuid.uuid4().hex}"
    _observed(upstream)
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 20, "stream": True, "messages": [{"role": "user", "content": content}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    events: Final = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]", response.text
    chunks: Final = [ChatChunk.model_validate_json(event) for event in events[:-1]]
    assert {chunk.object for chunk in chunks} == {"chat.completion.chunk"}, response.text
    assert len({chunk.id for chunk in chunks}) == 1, response.text
    assert _observed(upstream) == [_upstream_chat(content, streamed=True)], response.text
    return chunks[0].id


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
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
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
        base_request_id: Final = _chat_cost(gateway, upstream, model, key, 0.06)

        _patch(
            gateway,
            "/config/cost_discount_config",
            {"openai": 0.05},
            "Cost discount configuration updated successfully",
        )
        discounted_request_id: Final = _chat_cost(gateway, upstream, model, key, 0.057)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": 0.05},
            "Cost margin configuration updated successfully",
        )
        bare_margin_request_id: Final = _chat_cost(gateway, upstream, model, key, 0.06 * 0.95 * 1.05)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": {"fixed_amount": 0.001}},
            "Cost margin configuration updated successfully",
        )
        fixed_margin_request_id: Final = _chat_cost(gateway, upstream, model, key, 0.057 + 0.001)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": {"percentage": 0.1, "fixed_amount": 0.001}, "global": 0.05},
            "Cost margin configuration updated successfully",
        )
        provider_margin_request_id: Final = _chat_cost(gateway, upstream, model, key, 0.057 * 1.1 + 0.001)
        streamed_margin_request_id: Final = _streamed_chat(gateway, upstream, model, key)

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
        no_discount_request_id: Final = _chat_cost(gateway, upstream, model, key, 0.06 * 1.1 + 0.001)

        _patch(
            gateway,
            "/config/cost_margin_config",
            {"global": 1},
            "Cost margin configuration updated successfully",
        )
        global_margin_request_id: Final = _chat_cost(gateway, upstream, model, key, 0.12)
        streamed_global_request_id: Final = _streamed_chat(gateway, upstream, model, key)

        _assert_logged_spends(
            (
                (base_request_id, 0.06),
                (discounted_request_id, 0.057),
                (bare_margin_request_id, 0.06 * 0.95 * 1.05),
                (fixed_margin_request_id, 0.057 + 0.001),
                (provider_margin_request_id, 0.057 * 1.1 + 0.001),
                (streamed_margin_request_id, 0.057 * 1.1 + 0.001),
                (no_discount_request_id, 0.06 * 1.1 + 0.001),
                (global_margin_request_id, 0.12),
                (streamed_global_request_id, 0.12),
            )
        )

        rejections: Final = (
            (
                "/config/cost_margin_config",
                {"not_a_provider": 0.1},
                {
                    "detail": {
                        "error": "Invalid provider(s): not_a_provider. Must be valid LiteLLM providers or 'global'. "
                        "See https://docs.litellm.ai/docs/providers for the full list."
                    }
                },
            ),
            (
                "/config/cost_discount_config",
                {"openai": 1.5},
                {"detail": "Discount for openai must be between 0 and 1 (0% to 100%)"},
            ),
            (
                "/config/cost_discount_config",
                {"global": 0.1},
                {
                    "detail": {
                        "error": "Invalid provider(s): global. Must be valid LiteLLM providers. "
                        "See https://docs.litellm.ai/docs/providers for the full list."
                    }
                },
            ),
        )
        for path, body, error in rejections:
            rejected = gateway.request("PATCH", path, body)
            assert rejected.status_code == 400, rejected.text
            assert rejected.json() == error, rejected.text
        assert gateway.get("/config/cost_discount_config")["values"] == {}
        assert gateway.get("/config/cost_margin_config")["values"] == {"global": 1}
        final_settings: Final = _settings_row()
        assert final_settings["cost_discount_config"] == {}, final_settings
        assert final_settings["cost_margin_config"] == {"global": 1}, final_settings


def test_streamed_response_cost_header_matches_billed_spend(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: a streamed chat with a fixed cost margin returns x-litellm-response-cost equal to the fixed amount "
        "(computed before usage is known) while LiteLLM_SpendLogs bills the full margined cost"
    )
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        original_margin: Final = gateway.get("/config/cost_margin_config")["values"]
        scenario.cleanups.callback(
            _patch,
            gateway,
            "/config/cost_margin_config",
            original_margin,
            "Cost margin configuration updated successfully",
        )
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        _patch(
            gateway,
            "/config/cost_margin_config",
            {"openai": {"percentage": 0.1, "fixed_amount": 0.001}},
            "Cost margin configuration updated successfully",
        )
        content: Final = f"margin stream header {uuid.uuid4().hex}"
        _observed(upstream)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 20, "stream": True, "messages": [{"role": "user", "content": content}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        events: Final = [
            line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]", response.text
        request_id: Final = ChatChunk.model_validate_json(events[0]).id
        assert _observed(upstream) == [_upstream_chat(content, streamed=True)], response.text
        _assert_logged_spends(((request_id, 0.06 * 1.1 + 0.001),))
        advertised: Final = response.headers.get("x-litellm-response-cost")
        assert advertised is None or float(advertised) == pytest.approx(0.06 * 1.1 + 0.001), dict(response.headers)
