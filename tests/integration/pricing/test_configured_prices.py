import asyncio
import json
import os
import uuid
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from litellm import get_model_info
from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse
from tests.integration.pricing.test_realtime_cached_audio_pricing import one_realtime_turn

REALTIME_MODEL: Final = "gpt-realtime-2"
REALTIME_INPUT_TEXT_TOKENS: Final = 10
REALTIME_INPUT_AUDIO_TOKENS: Final = 20
REALTIME_OUTPUT_TEXT_TOKENS: Final = 5
REALTIME_OUTPUT_AUDIO_TOKENS: Final = 7


def _realtime_response_done() -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {
                "type": "response.done",
                "event_id": "evt_$REQUEST_ID",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "realtime.response",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "total_tokens": REALTIME_INPUT_TEXT_TOKENS
                        + REALTIME_INPUT_AUDIO_TOKENS
                        + REALTIME_OUTPUT_TEXT_TOKENS
                        + REALTIME_OUTPUT_AUDIO_TOKENS,
                        "input_tokens": REALTIME_INPUT_TEXT_TOKENS + REALTIME_INPUT_AUDIO_TOKENS,
                        "output_tokens": REALTIME_OUTPUT_TEXT_TOKENS + REALTIME_OUTPUT_AUDIO_TOKENS,
                        "input_token_details": {
                            "text_tokens": REALTIME_INPUT_TEXT_TOKENS,
                            "audio_tokens": REALTIME_INPUT_AUDIO_TOKENS,
                            "cached_tokens": 0,
                        },
                        "output_token_details": {
                            "text_tokens": REALTIME_OUTPUT_TEXT_TOKENS,
                            "audio_tokens": REALTIME_OUTPUT_AUDIO_TOKENS,
                        },
                    },
                },
            },
        ),
    )


@pytest.mark.parametrize(
    ("input_text_rate", "input_audio_rate", "output_text_rate", "output_audio_rate"),
    ((0.001, 0.002, 0.003, 0.004), (0.0, 0.0, 0.0, 0.0)),
    ids=("custom_rates", "zero_rated"),
)
def test_realtime_session_is_charged_at_the_deployment_configured_rates(
    gateway: Gateway,
    input_text_rate: float,
    input_audio_rate: float,
    output_text_rate: float,
    output_audio_rate: float,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"realtime-configured-price-{uuid.uuid4().hex[:12]}"
        handle: Final = register_scenario(scenario_id, _realtime_response_done())
        scenario.cleanups.callback(delete_scenario, handle)
        key: Final = scenario.key()
        model: Final = scenario.model(
            model=f"openai/{REALTIME_MODEL}",
            api_key=scenario_id,
            api_base=gateway.upstream_url.rstrip("/"),
            input_cost_per_token=input_text_rate,
            input_cost_per_audio_token=input_audio_rate,
            output_cost_per_token=output_text_rate,
            output_cost_per_audio_token=output_audio_rate,
        )
        session: Final = asyncio.run(one_realtime_turn(os.environ["INTEGRATION_PROXY_URL"].rstrip("/"), key, model))
        assert session.get("type") == "session.created", session
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, call_type FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["call_type"] == "_arealtime", rows
        assert float(str(rows[0]["spend"])) == pytest.approx(
            REALTIME_INPUT_TEXT_TOKENS * input_text_rate
            + REALTIME_INPUT_AUDIO_TOKENS * input_audio_rate
            + REALTIME_OUTPUT_TEXT_TOKENS * output_text_rate
            + REALTIME_OUTPUT_AUDIO_TOKENS * output_audio_rate,
            abs=1e-9,
        ), rows


@pytest.mark.covers("quota_management.spend_tracking.custom_price.matches_input_rates")
def test_custom_price_is_reported_and_charged(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "price control"}]}
        )
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(20 * 0.001 + 20 * 0.002)
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        matching: Final = tuple(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model)
        assert len(matching) == 1
        params: Final = object_value(matching[0]["litellm_params"])
        assert params["input_cost_per_token"] == 0.001
        assert params["output_cost_per_token"] == 0.002


@pytest.mark.covers("quota_management.cost_estimate.configured_price.reported_for_model_absent_from_cost_map")
def test_cost_estimate_reports_configured_prices_for_model_absent_from_cost_map(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/integration-on-prem-{uuid.uuid4().hex}",
            input_cost_per_token=0.003,
            output_cost_per_token=0.007,
        )
        response: Final = gateway.request(
            "POST",
            "/cost/estimate",
            {"model": model, "input_tokens": 1000, "output_tokens": 500, "num_requests_per_day": 10},
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        assert body["input_cost_per_token"] == pytest.approx(0.003), response.text
        assert body["output_cost_per_token"] == pytest.approx(0.007), response.text
        assert body["input_cost_per_request"] == pytest.approx(1000 * 0.003), response.text
        assert body["output_cost_per_request"] == pytest.approx(500 * 0.007), response.text
        margin: Final = body["margin_cost_per_request"]
        assert isinstance(margin, float), response.text
        assert body["cost_per_request"] == pytest.approx(1000 * 0.003 + 500 * 0.007 + margin), response.text
        assert body["daily_cost"] == pytest.approx(10 * (1000 * 0.003 + 500 * 0.007 + margin)), response.text


@pytest.mark.covers("quota_management.spend_tracking.default_prices.survive_nullable_sibling_reload")
def test_default_prices_survive_nullable_sibling_and_reload(gateway: Gateway) -> None:
    for registration_order in (("custom", "omitted", "nullable"), ("nullable", "omitted", "custom")):
        with gateway.scenario() as scenario:
            configured: Final = {
                "custom": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
                "omitted": {},
                "nullable": {"input_cost_per_token": None, "output_cost_per_token": None},
            }
            rates: Final = {
                "custom": (0.001, 0.002),
                "omitted": (0.00000015, 0.0000006),
                "nullable": (0.00000015, 0.0000006),
            }
            models: Final = {kind: scenario.model(**configured[kind]) for kind in registration_order}

            def observe_requests(
                registration_order: tuple[str, ...],
                models: Mapping[str, str],
                rates: Mapping[str, tuple[float, float]],
            ) -> Iterator[tuple[str, float]]:
                for generation in range(2):
                    entries: Final = gateway.get("/model/info")["data"]
                    assert isinstance(entries, list)
                    kinds: Final = tuple(reversed(registration_order)) if generation else registration_order
                    for index, kind in enumerate(kinds):
                        model: Final = models[kind]
                        target: Final = next(
                            object_value(entry) for entry in entries if object_value(entry)["model_name"] == model
                        )
                        info: Final = object_value(target["model_info"])
                        assert info["input_cost_per_token"] == rates[kind][0]
                        assert info["output_cost_per_token"] == rates[kind][1]
                        response: Final = gateway.request(
                            "POST",
                            "/v1/chat/completions",
                            {
                                "model": model,
                                "messages": [
                                    {"role": "user", "content": f"price {generation * len(registration_order) + index}"}
                                ],
                            },
                        )
                        assert response.status_code == 200, response.text
                        expected: Final = 20 * rates[kind][0] + 20 * rates[kind][1]
                        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(expected, rel=1e-6)
                        request_id: Final = string_value(object_value(response.json())["id"])
                        yield request_id, expected
                    target: Final = next(
                        object_value(entry)
                        for entry in entries
                        if object_value(entry)["model_name"] == models["nullable"]
                    )
                    identity: Final = string_value(object_value(target["model_info"])["id"])
                    updated: Final = gateway.request(
                        "PATCH", f"/model/{identity}/update", {"model_info": {"description": "reload price contract"}}
                    )
                    assert updated.status_code == 200, updated.text

            observations: Final = tuple(observe_requests(registration_order, models, rates))
            for request_id, expected in observations:
                rows: Final = eventually(
                    lambda request_id=request_id: read_rows(
                        'SELECT request_id, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
                        "WHERE request_id = %s",
                        (request_id,),
                    ),
                    lambda values: len(values) == 1,
                    seconds=70,
                )
                assert rows[0]["prompt_tokens"] == 20
                assert rows[0]["completion_tokens"] == 20
                assert float(rows[0]["spend"]) == pytest.approx(expected, rel=1e-6)


COST_MAP_DISPLAY_PRICING_KEYS: Final = frozenset(
    {
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
    }
)


def persisted_model_info(identity: str) -> dict[str, JsonValue]:
    rows: Final = read_rows('SELECT model_info FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,))
    assert len(rows) == 1, f"Deployment {identity} has {len(rows)} rows"
    stored: Final = rows[0]["model_info"]
    return object_value(json.loads(stored) if isinstance(stored, str) else stored)


@pytest.mark.covers("pricing.model_update.echoed_cost_map_price_is_not_persisted_as_override")
def test_saving_echoed_model_info_does_not_freeze_cost_map_price_into_deployment(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        target: Final = next(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model)
        displayed: Final = object_value(target["model_info"])
        identity: Final = string_value(displayed["id"])
        assert isinstance(displayed["input_cost_per_token"], float), displayed
        assert isinstance(displayed["output_cost_per_token"], float), displayed
        fresh: Final = persisted_model_info(identity)
        assert {key: value for key, value in fresh.items() if key in COST_MAP_DISPLAY_PRICING_KEYS} == {}, fresh
        saved: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"model_info": {**displayed, "description": "echoed ui save"}}
        )
        assert saved.status_code == 200, saved.text
        stored: Final = persisted_model_info(identity)
        assert stored["description"] == "echoed ui save", stored
        assert {key: value for key, value in stored.items() if key in COST_MAP_DISPLAY_PRICING_KEYS} == {}, stored


def displayed_model_info(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    target: Final = next(object_value(entry) for entry in entries if object_value(entry)["model_name"] == model)
    return object_value(target["model_info"])


@pytest.mark.covers("pricing.model_update.echoed_cost_map_metadata_is_not_persisted_as_override")
def test_saving_echoed_model_info_does_not_persist_cost_map_metadata_as_overrides(gateway: Gateway) -> None:
    catalog_entry: Final = get_model_info("openai/gpt-4o-mini")
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        displayed: Final = displayed_model_info(gateway, model)
        identity: Final = string_value(displayed["id"])
        assert displayed["key"] == catalog_entry["key"], displayed
        assert displayed["max_input_tokens"] == catalog_entry["max_input_tokens"], displayed
        saved: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"model_info": {**displayed, "description": "echoed ui save"}}
        )
        assert saved.status_code == 200, saved.text
        stored: Final = persisted_model_info(identity)
        assert stored["description"] == "echoed ui save", stored
        assert {key: value for key, value in stored.items() if key in catalog_entry} == {}, stored


@pytest.mark.covers("pricing.model_update.echoing_cost_map_value_back_clears_stored_override")
def test_saving_the_cost_map_value_back_over_a_stored_override_clears_it(gateway: Gateway, tmp_path: Path) -> None:
    catalog_limit: Final = get_model_info("openai/gpt-4o-mini")["max_input_tokens"]
    assert isinstance(catalog_limit, int) and catalog_limit != 4321, catalog_limit
    with owned_proxy(gateway, tmp_path, {}) as candidate, candidate.scenario() as scenario:
        overridden: Final = scenario.model(model_info={"max_input_tokens": 4321})
        displayed: Final = displayed_model_info(candidate, overridden)
        identity: Final = string_value(displayed["id"])
        assert displayed["max_input_tokens"] == 4321, displayed
        assert persisted_model_info(identity)["max_input_tokens"] == 4321
        saved: Final = candidate.request(
            "PATCH", f"/model/{identity}/update", {"model_info": {**displayed, "max_input_tokens": catalog_limit}}
        )
        assert saved.status_code == 200, saved.text
        stored: Final = persisted_model_info(identity)
        assert "max_input_tokens" not in stored, stored


@pytest.mark.covers("quota_management.spend_tracking.default_prices.loaded_router_preserves_cached_defaults")
def test_loaded_router_preserves_cached_defaults_during_real_requests(gateway: Gateway, tmp_path: Path) -> None:
    from litellm import Router

    aliases: Final = tuple(f"pricing-{uuid.uuid4().hex}" for _ in range(3))
    path: Final = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": alias,
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_key": "integration-provider-key",
                            "api_base": f"{gateway.upstream_url}/v1",
                        },
                        "model_info": {"id": alias, **pricing},
                    }
                    for alias, pricing in zip(
                        aliases,
                        (
                            {},
                            {"input_cost_per_token": None, "output_cost_per_token": None},
                            {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
                        ),
                        strict=True,
                    )
                ]
            }
        )
    )
    for reverse in (False, True):
        configured: Final = yaml.safe_load(path.read_text())["model_list"]
        router: Final = Router(model_list=list(reversed(configured)) if reverse else configured, num_retries=0)
        try:
            for alias in (*aliases, *reversed(aliases)):
                result: Final = router.completion(
                    model=alias, messages=[{"role": "user", "content": "router price control"}]
                )
                assert result.usage.prompt_tokens == 20
                assert result.usage.completion_tokens == 20
                expected_cost: Final = 0.0 if alias == aliases[2] else 20 * 0.00000015 + 20 * 0.0000006
                assert result._hidden_params["response_cost"] == pytest.approx(expected_cost, rel=1e-6)
                deployment: Final = router.get_deployment(model_id=alias)
                assert deployment is not None
                info: Final = router.get_router_model_info(deployment=deployment, received_model_name=alias)
                assert info["input_cost_per_token"] == (0.0 if alias == aliases[2] else 0.00000015)
                assert info["output_cost_per_token"] == (0.0 if alias == aliases[2] else 0.0000006)
        finally:
            router.reset()
