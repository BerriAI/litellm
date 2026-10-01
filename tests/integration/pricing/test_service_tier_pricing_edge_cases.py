from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final, cast

import pytest
from pydantic import JsonValue

import tests.integration._support.service_tier_pricing as pricing
from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import JsonResponse


@pytest.mark.parametrize(
    ("label", "service_tier"),
    pricing.INVALID_TIER_VALUES,
    ids=tuple(value[0] for value in pricing.INVALID_TIER_VALUES),
)
def test_malformed_or_unrecognized_tier_inputs_keep_the_observed_result(
    gateway: Gateway,
    label: str,
    service_tier: JsonValue,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(service_tier),
            identifier=f"tier-matrix-malformed-tier-{label}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload(service_tier)},
            key=key,
        )
        assert outcome.status == 200, outcome.text
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(
                service_tier if isinstance(service_tier, str) else None,
                pricing.MATRIX_PROMPT_TOKENS,
                pricing.MATRIX_COMPLETION_TOKENS,
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_upstream(
            gateway,
            expected_tier=service_tier,
            tier_present=True,
        )


def test_uppercase_priority_tier_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("PRIORITY"),
            identifier=f"tier-matrix-uppercase-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("PRIORITY")},
            key=key,
        )
        pricing.assert_upstream(gateway, expected_tier="PRIORITY", tier_present=True)
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(
                "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )


def test_scripted_upstream_failure_records_one_failure_spend_row(gateway: Gateway) -> None:
    expected_attempt_count: Final = 1
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=JsonResponse(
                content_type="application/json",
                body={"error": {"message": "scripted upstream failure", "type": "server_error"}},
                status=500,
            ),
            identifier=f"tier-matrix-upstream-failure-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("priority")},
            key=key,
        )
        assert outcome.status == 500, outcome
        observed: Final = pricing.observations(gateway)
        assert len(observed) == expected_attempt_count, observed
        for request in observed:
            pricing.assert_forwarded_body(
                object_value(request["body"]),
                expected_tier="priority",
                tier_present=True,
            )
        rows: Final = eventually(
            lambda: pricing.spend_rows_for_key(key),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["status"] == "failure", rows
        assert float(str(rows[0]["spend"])) == 0, rows


def test_null_explicit_priority_input_is_treated_as_missing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-null-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            model_info={"input_cost_per_token_priority": None},
        )
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("priority")},
            key=key,
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier="priority",
            expected_cost=pricing.expected_cost(
                "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
        )


def test_unauthenticated_priority_request_is_rejected_without_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-unauthenticated-request-{uuid.uuid4().hex}",
        )
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        invalid_key: Final = f"sk-invalid-tier-matrix-{uuid.uuid4().hex}"
        response: Final = gateway.request(
            "POST",
            pricing.MATRIX_CHAT_PATH,
            {"model": model, **pricing.chat_payload("priority")},
            key=invalid_key,
        )
        assert response.status_code == 401, response.text
        assert (
            read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                (sha256(invalid_key.encode()).hexdigest(),),
            )
            == []
        )
        assert pricing.observations(gateway) == []


def test_three_repeated_priority_requests_create_three_identical_cost_rows(gateway: Gateway) -> None:
    expected_cost: Final = pricing.expected_cost(
        "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
    )
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-repeated-priority-requests-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcomes: Final = tuple(
            pricing.invoke_http(
                gateway,
                path=pricing.MATRIX_CHAT_PATH,
                payload={"model": model, **pricing.chat_payload("priority")},
                key=key,
            )
            for _ in range(3)
        )
        assert all(outcome.status == 200 for outcome in outcomes), outcomes
        ids: Final = tuple(outcome.request_id for outcome in outcomes)
        assert None not in ids, outcomes
        assert len(set(ids)) == 3, ids
        observed: Final = pricing.observations(gateway)
        assert len(observed) == 3, observed
        for request in observed:
            pricing.assert_forwarded_body(
                object_value(request["body"]),
                expected_tier="priority",
                tier_present=True,
            )
        rows: Final = tuple(pricing.spend_rows(cast(str, request_id))[0] for request_id in ids)
        assert len(rows) == 3, rows
        for outcome in outcomes:
            pricing.assert_cost_header(outcome, expected_cost=expected_cost)
        assert all(float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6) for row in rows), rows
        assert all(int(str(row["prompt_tokens"])) == pricing.MATRIX_PROMPT_TOKENS for row in rows), rows
        assert all(int(str(row["completion_tokens"])) == pricing.MATRIX_COMPLETION_TOKENS for row in rows), rows


def test_priority_burst_uses_catalog_rates_during_standard_price_update(gateway: Gateway) -> None:
    updated_input_rate: Final = 0.00037
    updated_output_rate: Final = 0.00082
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_sse_response(frame_delay_ms=80),
            identifier=f"tier-matrix-standard-update-priority-burst-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model, model_id = pricing.register_db_model(
            gateway,
            scenario,
            model=f"openai/{pricing.MATRIX_BACKEND_MODEL}",
            scenario_id=scenario_id,
            api_base=api_base,
        )
        requests: Final = tuple(
            {
                "model": model,
                **pricing.chat_payload("priority", stream=True),
            }
            for _ in range(20)
        )
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures: Final = tuple(
                executor.submit(
                    pricing.invoke_http,
                    gateway,
                    path=pricing.MATRIX_CHAT_PATH,
                    payload=request,
                    key=key,
                    stream=True,
                    surface="chat",
                )
                for request in requests
            )
            updated: Final = gateway.request(
                "POST",
                "/model/update",
                {
                    "model_info": {"id": model_id},
                    "litellm_params": {
                        "input_cost_per_token": updated_input_rate,
                        "output_cost_per_token": updated_output_rate,
                    },
                },
            )
            assert updated.status_code == 200, updated.text
            outcomes: Final = tuple(future.result(timeout=90) for future in futures)
        expected_cost: Final = pricing.expected_cost(
            "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
        )
        assert all(outcome.status == 200 for outcome in outcomes), outcomes
        for outcome in outcomes:
            pricing.assert_cost_header(outcome, expected_cost=expected_cost, allow_missing_header=True)
        ids: Final = tuple(outcome.request_id for outcome in outcomes)
        assert None not in ids, outcomes
        rows: Final = tuple(pricing.spend_rows(cast(str, request_id))[0] for request_id in ids)
        assert len(rows) == 20, rows
        assert all(float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6) for row in rows), rows
        observed: Final = pricing.observations(gateway)
        assert len(observed) == 20, observed
        assert all(object_value(request["body"]).get("service_tier") == "priority" for request in observed), observed
        assert all(
            not {
                field
                for field in object_value(request["body"])
                if "cost_per_token" in field or field.startswith("cache_read_input_token_cost")
            }
            for request in observed
        ), observed


def test_cache_hit_keeps_the_observed_second_request_billing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-cache-hit-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        control_payload: Final = {
            "model": model,
            **pricing.chat_payload("priority", no_cache=False),
            "messages": [{"role": "user", "content": "cache billing control"}],
        }
        control: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload=control_payload,
            key=key,
        )
        assert control.cost_header is not None, control
        control_cost: Final = control.cost_header
        pricing.assert_billing(
            control,
            expected_cost=control_cost,
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)
        payload: Final = {
            "model": model,
            **pricing.chat_payload("priority", no_cache=False),
            "messages": [{"role": "user", "content": "cache hit billing"}],
        }
        first: Final = pricing.invoke_http(gateway, path=pricing.MATRIX_CHAT_PATH, payload=payload, key=key)
        second: Final = pricing.invoke_http(gateway, path=pricing.MATRIX_CHAT_PATH, payload=payload, key=key)
        assert first.status == second.status == 200, (first, second)
        assert first.request_id is not None and second.request_id is not None, (first, second)
        assert first.request_id == second.request_id, (first, second)
        expected_first_cost: Final = control_cost
        pricing.assert_cost_header(first, expected_cost=expected_first_cost)
        pricing.assert_cost_header(second, expected_cost=expected_first_cost)
        rows: Final = pricing.spend_rows(first.request_id)
        assert float(str(rows[0]["spend"])) == pytest.approx(expected_first_cost, rel=1e-6), rows
        cache_hit_rows: Final = pricing.cache_hit_spend_rows(second.request_id)
        assert cache_hit_rows[0]["cache_hit"] in ("True", True), cache_hit_rows
        assert float(str(cache_hit_rows[0]["spend"])) == 0, cache_hit_rows
        observed: Final = pricing.observations(gateway)
        assert len(observed) == 1, observed
        pricing.assert_forwarded_body(
            object_value(observed[0]["body"]),
            expected_tier="priority",
            tier_present=True,
        )


def test_response_priority_matches_catalog_priced_rule(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(None, response_service_tier="priority"),
            identifier=f"tier-matrix-response-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        custom_model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
        )
        catalog_model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            custom_rates=False,
        )
        custom_outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": custom_model, **pricing.chat_payload(None)},
            key=key,
        )
        custom_observed: Final = pricing.assert_upstream(gateway)
        assert "service_tier" not in custom_observed
        catalog_outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": catalog_model, **pricing.chat_payload(None)},
            key=key,
        )
        catalog_observed: Final = pricing.assert_upstream(gateway)
        assert "service_tier" not in catalog_observed
        priority_outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": custom_model, **pricing.chat_payload("priority")},
            key=key,
        )
        priority_observed: Final = pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)
        catalog_cost: Final = pricing.expected_cost(
            "priority",
            pricing.MATRIX_PROMPT_TOKENS,
            pricing.MATRIX_COMPLETION_TOKENS,
            custom_standard=False,
        )
        assert custom_outcome.body.get("service_tier") == "priority", custom_outcome.body
        assert catalog_outcome.body.get("service_tier") == "priority", catalog_outcome.body
        assert priority_outcome.body.get("service_tier") == "priority", priority_outcome.body
        assert all(
            not any("cost_per_token" in field or field.startswith("cache_read_input_token_cost") for field in request)
            for request in (custom_observed, catalog_observed, priority_observed)
        ), (custom_observed, catalog_observed, priority_observed)
        pricing.assert_billing(
            custom_outcome,
            expected_cost=catalog_cost,
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_billing(
            catalog_outcome,
            expected_cost=catalog_cost,
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_billing(
            priority_outcome,
            expected_cost=catalog_cost,
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
