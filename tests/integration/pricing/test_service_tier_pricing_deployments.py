from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest

import tests.integration._support.service_tier_pricing as pricing
from tests.integration._support.client import Gateway, object_value, string_value
from tests.integration._support.process import owned_proxy_process


def test_cached_prompt_tokens_use_priority_cache_rate(gateway: Gateway) -> None:
    cached_tokens: Final = 5
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority", cached_tokens=cached_tokens),
            identifier=f"tier-matrix-chat-priority-cache-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("priority")},
            key=key,
        )
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.expected_cost(
                "priority",
                pricing.MATRIX_PROMPT_TOKENS,
                pricing.MATRIX_COMPLETION_TOKENS,
                cached_tokens=cached_tokens,
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


@pytest.mark.parametrize(
    ("service_tier", "rate_tier"),
    pricing.LONG_CONTEXT_CASES,
    ids=("ultrafast", "priority"),
)
def test_long_context_tier_rate_is_inherited_for_custom_deployments(
    gateway: Gateway,
    service_tier: str,
    rate_tier: str,
) -> None:
    prompt_tokens: Final = 300_000
    completion_tokens: Final = 100
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(
                service_tier,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            identifier=f"tier-matrix-long-context-{service_tier}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload(service_tier)},
            key=key,
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            expected_cost=pricing.expected_cost(
                rate_tier,
                prompt_tokens,
                completion_tokens,
                long_context=True,
            ),
        )


def test_long_context_no_tier_keeps_custom_standard_rates(gateway: Gateway) -> None:
    prompt_tokens: Final = 300_000
    completion_tokens: Final = 100
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(
                None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            identifier=f"tier-matrix-long-context-no-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload(None)},
            key=key,
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            expected_cost=pricing.expected_cost(None, prompt_tokens, completion_tokens),
        )


def test_explicit_priority_input_and_output_prices_win(gateway: Gateway) -> None:
    explicit_input_rate: Final = 0.00091
    explicit_output_rate: Final = 0.00173
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-explicit-priority-rates-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            input_cost_per_token_priority=explicit_input_rate,
            output_cost_per_token_priority=explicit_output_rate,
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
            expected_cost=pricing.MATRIX_PROMPT_TOKENS * explicit_input_rate
            + pricing.MATRIX_COMPLETION_TOKENS * explicit_output_rate,
        )


def test_explicit_priority_input_wins_while_missing_output_is_inherited(gateway: Gateway) -> None:
    explicit_input_rate: Final = 0.00091
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-explicit-priority-input-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            input_cost_per_token_priority=explicit_input_rate,
        )
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("priority")},
            key=key,
        )
        expected_cost: Final = (
            pricing.MATRIX_PROMPT_TOKENS * explicit_input_rate
            + pricing.MATRIX_COMPLETION_TOKENS
            * pricing.bundled_rate(pricing.MATRIX_BACKEND_MODEL, "output_cost_per_token_priority")
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier="priority",
            expected_cost=expected_cost,
        )


def test_standard_input_only_deployment_inherits_catalog_priority_rates(gateway: Gateway) -> None:
    explicit_standard_input_rate: Final = 0.00071
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-standard-input-only-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            custom_rates=False,
            input_cost_per_token=explicit_standard_input_rate,
        )
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("priority")},
            key=key,
        )
        expected_cost: Final = pricing.expected_cost(
            "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier="priority",
            expected_cost=expected_cost,
        )


@pytest.mark.parametrize(
    "service_tier",
    ("priority", None),
    ids=("priority", "no-tier"),
)
def test_catalog_priced_deployment_keeps_its_catalog_rates(
    gateway: Gateway,
    service_tier: str | None,
) -> None:
    scenario_name: Final = "priority" if service_tier is not None else "no-tier"
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(service_tier),
            identifier=f"tier-matrix-catalog-priced-{scenario_name}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            custom_rates=False,
        )
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload(service_tier)},
            key=key,
        )
        pricing.assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            expected_cost=pricing.expected_cost(
                service_tier,
                pricing.MATRIX_PROMPT_TOKENS,
                pricing.MATRIX_COMPLETION_TOKENS,
                custom_standard=False,
            ),
        )


def test_unknown_custom_model_uses_custom_standard_rates_without_proxy_error_log(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    unknown_model: Final = f"openai/unknown-tier-priced-model-{uuid.uuid4().hex}"
    with owned_proxy_process(gateway, tmp_path, {}) as owned:
        candidate: Final = owned.gateway
        prior_log: Final = owned.log.read_text() if owned.log.exists() else ""
        with candidate.scenario() as scenario:
            scenario_id, api_base = pricing.register_upstream(
                gateway,
                scenario,
                response=pricing.chat_json_response(None),
                identifier=f"tier-matrix-unknown-model-standard-pricing-{uuid.uuid4().hex}",
            )
            key: Final = scenario.key()
            model: Final = pricing.scenario_model(
                scenario,
                scenario_id=scenario_id,
                api_base=api_base,
                model=unknown_model,
            )
            outcome: Final = pricing.invoke_http(
                candidate,
                path=pricing.MATRIX_CHAT_PATH,
                payload={"model": model, **pricing.chat_payload(None)},
                key=key,
            )
            pricing.assert_success(
                gateway,
                outcome,
                service_tier=None,
                expected_cost=pricing.expected_cost(
                    None, pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
                ),
            )
        current_log: Final = owned.log.read_text() if owned.log.exists() else ""
        added_log: Final = current_log.removeprefix(prior_log)
        assert "ERROR" not in added_log, added_log


def test_azure_alias_inherits_priority_rates_from_model_info_base_model(gateway: Gateway) -> None:
    input_rate: Final = pricing.bundled_rate(pricing.MATRIX_AZURE_CATALOG_MODEL, "input_cost_per_token_priority")
    output_rate: Final = pricing.bundled_rate(pricing.MATRIX_AZURE_CATALOG_MODEL, "output_cost_per_token_priority")
    identifier: Final = f"tier-matrix-azure-alias-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=identifier,
        )
        key: Final = scenario.key()
        model: Final = pricing.scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            model=pricing.MATRIX_AZURE_ALIAS,
            model_info={"base_model": pricing.MATRIX_AZURE_CATALOG_MODEL},
            api_version="2024-10-21",
        )
        outcome: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("priority")},
            key=key,
        )
        pricing.assert_azure_upstream(gateway, scenario_id=identifier)
        pricing.assert_billing(
            outcome,
            expected_cost=pricing.MATRIX_PROMPT_TOKENS * input_rate + pricing.MATRIX_COMPLETION_TOKENS * output_rate,
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )


def test_ptu_deployment_keeps_zero_per_request_cost_with_attribution_enabled(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-ptu-zero-cost-{uuid.uuid4().hex}",
        )
        team_id: Final = scenario.team()
        model: Final = f"tier-priced-ptu-{uuid.uuid4().hex}"
        config: Final = pricing.write_proxy_config(
            tmp_path,
            models=(
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": f"openai/{pricing.MATRIX_BACKEND_MODEL}",
                        "api_key": scenario_id,
                        "api_base": api_base,
                    },
                    "model_info": {
                        "id": uuid.uuid4().hex,
                        "team_id": team_id,
                        "ptu_count": 4,
                        "cost_per_ptu_per_hour": 0.0,
                        "ptu_effective_from": datetime.now(UTC).isoformat(),
                    },
                },
            ),
            filename="service-tier-ptu.yaml",
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "true"},
            config=config,
        ) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as candidate_scenario:
                key: Final = candidate_scenario.key(team_id=team_id)
                outcome: Final = pricing.invoke_http(
                    candidate,
                    path=pricing.MATRIX_CHAT_PATH,
                    payload={"model": model, **pricing.chat_payload("priority")},
                    key=key,
                )
                pricing.assert_billing(
                    outcome,
                    expected_cost=0.0,
                    prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
                    completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
                    allow_missing_header=True,
                )
                pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_yaml_configured_deployment_inherits_priority_rates(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-yaml-config-priority-{uuid.uuid4().hex}",
        )
        model_name: Final = f"custom-priced-tier-yaml-{uuid.uuid4().hex}"
        model: Final = {
            "model_name": model_name,
            "litellm_params": {
                "model": f"openai/{pricing.MATRIX_BACKEND_MODEL}",
                "api_key": scenario_id,
                "api_base": api_base,
                "input_cost_per_token": pricing.CUSTOM_STANDARD_INPUT_RATE,
                "output_cost_per_token": pricing.CUSTOM_STANDARD_OUTPUT_RATE,
            },
            "model_info": {},
        }
        config: Final = pricing.write_proxy_config(
            tmp_path,
            models=(model,),
            filename="service-tier-yaml.yaml",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as candidate_scenario:
                key: Final = candidate_scenario.key()
                outcome: Final = pricing.invoke_http(
                    candidate,
                    path=pricing.MATRIX_CHAT_PATH,
                    payload={"model": model_name, **pricing.chat_payload("priority")},
                    key=key,
                )
                pricing.assert_billing(
                    outcome,
                    expected_cost=pricing.expected_cost(
                        "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
                    ),
                    prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
                    completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
                )
                pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_priority_behavior_survives_owned_proxy_restart_after_database_registration(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-database-restart-priority-{uuid.uuid4().hex}",
        )
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as first:
            key_record: Final = first.gateway.post("/key/generate", {})
            key: Final = string_value(key_record["key"])
            model_name: Final = f"custom-priced-tier-restart-{uuid.uuid4().hex}"
            created: Final = first.gateway.post(
                "/model/new",
                {
                    "model_name": model_name,
                    "litellm_params": {
                        "model": f"openai/{pricing.MATRIX_BACKEND_MODEL}",
                        "api_key": scenario_id,
                        "api_base": api_base,
                        "input_cost_per_token": pricing.CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": pricing.CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                    "model_info": {},
                },
            )
            model_id: Final = string_value(object_value(created["model_info"])["id"])
            scenario.cleanups.callback(scenario.delete_model, model_id)
            scenario.cleanups.callback(scenario.delete_key, key)
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as restarted:
            outcome: Final = pricing.invoke_http(
                restarted.gateway,
                path=pricing.MATRIX_CHAT_PATH,
                payload={"model": model_name, **pricing.chat_payload("priority")},
                key=key,
            )
            pricing.assert_billing(
                outcome,
                expected_cost=pricing.expected_cost(
                    "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
                ),
                prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
                completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
            )
            pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_model_update_changes_no_tier_standard_and_keeps_priority_inheritance(gateway: Gateway) -> None:
    updated_input_rate: Final = 0.00037
    updated_output_rate: Final = 0.00082
    with gateway.scenario() as scenario:
        scenario_id, api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response("priority"),
            identifier=f"tier-matrix-model-update-priority-{uuid.uuid4().hex}",
        )
        no_tier_scenario_id, no_tier_api_base = pricing.register_upstream(
            gateway,
            scenario,
            response=pricing.chat_json_response(None),
            identifier=f"tier-matrix-model-update-no-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model, model_id = pricing.register_db_model(
            gateway,
            scenario,
            model=f"openai/{pricing.MATRIX_BACKEND_MODEL}",
            scenario_id=scenario_id,
            api_base=api_base,
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
        priority: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload("priority")},
            key=key,
        )
        pricing.assert_upstream(gateway, expected_tier="priority", tier_present=True)
        no_tier_update: Final = gateway.request(
            "POST",
            "/model/update",
            {
                "model_info": {"id": model_id},
                "litellm_params": {
                    "api_base": no_tier_api_base,
                    "api_key": no_tier_scenario_id,
                },
            },
        )
        assert no_tier_update.status_code == 200, no_tier_update.text
        no_tier: Final = pricing.invoke_http(
            gateway,
            path=pricing.MATRIX_CHAT_PATH,
            payload={"model": model, **pricing.chat_payload(None)},
            key=key,
        )
        pricing.assert_billing(
            no_tier,
            expected_cost=pricing.MATRIX_PROMPT_TOKENS * updated_input_rate
            + pricing.MATRIX_COMPLETION_TOKENS * updated_output_rate,
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
        pricing.assert_upstream(gateway, expected_tier=None, tier_present=False)
        pricing.assert_billing(
            priority,
            expected_cost=pricing.expected_cost(
                "priority", pricing.MATRIX_PROMPT_TOKENS, pricing.MATRIX_COMPLETION_TOKENS
            ),
            prompt_tokens=pricing.MATRIX_PROMPT_TOKENS,
            completion_tokens=pricing.MATRIX_COMPLETION_TOKENS,
        )
