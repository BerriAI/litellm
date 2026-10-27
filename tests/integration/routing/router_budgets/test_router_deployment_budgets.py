"""Deployment budgets (``litellm_params.max_budget`` + ``budget_duration``) on a two-worker proxy.

Each cell owns its deployments, so the ``deployment_spend:<model_id>`` keys never collide. Probes carry
``PROVIDER_FAILURE`` and are never charged. Response caching is on so the cache-hit cell runs against the
same proxy; every other request carries unique text and never hits the cache.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Wire, wire_server
from pydantic import JsonValue
from integration.routing.router_budgets import _rig as rig

ISSUE_43214: Final = "https://github.com/BerriAI/litellm/issues/43214"


@dataclass(frozen=True, slots=True)
class DeploymentRig:
    gateway: Gateway
    upstream: Wire
    budget: rig.BudgetRig


def _capped(
    name: str, model_id: str, url: str, cap: float, duration: str | None = "1d", **extra: JsonValue
) -> dict[str, JsonValue]:
    return rig.deployment(
        name,
        f"hosted_vllm/{model_id}",
        url,
        model_id=model_id,
        max_budget=cap,
        **({"budget_duration": duration} if duration is not None else {}),
        **extra,
    )


@pytest.fixture(scope="module")
def deployments(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DeploymentRig]:
    tmp_path: Final = tmp_path_factory.mktemp("deployment-budgets")
    with gateway_from_environment() as gateway, wire_server(rig.upstream) as upstream:
        url: Final = upstream.url
        path: Final = rig.write_config(
            tmp_path / "deployments.yaml",
            (
                _capped("dep-at-cap", "dep-at-cap", url, 2 * rig.CALL_COST),
                _capped("dep-tiny", "dep-tiny", url, 0.01),
                _capped("dep-zero", "dep-zero", url, 0),
                _capped("dep-no-duration", "dep-no-duration", url, 0.01, None),
                _capped("dep-pair", "dep-pair-capped", url, 0.01, order=1),
                rig.deployment("dep-pair", "hosted_vllm/dep-pair-sibling", url, model_id="dep-pair-sibling", order=2),
                _capped("dep-solo", "dep-solo", url, 0.01),
                rig.deployment("dep-backup", "hosted_vllm/dep-backup", url, model_id="dep-backup"),
                _capped("dep-message", "dep-message", url, 0.01),
                _capped("dep-accounting", "dep-accounting", url, 2 * rig.CALL_COST),
                _capped("dep-cached", "dep-cached", url, 2 * rig.CALL_COST),
            ),
            litellm_settings={
                "cache": True,
                "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
            },
        )
        config: Final = yaml.safe_load(path.read_text())
        config["router_settings"]["fallbacks"] = [{"dep-solo": ["dep-backup"]}]
        path.write_text(yaml.safe_dump(config))
        with rig.budget_proxy(gateway, tmp_path, path) as budget:
            yield DeploymentRig(budget.gateway, upstream, budget)


def test_spend_reaching_exactly_the_deployment_cap_blocks_the_next_request(deployments: DeploymentRig) -> None:
    first: Final = rig.chat(deployments.gateway, "dep-at-cap", "deployment at cap first")
    second: Final = rig.chat(deployments.gateway, "dep-at-cap", "deployment at cap second")
    assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)

    blocked: Final = rig.until_rejected(deployments.gateway, "dep-at-cap", rig.probe_text("deployment at cap"))

    assert blocked.status_code == 429, blocked.text
    assert "model_id: dep-at-cap" in blocked.text
    assert deployments.budget.settled("deployment_spend:dep-at-cap:1d", 2 * rig.CALL_COST) == 2 * rig.CALL_COST


def test_spend_over_a_tiny_deployment_cap_rejects_with_429(deployments: DeploymentRig) -> None:
    first: Final = rig.chat(deployments.gateway, "dep-tiny", "deployment tiny first")
    assert first.status_code == 200, first.text

    blocked: Final = rig.until_rejected(deployments.gateway, "dep-tiny", rig.probe_text("deployment tiny"))

    assert blocked.status_code == 429, blocked.text
    assert "Exceeded budget for deployment model_name: dep-tiny" in blocked.text


@pytest.mark.xfail(strict=True, reason=f"max_budget 0 is treated as unlimited: {ISSUE_43214}")
def test_a_zero_deployment_cap_rejects_the_first_request(deployments: DeploymentRig) -> None:
    response: Final = rig.chat(deployments.gateway, "dep-zero", rig.probe_text("deployment zero"))

    assert rig.is_budget_rejection(response), response.text


@pytest.mark.xfail(strict=True, reason="max_budget without budget_duration loads without error and is never enforced")
def test_a_deployment_cap_without_budget_duration_is_still_enforced(deployments: DeploymentRig) -> None:
    first: Final = rig.chat(deployments.gateway, "dep-no-duration", "deployment without duration")
    assert first.status_code == 200, first.text

    rig.until_rejected(deployments.gateway, "dep-no-duration", rig.probe_text("deployment no duration"), seconds=10)


@pytest.mark.xfail(strict=True, reason="the deployment rejection prints budget_duration where the cap belongs")
def test_the_deployment_rejection_names_the_cap_it_crossed(deployments: DeploymentRig) -> None:
    first: Final = rig.chat(deployments.gateway, "dep-message", "deployment message first")
    assert first.status_code == 200, first.text

    blocked: Final = rig.until_rejected(deployments.gateway, "dep-message", rig.probe_text("deployment message"))

    assert f"{rig.CALL_COST} >= 0.01" in blocked.text, blocked.text


def test_traffic_converges_on_the_uncapped_sibling_deployment(deployments: DeploymentRig) -> None:
    first: Final = rig.chat(deployments.gateway, "dep-pair", "dep pair first")
    assert first.status_code == 200, first.text
    assert first.headers["x-litellm-model-id"] == "dep-pair-capped"

    def batch() -> tuple[tuple[int, str | None], ...]:
        return tuple(
            (response.status_code, response.headers.get("x-litellm-model-id"))
            for response in (
                rig.chat(deployments.gateway, "dep-pair", f"dep pair {uuid.uuid4().hex}") for _ in range(6)
            )
        )

    eventually(batch, lambda outcomes: all(outcome == (200, "dep-pair-sibling") for outcome in outcomes), seconds=30)

    eventually(
        lambda: deployments.budget.redis_float("deployment_spend:dep-pair-capped:1d"),
        lambda spend: spend is not None and spend >= rig.CALL_COST,
    )
    assert deployments.budget.redis_float("deployment_spend:dep-pair-sibling:1d") is None


def test_an_over_budget_group_falls_back_to_the_configured_fallback_group(deployments: DeploymentRig) -> None:
    first: Final = rig.chat(deployments.gateway, "dep-solo", "deployment solo first")
    assert first.status_code == 200, first.text
    assert first.headers["x-litellm-model-id"] == "dep-solo"

    fallen_back: Final = eventually(
        lambda: rig.chat(deployments.gateway, "dep-solo", f"deployment solo {uuid.uuid4().hex}"),
        lambda response: response.headers.get("x-litellm-model-id") == "dep-backup",
        seconds=30,
    )

    assert fallen_back.status_code == 200, fallen_back.text
    assert fallen_back.headers["x-litellm-attempted-fallbacks"] == "1"


def test_failed_calls_are_free_and_a_success_charges_exactly_its_spend_log_cost(deployments: DeploymentRig) -> None:
    deployments.upstream.drain()
    failures: Final = tuple(
        rig.chat(deployments.gateway, "dep-accounting", rig.probe_text("accounting")) for _ in range(5)
    )
    assert all(response.status_code == 500 for response in failures), [response.text for response in failures]
    success: Final = rig.chat(deployments.gateway, "dep-accounting", "accounting success")
    assert success.status_code == 200, success.text

    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, model_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (string_value(success.json()["id"]),)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )

    assert rows[0]["spend"] == rig.CALL_COST
    assert rows[0]["model_id"] == "dep-accounting"
    assert deployments.budget.settled("deployment_spend:dep-accounting:1d", rig.CALL_COST) == rig.CALL_COST
    probe: Final = rig.chat(deployments.gateway, "dep-accounting", rig.probe_text("accounting after"))
    assert probe.status_code == 500, probe.text
    assert len(deployments.upstream.drain()) == 7


def test_a_response_cache_hit_does_not_charge_the_deployment_budget(deployments: DeploymentRig) -> None:
    text: Final = f"cached {uuid.uuid4().hex}"
    first: Final = rig.chat(deployments.gateway, "dep-cached", text)
    assert first.status_code == 200, first.text
    eventually(
        lambda: deployments.budget.redis_float("deployment_spend:dep-cached:1d"), lambda spend: spend == rig.CALL_COST
    )
    deployments.upstream.drain()

    hit: Final = eventually(
        lambda: rig.chat(deployments.gateway, "dep-cached", text),
        lambda response: response.headers.get("x-litellm-cache-key") is not None,
    )

    assert hit.json()["id"] == first.json()["id"]
    assert deployments.upstream.drain() == ()
    second: Final = rig.chat(deployments.gateway, "dep-cached", "cached sibling request")
    assert second.status_code == 200, second.text
    assert deployments.budget.settled("deployment_spend:dep-cached:1d", 2 * rig.CALL_COST) == 2 * rig.CALL_COST
    rig.until_rejected(deployments.gateway, "dep-cached", rig.probe_text("cached probe"))


def test_a_budgeted_deployment_added_and_raised_at_runtime_is_enforced(deployments: DeploymentRig) -> None:
    model_id: Final = f"runtime-{uuid.uuid4().hex}"
    model_name: Final = f"runtime-{uuid.uuid4().hex}"
    created: Final = deployments.gateway.post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": {
                "model": f"hosted_vllm/{model_id}",
                "api_base": f"{deployments.upstream.url}/v1",
                "api_key": "router-budget-provider-key",
                "input_cost_per_token": rig.PRICE,
                "output_cost_per_token": rig.PRICE,
                "max_budget": 0.01,
                "budget_duration": "1d",
            },
            "model_info": {"id": model_id},
        },
    )
    assert object_value(created["model_info"])["id"] == model_id
    try:
        first: Final = eventually(
            lambda: rig.chat(deployments.gateway, model_name, "runtime first"),
            lambda response: response.status_code == 200,
            seconds=30,
        )
        assert first.headers["x-litellm-model-id"] == model_id

        rig.until_rejected(deployments.gateway, model_name, rig.probe_text("runtime"))

        raised: Final = deployments.gateway.request(
            "POST", "/model/update", {"model_info": {"id": model_id}, "litellm_params": {"max_budget": 100}}
        )
        assert raised.status_code == 200, raised.text

        def five_probes() -> tuple[int, ...]:
            return tuple(
                rig.chat(deployments.gateway, model_name, rig.probe_text("runtime raised")).status_code
                for _ in range(5)
            )

        eventually(five_probes, lambda statuses: statuses == (500,) * 5, seconds=30)
    finally:
        deployments.gateway.post("/model/delete", {"id": model_id})
