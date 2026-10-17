"""Provider budgets (``router_settings.provider_budget_config``) on a two-worker proxy.

Every provider in the config belongs to exactly one cell, so the global ``provider_spend:<provider>`` keys
never collide and the cells run in any order. A probe carries ``PROVIDER_FAILURE``: when the filter admits
it the upstream answers 500 and nothing is charged, so probing never moves spend across the cap. The
Prometheus cell runs one worker because the registry is per process and the rig sets no multiprocess dir,
and the boot-failure cell runs one worker so the failed boot exits instead of respawning.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.wire import Wire, wire_server
from integration.routing.router_budgets import _rig as rig

ISSUE_43214: Final = "https://github.com/BerriAI/litellm/issues/43214"


@dataclass(frozen=True, slots=True)
class ProviderRig:
    gateway: Gateway
    upstream: Wire
    budget: rig.BudgetRig


@pytest.fixture(scope="module")
def providers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ProviderRig]:
    tmp_path: Final = tmp_path_factory.mktemp("provider-budgets")
    with gateway_from_environment() as gateway, wire_server(rig.upstream) as upstream:
        url: Final = upstream.url
        config: Final = rig.write_config(
            tmp_path / "providers.yaml",
            (
                rig.deployment("at-cap", "openai/budget-at-cap", url, model_id="provider-at-cap"),
                rig.deployment("tiny-cap", "hosted_vllm/budget-tiny", url, model_id="provider-tiny"),
                rig.deployment("zero-cap", "deepseek/budget-zero", url, model_id="provider-zero"),
                rig.deployment("no-limit", "groq/budget-no-limit", url, model_id="provider-no-limit"),
                rig.deployment("no-period", "together_ai/budget-no-period", url, model_id="provider-no-period"),
                rig.deployment("fw-solo", "fireworks_ai/budget-fw-solo", url, model_id="provider-fw-solo"),
                rig.deployment("fw-pair", "fireworks_ai/budget-fw-pair", url, model_id="provider-fw-pair"),
                rig.deployment("fw-pair", "lm_studio/budget-sibling", url, model_id="provider-sibling"),
                rig.deployment("reported", "deepinfra/budget-reported", url, model_id="provider-reported"),
            ),
            provider_budget_config={
                "openai": {"budget_limit": 2 * rig.CALL_COST, "time_period": "1d"},
                "hosted_vllm": {"budget_limit": 0.01, "time_period": "1d"},
                "deepseek": {"budget_limit": 0, "time_period": "1d"},
                "groq": {"time_period": "1d"},
                "together_ai": {"budget_limit": 0.01},
                "fireworks_ai": {"budget_limit": 0.01, "time_period": "1d"},
                "deepinfra": {"budget_limit": 1, "time_period": "1d"},
            },
        )
        with rig.budget_proxy(gateway, tmp_path, config) as budget:
            yield ProviderRig(budget.gateway, upstream, budget)


def _served_models(upstream: Wire) -> tuple[str, ...]:
    return tuple(
        str(rig.json_body(request)["model"])
        for request in upstream.drain()
        if rig.PROVIDER_FAILURE.encode() not in request.body
    )


def test_spend_reaching_exactly_the_provider_cap_blocks_the_next_request(providers: ProviderRig) -> None:
    providers.upstream.drain()
    first: Final = rig.chat(providers.gateway, "at-cap", "provider at cap first")
    second: Final = rig.chat(providers.gateway, "at-cap", "provider at cap second")
    assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)

    blocked: Final = rig.until_rejected(providers.gateway, "at-cap", rig.probe_text("provider at cap"))

    assert blocked.status_code == 429, blocked.text
    assert object_value(blocked.json()["error"])["message"] == (
        f"{rig.BUDGET_ERROR}: Exceeded budget for provider openai: {2 * rig.CALL_COST} >= {2 * rig.CALL_COST}\n"
    )
    assert _served_models(providers.upstream) == ("budget-at-cap", "budget-at-cap")
    assert providers.budget.settled("provider_spend:openai:1d", 2 * rig.CALL_COST) == 2 * rig.CALL_COST


def test_spend_over_a_tiny_provider_cap_rejects_with_429(providers: ProviderRig) -> None:
    first: Final = rig.chat(providers.gateway, "tiny-cap", "provider tiny first")
    assert first.status_code == 200, first.text

    blocked: Final = rig.until_rejected(providers.gateway, "tiny-cap", rig.probe_text("provider tiny"))

    assert blocked.status_code == 429, blocked.text
    assert "Exceeded budget for provider hosted_vllm" in blocked.text
    assert ">= 0.01" in blocked.text


@pytest.mark.xfail(strict=True, reason=f"budget_limit 0 is treated as unlimited: {ISSUE_43214}")
def test_a_zero_provider_cap_rejects_the_first_request(providers: ProviderRig) -> None:
    response: Final = rig.chat(providers.gateway, "zero-cap", rig.probe_text("provider zero"))

    assert rig.is_budget_rejection(response), response.text


@pytest.mark.xfail(
    strict=True,
    reason=f"a provider entry without budget_limit drops every deployment of that provider: {ISSUE_43214}",
)
def test_a_provider_entry_without_budget_limit_leaves_the_provider_uncapped(providers: ProviderRig) -> None:
    response: Final = rig.chat(providers.gateway, "no-limit", "provider without budget_limit")

    assert response.status_code == 200, response.text


@pytest.mark.xfail(
    strict=True,
    reason="a provider budget_limit without time_period is accepted at boot but never counted or enforced",
)
def test_a_provider_cap_without_time_period_is_still_enforced(providers: ProviderRig) -> None:
    first: Final = rig.chat(providers.gateway, "no-period", "provider without time_period")
    assert first.status_code == 200, first.text

    rig.until_rejected(providers.gateway, "no-period", rig.probe_text("provider no period"), seconds=10)


def test_traffic_moves_to_an_uncapped_sibling_once_the_provider_is_over_budget(providers: ProviderRig) -> None:
    exhaust: Final = rig.chat(providers.gateway, "fw-solo", "fireworks spend")
    assert exhaust.status_code == 200, exhaust.text
    rig.until_rejected(providers.gateway, "fw-solo", rig.probe_text("fireworks solo"))

    def batch() -> tuple[tuple[int, str | None], ...]:
        return tuple(
            (response.status_code, response.headers.get("x-litellm-model-id"))
            for response in (rig.chat(providers.gateway, "fw-pair", f"pair {index}") for index in range(6))
        )

    served: Final = eventually(
        batch, lambda outcomes: all(outcome == (200, "provider-sibling") for outcome in outcomes), seconds=30
    )

    assert len(served) == 6


def test_provider_budgets_endpoint_reports_the_spend_redis_holds(providers: ProviderRig) -> None:
    before: Final = datetime.now(timezone.utc)
    response: Final = rig.chat(providers.gateway, "reported", "reported spend")
    assert response.status_code == 200, response.text

    report: Final = eventually(
        lambda: object_value(object_value(providers.gateway.get("/provider/budgets")["providers"])["deepinfra"]),
        lambda entry: entry["spend"] == rig.CALL_COST,
    )

    assert report["budget_limit"] == 1.0
    assert report["time_period"] == "1d"
    assert providers.budget.redis_float("provider_spend:deepinfra:1d") == rig.CALL_COST
    reset_at: Final = datetime.fromisoformat(str(report["budget_reset_at"]))
    assert before < reset_at <= before + timedelta(days=1, minutes=1)


def test_a_null_provider_entry_fails_the_proxy_boot_with_a_named_error(tmp_path: Path) -> None:
    provider: Final = f"null-provider-{uuid.uuid4().hex}"
    with gateway_from_environment() as gateway, wire_server(rig.upstream) as upstream:
        config: Final = rig.write_config(
            tmp_path / "null-provider.yaml",
            (rig.deployment("null-provider", "openai/null-provider", upstream.url, model_id=provider),),
            provider_budget_config={provider: None},
        )
        with pytest.raises(AssertionError, match="Owned proxy exited before readiness"):
            with rig.budget_proxy(gateway, tmp_path, config, workers=1):
                pass

    assert len(rig.proxy_logs_mentioning(tmp_path, f"No budget config found for provider {provider}")) == 1


@pytest.mark.xfail(strict=True, reason="GET /provider/budgets without provider_budget_config answers 500")
def test_provider_budgets_endpoint_without_provider_config_is_a_client_error(tmp_path: Path) -> None:
    with gateway_from_environment() as gateway, wire_server(rig.upstream) as upstream:
        config: Final = rig.write_config(
            tmp_path / "deployment-only.yaml",
            (
                rig.deployment(
                    "deployment-only",
                    "openai/deployment-only",
                    upstream.url,
                    model_id="deployment-only",
                    max_budget=1,
                    budget_duration="1d",
                ),
            ),
        )
        with rig.budget_proxy(gateway, tmp_path, config) as budget:
            response: Final[httpx.Response] = budget.gateway.request("GET", "/provider/budgets")

    assert 400 <= response.status_code < 500, response.text


def _remaining_budget(gateway: Gateway) -> float | None:
    scrape: Final = gateway.request("GET", "/metrics/").text
    prefix: Final = 'litellm_provider_remaining_budget_metric{api_provider="openai"} '
    return next((float(line.removeprefix(prefix)) for line in scrape.splitlines() if line.startswith(prefix)), None)


@pytest.mark.xfail(
    strict=True,
    reason="litellm_provider_remaining_budget_metric is set only while routing, so it lags one request behind spend",
)
def test_the_prometheus_remaining_budget_follows_spend_after_a_call(tmp_path: Path) -> None:
    with gateway_from_environment() as gateway, wire_server(rig.upstream) as upstream:
        config: Final = rig.write_config(
            tmp_path / "prometheus.yaml",
            (rig.deployment("metered", "openai/metered", upstream.url, model_id="metered"),),
            provider_budget_config={"openai": {"budget_limit": 1, "time_period": "1d"}},
            litellm_settings={"callbacks": ["prometheus"]},
        )
        with rig.budget_proxy(gateway, tmp_path, config, workers=1) as budget:
            response: Final = rig.chat(budget.gateway, "metered", "metered call")
            assert response.status_code == 200, response.text
            assert budget.settled("provider_spend:openai:1d", rig.CALL_COST) == rig.CALL_COST

            remaining: Final = eventually(
                lambda: _remaining_budget(budget.gateway),
                lambda value: value == 1 - rig.CALL_COST,
                seconds=10,
                return_last_on_timeout=True,
            )

    assert remaining == 1 - rig.CALL_COST
