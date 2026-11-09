from types import SimpleNamespace
from typing import Final
from unittest.mock import Mock

import pytest

import litellm
from litellm._auth.budget_pricing import ModelPriceFacts
from litellm.proxy.auth import budget_pricing_adapter as native
from tests.unit.proxy.auth import phase1_budget_auth_reference as reference


def router(input_price: float | None, output_price: float | None, deployments: tuple[object, ...] = ()) -> Mock:
    view: Final = Mock()
    view.group.return_value = SimpleNamespace(input_cost_per_token=input_price, output_cost_per_token=output_price)
    view.zero_cost.return_value = {}
    view.deployments.return_value = deployments
    return Mock(registry=Mock(return_value=view))


@pytest.mark.parametrize(("input_price", "output_price"), ((None, 0), (0, None), (1, 0), (0, 1)))
def test_priced_or_undefined_group_does_not_read_deployments(
    input_price: float | None, output_price: float | None
) -> None:
    fake: Final = router(input_price, output_price)
    facts: Final = native.read_model_price_facts("model", fake)
    assert facts == ModelPriceFacts(group_found=True, input_price=input_price, output_price=output_price)
    fake.registry().deployments.assert_not_called()


@pytest.mark.parametrize(
    "route",
    (
        "/v1/models",
        "/model/info",
        "/health/services",
        "/chat/completions",
        "/mcp",
        "/mcp-rest/tools/list",
        "/mcp/tools/call",
        "/key/info",
    ),
)
@pytest.mark.parametrize("body", (None, {}, {"method": "tools/list"}, {"method": "tools/call"}, {"method": 1}))
def test_route_and_mcp_decision_matches_production_reference(route: str, body: dict[str, object] | None) -> None:
    assert native.request_skips_budget_checks(
        route, None, None, request_body=body
    ) == reference.request_skips_budget_checks(route, None, None, request_body=body)
    assert native.route_skips_budget_checks(route) == reference.route_skips_budget_checks(route)


def test_missing_models_and_empty_model_list_preserve_exemption_behavior() -> None:
    fake: Final = router(0, 0)
    assert native.is_model_cost_zero(None, fake) is False
    assert native.is_model_cost_zero("model", None) is False
    assert native.is_model_cost_zero([], fake) is True


@pytest.mark.parametrize(
    ("prices", "reads", "exempt"), (((0, 1, 0), 2, False), ((0, 0, 0), 3, True), ((1, 0, 0), 1, False))
)
def test_raw_exemption_prices_only_native_selected_models_and_preserves_cache_writes(
    monkeypatch: pytest.MonkeyPatch, prices: tuple[float, ...], reads: int, exempt: bool
) -> None:
    from litellm._auth import budget_pricing

    old: Final = router(0, 0)
    new: Final = router(0, 0)
    monkeypatch.setitem(
        litellm.model_cost, "budget-zero-fixture", {"input_cost_per_token": 0, "output_cost_per_token": 0}
    )
    deployment: Final = {
        "model_name": "model",
        "litellm_params": {"model": "openai/fixture"},
        "model_info": {"id": "budget-zero-fixture"},
    }
    old.registry().deployments.return_value = (deployment,)
    new.registry().deployments.return_value = (deployment,)
    old.registry().resolve.side_effect = lambda model: model
    new.registry().resolve.side_effect = lambda model: model
    old.registry().group.side_effect = tuple(
        SimpleNamespace(input_cost_per_token=p, output_cost_per_token=0) for p in prices
    )
    new.registry().group.side_effect = tuple(
        SimpleNamespace(input_cost_per_token=p, output_cost_per_token=0) for p in prices
    )
    old_cache: Final[dict[str, bool]] = {}
    new_cache: Final[dict[str, bool]] = {}
    old.registry().zero_cost.return_value = old_cache
    new.registry().zero_cost.return_value = new_cache
    models: Final = ["first", "second", "third"]
    before: Final = reference.request_skips_budget_checks("/chat/completions", models, old)
    facts: Final = native.read_budget_exemption_facts("/chat/completions", models, new)
    assert budget_pricing.exemption(facts).exempt is before is exempt
    assert len(facts.prices) == reads
    assert old.registry().group.call_count == new.registry().group.call_count == reads
    assert old_cache == new_cache


def test_raw_exemption_route_skip_does_not_touch_model_registry() -> None:
    fake: Final = router(0, 0)
    facts: Final = native.read_budget_exemption_facts("/v1/models", ["first", "second"], fake)
    assert facts.prices == ()
    fake.registry.assert_not_called()


def test_raw_cached_price_does_not_read_model_group_or_write_again() -> None:
    fake: Final = router(0, 0)
    fake.registry().zero_cost.return_value = {"model": False}
    facts: Final = native.read_model_cost_facts("model", fake)
    assert facts.cached is False
    fake.registry().group.assert_not_called()


def test_empty_model_list_still_reads_zero_cost_cache() -> None:
    fake: Final = router(0, 0)
    facts: Final = native.read_budget_exemption_facts("/chat/completions", [], fake)
    assert facts.prices == ()
    assert facts.cache_read is True
    fake.registry().zero_cost.assert_called_once_with()
    fake.registry().group.assert_not_called()
