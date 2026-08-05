import pytest

from litellm.router import Router
from litellm.router_strategy.savings_baseline import (
    Baseline,
    canonical_model,
    _models_in,
    _most_expensive,
    resolve_baseline,
)


@pytest.fixture
def parent() -> Router:
    return Router(
        model_list=[
            {"model_name": "cheap", "litellm_params": {"model": "anthropic/claude-haiku-4-5"}},
            {"model_name": "top", "litellm_params": {"model": "anthropic/claude-opus-5"}},
            {"model_name": "pool", "litellm_params": {"model": "anthropic/claude-haiku-4-5"}},
            {"model_name": "pool", "litellm_params": {"model": "anthropic/claude-opus-5"}},
        ]
    )


class TestCanonicalModel:
    def test_qualifies_a_bare_name_with_the_provider_that_owns_it(self):
        assert canonical_model("claude-opus-5") == "anthropic/claude-opus-5"

    def test_keeps_an_already_qualified_name_qualified(self):
        assert canonical_model("anthropic/claude-opus-5") == "anthropic/claude-opus-5"

    def test_honours_a_separately_declared_provider(self):
        assert canonical_model("claude-opus-5", "openai") == "openai/claude-opus-5"

    def test_returns_none_for_a_name_no_provider_claims(self):
        assert canonical_model("") is None


class TestModelsForGroup:
    def test_resolves_a_group_to_the_models_its_deployments_call(self, parent):
        assert [c.model for c in _models_in(parent, "cheap")] == ["anthropic/claude-haiku-4-5"]

    def test_returns_every_deployment_in_a_pooled_group(self, parent):
        assert sorted(c.model for c in _models_in(parent, "pool")) == [
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-opus-5",
        ]

    def test_treats_an_unknown_group_as_a_model_name(self, parent):
        """A tier can point straight at a provider model rather than a configured group."""
        assert [c.model for c in _models_in(parent, "claude-opus-5")] == ["anthropic/claude-opus-5"]


class TestMostExpensive:
    """Ranking runs through the router, because what a deployment costs is the
    router's answer to give: it merges configured prices over the built-in map."""

    def test_picks_by_output_rate(self, parent):
        picked = _most_expensive(parent, [Baseline("anthropic/claude-haiku-4-5"), Baseline("anthropic/claude-opus-5")])
        assert picked.model == "anthropic/claude-opus-5"

    def test_ignores_models_with_no_per_token_price(self, parent):
        """A free model as baseline would report the whole real spend as a loss."""
        picked = _most_expensive(
            parent, [Baseline("not-a-real-model-anywhere"), Baseline("anthropic/claude-haiku-4-5")]
        )
        assert picked.model == "anthropic/claude-haiku-4-5"

    def test_returns_none_when_nothing_can_be_priced(self, parent):
        assert _most_expensive(parent, [Baseline("not-a-real-model-anywhere")]) is None

    def test_returns_none_for_an_empty_candidate_set(self, parent):
        assert _most_expensive(parent, []) is None


class TestResolveBaseline:
    def test_derives_the_priciest_candidate(self, parent):
        assert resolve_baseline(parent, ["cheap", "top"]).model == "anthropic/claude-opus-5"

    def test_never_raises_so_a_metric_cannot_fail_a_live_request(self):
        """Read on the routing path while decorating a request that is about to be
        served; a dashboard counterfactual must not be able to take routing down."""

        class Exploding:
            @property
            def model_name_to_deployment_indices(self):
                raise RuntimeError("router is mid-reload")

        assert resolve_baseline(Exploding(), ["anything"]) is None

    def test_an_empty_candidate_set_zeroes_the_driver_rather_than_inventing_one(self, parent):
        assert resolve_baseline(parent, []) is None


class TestDeploymentsPricedByBaseModel:
    """`litellm_params.model` is not always a model.

    On Azure it is the deployment name, which is absent from the cost map, so pricing it
    directly drops the candidate. If that candidate was the priciest, the baseline quietly
    becomes the second priciest and every saving is understated; if the whole pool is
    Azure, nothing prices and the driver reports zero with nothing at default log level
    saying why. `model_info.base_model` is what names the real model, which is the chain
    router.py already resolves pricing through.
    """

    @staticmethod
    def _router(*deployments: dict) -> Router:
        return Router(model_list=list(deployments))

    def test_model_info_base_model_is_preferred_over_the_deployment_name(self):
        router = self._router(
            {
                "model_name": "big",
                "litellm_params": {"model": "azure/my-gpt5-deployment"},
                "model_info": {"base_model": "azure/gpt-4.1"},
            },
        )
        assert [c.model for c in _models_in(router, "big")] == ["azure/gpt-4.1"]

    def test_litellm_params_base_model_is_the_other_accepted_spelling(self):
        router = self._router(
            {
                "model_name": "big",
                "litellm_params": {"model": "azure/my-gpt5-deployment", "base_model": "azure/gpt-4.1"},
            },
        )
        assert [c.model for c in _models_in(router, "big")] == ["azure/gpt-4.1"]

    def test_a_deployment_without_a_base_model_still_prices_by_its_model(self):
        router = self._router({"model_name": "big", "litellm_params": {"model": "anthropic/claude-opus-5"}})
        assert [c.model for c in _models_in(router, "big")] == ["anthropic/claude-opus-5"]

    def test_an_azure_deployment_can_win_the_priciest_candidate(self):
        """Without the base_model hop the Azure candidate never prices, so the cheaper
        model wins by default and the reported saving shrinks."""
        router = self._router(
            {"model_name": "cheap", "litellm_params": {"model": "anthropic/claude-haiku-4-5"}},
            {
                "model_name": "big",
                "litellm_params": {"model": "azure/my-gpt5-deployment"},
                "model_info": {"base_model": "azure/gpt-4.1"},
            },
        )
        assert resolve_baseline(router, ["cheap", "big"]).model == "azure/gpt-4.1"

    def test_an_all_azure_pool_still_has_a_baseline(self):
        """Otherwise nothing prices, the driver is disabled and the card reads $0.00."""
        router = self._router(
            {
                "model_name": "big",
                "litellm_params": {"model": "azure/my-gpt5-deployment"},
                "model_info": {"base_model": "azure/gpt-4.1"},
            },
        )
        assert resolve_baseline(router, ["big"]).model == "azure/gpt-4.1"


class TestDeploymentPricingOverrides:
    """A deployment may not be charged the public rate for the model it names."""

    def test_a_configured_price_decides_the_baseline_not_the_public_rate(self):
        """A deployment configured far above its public rate is what the traffic would
        really have cost. Ranking on the public rate picks the wrong counterfactual and
        then prices it at a rate nobody pays."""
        router = Router(
            model_list=[
                {"model_name": "cheap", "litellm_params": {"model": "anthropic/claude-haiku-4-5"}},
                {"model_name": "top", "litellm_params": {"model": "anthropic/claude-opus-5"}},
            ]
        )
        assert resolve_baseline(router, ["cheap", "top"]).model == "anthropic/claude-opus-5"

        overridden = Router(
            model_list=[
                {
                    "model_name": "cheap",
                    "litellm_params": {
                        "model": "anthropic/claude-haiku-4-5",
                        "input_cost_per_token": 0.001,
                        "output_cost_per_token": 0.002,
                    },
                },
                {"model_name": "top", "litellm_params": {"model": "anthropic/claude-opus-5"}},
            ]
        )
        assert resolve_baseline(overridden, ["cheap", "top"]).model == "anthropic/claude-haiku-4-5"
