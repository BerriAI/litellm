"""
Test that models not in the cost map do NOT bypass budget enforcement.

Regression test for the bug where unmapped models got fallback costs of 0,
causing _is_model_cost_zero() to return True and skip all budget checks.

See: https://github.com/BerriAI/litellm/issues/24770
"""

import copy

import pytest

import litellm
from litellm.proxy.auth.auth_checks import _is_model_cost_zero
from litellm.router import Router


UNPRICED_ZERO_COST_MODEL = "ollama/unpriced-zero-cost-target"


def _explicitly_free(name: str, model: str = "gpt-3.5-turbo", **model_info: float) -> dict:
    return {
        "model_name": name,
        "litellm_params": {
            "model": model,
            "api_key": "sk-fake",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        "model_info": {"id": f"{name}-id", **model_info},
    }


def _free_by_cost_map_only(name: str) -> dict:
    litellm.model_cost[UNPRICED_ZERO_COST_MODEL] = {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "ollama",
        "mode": "chat",
    }
    return {
        "model_name": name,
        "litellm_params": {"model": UNPRICED_ZERO_COST_MODEL, "api_base": "http://localhost:11434"},
        "model_info": {"id": f"{name}-id"},
    }


def _explicitly_free_ollama_wildcard() -> dict:
    return {
        "model_name": "ollama/*",
        "litellm_params": {
            "model": "ollama/*",
            "api_base": "http://localhost:11434",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        "model_info": {"id": "free-wildcard-id"},
    }


def _served_model(router: Router, name: str) -> str:
    deployment = router.get_available_deployment(model=name, messages=[{"role": "user", "content": "hi"}])
    return deployment["litellm_params"]["model"]


class TestUnmappedModelBudgetEnforcement:
    """Unmapped models must NOT bypass budget checks."""

    def setup_method(self):
        """Snapshot litellm.model_cost before each test."""
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        """Restore litellm.model_cost after each test."""
        litellm.model_cost = self._saved_model_cost

    def test_unmapped_model_enforces_budget(self):
        """A model not in litellm.model_cost should have budget enforced."""
        router = Router(
            model_list=[
                {
                    "model_name": "custom-model",
                    "litellm_params": {
                        "model": "openai/totally-nonexistent-model-xyz",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="custom-model", llm_router=router)
        assert result is False, "Unmapped model should enforce budget (return False), not bypass it (return True)"

    def test_explicitly_free_model_bypasses_budget(self):
        """A model with explicit cost=0 in model_info should bypass budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "free-model-id",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="free-model", llm_router=router)
        assert result is True, "Explicitly free model should bypass budget (return True)"

    def test_known_paid_model_enforces_budget(self):
        """A model in the cost map with non-zero costs should enforce budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "paid-model",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="paid-model", llm_router=router)
        assert result is False, "Known paid model should enforce budget (return False)"

    def test_unmapped_model_with_litellm_params_pricing(self):
        """A model with cost=0 in litellm_params (not model_info) should bypass budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-via-params",
                    "litellm_params": {
                        "model": "openai/nonexistent-but-free-model",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        result = _is_model_cost_zero(model="free-via-params", llm_router=router)
        assert result is True, "Model with explicit cost=0 in litellm_params should bypass budget"

    def test_cache_invalidates_on_in_place_pricing_update(self):
        """
        Regression test for the stale-cache bug surfaced in PR review:
        upgrading an explicitly free deployment to paid via ``upsert_deployment``
        (same deployment count, same router instance) must invalidate the
        cached ``_is_model_cost_zero=True`` answer so budget checks resume
        immediately — not after the next proxy restart.
        """
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = Router(
            model_list=[
                {
                    "model_name": "ramping-model",
                    "litellm_params": {
                        "model": "openai/ramping-deploy",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "ramping-deploy-id",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        # Warm the cache as zero-cost.
        assert _is_model_cost_zero(model="ramping-model", llm_router=router) is True
        assert router._zero_cost_cache.get("ramping-model") is True

        # In-place pricing update: same deployment count, same router id,
        # same model name. The pre-fix cache key was
        # ``(id(router), len(model_list), model_name)`` and would not change.
        router.upsert_deployment(
            deployment=Deployment(
                model_name="ramping-model",
                litellm_params=LiteLLM_Params(
                    model="openai/ramping-deploy",
                    api_key="sk-fake",
                    input_cost_per_token=0.000002,
                    output_cost_per_token=0.000008,
                ),
                model_info=ModelInfo(
                    id="ramping-deploy-id",
                    input_cost_per_token=0.000002,
                    output_cost_per_token=0.000008,
                ),
            )
        )

        # Cache must have been cleared by ``_invalidate_model_group_info_cache``.
        assert router._zero_cost_cache == {}
        # Subsequent call sees the new pricing and enforces budget.
        assert _is_model_cost_zero(model="ramping-model", llm_router=router) is False

    def test_strategy_router_alias_with_zero_pricing_enforces_budget(self):
        """An auto-router alias is never the deployment that gets called or
        billed, so zero pricing configured on it must not waive budget checks
        for requests that route to (and bill as) a real paid deployment."""
        router = Router(
            model_list=[
                {
                    "model_name": "smart-router",
                    "litellm_params": {
                        "model": "auto_router/complexity_router/smart-router",
                        "complexity_router_default_model": "paid-model",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                        "complexity_router_config": {"tiers": {"simple": "paid-model"}},
                    },
                    "model_info": {"id": "alias-id"},
                },
                {
                    "model_name": "paid-model",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                    "model_info": {"id": "paid-id"},
                },
            ]
        )

        assert "input_cost_per_token" not in litellm.model_cost.get("alias-id", {})
        assert _is_model_cost_zero(model="smart-router", llm_router=router) is False

    def test_model_group_alias_to_free_model_bypasses_budget(self):
        """A zero-cost group reached through model_group_alias bypasses budget, like its own name.

        Both names route to the same deployment and add nothing to spend, so refusing one of
        them denies a request on spend it cannot produce.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"free-model-alias": "free-model"},
        )

        assert _is_model_cost_zero(model="free-model", llm_router=router) is True
        assert _is_model_cost_zero(model="free-model-alias", llm_router=router) is True, (
            "An alias pointing at an explicitly-zero-cost group must be read as free, like its own name"
        )

    def test_model_group_alias_item_form_bypasses_budget(self):
        """The dict alias form ({"model": ..., "hidden": False}) resolves like the string form."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"free-model-alias": {"model": "free-model", "hidden": False}},
        )

        assert _is_model_cost_zero(model="free-model-alias", llm_router=router) is True

    def test_model_group_alias_to_paid_model_enforces_budget(self):
        """An alias does not turn a priced group into a free one."""
        router = Router(
            model_list=[
                {
                    "model_name": "paid-model",
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-fake"},
                    "model_info": {"id": "paid-model-id"},
                },
            ],
            model_group_alias={"paid-model-alias": "paid-model"},
        )

        assert _is_model_cost_zero(model="paid-model-alias", llm_router=router) is False

    def test_model_group_alias_to_ptu_flat_cost_enforces_budget(self):
        """A PTU group keeps budget enforced through an alias.

        Its explicit zero per-token price exists so the flat capacity cost is not charged twice,
        so the PTU check has to resolve the alias too — resolving only the explicit-cost gate
        would let this through as free.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "ptu-model",
                    "litellm_params": {
                        "model": "azure/ptu-deployment",
                        "api_base": "https://fake.openai.azure.com",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "ptu-model-id",
                        "ptu_count": 100,
                        "cost_per_ptu_per_hour": 2.0,
                    },
                },
            ],
            model_group_alias={"ptu-model-alias": "ptu-model"},
        )

        assert _is_model_cost_zero(model="ptu-model", llm_router=router) is False
        assert _is_model_cost_zero(model="ptu-model-alias", llm_router=router) is False, (
            "An aliased PTU group must not be read as free"
        )

    def test_hidden_model_group_alias_to_free_model_bypasses_budget(self):
        """A hidden alias to an explicitly free group bypasses budget, like the group itself.

        ``get_model_group_info`` returns None for hidden aliases, so the alias must be
        resolved to its target group before the cost lookup.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"hidden-alias": {"model": "free-model", "hidden": True}},
        )

        assert _is_model_cost_zero(model="hidden-alias", llm_router=router) is True

    def test_hidden_model_group_alias_to_paid_model_enforces_budget(self):
        """A hidden alias to a priced group keeps budget enforced."""
        router = Router(
            model_list=[
                {
                    "model_name": "paid-model",
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-fake"},
                    "model_info": {"id": "paid-model-id"},
                },
            ],
            model_group_alias={"hidden-paid-alias": {"model": "paid-model", "hidden": True}},
        )

        assert _is_model_cost_zero(model="hidden-paid-alias", llm_router=router) is False

    def test_dangling_model_group_alias_enforces_budget(self):
        """An alias pointing at a group that does not exist keeps budget enforced."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"dangling-alias": "model-that-does-not-exist"},
        )

        assert _is_model_cost_zero(model="dangling-alias", llm_router=router) is False

    def test_repointed_hidden_alias_does_not_reuse_cached_free_result(self):
        """Repointing a hidden alias from a free group to a paid group re-evaluates the cost.

        ``Router.update_settings`` is the one runtime path that rewrites the alias map (the
        proxy's config update applies ``router_settings`` through it), so the cached verdict
        has to drop there.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
                {
                    "model_name": "paid-model",
                    "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "sk-fake"},
                    "model_info": {"id": "paid-model-id"},
                },
            ],
            model_group_alias={"hidden-alias": {"model": "free-model", "hidden": True}},
        )

        assert _is_model_cost_zero(model="hidden-alias", llm_router=router) is True
        router.update_settings(model_group_alias={"hidden-alias": {"model": "paid-model", "hidden": True}})
        assert _is_model_cost_zero(model="hidden-alias", llm_router=router) is False

    @pytest.mark.parametrize("alias_name_first", [True, False])
    def test_alias_shadowing_a_real_group_answers_for_its_target_in_either_order(self, alias_name_first: bool):
        """An alias whose name is also a real PTU-priced group is judged by the free target it routes to.

        The router resolves the alias before it looks at deployments, so the PTU deployment sharing
        the alias's name is never served under it. The verdict is cached per requested name, so
        whichever name is asked first, both names read as free.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
                {
                    "model_name": "ptu-model",
                    "litellm_params": {
                        "model": "azure/ptu-deployment",
                        "api_base": "https://fake.openai.azure.com",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "ptu-model-id", "ptu_count": 100, "cost_per_ptu_per_hour": 2.0},
                },
            ],
            model_group_alias={"ptu-model": "free-model"},
        )
        order = ("ptu-model", "free-model") if alias_name_first else ("free-model", "ptu-model")
        expected = {"ptu-model": True, "free-model": True}

        assert [_is_model_cost_zero(model=name, llm_router=router) for name in order] == [
            expected[name] for name in order
        ]
        assert [_is_model_cost_zero(model=name, llm_router=router) for name in order] == [
            expected[name] for name in order
        ], "the cached verdicts must match the first evaluation"

    @pytest.mark.parametrize(
        "alias_entry",
        ["unpriced-target", {"model": "unpriced-target", "hidden": True}],
        ids=["plain_alias", "hidden_alias"],
    )
    def test_alias_shadowing_an_explicitly_free_group_answers_for_its_unpriced_target(
        self, alias_entry: str | dict[str, str | bool]
    ):
        """An alias keyed like an explicitly free real group is judged by its target alone, hidden or not.

        The router serves the alias name from its target, a group whose cost-map price is zero with
        no explicit price on the deployment, so the budget stays enforced exactly as it is for the
        target by name. The shadowed explicitly free deployment is never served under that name and
        must not lend it the bypass.
        """
        litellm.model_cost["ollama/unpriced-zero-cost-target"] = {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "ollama",
            "mode": "chat",
        }
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
                {
                    "model_name": "unpriced-target",
                    "litellm_params": {
                        "model": "ollama/unpriced-zero-cost-target",
                        "api_base": "http://localhost:11434",
                    },
                    "model_info": {"id": "unpriced-target-id"},
                },
            ],
            model_group_alias={"free-model": alias_entry},
        )

        assert _is_model_cost_zero(model="unpriced-target", llm_router=router) is False
        assert _is_model_cost_zero(model="free-model", llm_router=router) is False, (
            "the alias routes to the unpriced target, so it must be refused like the target by name"
        )

    def test_hidden_alias_shadowing_a_ptu_group_answers_for_its_free_target(self):
        """A hidden alias keyed like a PTU-priced real group reads as free when its target is free.

        The PTU deployment sharing the alias's name is never served under it, so its flat cost must
        not keep the alias enforced while the router serves every call from the free target.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "ptu-model",
                    "litellm_params": {
                        "model": "azure/ptu-deployment",
                        "api_base": "https://fake.openai.azure.com",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "ptu-model-id", "ptu_count": 100, "cost_per_ptu_per_hour": 2.0},
                },
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"ptu-model": {"model": "free-model", "hidden": True}},
        )

        assert _is_model_cost_zero(model="free-model", llm_router=router) is True
        assert _is_model_cost_zero(model="ptu-model", llm_router=router) is True, (
            "the alias routes to the free target, so it must bypass budget like the target by name"
        )

    def test_hidden_alias_shadowing_an_explicitly_free_group_to_a_priced_target_enforces_budget(self):
        """A hidden alias keyed like an explicitly free real group stays enforced when its target is priced."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
                {
                    "model_name": "paid-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0000002,
                        "output_cost_per_token": 0.0000012,
                    },
                    "model_info": {"id": "paid-model-id"},
                },
            ],
            model_group_alias={"free-model": {"model": "paid-model", "hidden": True}},
        )

        assert _is_model_cost_zero(model="free-model", llm_router=router) is False

    def test_alias_chain_through_a_priced_group_enforces_budget(self):
        """An alias to a group that is itself an alias key resolves one hop, like the router does.

        The router serves ``chain-smart`` with the real ``chain-legacy`` deployment, which is priced,
        so following the second hop to the free group would waive the budget for a paid call.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "chain-legacy",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0000002,
                        "output_cost_per_token": 0.0000012,
                    },
                    "model_info": {"id": "chain-legacy-id"},
                },
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "ollama/llama2",
                        "api_base": "http://localhost:11434",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"chain-smart": "chain-legacy", "chain-legacy": "free-model"},
        )

        assert _is_model_cost_zero(model="chain-smart", llm_router=router) is False

    @pytest.mark.parametrize("hidden", [False, True], ids=["plain_alias", "hidden_alias"])
    def test_alias_to_an_unpriced_group_that_is_also_an_alias_enforces_budget(self, hidden: bool):
        """An alias is judged by the group it routes to, never by where that group's own alias points.

        The router serves ``chain-entry`` from the real ``chain-middle`` deployment, whose zero price
        comes from the cost map alone. ``chain-middle`` is also an alias key to an explicitly free
        group, a second hop the router never takes for ``chain-entry``, so that group must not lend
        it the bypass.
        """

        def alias(target: str) -> str | dict[str, str | bool]:
            return {"model": target, "hidden": True} if hidden else target

        router = Router(
            model_list=[_free_by_cost_map_only("chain-middle"), _explicitly_free("free-model")],
            model_group_alias={"chain-entry": alias("chain-middle"), "chain-middle": alias("free-model")},
        )

        assert _served_model(router, "chain-entry") == UNPRICED_ZERO_COST_MODEL
        assert _is_model_cost_zero(model="chain-entry", llm_router=router) is False
        assert _is_model_cost_zero(model="chain-middle", llm_router=router) is True, (
            "asked by its own name, chain-middle routes to the explicitly free group"
        )

    def test_alias_to_a_free_group_that_is_also_an_alias_to_a_ptu_group_bypasses_budget(self):
        """A PTU group one alias hop past the group that serves the request does not enforce the budget.

        The router serves ``chain-entry`` from the real, explicitly free ``chain-middle`` deployment.
        ``chain-middle`` is also an alias key to a PTU-priced group, which only a request for
        ``chain-middle`` itself routes to.
        """
        router = Router(
            model_list=[
                _explicitly_free("chain-middle"),
                _explicitly_free("ptu-model", model="azure/ptu-deployment", ptu_count=100, cost_per_ptu_per_hour=2.0),
            ],
            model_group_alias={"chain-entry": "chain-middle", "chain-middle": "ptu-model"},
        )

        assert _served_model(router, "chain-entry") == "gpt-3.5-turbo"
        assert _is_model_cost_zero(model="chain-entry", llm_router=router) is True
        assert _is_model_cost_zero(model="chain-middle", llm_router=router) is False, (
            "asked by its own name, chain-middle routes to the PTU group"
        )

    def test_alias_chain_served_by_a_paid_wildcard_route_enforces_budget(self):
        """An alias whose target is only an alias key is never judged by that second alias's free group.

        ``chain-entry`` resolves one hop to ``gpt-4o-mini``, which is no deployment's name, so the
        router serves it from the paid wildcard route. The free group ``gpt-4o-mini`` is aliased
        to is only reached by a request for ``gpt-4o-mini`` itself.
        """
        router = Router(
            model_list=[
                _explicitly_free("free-model", model="ollama/llama2"),
                {"model_name": "*", "litellm_params": {"model": "openai/*", "api_key": "sk-fake"}},
            ],
            model_group_alias={"chain-entry": "gpt-4o-mini", "gpt-4o-mini": "free-model"},
        )

        assert _served_model(router, "chain-entry") == "openai/gpt-4o-mini"
        assert _is_model_cost_zero(model="chain-entry", llm_router=router) is False

    @pytest.mark.parametrize("alias_name", ["openai/smart", "smart"], ids=["alias_on_pattern", "alias_off_pattern"])
    def test_alias_chain_served_by_an_explicitly_priced_wildcard_route_enforces_budget(self, alias_name: str):
        """A chain the router serves from a priced wildcard route is budgeted at that route's price.

        The group's price reads $0 through the second alias to the free group, and the wildcard
        route's cost-map entry carries explicit prices, so only their sign tells that the served
        deployment is not free.
        """
        router = Router(
            model_list=[
                _explicitly_free("free-model", model="ollama/llama2"),
                {
                    "model_name": "openai/*",
                    "litellm_params": {
                        "model": "openai/*",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.00001,
                        "output_cost_per_token": 0.00002,
                    },
                    "model_info": {"id": "priced-wildcard-id"},
                },
            ],
            model_group_alias={alias_name: "openai/gpt-4o-mini", "openai/gpt-4o-mini": "free-model"},
        )

        assert _served_model(router, alias_name) == "openai/gpt-4o-mini"
        assert _is_model_cost_zero(model=alias_name, llm_router=router) is False

    def test_alias_shadowing_a_free_group_is_judged_by_its_unpriced_target_through_an_alias_chain(self):
        """A shadowing alias stays enforced when its unpriced target is itself an alias key to a free group."""
        router = Router(
            model_list=[
                _explicitly_free("shadowed-free"),
                _free_by_cost_map_only("chain-middle"),
                _explicitly_free("free-model"),
            ],
            model_group_alias={"shadowed-free": "chain-middle", "chain-middle": "free-model"},
        )

        assert _served_model(router, "shadowed-free") == UNPRICED_ZERO_COST_MODEL
        assert _is_model_cost_zero(model="shadowed-free", llm_router=router) is False

    @pytest.mark.parametrize("alias_name", ["ollama/fast", "fast"], ids=["alias_on_pattern", "alias_off_pattern"])
    def test_alias_to_a_name_served_by_an_explicitly_free_wildcard_route_bypasses_budget(self, alias_name: str):
        """An alias to a name only a wildcard route serves reads that route's deployment, like the name itself."""
        router = Router(
            model_list=[_explicitly_free_ollama_wildcard()],
            model_group_alias={alias_name: "ollama/llama3"},
        )

        assert _served_model(router, alias_name) == "ollama/llama3"
        assert _is_model_cost_zero(model="ollama/llama3", llm_router=router) is True
        assert _is_model_cost_zero(model=alias_name, llm_router=router) is True

    def test_alias_chain_to_a_name_served_by_an_explicitly_free_wildcard_route_bypasses_budget(self):
        """An alias to an alias key no deployment is named after reads the wildcard route serving it.

        ``ollama/fast`` resolves one hop to ``ollama/llama3``, which is only an alias key, so the
        router serves it from the explicitly free wildcard route. The group ``ollama/llama3`` is
        aliased to is only reached by a request for ``ollama/llama3`` itself.
        """
        router = Router(
            model_list=[_explicitly_free_ollama_wildcard(), _free_by_cost_map_only("unpriced-model")],
            model_group_alias={"ollama/fast": "ollama/llama3", "ollama/llama3": "unpriced-model"},
        )

        assert _served_model(router, "ollama/fast") == "ollama/llama3"
        assert _is_model_cost_zero(model="ollama/fast", llm_router=router) is True
        assert _is_model_cost_zero(model="ollama/llama3", llm_router=router) is False, (
            "asked by its own name, ollama/llama3 routes to the unpriced group"
        )

    def test_handles_router_without_zero_cost_cache_attribute(self):
        """Tolerate router-like objects (e.g. ``MagicMock`` stand-ins) that
        do not expose ``_zero_cost_cache`` — the auth check must still
        compute a correct answer, just without caching."""
        from unittest.mock import MagicMock

        from litellm.types.router import ModelGroupInfo

        mock_router = MagicMock(spec=Router)
        mock_router.model_list = []
        mock_router.get_model_group_info.return_value = ModelGroupInfo(
            model_group="paid-model",
            providers=["openai"],
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        # Strip the attribute so the helper falls back to the no-cache path.
        del mock_router._zero_cost_cache

        result = _is_model_cost_zero(model="paid-model", llm_router=mock_router)
        assert result is False
