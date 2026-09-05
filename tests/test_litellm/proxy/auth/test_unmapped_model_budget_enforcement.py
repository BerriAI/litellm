"""
Test that models not in the cost map do NOT bypass budget enforcement.

Regression test for the bug where unmapped models got fallback costs of 0,
causing _is_model_cost_zero() to return True and skip all budget checks.

See: https://github.com/BerriAI/litellm/issues/24770
"""

import copy

import litellm
from litellm.proxy.auth.auth_checks import _is_model_cost_zero
from litellm.router import Router


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
        assert result is False, (
            "Unmapped model should enforce budget (return False), "
            "not bypass it (return True)"
        )

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
        assert (
            result is True
        ), "Explicitly free model should bypass budget (return True)"

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
        assert (
            result is True
        ), "Model with explicit cost=0 in litellm_params should bypass budget"

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

    def test_model_group_alias_to_free_model_bypasses_budget(self):
        """An explicitly free model reached through model_group_alias should
        bypass budget, same as when it is called by its own name."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "openai/nonexistent-but-free-model",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"free-model-alias": "free-model"},
        )
        assert _is_model_cost_zero(model="free-model", llm_router=router) is True
        result = _is_model_cost_zero(model="free-model-alias", llm_router=router)
        assert result is True, "Alias of an explicitly free model should bypass budget (return True)"

    def test_model_group_alias_to_paid_model_enforces_budget(self):
        """An alias must not waive budget checks for a paid model group."""
        router = Router(
            model_list=[
                {
                    "model_name": "paid-model",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-fake",
                    },
                },
            ],
            model_group_alias={"paid-model-alias": "paid-model"},
        )
        result = _is_model_cost_zero(model="paid-model-alias", llm_router=router)
        assert result is False, "Alias of a paid model should enforce budget"

    def test_hidden_model_group_alias_enforces_budget(self):
        """A hidden alias has no resolvable model group, so budget stays enforced."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "openai/nonexistent-but-free-model",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"hidden-alias": {"model": "free-model", "hidden": True}},
        )
        result = _is_model_cost_zero(model="hidden-alias", llm_router=router)
        assert result is False, "Hidden alias should enforce budget"

    def test_dangling_model_group_alias_enforces_budget(self):
        """An alias pointing at a model group that does not exist must not bypass budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-model",
                    "litellm_params": {
                        "model": "openai/nonexistent-but-free-model",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {"id": "free-model-id"},
                },
            ],
            model_group_alias={"dangling-alias": "no-such-model-group"},
        )
        result = _is_model_cost_zero(model="dangling-alias", llm_router=router)
        assert result is False, "Alias to a missing model group should enforce budget"

    def test_model_group_alias_to_ptu_flat_cost_enforces_budget(self):
        """A PTU deployment bills reserved capacity as a flat cost and carries an explicit
        zero per-token price. Reaching it through an alias must still enforce budget."""
        router = Router(
            model_list=[
                {
                    "model_name": "ptu-model",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_key": "sk-fake",
                        "api_base": "https://example.openai.azure.com",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "ptu-model-id",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                        "ptu_count": 100,
                        "cost_per_ptu_per_hour": 2.0,
                    },
                },
            ],
            model_group_alias={"ptu-model-alias": "ptu-model"},
        )
        assert _is_model_cost_zero(model="ptu-model", llm_router=router) is False
        result = _is_model_cost_zero(model="ptu-model-alias", llm_router=router)
        assert result is False, "Alias of a PTU flat-cost model should enforce budget"
