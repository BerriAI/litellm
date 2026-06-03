"""
Test that models not in the cost map do NOT bypass budget enforcement.

Regression test for the bug where unmapped models got fallback costs of 0,
causing _is_model_cost_zero() to return True and skip all budget checks.

See: https://github.com/BerriAI/litellm/issues/24770
"""

import copy

import litellm
from litellm.proxy.auth.auth_checks import _is_model_cost_zero
from litellm.proxy.auth.user_api_key_auth import (
    _get_budget_relevant_fallback_models,
    _should_skip_budget_checks,
)
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

    def test_zero_cost_model_with_paid_router_fallback_enforces_budget(self):
        """A free primary must not bypass budgets when router fallback is paid."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-fallback",
                    "litellm_params": {
                        "model": "openai/paid-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[{"free-primary": ["paid-fallback"]}],
        )

        assert (
            _is_model_cost_zero(model="free-primary", llm_router=router) is True
        )
        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_budget_checks_run_without_router_context(self):
        """Missing router context must not skip budget checks."""
        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=None,
            )
            is False
        )

    def test_paid_primary_model_enforces_budget_without_checking_fallbacks(self):
        """Paid primary models keep budget checks enabled regardless of fallbacks."""
        router = Router(
            model_list=[
                {
                    "model_name": "paid-primary",
                    "litellm_params": {
                        "model": "openai/paid-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                }
            ]
        )

        assert (
            _should_skip_budget_checks(
                request_data={"model": "paid-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_zero_cost_router_fallback_skips_budget(self):
        """All-zero primary/fallback chain keeps the existing skip behavior."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-fallback",
                    "litellm_params": {
                        "model": "openai/free-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ],
            fallbacks=[{"free-primary": ["free-fallback"]}],
        )

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is True
        )

    def test_zero_cost_model_with_transitive_paid_fallback_enforces_budget(self):
        """A paid fallback later in the router chain keeps budget checks enabled."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-fallback",
                    "litellm_params": {
                        "model": "openai/free-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-fallback",
                    "litellm_params": {
                        "model": "openai/paid-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[
                {"free-primary": ["free-fallback"]},
                {"free-fallback": ["paid-fallback"]},
            ],
        )

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_paid_request_fallback_enforces_budget(self):
        """Client-provided paid fallbacks also keep budget checks enabled."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-fallback",
                    "litellm_params": {
                        "model": "openai/paid-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ]
        )

        assert (
            _should_skip_budget_checks(
                request_data={
                    "model": "free-primary",
                    "fallbacks": [{"free-primary": ["paid-fallback"]}],
                },
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_request_fallback_with_transitive_paid_router_fallback_enforces_budget(
        self,
    ):
        """Request fallback models are also checked through router fallback chains."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-request-fallback",
                    "litellm_params": {
                        "model": "openai/free-request-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-router-fallback",
                    "litellm_params": {
                        "model": "openai/paid-router-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[
                {"free-request-fallback": ["paid-router-fallback"]},
            ],
        )

        assert (
            _should_skip_budget_checks(
                request_data={
                    "model": "free-primary",
                    "fallbacks": [{"free-primary": ["free-request-fallback"]}],
                },
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_paid_generic_router_fallback_enforces_budget(self):
        """Generic router fallback lists are checked beyond the first entry."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "free-generic-fallback",
                    "litellm_params": {
                        "model": "openai/free-generic-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-generic-fallback",
                    "litellm_params": {
                        "model": "openai/paid-generic-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
        )
        router.fallbacks = ["free-generic-fallback", "paid-generic-fallback"]

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_paid_generic_dict_fallback_enforces_budget(self):
        """Generic router fallback dict entries are included in budget checks."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-dict-fallback",
                    "litellm_params": {
                        "model": "openai/paid-dict-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ],
            fallbacks=[{"model": "paid-dict-fallback"}],
        )

        assert (
            _should_skip_budget_checks(
                request_data={"model": "free-primary"},
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_zero_cost_model_with_paid_override_fallback_enforces_budget(self):
        """Per-request router overrides are included in budget checks."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
                {
                    "model_name": "paid-override-fallback",
                    "litellm_params": {
                        "model": "openai/paid-override-fallback",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.000001,
                        "output_cost_per_token": 0.000002,
                    },
                },
            ]
        )

        assert (
            _should_skip_budget_checks(
                request_data={
                    "model": "free-primary",
                    "router_settings_override": {
                        "fallbacks": [{"free-primary": ["paid-override-fallback"]}]
                    },
                },
                route="/chat/completions",
                request=None,
                llm_router=router,
            )
            is False
        )

    def test_budget_fallback_discovery_ignores_non_string_model_entries(self):
        """Defensive fallback traversal tolerates malformed model lists."""
        router = Router(
            model_list=[
                {
                    "model_name": "free-primary",
                    "litellm_params": {
                        "model": "openai/free-primary",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                }
            ]
        )

        assert (
            _get_budget_relevant_fallback_models(
                model=["free-primary", {"bad": "shape"}],
                request_data={},
                llm_router=router,
            )
            == []
        )
