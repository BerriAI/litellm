"""
Test that models not in the cost map do NOT bypass budget enforcement.

Regression test for the bug where unmapped models got fallback costs of 0,
causing _is_model_cost_zero() to return True and skip all budget checks.

See: https://github.com/BerriAI/litellm/issues/24770
"""

import copy
from typing import Optional
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    ModelCostClass,
    _block_unknown_cost_models_check,
    _classify_model_group_cost,
    _is_model_cost_zero,
    _request_has_unknown_cost_model,
    common_checks,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router


def _router_with_mixed_costs() -> Router:
    """Router exposing the three cost classes through one wildcard plus two
    explicitly-priced deployments, so tests don't depend on the live cost map:

    * ``anthropic/*`` -> unmapped concrete models resolve to UNKNOWN cost
    * ``paid-model``  -> explicit positive pricing (KNOWN_POSITIVE)
    * ``free-model``  -> explicit $0 pricing (EXPLICIT_ZERO)
    """
    return Router(
        model_list=[
            {
                "model_name": "anthropic/*",
                "litellm_params": {"model": "anthropic/*", "api_key": "sk-fake"},
            },
            {
                "model_name": "paid-model",
                "litellm_params": {
                    "model": "openai/some-paid-model",
                    "api_key": "sk-fake",
                    "input_cost_per_token": 0.000002,
                    "output_cost_per_token": 0.000008,
                },
                "model_info": {
                    "id": "paid-model-id",
                    "input_cost_per_token": 0.000002,
                    "output_cost_per_token": 0.000008,
                },
            },
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


# A model id that cannot appear in any cost map (neither remote nor backup),
# so it deterministically resolves to UNKNOWN cost through the wildcard route.
_UNMAPPED_MODEL = "anthropic/brand-new-unmapped-model-xyz"


@pytest.fixture
def mock_proxy_logging() -> ProxyLogging:
    proxy_logging = ProxyLogging(user_api_key_cache=None)

    async def _noop_budget_alerts(*args, **kwargs):
        return None

    proxy_logging.budget_alerts = _noop_budget_alerts
    return proxy_logging


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
        assert (
            router._model_cost_class_cache.get("ramping-model")
            == ModelCostClass.EXPLICIT_ZERO.value
        )

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
        assert router._model_cost_class_cache == {}
        # Subsequent call sees the new pricing and enforces budget.
        assert _is_model_cost_zero(model="ramping-model", llm_router=router) is False

    def test_handles_router_without_cost_class_cache_attribute(self):
        """Tolerate router-like objects (e.g. ``MagicMock`` stand-ins) that
        do not expose ``_model_cost_class_cache`` — the auth check must still
        compute a correct answer, just without caching."""
        from unittest.mock import MagicMock

        mock_router = MagicMock(spec=Router)
        mock_router.get_model_list.return_value = [
            {
                "model_name": "paid-model",
                "litellm_params": {"model": "openai/some-paid-model"},
                "model_info": {"id": "paid-id"},
            }
        ]
        mock_router.get_deployment_model_info.return_value = {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
        }
        # Strip the attribute so the helper falls back to the no-cache path.
        del mock_router._model_cost_class_cache

        result = _is_model_cost_zero(model="paid-model", llm_router=mock_router)
        assert result is False


class TestClassifyModelGroupCost:
    """The tri-state classifier must distinguish 'free' from 'unknown' cost."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    def test_unmapped_wildcard_model_is_unknown(self):
        router = _router_with_mixed_costs()
        assert (
            _classify_model_group_cost(_UNMAPPED_MODEL, router)
            == ModelCostClass.UNKNOWN
        )

    def test_explicitly_priced_model_is_known_positive(self):
        router = _router_with_mixed_costs()
        assert (
            _classify_model_group_cost("paid-model", router)
            == ModelCostClass.KNOWN_POSITIVE
        )

    def test_explicitly_free_model_is_explicit_zero(self):
        router = _router_with_mixed_costs()
        assert (
            _classify_model_group_cost("free-model", router)
            == ModelCostClass.EXPLICIT_ZERO
        )


class TestRequestHasUnknownCostModel:
    """Detector backing the fail-closed check."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    def test_unmapped_model_is_unknown(self):
        router = _router_with_mixed_costs()
        assert _request_has_unknown_cost_model(_UNMAPPED_MODEL, router) is True

    def test_known_paid_model_is_not_unknown(self):
        router = _router_with_mixed_costs()
        assert _request_has_unknown_cost_model("paid-model", router) is False

    def test_explicitly_free_model_is_not_unknown(self):
        router = _router_with_mixed_costs()
        assert _request_has_unknown_cost_model("free-model", router) is False

    def test_any_unmapped_model_in_list_is_unknown(self):
        router = _router_with_mixed_costs()
        assert (
            _request_has_unknown_cost_model(["paid-model", _UNMAPPED_MODEL], router)
            is True
        )

    def test_none_model_is_not_unknown(self):
        router = _router_with_mixed_costs()
        assert _request_has_unknown_cost_model(None, router) is False


class TestBlockUnknownCostModelsCheck:
    """Fail-closed gate (OWASP LLM10 Denial of Wallet) on unknown-cost models."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    def test_flag_off_allows_unmapped_model(self):
        router = _router_with_mixed_costs()
        # No exception: the flag is opt-in, default behavior is unchanged.
        _block_unknown_cost_models_check(
            enabled=False,
            model=_UNMAPPED_MODEL,
            llm_router=router,
            route="/v1/chat/completions",
        )

    def test_flag_on_blocks_unmapped_model(self):
        router = _router_with_mixed_costs()
        with pytest.raises(ProxyException) as exc_info:
            _block_unknown_cost_models_check(
                enabled=True,
                model=_UNMAPPED_MODEL,
                llm_router=router,
                route="/v1/chat/completions",
            )
        assert str(exc_info.value.code) == "400"
        assert "brand-new-unmapped-model-xyz" in exc_info.value.message
        assert "block_unknown_cost_models" in exc_info.value.message

    def test_flag_on_allows_known_paid_model(self):
        router = _router_with_mixed_costs()
        _block_unknown_cost_models_check(
            enabled=True,
            model="paid-model",
            llm_router=router,
            route="/v1/chat/completions",
        )

    def test_flag_on_allows_explicitly_free_model(self):
        router = _router_with_mixed_costs()
        _block_unknown_cost_models_check(
            enabled=True,
            model="free-model",
            llm_router=router,
            route="/v1/chat/completions",
        )

    def test_flag_on_ignores_non_llm_route(self):
        router = _router_with_mixed_costs()
        # Management routes never call an LLM, so there is no spend to enforce.
        _block_unknown_cost_models_check(
            enabled=True,
            model=_UNMAPPED_MODEL,
            llm_router=router,
            route="/key/generate",
        )

    def test_flag_on_blocks_list_containing_unmapped_model(self):
        router = _router_with_mixed_costs()
        with pytest.raises(ProxyException):
            _block_unknown_cost_models_check(
                enabled=True,
                model=["paid-model", _UNMAPPED_MODEL],
                llm_router=router,
                route="/v1/chat/completions",
            )

    def test_flag_on_allows_wildcard_with_explicit_zero_cost(self):
        """
        Wildcard deployments with explicit ``input_cost_per_token=0`` /
        ``output_cost_per_token=0`` must classify as EXPLICIT_ZERO for any
        concrete model they match, so the fail-closed guard does not block
        legitimately free traffic. Before the fix, ``_is_cost_explicitly_configured``
        compared the requested model against ``deployment["model_name"]`` directly,
        which never matched a wildcard pattern like ``free-wildcard/*`` and
        misclassified the request as UNKNOWN.
        """
        router = Router(
            model_list=[
                {
                    "model_name": "free-wildcard/*",
                    "litellm_params": {
                        "model": "openai/free-wildcard/*",
                        "api_key": "sk-fake",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "model_info": {
                        "id": "free-wildcard-id",
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                },
            ]
        )
        _block_unknown_cost_models_check(
            enabled=True,
            model="free-wildcard/any-concrete-model",
            llm_router=router,
            route="/v1/chat/completions",
        )

    def test_flag_on_blocks_mixed_group_with_unpriced_deployment(self):
        """A model group that load balances across a priced and an unpriced
        deployment must be blocked: get_model_group_info reports the maximum
        cost across deployments, so the priced one would mask the unpriced one,
        yet a call routed to the unpriced deployment records $0 and bypasses
        budgets."""
        router = Router(
            model_list=[
                {
                    "model_name": "mixed-group",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-fake",
                    },
                    "model_info": {"id": "mixed-priced"},
                },
                {
                    "model_name": "mixed-group",
                    "litellm_params": {
                        "model": "openai/totally-unmapped-deployment-zzz",
                        "api_key": "sk-fake",
                    },
                    "model_info": {"id": "mixed-unpriced"},
                },
            ]
        )
        assert (
            _classify_model_group_cost("mixed-group", router) == ModelCostClass.UNKNOWN
        )
        with pytest.raises(ProxyException) as exc_info:
            _block_unknown_cost_models_check(
                enabled=True,
                model="mixed-group",
                llm_router=router,
                route="/v1/chat/completions",
            )
        assert "mixed-group" in exc_info.value.message


class TestBlockUnknownCostModelsViaCommonChecks:
    """End-to-end wiring through common_checks (the actual auth call site)."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    async def _run_common_checks(
        self,
        model: str,
        general_settings: dict,
        proxy_logging,
        request_body: Optional[dict] = None,
    ):
        router = _router_with_mixed_costs()
        return await common_checks(
            request_body=request_body if request_body is not None else {"model": model},
            team_object=None,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings=general_settings,
            route="/v1/chat/completions",
            llm_router=router,
            proxy_logging_obj=proxy_logging,
            valid_token=UserAPIKeyAuth(token="test-token"),
            request=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_unmapped_model_allowed_when_flag_off(self, mock_proxy_logging):
        # Demonstrates the Denial-of-Wallet gap: without the flag, an unmapped
        # model (whose spend can't be tracked) passes auth unimpeded.
        result = await self._run_common_checks(
            model=_UNMAPPED_MODEL,
            general_settings={},
            proxy_logging=mock_proxy_logging,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_unmapped_model_blocked_when_flag_on(self, mock_proxy_logging):
        with pytest.raises(ProxyException) as exc_info:
            await self._run_common_checks(
                model=_UNMAPPED_MODEL,
                general_settings={"block_unknown_cost_models": True},
                proxy_logging=mock_proxy_logging,
            )
        assert "brand-new-unmapped-model-xyz" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_known_model_allowed_when_flag_on(self, mock_proxy_logging):
        result = await self._run_common_checks(
            model="paid-model",
            general_settings={"block_unknown_cost_models": True},
            proxy_logging=mock_proxy_logging,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_unmapped_completion_model_default_blocked_when_flag_on(
        self, mock_proxy_logging
    ):
        """When a request omits ``model`` but ``general_settings.completion_model``
        names an unpriced model, the fail-closed guard must classify the
        server default; otherwise ``common_request_processing`` would silently
        substitute the default after auth, reopening the Denial-of-Wallet gap."""
        with pytest.raises(ProxyException) as exc_info:
            await self._run_common_checks(
                model="",
                general_settings={
                    "block_unknown_cost_models": True,
                    "completion_model": _UNMAPPED_MODEL,
                },
                proxy_logging=mock_proxy_logging,
                request_body={},
            )
        assert "brand-new-unmapped-model-xyz" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_known_completion_model_default_allowed_when_flag_on(
        self, mock_proxy_logging
    ):
        result = await self._run_common_checks(
            model="",
            general_settings={
                "block_unknown_cost_models": True,
                "completion_model": "paid-model",
            },
            proxy_logging=mock_proxy_logging,
            request_body={},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_unpriced_completion_model_overrides_priced_request_model(
        self, mock_proxy_logging
    ):
        """``completion_model`` takes precedence over the request body model in
        ``common_request_processing``, so an unpriced server default must be
        blocked even when the request itself names a priced model; otherwise the
        priced model in the body masks the unpriced model that actually runs."""
        with pytest.raises(ProxyException) as exc_info:
            await self._run_common_checks(
                model="paid-model",
                general_settings={
                    "block_unknown_cost_models": True,
                    "completion_model": _UNMAPPED_MODEL,
                },
                proxy_logging=mock_proxy_logging,
                request_body={"model": "paid-model"},
            )
        assert "brand-new-unmapped-model-xyz" in exc_info.value.message


class TestCostClassificationCaching:
    """The free-model bypass and the unknown-cost block check must share one
    per-router classification, so a model is resolved at most once per request."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    def test_both_callers_share_a_single_deployment_lookup(self):
        router = _router_with_mixed_costs()
        lookups = {"count": 0}
        original_get_model_list = router.get_model_list

        def counting_get_model_list(*args, **kwargs):
            lookups["count"] += 1
            return original_get_model_list(*args, **kwargs)

        router.get_model_list = counting_get_model_list

        # First caller resolves and caches the classification (one lookup).
        assert _request_has_unknown_cost_model(_UNMAPPED_MODEL, router) is True
        lookups_after_first = lookups["count"]
        assert lookups_after_first >= 1

        # The other caller for the same model must reuse the cached class with
        # no additional router lookup (this is the double-lookup regression).
        assert _is_model_cost_zero(model=_UNMAPPED_MODEL, llm_router=router) is False
        assert lookups["count"] == lookups_after_first

        assert (
            router._model_cost_class_cache[_UNMAPPED_MODEL]
            == ModelCostClass.UNKNOWN.value
        )

    def test_cache_hit_does_not_call_router_again(self):
        router = _router_with_mixed_costs()
        assert (
            _classify_model_group_cost("paid-model", router)
            == ModelCostClass.KNOWN_POSITIVE
        )

        # Once cached, a stale router lookup must never be consulted again.
        def fail_if_called(*args, **kwargs):
            raise AssertionError("classification cache was bypassed")

        router.get_model_list = fail_if_called
        assert (
            _classify_model_group_cost("paid-model", router)
            == ModelCostClass.KNOWN_POSITIVE
        )


class TestBlockUnknownCostModelsWithoutRouter:
    """A proxy that routes directly through litellm (no Router) must still
    enforce the guard via the global cost map, not silently pass the request."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    def test_no_router_blocks_unmapped_model(self):
        # No router and no cost map entry: spend can't be measured, so the
        # Denial-of-Wallet guard must fail closed instead of waving it through.
        with pytest.raises(ProxyException) as exc_info:
            _block_unknown_cost_models_check(
                enabled=True,
                model=_UNMAPPED_MODEL,
                llm_router=None,
                route="/v1/chat/completions",
            )
        assert "brand-new-unmapped-model-xyz" in exc_info.value.message

    def test_no_router_allows_cost_mapped_model(self):
        # A model whose cost litellm knows is measurable, so it must pass even
        # without a router.
        litellm.model_cost["nrouter-known-model"] = {
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
            "litellm_provider": "openai",
            "mode": "chat",
        }
        _block_unknown_cost_models_check(
            enabled=True,
            model="nrouter-known-model",
            llm_router=None,
            route="/v1/chat/completions",
        )
