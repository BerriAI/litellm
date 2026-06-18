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
    _as_cost,
    _block_unknown_cost_models_check,
    _classify_deployment_cost,
    _classify_model_group_cost,
    _cost_class_cache_key,
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
            router._model_cost_class_cache.get(
                _cost_class_cache_key("ramping-model", None)
            )
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

    def test_wildcard_route_to_mapped_model_is_known_positive(self):
        """A wildcard deployment reaching a model that IS in the cost map must
        classify KNOWN_POSITIVE (and stay usable), while a genuinely unmapped
        concrete model on the same wildcard stays UNKNOWN. Guards against the
        wildcard route being classified UNKNOWN wholesale."""
        # Inject a mapped concrete model so the test doesn't depend on the live
        # cost map; setup/teardown snapshot and restore litellm.model_cost.
        litellm.model_cost["anthropic/mapped-via-wildcard-test"] = {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000004,
            "litellm_provider": "anthropic",
            "mode": "chat",
        }
        router = Router(
            model_list=[
                {
                    "model_name": "anthropic/*",
                    "litellm_params": {
                        "model": "anthropic/*",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        assert (
            _classify_model_group_cost("anthropic/mapped-via-wildcard-test", router)
            == ModelCostClass.KNOWN_POSITIVE
        )
        assert (
            _classify_model_group_cost("anthropic/brand-new-unmapped-zzz", router)
            == ModelCostClass.UNKNOWN
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

    @pytest.mark.asyncio
    async def test_free_request_model_does_not_skip_guard_for_unpriced_default(
        self, mock_proxy_logging
    ):
        """skip_budget_checks is derived from the request model upstream, but
        completion_model overrides that model in common_request_processing. A
        caller sending an explicitly-free model (so skip_budget_checks=True) must
        still hit the guard when the server default is unpriced; gating the guard
        on skip_budget_checks would let the unpriced default run untracked."""
        router = _router_with_mixed_costs()
        with pytest.raises(ProxyException) as exc_info:
            await common_checks(
                request_body={"model": "free-model"},
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={
                    "block_unknown_cost_models": True,
                    "completion_model": _UNMAPPED_MODEL,
                },
                route="/v1/chat/completions",
                llm_router=router,
                proxy_logging_obj=mock_proxy_logging,
                valid_token=UserAPIKeyAuth(token="test-token"),
                request=MagicMock(),
                skip_budget_checks=True,
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
            router._model_cost_class_cache[_cost_class_cache_key(_UNMAPPED_MODEL, None)]
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


class TestClassifyDeploymentCost:
    """Per-deployment classifier branches the group-level tests don't reach,
    including the fallback that runs when a deployment never resolves through
    ``get_deployment_model_info`` (no ``model_info.id``)."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    def test_positive_explicit_cost_without_model_id_is_known_positive(self):
        # Regression: a deployment that never reaches get_deployment_model_info
        # (no model_info.id) but carries a positive explicit price must be
        # KNOWN_POSITIVE, not EXPLICIT_ZERO. Misclassifying it free would make
        # _is_model_cost_zero return True and silently skip every budget check.
        router = MagicMock(spec=Router)
        deployment = {
            "model_name": "no-id-paid",
            "litellm_params": {
                "model": "openai/some-paid-model",
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000008,
            },
            "model_info": {},
        }
        assert (
            _classify_deployment_cost(deployment, router)
            == ModelCostClass.KNOWN_POSITIVE
        )
        router.get_deployment_model_info.assert_not_called()

    def test_zero_explicit_cost_without_model_id_is_explicit_zero(self):
        router = MagicMock(spec=Router)
        deployment = {
            "model_name": "no-id-free",
            "litellm_params": {
                "model": "openai/some-free-model",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "model_info": {},
        }
        assert (
            _classify_deployment_cost(deployment, router)
            == ModelCostClass.EXPLICIT_ZERO
        )

    def test_no_explicit_cost_and_no_resolution_is_unknown(self):
        # No model_info.id, no explicit price, nothing to resolve: fail closed
        # to UNKNOWN rather than silently treating it as free.
        router = MagicMock(spec=Router)
        deployment = {
            "model_name": "no-id-unpriced",
            "litellm_params": {"model": "openai/whatever"},
            "model_info": None,
        }
        assert _classify_deployment_cost(deployment, router) == ModelCostClass.UNKNOWN

    def test_explicit_price_short_circuits_resolution(self):
        # An explicit deployment price is authoritative, so the classifier honors
        # it without consulting (or being broken by) the cost resolver.
        router = MagicMock(spec=Router)
        router.get_deployment_model_info.side_effect = RuntimeError("boom")
        deployment = {
            "model_name": "err-priced",
            "litellm_params": {
                "model": "openai/some-paid-model",
                "input_cost_per_token": 0.000002,
                "output_cost_per_token": 0.000008,
            },
            "model_info": {"id": "err-priced-id"},
        }
        assert (
            _classify_deployment_cost(deployment, router)
            == ModelCostClass.KNOWN_POSITIVE
        )
        router.get_deployment_model_info.assert_not_called()

    def test_resolution_error_is_treated_as_unknown(self):
        # With no explicit price, a cost-resolution error must fail closed to
        # UNKNOWN (the underlying model is unmapped), never propagate or assume free.
        router = MagicMock(spec=Router)
        router.get_deployment_model_info.side_effect = RuntimeError("boom")
        deployment = {
            "model_name": "err-unpriced",
            "litellm_params": {"model": "openai/totally-unmapped-err-zzz"},
            "model_info": {"id": "err-unpriced-id"},
        }
        assert _classify_deployment_cost(deployment, router) == ModelCostClass.UNKNOWN
        router.get_deployment_model_info.assert_called_once()

    def test_explicit_cost_from_model_info_when_absent_in_litellm_params(self):
        # The explicit price lives only in model_info; the model_info fallback
        # must still classify it as explicitly free.
        router = MagicMock(spec=Router)
        router.get_deployment_model_info.return_value = None
        deployment = {
            "model_name": "mi-free",
            "litellm_params": {"model": "openai/mi-free"},
            "model_info": {
                "id": "mi-free-id",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        }
        assert (
            _classify_deployment_cost(deployment, router)
            == ModelCostClass.EXPLICIT_ZERO
        )

    def test_resolved_positive_cost_is_known_positive(self):
        router = MagicMock(spec=Router)
        router.get_deployment_model_info.return_value = {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000009,
        }
        deployment = {
            "model_name": "resolved-paid",
            "litellm_params": {"model": "openai/resolved-paid"},
            "model_info": {"id": "resolved-paid-id"},
        }
        assert (
            _classify_deployment_cost(deployment, router)
            == ModelCostClass.KNOWN_POSITIVE
        )

    def test_unresolved_deployment_falls_back_to_cost_map(self):
        # When per-deployment resolution yields nothing (wildcard/unregistered),
        # the concrete model is priced from the global cost map: mapped models
        # stay usable, unmapped ones fail closed.
        litellm.model_cost["anthropic/fallback-mapped-test"] = {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000004,
            "litellm_provider": "anthropic",
            "mode": "chat",
        }
        router = MagicMock(spec=Router)
        router.get_deployment_model_info.return_value = None
        mapped = {
            "model_name": "anthropic/fallback-mapped-test",
            "litellm_params": {"model": "anthropic/fallback-mapped-test"},
            "model_info": {"id": "wc-id"},
        }
        assert (
            _classify_deployment_cost(mapped, router) == ModelCostClass.KNOWN_POSITIVE
        )
        unmapped = {
            "model_name": "anthropic/fallback-unmapped-zzz",
            "litellm_params": {"model": "anthropic/fallback-unmapped-zzz"},
            "model_info": {"id": "wc-id2"},
        }
        assert _classify_deployment_cost(unmapped, router) == ModelCostClass.UNKNOWN

    def test_retained_wildcard_pattern_uses_concrete_model_name(self):
        # If litellm_params still holds the wildcard pattern, the concrete
        # model_name the router resolved is used for pricing.
        litellm.model_cost["anthropic/retained-pattern-mapped"] = {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000004,
            "litellm_provider": "anthropic",
            "mode": "chat",
        }
        router = MagicMock(spec=Router)
        router.get_deployment_model_info.return_value = None
        deployment = {
            "model_name": "anthropic/retained-pattern-mapped",
            "litellm_params": {"model": "anthropic/*"},
            "model_info": {"id": "wc-id3"},
        }
        assert (
            _classify_deployment_cost(deployment, router)
            == ModelCostClass.KNOWN_POSITIVE
        )


class TestAsCost:
    """``_as_cost`` coerces only real numbers, so non-numeric pricing values
    can't masquerade as a measurable cost."""

    def test_numbers_pass_through_as_float(self):
        assert _as_cost(0) == 0.0
        assert _as_cost(5) == 5.0
        assert _as_cost(0.000002) == 0.000002

    def test_non_numbers_become_none(self):
        assert _as_cost(None) is None
        assert _as_cost("0.0") is None
        assert _as_cost({"input_cost_per_token": 1}) is None

    def test_bool_is_not_a_cost(self):
        # bool subclasses int, but a misconfigured `cost: true` is not a $1 price.
        assert _as_cost(True) is None
        assert _as_cost(False) is None


class TestTeamScopedCostClassification:
    """A team can shadow a priced public model name with its own unpriced
    deployment; the guard must classify the deployment set the router resolves
    for that team, not the global one, or the team route bypasses budgets."""

    def setup_method(self):
        self._saved_model_cost = copy.deepcopy(litellm.model_cost)

    def teardown_method(self):
        litellm.model_cost = self._saved_model_cost

    def _router_with_team_shadow(self) -> Router:
        # Global "shared-model" is priced; team "teamX" exposes the same public
        # name backed by its own unmapped (unpriced) deployment.
        return Router(
            model_list=[
                {
                    "model_name": "shared-model",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "sk-fake",
                    },
                    "model_info": {"id": "global-priced"},
                },
                {
                    "model_name": "shared-model-team-teamX",
                    "litellm_params": {
                        "model": "openai/team-unmapped-zzz",
                        "api_key": "sk-fake",
                    },
                    "model_info": {
                        "id": "team-unpriced",
                        "team_id": "teamX",
                        "team_public_model_name": "shared-model",
                    },
                },
            ]
        )

    def test_team_unpriced_shadow_is_blocked(self):
        router = self._router_with_team_shadow()
        # The global view only sees the priced deployment, which is why a
        # team-unaware guard would wrongly let the team request through.
        assert (
            _classify_model_group_cost("shared-model", router, None)
            == ModelCostClass.KNOWN_POSITIVE
        )
        assert (
            _classify_model_group_cost("shared-model", router, "teamX")
            == ModelCostClass.UNKNOWN
        )
        with pytest.raises(ProxyException) as exc_info:
            _block_unknown_cost_models_check(
                enabled=True,
                model="shared-model",
                llm_router=router,
                route="/v1/chat/completions",
                team_id="teamX",
            )
        assert "shared-model" in exc_info.value.message

    def test_non_team_request_not_overblocked(self):
        router = self._router_with_team_shadow()
        # A non-team caller routes to the priced global deployment, so the guard
        # must not block it just because some team shadows the name.
        _block_unknown_cost_models_check(
            enabled=True,
            model="shared-model",
            llm_router=router,
            route="/v1/chat/completions",
            team_id=None,
        )
