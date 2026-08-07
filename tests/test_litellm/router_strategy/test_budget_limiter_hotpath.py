import asyncio

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import BudgetConfig


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_does_not_require_litellm_params_instantiation(
    disable_budget_sync, monkeypatch
):
    class RaiseOnInit:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "LiteLLM_Params should not be instantiated in hot path"
            )

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.LiteLLM_Params",
        RaiseOnInit,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {"litellm_params": {"model": "openai/gpt-4o-mini"}}
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_view_supports_mapping_and_attr_access(
    disable_budget_sync, monkeypatch
):
    observed = {}

    def _future_style_get_llm_provider(
        model,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
        litellm_params=None,
    ):
        assert litellm_params is not None
        observed["model_attr"] = litellm_params.model
        observed["provider_get"] = litellm_params.get("custom_llm_provider")
        observed["api_base_item"] = litellm_params["api_base"]
        observed["has_api_key"] = "api_key" in litellm_params
        observed["model_dump"] = litellm_params.model_dump()
        return model, "openai", None, None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.litellm.get_llm_provider",
        _future_style_get_llm_provider,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "custom_llm_provider": "openai",
            "api_base": "https://api.openai.com/v1",
        }
    }
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"
    assert observed["model_attr"] == "openai/gpt-4o-mini"
    assert observed["provider_get"] == "openai"
    assert observed["api_base_item"] == "https://api.openai.com/v1"
    assert observed["has_api_key"] is False
    assert observed["model_dump"]["model"] == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_filter_deployments_resolves_provider_once_per_deployment(
    disable_budget_sync, monkeypatch
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    provider_resolution_calls = 0

    def _count_provider_calls(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return "openai"

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _count_provider_calls,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


@pytest.mark.asyncio
async def test_async_filter_deployments_does_not_recompute_provider_when_resolved_none(
    disable_budget_sync, monkeypatch
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "max_budget": 100.0,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "deployment-1"},
            }
        ],
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "unknown-provider/model"},
            "model_info": {"id": "deployment-1"},
        }
    ]

    provider_resolution_calls = 0

    def _provider_returns_none(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return None

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _provider_returns_none,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


class _OrderRecordingRedisCache:
    """
    Fake redis cache that records the order of increment-pipeline writes relative
    to read-backs, so we can prove the sync flushes queued spend before reading.
    """

    def __init__(self):
        self.events: list[str] = []
        self.batch_get_seen_values: list[float] = []
        self._store: dict[str, float] = {}

    async def async_increment_pipeline(self, increment_list, **kwargs):
        self.events.append("pipeline_start")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        results = []
        for op in increment_list:
            self._store[op["key"]] = self._store.get(op["key"], 0.0) + op["increment_value"]
            results.append(self._store[op["key"]])
        self.events.append("pipeline_end")
        return results

    async def async_batch_get_cache(self, key_list, **kwargs):
        self.events.append("batch_get")
        values = {key: self._store.get(key, 0.0) for key in key_list}
        self.batch_get_seen_values = list(values.values())
        return values

    async def async_get_cache(self, key, **kwargs):
        return self._store.get(key, 0.0)

    async def async_set_cache(self, key, value, **kwargs):
        self._store[key] = value


@pytest.mark.asyncio
async def test_sync_flushes_increments_to_redis_before_reading_back(disable_budget_sync):
    """
    Regression for router budget sync race (#32614): _push_in_memory_increments_to_redis
    must await the pipeline write instead of firing a background task. Otherwise the
    subsequent read-back sees stale Redis spend and overwrites fresher in-memory spend.
    """
    dual_cache = DualCache()
    redis = _OrderRecordingRedisCache()
    dual_cache.redis_cache = redis

    budget_limiter = RouterBudgetLimiting(
        dual_cache=dual_cache,
        provider_budget_config={"openai": BudgetConfig(budget_duration="1d", max_budget=100.0)},
    )
    spend_key = "provider_spend:openai:1d"
    budget_limiter.redis_increment_operation_queue = [
        RedisPipelineIncrementOperation(key=spend_key, increment_value=7.0, ttl=3600),
    ]

    await budget_limiter._sync_in_memory_spend_with_redis()

    assert redis.events == ["pipeline_start", "pipeline_end", "batch_get"]
    assert redis.batch_get_seen_values == [7.0]
    assert budget_limiter.redis_increment_operation_queue == []


def _legacy_provider_resolution(deployment):
    """
    Reference implementation used before hot-path optimization.
    """
    try:
        _litellm_params = LiteLLM_Params(
            **deployment.get("litellm_params", {"model": ""})
        )
        _, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=_litellm_params.model,
            litellm_params=_litellm_params,
        )
    except Exception:
        return None
    return custom_llm_provider


@pytest.mark.parametrize(
    "deployment",
    [
        {"litellm_params": {"model": "openai/gpt-4o-mini"}},
        {"litellm_params": {"model": "gpt-4o-mini", "custom_llm_provider": "openai"}},
        {"litellm_params": {"model": "unknown-provider/model"}},
    ],
)
@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_matches_legacy_behavior(
    disable_budget_sync, deployment
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    current_provider = provider_budget._get_llm_provider_for_deployment(deployment)
    legacy_provider = _legacy_provider_resolution(deployment)

    assert current_provider == legacy_provider


def test_register_deployment_budget_for_runtime_added_deployment(
    disable_budget_sync, monkeypatch
):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )
    model_id = "dynamic-deployment-id"
    budget_limiter.register_deployment_budget(
        deployment={
            "model_name": "dynamic-budget-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "max_budget": 0.000000000001,
                "budget_duration": "1d",
            },
            "model_info": {"id": model_id},
        }
    )

    config = budget_limiter._get_budget_config_for_deployment(model_id)
    assert config is not None
    assert config.max_budget == 0.000000000001
    assert config.budget_duration == "1d"

    budget_limiter.unregister_deployment_budget(model_id=model_id)
    assert budget_limiter._get_budget_config_for_deployment(model_id) is None


def test_router_add_deployment_registers_deployment_budget(
    disable_budget_sync, monkeypatch
):
    import asyncio

    from litellm import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)

    router = Router(
        model_list=[],
        optional_pre_call_checks=[],
    )

    router.add_deployment(
        deployment=Deployment(
            model_name="dynamic-budget-model",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4o-mini",
                api_key="fake-key",
                max_budget=0.000000000001,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="runtime-budget-deployment"),
        )
    )

    budget_limiter = router._get_router_deployment_budget_limiter()
    assert budget_limiter is not None
    config = budget_limiter._get_budget_config_for_deployment(
        "runtime-budget-deployment"
    )
    assert config is not None
    assert config.max_budget == 0.000000000001
