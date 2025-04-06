import asyncio

import pytest
from unittest.mock import patch
import pytest_asyncio

from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.utils import BudgetConfig, GenericBudgetConfigType


def compare_metrics(func):
    def get_metrics():
        metrics = {}
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                metrics[sample.name] = sample.value
        return metrics

    async def wrapper(*args, **kwargs):
        initial_metrics = get_metrics()
        await func(*args, **kwargs)
        await asyncio.sleep(2)
        updated_metrics = get_metrics()

        return {
            metric: updated_metrics.get(metric, 0) - initial_metrics.get(metric, 0)
            for metric in set(initial_metrics) | set(updated_metrics)
        }

    return wrapper


@pytest_asyncio.fixture(scope="function")
async def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

    with patch("litellm.proxy.proxy_server.premium_user", True):
        yield PrometheusLogger()


@pytest.mark.asyncio
async def test_async_prometheus_success_logging_with_callbacks(prometheus_logger):
    litellm.callbacks = [prometheus_logger]

    @compare_metrics
    async def op():
        await litellm.acompletion(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            mock_response="hi",
            temperature=0.2,
        )

    diff = await op()
    assert diff["litellm_requests_metric_total"] == 1.0


@pytest.mark.asyncio
async def test_async_prometheus_budget_logging_with_callbacks(prometheus_logger):
    litellm.callbacks = [prometheus_logger]

    @compare_metrics
    async def op():
        provider_budget_config: GenericBudgetConfigType = {
            "openai": BudgetConfig(time_period="1d", budget_limit=50),
        }

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "openai/gpt-3.5-turbo",
                        "api_key": "mock-key",
                    },
                }
            ],
            provider_budget_config=provider_budget_config,
        )

        await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "llm?"}],
            mock_response="openai",
            metadata={
                "user_api_key_team_id": "team-1",
                "user_api_key_team_alias": "test-team",
                "user_api_key": "test-key",
                "user_api_key_alias": "test-key-alias",
            },
        )

    diff = await op()

    # TODO: Should implement `litellm_provider_remaining_budget_metric` in prometheus.py
    assert diff.get("litellm_provider_remaining_budget_metric", 50.0) == 50.0
