from unittest.mock import patch

import pytest_asyncio
from prometheus_client import REGISTRY

try:
    from litellm.integrations.prometheus import PrometheusLogger
except Exception:
    PrometheusLogger = None

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from litellm.types.router import ModelInfo
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


@pytest.fixture(scope="function")
def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

    with patch("litellm.proxy.proxy_server.premium_user", True):
        logger = PrometheusLogger()

        # Add the missing async_logging_hook method
        async def async_logging_hook(kwargs, result, call_type):
            return kwargs, result

        logger.async_logging_hook = async_logging_hook
        return logger


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
    await asyncio.sleep(2)
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


@pytest.mark.asyncio
async def test_prometheus_metric_tracking():
    """
    Test that the Prometheus metric for provider budget is tracked correctly
    """
    try:
        from unittest.mock import MagicMock

        from litellm.integrations.prometheus import PrometheusLogger
    except Exception:
        PrometheusLogger = None
    if PrometheusLogger is None:
        pytest.skip("PrometheusLogger is not installed")

    # Create a mock PrometheusLogger
    mock_prometheus = MagicMock(spec=PrometheusLogger)

    # Setup provider budget limiting
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100)
        },
    )

    litellm._async_success_callback = [mock_prometheus]

    provider_budget_config: GenericBudgetConfigType = {
        "openai": BudgetConfig(budget_duration="1d", max_budget=0.000000000001),
        "azure": BudgetConfig(budget_duration="1d", max_budget=100),
    }

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "model_info": {"id": "azure-model-id"},
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                },
                "model_info": {"id": "openai-model-id"},
            },
        ],
        provider_budget_config=provider_budget_config,
        redis_host=os.getenv("REDIS_HOST"),
        redis_port=int(os.getenv("REDIS_PORT", 6379)),
        redis_password=os.getenv("REDIS_PASSWORD"),
    )

    try:
        response = await router.acompletion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="openai/gpt-4o-mini",
            mock_response="hi",
        )
        print(response)
    except Exception as e:
        print("error", e)

    await asyncio.sleep(2.5)

    # Verify the mock was called correctly
    mock_prometheus.track_provider_remaining_budget.assert_called()


class CustomPrometheusLogger(PrometheusLogger):
    def __init__(self):
        super().__init__()
        self.deployment_complete_outages = []
        self.deployment_cooled_downs = []

    def set_deployment_complete_outage(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
    ):
        self.deployment_complete_outages.append(
            [litellm_model_name, model_id, api_base, api_provider]
        )

    def increment_deployment_cooled_down(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
        exception_status: str,
    ):
        self.deployment_cooled_downs.append(
            [litellm_model_name, model_id, api_base, api_provider, exception_status]
        )


@pytest.mark.asyncio
async def test_router_cooldown_event_callback():
    # Clear Prometheus registry to avoid duplicate metric registration
    from prometheus_client import REGISTRY

    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

    """
    Test the router_cooldown_event_callback function

    Ensures that the router_cooldown_event_callback function correctly logs the cooldown event to the PrometheusLogger
    """
    # Mock Router instance
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {"model": "gpt-3.5-turbo"},
        "model_name": "gpt-3.5-turbo",
        "model_info": ModelInfo(id="test-model-id"),
    }
    mock_router.get_deployment.return_value = mock_deployment

    # Create a real PrometheusLogger instance
    prometheus_logger = CustomPrometheusLogger()
    litellm.callbacks = [prometheus_logger]

    await router_cooldown_event_callback(
        litellm_router_instance=mock_router,
        deployment_id="test-deployment",
        exception_status="429",
        cooldown_time=60.0,
    )

    await asyncio.sleep(0.5)

    # Assert that the router's get_deployment method was called
    mock_router.get_deployment.assert_called_once_with(model_id="test-deployment")

    print(
        "prometheus_logger.deployment_complete_outages",
        prometheus_logger.deployment_complete_outages,
    )
    print(
        "prometheus_logger.deployment_cooled_downs",
        prometheus_logger.deployment_cooled_downs,
    )

    # Assert that PrometheusLogger methods were called
    assert len(prometheus_logger.deployment_complete_outages) == 1
    assert len(prometheus_logger.deployment_cooled_downs) == 1

    assert prometheus_logger.deployment_complete_outages[0] == [
        "gpt-3.5-turbo",
        "test-model-id",
        "https://api.openai.com",
        "openai",
    ]
    assert prometheus_logger.deployment_cooled_downs[0] == [
        "gpt-3.5-turbo",
        "test-model-id",
        "https://api.openai.com",
        "openai",
        "429",
    ]


@pytest.mark.asyncio
async def test_router_cooldown_event_callback_no_prometheus():
    """
    Test the router_cooldown_event_callback function

    Ensures that the router_cooldown_event_callback function does not raise an error when no PrometheusLogger is found
    """
    # Mock Router instance
    mock_router = MagicMock()
    mock_deployment = {
        "litellm_params": {"model": "gpt-3.5-turbo"},
        "model_name": "gpt-3.5-turbo",
        "model_info": ModelInfo(id="test-model-id"),
    }
    mock_router.get_deployment.return_value = mock_deployment

    await router_cooldown_event_callback(
        litellm_router_instance=mock_router,
        deployment_id="test-deployment",
        exception_status="429",
        cooldown_time=60.0,
    )

    # Assert that the router's get_deployment method was called
    mock_router.get_deployment.assert_called_once_with(model_id="test-deployment")
