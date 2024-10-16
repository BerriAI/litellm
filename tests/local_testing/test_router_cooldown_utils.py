import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params, ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
from unittest.mock import AsyncMock, MagicMock
from litellm.integrations.prometheus import PrometheusLogger
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback


load_dotenv()


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


async def test_router_cooldown_event_callback_no_deployment():
    """
    Test the router_cooldown_event_callback function

    Ensures that the router_cooldown_event_callback function does not raise an error when no deployment is found

    In this scenario it should do nothing
    """
    # Mock Router instance
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    await router_cooldown_event_callback(
        litellm_router_instance=mock_router,
        deployment_id="test-deployment",
        exception_status="429",
        cooldown_time=60.0,
    )

    # Assert that the router's get_deployment method was called
    mock_router.get_deployment.assert_called_once_with(model_id="test-deployment")
