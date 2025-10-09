# What is this?
## Unit Tests for prometheus service monitoring

import json
import sys
import os
import io, asyncio

sys.path.insert(0, os.path.abspath("../.."))
import pytest
from litellm import acompletion, Cache
from litellm._service_logger import ServiceLogging
from litellm.integrations.prometheus_services import PrometheusServicesLogger
from litellm.proxy.utils import ServiceTypes
from unittest.mock import patch, AsyncMock
import litellm

"""
- Check if it receives a call when redis is used 
- Check if it fires messages accordingly
"""


@pytest.mark.asyncio
async def test_init_prometheus():
    """
    - Run completion with caching
    - Assert success callback gets called
    """

    pl = PrometheusServicesLogger(mock_testing=True)


@pytest.mark.asyncio
async def test_completion_with_caching():
    """
    - Run completion with caching
    - Assert success callback gets called
    """

    litellm.set_verbose = True
    litellm.cache = Cache(type="redis")
    litellm.service_callback = ["prometheus_system"]

    sl = ServiceLogging(mock_testing=True)
    sl.prometheusServicesLogger.mock_testing = True
    litellm.cache.cache.service_logger_obj = sl

    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response1 = await acompletion(
        model="gpt-3.5-turbo", messages=messages, caching=True
    )
    response1 = await acompletion(
        model="gpt-3.5-turbo", messages=messages, caching=True
    )

    assert sl.mock_testing_async_success_hook > 0
    assert sl.prometheusServicesLogger.mock_testing_success_calls > 0
    assert sl.mock_testing_sync_failure_hook == 0
    assert sl.mock_testing_async_failure_hook == 0


@pytest.mark.asyncio
async def test_completion_with_caching_bad_call():
    """
    - Run completion with caching (incorrect credentials)
    - Assert failure callback gets called
    """
    litellm.set_verbose = True

    try:
        from litellm.caching.caching import RedisCache

        litellm.service_callback = ["prometheus_system"]
        sl = ServiceLogging(mock_testing=True)

        RedisCache(host="hello-world", service_logger_obj=sl)
    except Exception as e:
        print(f"Receives exception = {str(e)}")

    await asyncio.sleep(5)
    assert sl.mock_testing_async_failure_hook > 0
    assert sl.mock_testing_async_success_hook == 0
    assert sl.mock_testing_sync_success_hook == 0


@pytest.mark.asyncio
async def test_router_with_caching():
    """
    - Run router with usage-based-routing-v2
    - Assert success callback gets called
    """
    try:

        def get_openai_params():
            params = {
                "model": "gpt-4.1-nano",
                "api_key": os.environ["OPENAI_API_KEY"],
            }
            return params

        model_list = [
            {
                "model_name": "azure/gpt-4",
                "litellm_params": get_openai_params(),
                "tpm": 100,
            },
            {
                "model_name": "azure/gpt-4",
                "litellm_params": get_openai_params(),
                "tpm": 1000,
            },
        ]

        router = litellm.Router(
            model_list=model_list,
            set_verbose=True,
            debug_level="DEBUG",
            routing_strategy="usage-based-routing-v2",
            redis_host=os.environ["REDIS_HOST"],
            redis_port=os.environ["REDIS_PORT"],
            redis_password=os.environ["REDIS_PASSWORD"],
        )

        litellm.service_callback = ["prometheus_system"]

        sl = ServiceLogging(mock_testing=True)
        sl.prometheusServicesLogger.mock_testing = True
        router.cache.redis_cache.service_logger_obj = sl

        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        response1 = await router.acompletion(model="azure/gpt-4", messages=messages)
        response1 = await router.acompletion(model="azure/gpt-4", messages=messages)

        assert sl.mock_testing_async_success_hook > 0
        assert sl.mock_testing_sync_failure_hook == 0
        assert sl.mock_testing_async_failure_hook == 0
        assert sl.prometheusServicesLogger.mock_testing_success_calls > 0

    except Exception as e:
        pytest.fail(f"An exception occured - {str(e)}")


@pytest.mark.asyncio
async def test_service_logger_db_monitoring():
    """
    Test prometheus monitoring for database operations
    """
    litellm.service_callback = ["prometheus_system"]
    sl = ServiceLogging()

    # Create spy on prometheus logger's async_service_success_hook
    with patch.object(
        sl.prometheusServicesLogger,
        "async_service_success_hook",
        new_callable=AsyncMock,
    ) as mock_prometheus_success:
        # Test DB success monitoring
        await sl.async_service_success_hook(
            service=ServiceTypes.DB,
            duration=0.3,
            call_type="query",
            event_metadata={"query_type": "SELECT", "table": "api_keys"},
        )

        # Assert prometheus logger's success hook was called
        mock_prometheus_success.assert_called_once()
        # Optionally verify the payload
        actual_payload = mock_prometheus_success.call_args[1]["payload"]
        print("actual_payload sent to prometheus: ", actual_payload)
        assert actual_payload.service == ServiceTypes.DB
        assert actual_payload.duration == 0.3
        assert actual_payload.call_type == "query"
        assert actual_payload.is_error is False


@pytest.mark.asyncio
async def test_service_logger_db_monitoring_failure():
    """
    Test prometheus monitoring for failed database operations
    """
    litellm.service_callback = ["prometheus_system"]
    sl = ServiceLogging()

    # Create spy on prometheus logger's async_service_failure_hook
    with patch.object(
        sl.prometheusServicesLogger,
        "async_service_failure_hook",
        new_callable=AsyncMock,
    ) as mock_prometheus_failure:
        # Test DB failure monitoring
        test_error = Exception("Database connection failed")
        await sl.async_service_failure_hook(
            service=ServiceTypes.DB,
            duration=0.3,
            error=test_error,
            call_type="query",
            event_metadata={"query_type": "SELECT", "table": "api_keys"},
        )

        # Assert prometheus logger's failure hook was called
        mock_prometheus_failure.assert_called_once()
        # Verify the payload
        actual_payload = mock_prometheus_failure.call_args[1]["payload"]
        print("actual_payload sent to prometheus: ", actual_payload)
        assert actual_payload.service == ServiceTypes.DB
        assert actual_payload.duration == 0.3
        assert actual_payload.call_type == "query"
        assert actual_payload.is_error is True
        assert actual_payload.error == "Database connection failed"


def test_get_metric_existing():
    """Test _get_metric when metric exists. _get_metric should return the metric object"""
    pl = PrometheusServicesLogger()
    # Create a metric first
    hist = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    # Test retrieving existing metric
    retrieved_metric = pl._get_metric("litellm_test_service_test_type_of_request")
    assert retrieved_metric is hist
    assert retrieved_metric is not None


def test_get_metric_non_existing():
    """Test _get_metric when metric doesn't exist, returns None"""
    pl = PrometheusServicesLogger()

    # Test retrieving non-existent metric
    non_existent = pl._get_metric("non_existent_metric")
    assert non_existent is None


def test_create_histogram_new():
    """Test creating a new histogram"""
    pl = PrometheusServicesLogger()

    # Create new histogram
    hist = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert hist is not None
    assert pl._get_metric("litellm_test_service_test_type_of_request") is hist


def test_create_histogram_existing():
    """Test creating a histogram that already exists"""
    pl = PrometheusServicesLogger()

    # Create initial histogram
    hist1 = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    # Create same histogram again
    hist2 = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert hist2 is hist1  # same object
    assert pl._get_metric("litellm_test_service_test_type_of_request") is hist1


def test_create_counter_new():
    """Test creating a new counter"""
    pl = PrometheusServicesLogger()

    # Create new counter
    counter = pl.create_counter(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert counter is not None
    assert pl._get_metric("litellm_test_service_test_type_of_request") is counter


def test_create_counter_existing():
    """Test creating a counter that already exists"""
    pl = PrometheusServicesLogger()

    # Create initial counter
    counter1 = pl.create_counter(
        service="test_service", type_of_request="test_type_of_request"
    )

    # Create same counter again
    counter2 = pl.create_counter(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert counter2 is counter1
    assert pl._get_metric("litellm_test_service_test_type_of_request") is counter1
