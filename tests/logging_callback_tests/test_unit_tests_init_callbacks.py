import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from typing import Literal

import pytest
import litellm
import asyncio
import logging
from litellm._logging import verbose_logger
from prometheus_client import REGISTRY, CollectorRegistry

from litellm.integrations.lago import LagoLogger
from litellm.integrations.openmeter import OpenMeterLogger
from litellm.integrations.braintrust_logging import BraintrustLogger
from litellm.integrations.pagerduty.pagerduty import PagerDutyAlerting
from litellm.integrations.galileo import GalileoObserve
from litellm.integrations.langsmith import LangsmithLogger
from litellm.integrations.literal_ai import LiteralAILogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger
from litellm.integrations.opik.opik import OpikLogger
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.integrations.mlflow import MlflowLogger
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger
from litellm.integrations.humanloop import HumanloopLogger
from litellm.proxy.hooks.dynamic_rate_limiter import _PROXY_DynamicRateLimitHandler
from unittest.mock import patch

# clear prometheus collectors / registry
collectors = list(REGISTRY._collector_to_names.keys())
for collector in collectors:
    REGISTRY.unregister(collector)
######################################

callback_class_str_to_classType = {
    "lago": LagoLogger,
    "openmeter": OpenMeterLogger,
    "braintrust": BraintrustLogger,
    "galileo": GalileoObserve,
    "langsmith": LangsmithLogger,
    "literalai": LiteralAILogger,
    "prometheus": PrometheusLogger,
    "datadog": DataDogLogger,
    "datadog_llm_observability": DataDogLLMObsLogger,
    "gcs_bucket": GCSBucketLogger,
    "opik": OpikLogger,
    "argilla": ArgillaLogger,
    "opentelemetry": OpenTelemetry,
    "azure_storage": AzureBlobStorageLogger,
    "humanloop": HumanloopLogger,
    # OTEL compatible loggers
    "logfire": OpenTelemetry,
    "arize": OpenTelemetry,
    "langtrace": OpenTelemetry,
    "mlflow": MlflowLogger,
    "langfuse": LangfusePromptManagement,
    "otel": OpenTelemetry,
    "pagerduty": PagerDutyAlerting,
    "gcs_pubsub": GcsPubSubLogger,
}

expected_env_vars = {
    "LAGO_API_KEY": "api_key",
    "LAGO_API_BASE": "mock_base",
    "LAGO_API_EVENT_CODE": "mock_event_code",
    "OPENMETER_API_KEY": "openmeter_api_key",
    "BRAINTRUST_API_KEY": "braintrust_api_key",
    "GALILEO_API_KEY": "galileo_api_key",
    "LITERAL_API_KEY": "literal_api_key",
    "DD_API_KEY": "datadog_api_key",
    "DD_SITE": "datadog_site",
    "GOOGLE_APPLICATION_CREDENTIALS": "gcs_credentials",
    "OPIK_API_KEY": "opik_api_key",
    "LANGTRACE_API_KEY": "langtrace_api_key",
    "LOGFIRE_TOKEN": "logfire_token",
    "ARIZE_SPACE_KEY": "arize_space_key",
    "ARIZE_API_KEY": "arize_api_key",
    "ARGILLA_API_KEY": "argilla_api_key",
    "PAGERDUTY_API_KEY": "pagerduty_api_key",
    "GCS_PUBSUB_TOPIC_ID": "gcs_pubsub_topic_id",
    "GCS_PUBSUB_PROJECT_ID": "gcs_pubsub_project_id",
}


def reset_all_callbacks():
    litellm.callbacks = []
    litellm.input_callback = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []


initial_env_vars = {}


def init_env_vars():
    for env_var, value in expected_env_vars.items():
        if env_var not in os.environ:
            os.environ[env_var] = value
        else:
            initial_env_vars[env_var] = os.environ[env_var]


def reset_env_vars():
    for env_var, value in initial_env_vars.items():
        os.environ[env_var] = value


all_callback_required_env_vars = []


async def use_callback_in_llm_call(
    callback: str, used_in: Literal["callbacks", "success_callback"]
):
    if callback == "dynamic_rate_limiter":
        # internal CustomLogger class that expects internal_usage_cache passed to it, it always fails when tested in this way
        return
    elif callback == "argilla":
        litellm.argilla_transformation_object = {}
    elif callback == "openmeter":
        # it's currently handled in jank way, TODO: fix openmete and then actually run it's test
        return
    elif callback == "prometheus":
        # pytest teardown - clear existing prometheus collectors
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)

    # Mock the httpx call for Argilla dataset retrieval
    if callback == "argilla":
        import httpx

        mock_response = httpx.Response(
            status_code=200, json={"items": [{"id": "mocked_dataset_id"}]}
        )
        patch.object(
            litellm.module_level_client, "get", return_value=mock_response
        ).start()

    # Mock the httpx call for Argilla dataset retrieval
    if callback == "argilla":
        import httpx

        mock_response = httpx.Response(
            status_code=200, json={"items": [{"id": "mocked_dataset_id"}]}
        )
        patch.object(
            litellm.module_level_client, "get", return_value=mock_response
        ).start()

    if used_in == "callbacks":
        litellm.callbacks = [callback]
    elif used_in == "success_callback":
        litellm.success_callback = [callback]

    for _ in range(5):
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            mock_response="hello",
        )

        await asyncio.sleep(0.5)

        expected_class = callback_class_str_to_classType[callback]

        if used_in == "callbacks":
            assert isinstance(litellm._async_success_callback[0], expected_class)
            assert isinstance(litellm._async_failure_callback[0], expected_class)
            assert isinstance(litellm.success_callback[0], expected_class)
            assert isinstance(litellm.failure_callback[0], expected_class)

            assert (
                len(litellm._async_success_callback) == 1
            ), f"Got={litellm._async_success_callback}"
            assert len(litellm._async_failure_callback) == 1
            assert len(litellm.success_callback) == 1
            assert len(litellm.failure_callback) == 1
            assert len(litellm.callbacks) == 1
        elif used_in == "success_callback":
            print(f"litellm.success_callback: {litellm.success_callback}")
            print(f"litellm._async_success_callback: {litellm._async_success_callback}")
            assert isinstance(litellm.success_callback[0], expected_class)
            assert len(litellm.success_callback) == 1  # ["lago", LagoLogger]
            assert isinstance(litellm._async_success_callback[0], expected_class)
            assert len(litellm._async_success_callback) == 1

            # TODO also assert that it's not set for failure_callback
            # As of Oct 21 2024, it's currently set
            # 1st hoping to add test coverage for just setting in success_callback/_async_success_callback

        if callback == "argilla":
            patch.stopall()

        if callback == "argilla":
            patch.stopall()


@pytest.mark.asyncio
async def test_init_custom_logger_compatible_class_as_callback():
    init_env_vars()

    # used like litellm.callbacks = ["prometheus"]
    for callback in litellm._known_custom_logger_compatible_callbacks:
        print(f"Testing callback: {callback}")
        reset_all_callbacks()

        await use_callback_in_llm_call(callback, used_in="callbacks")

    # used like this litellm.success_callback = ["prometheus"]
    for callback in litellm._known_custom_logger_compatible_callbacks:
        print(f"Testing callback: {callback}")
        reset_all_callbacks()

        await use_callback_in_llm_call(callback, used_in="success_callback")

    reset_env_vars()


def test_dynamic_logging_global_callback():
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.types.utils import ModelResponse, Choices, Message, Usage

    cl = CustomLogger()

    litellm_logging = LiteLLMLoggingObj(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="123",
        function_id="456",
        kwargs={
            "langfuse_public_key": "my-mock-public-key",
            "langfuse_secret_key": "my-mock-secret-key",
        },
        dynamic_success_callbacks=["langfuse"],
    )

    with patch.object(cl, "log_success_event") as mock_log_success_event:
        cl.log_success_event = mock_log_success_event
        litellm.success_callback = [cl]

        try:
            litellm_logging.success_handler(
                result=ModelResponse(
                    id="chatcmpl-5418737b-ab14-420b-b9c5-b278b6681b70",
                    created=1732306261,
                    model="claude-3-opus-20240229",
                    object="chat.completion",
                    system_fingerprint=None,
                    choices=[
                        Choices(
                            finish_reason="stop",
                            index=0,
                            message=Message(
                                content="hello",
                                role="assistant",
                                tool_calls=None,
                                function_call=None,
                            ),
                        )
                    ],
                    usage=Usage(
                        completion_tokens=20,
                        prompt_tokens=10,
                        total_tokens=30,
                        completion_tokens_details=None,
                        prompt_tokens_details=None,
                    ),
                ),
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_log_success_event.assert_called_once()


def test_get_combined_callback_list():
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    _logging = LiteLLMLoggingObj(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="123",
        function_id="456",
    )

    assert "langfuse" in _logging.get_combined_callback_list(
        dynamic_success_callbacks=["langfuse"], global_callbacks=["lago"]
    )
    assert "lago" in _logging.get_combined_callback_list(
        dynamic_success_callbacks=["langfuse"], global_callbacks=["lago"]
    )
