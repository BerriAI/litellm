import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import time

from litellm.constants import SENTRY_DENYLIST, SENTRY_PII_DENYLIST
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.litellm_core_utils.litellm_logging import set_callbacks
from litellm.types.utils import ModelResponse


@pytest.fixture
def logging_obj():
    return LitellmLogging(
        model="bedrock/claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    )


def test_get_masked_api_base(logging_obj):
    api_base = "https://api.openai.com/v1"
    masked_api_base = logging_obj._get_masked_api_base(api_base)
    assert masked_api_base == "https://api.openai.com/v1"
    assert type(masked_api_base) == str


def test_sentry_sample_rate():
    existing_sample_rate = os.getenv("SENTRY_API_SAMPLE_RATE")
    try:
        # test with default value by removing the environment variable
        if existing_sample_rate:
            del os.environ["SENTRY_API_SAMPLE_RATE"]

        set_callbacks(["sentry"])
        # Check if the default sample rate is set to 1.0
        assert os.environ.get("SENTRY_API_SAMPLE_RATE") == "1.0"

        # test with custom value
        os.environ["SENTRY_API_SAMPLE_RATE"] = "0.5"

        set_callbacks(["sentry"])
        # Check if the custom sample rate is set correctly
        assert os.environ.get("SENTRY_API_SAMPLE_RATE") == "0.5"
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Restore the original environment variable
        if existing_sample_rate:
            os.environ["SENTRY_API_SAMPLE_RATE"] = existing_sample_rate
        else:
            if "SENTRY_API_SAMPLE_RATE" in os.environ:
                del os.environ["SENTRY_API_SAMPLE_RATE"]


def test_sentry_environment():
    """Test that SENTRY_ENVIRONMENT is properly handled during Sentry initialization"""
    existing_environment = os.getenv("SENTRY_ENVIRONMENT")
    existing_dsn = os.getenv("SENTRY_DSN")

    # Create mock sentry_sdk module
    mock_event_scrubber_instance = MagicMock()
    mock_event_scrubber_cls = MagicMock(return_value=mock_event_scrubber_instance)

    mock_scrubber_module = MagicMock()
    mock_scrubber_module.EventScrubber = mock_event_scrubber_cls

    mock_sentry_sdk = MagicMock()
    mock_sentry_sdk.scrubber = mock_scrubber_module
    mock_init = MagicMock()
    mock_sentry_sdk.init = mock_init

    # Inject mocks into sys.modules
    sys.modules["sentry_sdk"] = mock_sentry_sdk
    sys.modules["sentry_sdk.scrubber"] = mock_scrubber_module

    try:
        # Set a mock DSN to allow Sentry initialization
        os.environ["SENTRY_DSN"] = "https://test@sentry.io/123456"

        # Test with default value (no environment set)
        if existing_environment:
            del os.environ["SENTRY_ENVIRONMENT"]

        mock_init.reset_mock()
        set_callbacks(["sentry"])
        # Check that init was called with default environment "production"
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["environment"] == "production"

        # Test with custom environment value
        os.environ["SENTRY_ENVIRONMENT"] = "development"

        mock_init.reset_mock()
        set_callbacks(["sentry"])
        # Check that init was called with custom environment "development"
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["environment"] == "development"

        # Test with staging environment
        os.environ["SENTRY_ENVIRONMENT"] = "staging"

        mock_init.reset_mock()
        set_callbacks(["sentry"])
        # Check that init was called with custom environment "staging"
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["environment"] == "staging"

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Restore the original environment variables
        if existing_environment:
            os.environ["SENTRY_ENVIRONMENT"] = existing_environment
        else:
            if "SENTRY_ENVIRONMENT" in os.environ:
                del os.environ["SENTRY_ENVIRONMENT"]

        if existing_dsn:
            os.environ["SENTRY_DSN"] = existing_dsn
        else:
            if "SENTRY_DSN" in os.environ:
                del os.environ["SENTRY_DSN"]


def test_use_custom_pricing_for_model():
    from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model

    litellm_params = {
        "custom_llm_provider": "azure",
        "input_cost_per_pixel": 10,
    }
    assert use_custom_pricing_for_model(litellm_params) == True


def test_logging_prevent_double_logging(logging_obj):
    """
    When using a bridge, log only once from the underlying bridge call.
    This is to avoid double logging.
    """
    logging_obj.stream = False
    logging_obj.has_run_logging(event_type="sync_success")
    assert logging_obj.should_run_logging(event_type="sync_success") == False
    assert logging_obj.should_run_logging(event_type="sync_failure") == True
    assert logging_obj.should_run_logging(event_type="async_success") == True
    assert logging_obj.should_run_logging(event_type="async_failure") == True


@pytest.mark.asyncio
async def test_datadog_logger_not_shadowed_by_llm_obs(monkeypatch):
    """Ensure DataDog logger instantiates even when LLM Obs logger already cached."""

    # Ensure required env vars exist for Datadog loggers
    monkeypatch.setenv("DD_API_KEY", "test")
    monkeypatch.setenv("DD_SITE", "us5.datadoghq.com")

    from litellm.litellm_core_utils import litellm_logging as logging_module
    from litellm.integrations.datadog.datadog import DataDogLogger
    from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger

    logging_module._in_memory_loggers.clear()

    try:
        # Cache an LLM Obs logger first to mirror callbacks=["datadog_llm_observability", ...]
        obs_logger = DataDogLLMObsLogger()
        logging_module._in_memory_loggers.append(obs_logger)

        datadog_logger = logging_module._init_custom_logger_compatible_class(
            logging_integration="datadog",
            internal_usage_cache=None,
            llm_router=None,
            custom_logger_init_args={},
        )

        # Regression check: we expect a distinct DataDogLogger, not the LLM Obs logger
        assert type(datadog_logger) is DataDogLogger
        assert any(isinstance(cb, DataDogLLMObsLogger) for cb in logging_module._in_memory_loggers)
        assert any(type(cb) is DataDogLogger for cb in logging_module._in_memory_loggers)
    finally:
        logging_module._in_memory_loggers.clear()


@pytest.mark.asyncio
async def test_logfire_logger_accepts_env_vars_for_base_url(monkeypatch):
    """Ensure Logfire logger uses LOGFIRE_BASE_URL to build the OTLP HTTP endpoint (/v1/traces)."""

    # Required env vars for Logfire integration
    monkeypatch.setenv("LOGFIRE_TOKEN", "test-token")
    monkeypatch.setenv("LOGFIRE_BASE_URL", "https://logfire-api-custom.pydantic.dev")  # no trailing slash on purpose

    # Import after env vars are set (important if module-level caching exists)
    from litellm.litellm_core_utils import litellm_logging as logging_module
    from litellm.integrations.opentelemetry import OpenTelemetry  # logger class

    logging_module._in_memory_loggers.clear()

    try:
        # Instantiate via the same mechanism LiteLLM uses for callbacks=["logfire"]
        logger = logging_module._init_custom_logger_compatible_class(
            logging_integration="logfire",
            internal_usage_cache=None,
            llm_router=None,
            custom_logger_init_args={},
        )

        # Sanity: we got the right logger type and it is cached
        assert type(logger) is OpenTelemetry
        assert any(type(cb) is OpenTelemetry for cb in logging_module._in_memory_loggers)

        # Core regression check: base URL env var should influence the exporter endpoint.
        #
        # OpenTelemetry integration has historically stored config on the instance.
        # We defensively check a few common attribute names to avoid brittle coupling.
        cfg = (
            getattr(logger, "otel_config", None)
            or getattr(logger, "config", None)
            or getattr(logger, "_otel_config", None)
        )
        assert cfg is not None, "Expected OpenTelemetry logger to keep an otel config on the instance"

        endpoint = getattr(cfg, "endpoint", None) or getattr(cfg, "otlp_endpoint", None)
        assert endpoint is not None, "Expected otel config to expose the OTLP endpoint"

        assert endpoint == "https://logfire-api-custom.pydantic.dev/v1/traces"

    finally:
        logging_module._in_memory_loggers.clear()


@pytest.mark.asyncio
async def test_logging_result_for_bridge_calls(logging_obj):
    """
    When using a bridge, log only once from the underlying bridge call.
    This is to avoid double logging.
    """
    import asyncio

    import litellm

    with patch.object(
        litellm.litellm_core_utils.litellm_logging,
        "get_standard_logging_object_payload",
    ) as mock_should_run_logging:
        await litellm.anthropic_messages(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hey"}],
            model="openai/codex-mini-latest",
            mock_response="Hello, world!",
        )
        await asyncio.sleep(1)
        assert mock_should_run_logging.call_count == 2  # called twice per call


@pytest.mark.asyncio
async def test_logging_non_streaming_request():
    import asyncio

    from litellm.integrations.custom_logger import CustomLogger

    class MockPrometheusLogger(CustomLogger):
        pass

    import litellm

    # Save original callbacks to restore after test
    original_callbacks = getattr(litellm, "callbacks", [])

    try:
        mock_logging_obj = MockPrometheusLogger()

        litellm.callbacks = [mock_logging_obj]

        with patch.object(
            mock_logging_obj,
            "async_log_success_event",
        ) as mock_async_log_success_event:
            await litellm.acompletion(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hey"}],
                model="openai/codex-mini-latest",
                mock_response="Hello, world!",
            )
            await asyncio.sleep(1)
            
            # Filter calls to only count the one with the expected input message "Hey"
            # Bridge models may make internal calls that also log, so we filter by the actual input
            calls_with_expected_input = []
            for call in mock_async_log_success_event.call_args_list:
                messages = call.kwargs.get("kwargs", {}).get("messages", [])
                if messages and len(messages) > 0:
                    first_message_content = messages[0].get("content")
                    if first_message_content == "Hey":
                        calls_with_expected_input.append(call)
            
            # Assert that we have exactly one call with the expected input
            assert len(calls_with_expected_input) == 1, (
                f"Expected 1 call with input 'Hey', but got {len(calls_with_expected_input)}. "
                f"Total calls: {mock_async_log_success_event.call_count}"
            )
            
            # Use the filtered call for assertions
            call_args = calls_with_expected_input[0]
            standard_logging_object = call_args.kwargs["kwargs"][
                "standard_logging_object"
            ]
            assert standard_logging_object["stream"] is not True
    finally:
        # Restore original callbacks to ensure test isolation
        litellm.callbacks = original_callbacks


@pytest.mark.parametrize("async_flag", ["acompletion", "aresponses"])
def test_success_handler_skips_sync_callbacks_for_async_requests(logging_obj, async_flag):
    """Ensure sync success callbacks are skipped when async call type flags are set."""
    from litellm.integrations.custom_logger import CustomLogger

    class DummyLogger(CustomLogger):
        pass

    logging_obj.stream = False  # simulate non-streaming request where sync callbacks would normally run
    logging_obj.model_call_details["litellm_params"] = {async_flag: True}
    logging_obj.litellm_params = logging_obj.model_call_details["litellm_params"]

    dummy_logger = DummyLogger()
    dummy_logger.log_success_event = MagicMock()
    dummy_logger.log_stream_event = MagicMock()

    model_response = ModelResponse(
        id="resp-123",
        model="gpt-4o-mini",
        choices=[
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    with patch.object(
        logging_obj,
        "get_combined_callback_list",
        return_value=[dummy_logger],
    ):
        logging_obj.success_handler(result=model_response)

    dummy_logger.log_success_event.assert_not_called()
    dummy_logger.log_stream_event.assert_not_called()


@pytest.mark.parametrize("call_type", ["completion", "responses"])
def test_success_handler_runs_sync_callbacks_for_sync_requests(logging_obj, call_type):
    """Ensure sync success callbacks execute when call type is sync (completion/responses)."""
    from litellm.integrations.custom_logger import CustomLogger

    class DummyLogger(CustomLogger):
        pass

    logging_obj.stream = False
    logging_obj.call_type = call_type
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.litellm_params = {}

    dummy_logger = DummyLogger()
    dummy_logger.log_success_event = MagicMock()
    dummy_logger.log_stream_event = MagicMock()

    model_response = ModelResponse(
        id="resp-123",
        model="gpt-4o-mini",
        choices=[
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    with patch.object(
        logging_obj,
        "get_combined_callback_list",
        return_value=[dummy_logger],
    ):
        logging_obj.success_handler(result=model_response)

    dummy_logger.log_success_event.assert_called_once()
    dummy_logger.log_stream_event.assert_not_called()


def test_get_user_agent_tags():
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    tags = StandardLoggingPayloadSetup._get_user_agent_tags(
        proxy_server_request={
            "headers": {
                "user-agent": "litellm/0.1.0",
            }
        }
    )

    assert "User-Agent: litellm" in tags
    assert "User-Agent: litellm/0.1.0" in tags


def test_get_request_tags():
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={"metadata": {"tags": ["test-tag"]}},
        proxy_server_request={
            "headers": {
                "user-agent": "litellm/0.1.0",
            }
        },
    )

    assert "test-tag" in tags
    assert "User-Agent: litellm" in tags
    assert "User-Agent: litellm/0.1.0" in tags


def test_get_request_tags_from_metadata_and_litellm_metadata():
    """
    Test that _get_request_tags correctly picks tags from both 'metadata' and 'litellm_metadata'.

    Scenarios tested:
    1. Tags in metadata only
    2. Tags in litellm_metadata only
    3. Tags in both (metadata should take priority)
    4. No tags in either
    5. None values for metadata/litellm_metadata
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Test case 1: Tags in metadata only
    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={"metadata": {"tags": ["metadata-tag-1", "metadata-tag-2"]}},
        proxy_server_request={},
    )
    assert "metadata-tag-1" in tags
    assert "metadata-tag-2" in tags
    assert len([t for t in tags if not t.startswith("User-Agent:")]) == 2

    # Test case 2: Tags in litellm_metadata only
    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={
            "litellm_metadata": {
                "tags": ["litellm-metadata-tag-1", "litellm-metadata-tag-2"]
            }
        },
        proxy_server_request={},
    )
    assert "litellm-metadata-tag-1" in tags
    assert "litellm-metadata-tag-2" in tags
    assert len([t for t in tags if not t.startswith("User-Agent:")]) == 2

    # Test case 3: Tags in both - metadata should take priority
    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={
            "metadata": {"tags": ["metadata-tag"]},
            "litellm_metadata": {"tags": ["litellm-metadata-tag"]},
        },
        proxy_server_request={},
    )
    assert "metadata-tag" in tags
    assert "litellm-metadata-tag" not in tags
    assert len([t for t in tags if not t.startswith("User-Agent:")]) == 1

    # Test case 4: No tags in either
    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={"metadata": {}, "litellm_metadata": {}},
        proxy_server_request={},
    )
    assert len([t for t in tags if not t.startswith("User-Agent:")]) == 0

    # Test case 5: None values for metadata/litellm_metadata
    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={"metadata": None, "litellm_metadata": None},
        proxy_server_request={},
    )
    assert isinstance(tags, list)
    assert len([t for t in tags if not t.startswith("User-Agent:")]) == 0

    # Test case 6: Empty litellm_params
    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={},
        proxy_server_request={},
    )
    assert isinstance(tags, list)
    assert len([t for t in tags if not t.startswith("User-Agent:")]) == 0

    # Test case 7: Metadata tags combined with user-agent tags
    tags = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params={"metadata": {"tags": ["custom-tag"]}},
        proxy_server_request={
            "headers": {
                "user-agent": "litellm/1.0.0",
            }
        },
    )
    assert "custom-tag" in tags
    assert "User-Agent: litellm" in tags
    assert "User-Agent: litellm/1.0.0" in tags


def test_get_request_tags_does_not_mutate_original_tags():
    """
    Test that _get_request_tags does not mutate the original tags list in metadata.

    This is a regression test for a bug where calling _get_request_tags multiple times
    would cause User-Agent tags to be duplicated because the function was mutating
    the original tags list instead of creating a copy.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create metadata with original tags
    original_tags = ["custom-tag-1", "custom-tag-2"]
    metadata = {"tags": original_tags}
    litellm_params = {"metadata": metadata}
    proxy_server_request = {
        "headers": {
            "user-agent": "AsyncOpenAI/Python 1.99.9",
        }
    }

    # Call _get_request_tags multiple times (simulating multiple callbacks)
    tags1 = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params=litellm_params,
        proxy_server_request=proxy_server_request,
    )
    tags2 = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params=litellm_params,
        proxy_server_request=proxy_server_request,
    )
    tags3 = StandardLoggingPayloadSetup._get_request_tags(
        litellm_params=litellm_params,
        proxy_server_request=proxy_server_request,
    )

    # Verify the original tags list was NOT mutated
    assert original_tags == ["custom-tag-1", "custom-tag-2"], (
        f"Original tags list was mutated: {original_tags}"
    )
    assert metadata["tags"] == ["custom-tag-1", "custom-tag-2"], (
        f"metadata['tags'] was mutated: {metadata['tags']}"
    )

    # Verify each returned list has exactly 2 User-Agent tags (not duplicated)
    user_agent_count_1 = len([t for t in tags1 if t.startswith("User-Agent:")])
    user_agent_count_2 = len([t for t in tags2 if t.startswith("User-Agent:")])
    user_agent_count_3 = len([t for t in tags3 if t.startswith("User-Agent:")])

    assert user_agent_count_1 == 2, f"Expected 2 User-Agent tags, got {user_agent_count_1}"
    assert user_agent_count_2 == 2, f"Expected 2 User-Agent tags, got {user_agent_count_2}"
    assert user_agent_count_3 == 2, f"Expected 2 User-Agent tags, got {user_agent_count_3}"

    # Verify all returned lists are independent (different objects)
    assert tags1 is not tags2
    assert tags2 is not tags3
    assert tags1 is not original_tags


def test_get_extra_header_tags():
    """Test the _get_extra_header_tags method with various scenarios."""
    import litellm
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Store original value to restore later
    original_extra_headers = getattr(litellm, "extra_spend_tag_headers", None)

    try:
        # Test case 1: No extra headers configured
        litellm.extra_spend_tag_headers = None
        result = StandardLoggingPayloadSetup._get_extra_header_tags(
            proxy_server_request={"headers": {"x-custom": "value"}}
        )
        assert result is None

        # Test case 2: Empty extra headers list
        litellm.extra_spend_tag_headers = []
        result = StandardLoggingPayloadSetup._get_extra_header_tags(
            proxy_server_request={"headers": {"x-custom": "value"}}
        )
        assert result is None

        # Test case 3: Extra headers configured but request has no headers dict
        litellm.extra_spend_tag_headers = ["x-custom", "x-tenant"]
        result = StandardLoggingPayloadSetup._get_extra_header_tags(
            proxy_server_request={"headers": "not-a-dict"}
        )
        assert result is None

        # Test case 4: Extra headers configured but none match request headers
        litellm.extra_spend_tag_headers = ["x-custom", "x-tenant"]
        result = StandardLoggingPayloadSetup._get_extra_header_tags(
            proxy_server_request={
                "headers": {
                    "content-type": "application/json",
                    "authorization": "Bearer token",
                }
            }
        )
        assert result is None

        # Test case 5: Some extra headers match request headers
        litellm.extra_spend_tag_headers = ["x-custom", "x-tenant", "x-missing"]
        result = StandardLoggingPayloadSetup._get_extra_header_tags(
            proxy_server_request={
                "headers": {
                    "x-custom": "my-custom-value",
                    "x-tenant": "tenant-123",
                    "content-type": "application/json",
                }
            }
        )
        assert result is not None
        assert len(result) == 2
        assert "x-custom: my-custom-value" in result
        assert "x-tenant: tenant-123" in result
        assert "x-missing: " not in str(result)

        # Test case 6: All extra headers match request headers
        litellm.extra_spend_tag_headers = ["x-custom", "x-tenant"]
        result = StandardLoggingPayloadSetup._get_extra_header_tags(
            proxy_server_request={
                "headers": {
                    "x-custom": "my-custom-value",
                    "x-tenant": "tenant-123",
                    "content-type": "application/json",
                }
            }
        )
        assert result is not None
        assert len(result) == 2
        assert "x-custom: my-custom-value" in result
        assert "x-tenant: tenant-123" in result

        # Test case 7: Headers with empty values should not be included
        litellm.extra_spend_tag_headers = ["x-custom", "x-empty"]
        result = StandardLoggingPayloadSetup._get_extra_header_tags(
            proxy_server_request={"headers": {"x-custom": "my-value", "x-empty": ""}}
        )
        assert result is not None
        assert len(result) == 1
        assert "x-custom: my-value" in result
        assert "x-empty:" not in str(result)

    finally:
        # Restore original value
        if original_extra_headers is not None:
            litellm.extra_spend_tag_headers = original_extra_headers
        else:
            # Remove the attribute if it didn't exist before
            if hasattr(litellm, "extra_spend_tag_headers"):
                delattr(litellm, "extra_spend_tag_headers")


def test_response_cost_calculator_with_response_cost_in_hidden_params(logging_obj):
    from litellm import Router
    from litellm.litellm_core_utils.litellm_logging import Logging

    router = Router(
        model_list=[
            {
                "model_name": "DeepSeek-R1",
                "litellm_params": {
                    "model": "together_ai/deepseek-ai/DeepSeek-R1",
                },
                "model_info": {
                    "access_groups": ["agent-models"],
                    "supports_tool_choice": True,
                    "supports_function_calling": True,
                    "input_cost_per_token": 100,
                    "output_cost_per_token": 100,
                },
            }
        ]
    )

    mock_response = router.completion(
        model="DeepSeek-R1",
        messages=[{"role": "user", "content": "Hey"}],
        mock_response="Hello, world!",
    )

    response_cost = logging_obj._response_cost_calculator(
        result=mock_response,
    )

    assert response_cost is not None
    assert response_cost > 100


def test_sentry_event_scrubber_initialization(monkeypatch):
    # Step 1: Create a fake sentry_sdk.scrubber module
    mock_event_scrubber_instance = MagicMock()
    mock_event_scrubber_cls = MagicMock(return_value=mock_event_scrubber_instance)

    mock_scrubber_module = MagicMock()
    mock_scrubber_module.EventScrubber = mock_event_scrubber_cls

    # Step 2: Create a fake sentry_sdk module and insert into sys.modules
    mock_sentry_sdk = MagicMock()
    mock_sentry_sdk.scrubber = mock_scrubber_module
    mock_init = MagicMock()
    mock_sentry_sdk.init = mock_init

    # Step 3: Inject both into sys.modules BEFORE import occurs
    sys.modules["sentry_sdk"] = mock_sentry_sdk
    sys.modules["sentry_sdk.scrubber"] = mock_scrubber_module

    # Step 4: Run the actual sentry setup code
    set_callbacks(["sentry"])

    # Step 5: Assert the EventScrubber was constructed correctly
    mock_event_scrubber_cls.assert_called_once_with(
        denylist=SENTRY_DENYLIST,
        pii_denylist=SENTRY_PII_DENYLIST,
    )

    # Step 6: Assert the event_scrubber and PII args were passed
    mock_init.assert_called_once()
    call_args = mock_init.call_args[1]
    assert call_args["event_scrubber"] == mock_event_scrubber_instance
    assert call_args["send_default_pii"] is False


def test_get_masked_values():
    from litellm.litellm_core_utils.litellm_logging import _get_masked_values

    sensitive_object = {
        "mode": "pre_call",
        "api_key": "sensitive_api_key",
        "payload": True,
        "api_base": "sensitive_api_base",
        "dev_info": True,
        "metadata": None,
        "breakdown": True,
        "guardrail": "azure/text_moderations",
        "default_on": False,
        "guard_name": None,
        "project_id": None,
        "aws_role_name": None,
        "lasso_user_id": None,
        "aws_region_name": None,
        "aws_profile_name": None,
        "aws_session_name": None,
        "aws_sts_endpoint": None,
        "guardrailVersion": None,
        "output_parse_pii": None,
        "aws_access_key_id": None,
        "aws_session_token": None,
        "presidio_language": "en",
        "mock_redacted_text": None,
        "severity_threshold": "5",
        "category_thresholds": None,
        "guardrailIdentifier": None,
        "pangea_input_recipe": None,
        "pii_entities_config": {},
        "mask_request_content": None,
        "pangea_output_recipe": None,
        "aws_secret_access_key": None,
        "detect_secrets_config": None,
        "lasso_conversation_id": None,
        "mask_response_content": None,
        "aws_web_identity_token": None,
        "presidio_analyzer_api_base": None,
        "presidio_ad_hoc_recognizers": None,
        "aws_bedrock_runtime_endpoint": None,
        "presidio_anonymizer_api_base": None,
        "vertex_credentials": "{sensitive_api_key}",
    }
    masked_values = _get_masked_values(
        sensitive_object, unmasked_length=4, number_of_asterisks=4
    )
    assert masked_values["presidio_anonymizer_api_base"] is None
    assert masked_values["vertex_credentials"] == "{s****y}"


@pytest.mark.asyncio
async def test_e2e_generate_cold_storage_object_key_successful():
    """
    Test end-to-end generation of cold storage object key when cold storage is properly configured.
    """
    from datetime import datetime, timezone
    from unittest.mock import patch

    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create test data
    start_time = datetime(2025, 1, 15, 10, 30, 45, 123456, timezone.utc)
    response_id = "chatcmpl-test-12345"
    team_alias = "test-team"

    with patch("litellm.cold_storage_custom_logger", return_value="s3"), patch(
        "litellm.integrations.s3.get_s3_object_key"
    ) as mock_get_s3_key:

        # Mock the S3 object key generation to return a predictable result
        mock_get_s3_key.return_value = (
            "2025-01-15/time-10-30-45-123456_chatcmpl-test-12345.json"
        )

        # Call the function
        result = StandardLoggingPayloadSetup._generate_cold_storage_object_key(
            start_time=start_time, response_id=response_id, team_alias=team_alias
        )

        # Verify the S3 function was called with correct parameters
        mock_get_s3_key.assert_called_once_with(
            s3_path="",  # Empty path as default
            prefix="",  # No prefix for cold storage
            start_time=start_time,
            s3_file_name="time-10-30-45-123456_chatcmpl-test-12345",
        )

        # Verify the result
        assert result == "2025-01-15/time-10-30-45-123456_chatcmpl-test-12345.json"
        assert result is not None
        assert isinstance(result, str)


@pytest.mark.asyncio
async def test_e2e_generate_cold_storage_object_key_with_custom_logger_s3_path():
    """
    Test that _generate_cold_storage_object_key uses s3_path from custom logger instance.
    """
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch

    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create test data
    start_time = datetime(2025, 1, 15, 10, 30, 45, 123456, timezone.utc)
    response_id = "chatcmpl-test-12345"

    # Create mock custom logger with s3_path
    mock_custom_logger = MagicMock()
    mock_custom_logger.s3_path = "storage"

    with patch("litellm.cold_storage_custom_logger", "s3_v2"), patch(
        "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name"
    ) as mock_get_logger, patch(
        "litellm.integrations.s3.get_s3_object_key"
    ) as mock_get_s3_key:

        # Setup mocks
        mock_get_logger.return_value = mock_custom_logger
        mock_get_s3_key.return_value = (
            "storage/2025-01-15/time-10-30-45-123456_chatcmpl-test-12345.json"
        )

        # Call the function
        result = StandardLoggingPayloadSetup._generate_cold_storage_object_key(
            start_time=start_time, response_id=response_id
        )

        # Verify logger was queried correctly
        mock_get_logger.assert_called_once_with("s3_v2")

        # Verify the S3 function was called with the custom logger's s3_path
        mock_get_s3_key.assert_called_once_with(
            s3_path="storage",  # Should use custom logger's s3_path
            prefix="",
            start_time=start_time,
            s3_file_name="time-10-30-45-123456_chatcmpl-test-12345",
        )

        # Verify the result
        assert (
            result == "storage/2025-01-15/time-10-30-45-123456_chatcmpl-test-12345.json"
        )


@pytest.mark.asyncio
async def test_e2e_generate_cold_storage_object_key_with_logger_no_s3_path():
    """
    Test that _generate_cold_storage_object_key falls back to empty s3_path when logger has no s3_path.
    """
    from datetime import datetime, timezone
    from unittest.mock import MagicMock, patch

    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create test data
    start_time = datetime(2025, 1, 15, 10, 30, 45, 123456, timezone.utc)
    response_id = "chatcmpl-test-12345"

    # Create mock custom logger without s3_path
    mock_custom_logger = MagicMock()
    mock_custom_logger.s3_path = None  # or could be missing attribute

    with patch("litellm.cold_storage_custom_logger", "s3_v2"), patch(
        "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name"
    ) as mock_get_logger, patch(
        "litellm.integrations.s3.get_s3_object_key"
    ) as mock_get_s3_key:

        # Setup mocks
        mock_get_logger.return_value = mock_custom_logger
        mock_get_s3_key.return_value = (
            "2025-01-15/time-10-30-45-123456_chatcmpl-test-12345.json"
        )

        # Call the function
        result = StandardLoggingPayloadSetup._generate_cold_storage_object_key(
            start_time=start_time, response_id=response_id
        )

        # Verify the S3 function was called with empty s3_path (fallback)
        mock_get_s3_key.assert_called_once_with(
            s3_path="",  # Should fall back to empty string
            prefix="",
            start_time=start_time,
            s3_file_name="time-10-30-45-123456_chatcmpl-test-12345",
        )

        # Verify the result
        assert result == "2025-01-15/time-10-30-45-123456_chatcmpl-test-12345.json"


@pytest.mark.asyncio
async def test_e2e_generate_cold_storage_object_key_not_configured():
    """
    Test end-to-end generation of cold storage object key when cold storage is not configured.
    """
    from datetime import datetime, timezone
    from unittest.mock import patch

    import litellm
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create test data
    start_time = datetime(2025, 1, 15, 10, 30, 45, 123456, timezone.utc)
    response_id = "chatcmpl-test-67890"
    team_alias = "another-team"

    # Use patch to ensure test isolation
    with patch.object(litellm, "cold_storage_custom_logger", None):
        # Call the function
        result = StandardLoggingPayloadSetup._generate_cold_storage_object_key(
            start_time=start_time, response_id=response_id, team_alias=team_alias
        )

    # Verify the result is None when cold storage is not configured
    assert result is None


def test_get_final_response_obj_with_empty_response_obj_and_list_init():
    """
    Test get_final_response_obj when response_obj is empty dict and init_response_obj is a list.

    When response_obj is empty (falsy), the method should return init_response_obj if it's a list.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create test objects
    class TestObject1:
        def __init__(self):
            self.name = "Object1"

    class TestObject2:
        def __init__(self):
            self.name = "Object2"

    obj1 = TestObject1()
    obj2 = TestObject2()

    # Test case: empty response_obj, list init_response_obj
    response_obj = {}
    init_response_obj = [obj1, obj2]
    kwargs = {}

    # Call the method
    result = StandardLoggingPayloadSetup.get_final_response_obj(
        response_obj=response_obj, init_response_obj=init_response_obj, kwargs=kwargs
    )

    # Verify the result
    assert result == [obj1, obj2]
    assert result is init_response_obj  # Should be the exact same list object
    assert len(result) == 2
    assert result[0].name == "Object1"
    assert result[1].name == "Object2"


def test_append_system_prompt_messages():
    """
    Test append_system_prompt_messages prepends system message from kwargs to messages list.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Test case 1: system in kwargs with existing messages
    kwargs = {"system": "You are a helpful assistant"}
    messages = [{"role": "user", "content": "Hello"}]
    result = StandardLoggingPayloadSetup.append_system_prompt_messages(
        kwargs=kwargs, messages=messages
    )
    assert len(result) == 2
    assert result[0] == {"role": "system", "content": "You are a helpful assistant"}
    assert result[1] == {"role": "user", "content": "Hello"}

    # Test case 2: system in kwargs with None messages
    kwargs = {"system": "You are a helpful assistant"}
    result = StandardLoggingPayloadSetup.append_system_prompt_messages(
        kwargs=kwargs, messages=None
    )
    assert len(result) == 1
    assert result[0] == {"role": "system", "content": "You are a helpful assistant"}

    # Test case 3: system in kwargs with empty messages list
    kwargs = {"system": "You are a helpful assistant"}
    result = StandardLoggingPayloadSetup.append_system_prompt_messages(
        kwargs=kwargs, messages=[]
    )
    assert len(result) == 1
    assert result[0] == {"role": "system", "content": "You are a helpful assistant"}

    # Test case 4: duplicate system message should not be added
    kwargs = {"system": "You are a helpful assistant"}
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    result = StandardLoggingPayloadSetup.append_system_prompt_messages(
        kwargs=kwargs, messages=messages
    )
    assert len(result) == 2
    assert result[0] == {"role": "system", "content": "You are a helpful assistant"}

    # Test case 5: no system in kwargs returns messages unchanged
    kwargs = {}
    messages = [{"role": "user", "content": "Hello"}]
    result = StandardLoggingPayloadSetup.append_system_prompt_messages(
        kwargs=kwargs, messages=messages
    )
    assert result == messages

    # Test case 6: None kwargs returns messages unchanged
    result = StandardLoggingPayloadSetup.append_system_prompt_messages(
        kwargs=None, messages=messages
    )
    assert result == messages


@pytest.mark.asyncio
async def test_async_success_handler_sets_standard_logging_object_for_pass_through_endpoints():
    """
    Test that async_success_handler sets standard_logging_object for pass-through endpoints
    even when complete_streaming_response is None.

    This is a regression test for the bug where pass-through endpoints (like vLLM classify)
    would not set standard_logging_object, causing model_max_budget_limiter to raise
    ValueError("standard_logging_payload is required").

    The fix adds an elif branch in async_success_handler to set standard_logging_object
    for pass-through endpoints when complete_streaming_response is None.
    """
    from datetime import datetime
    from unittest.mock import patch

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import StandardPassThroughResponseObject

    # Create a logging object for a pass-through endpoint
    logging_obj = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )

    # Set up model_call_details with required fields
    logging_obj.model_call_details = {
        "litellm_params": {
            "metadata": {},
            "proxy_server_request": {},
        },
        "litellm_call_id": "test-call-id",
    }

    # Create a pass-through response object (not a ModelResponse)
    result = StandardPassThroughResponseObject(response='{"status": "success"}')

    start_time = datetime.now()
    end_time = datetime.now()

    # Mock the callbacks to avoid actual logging
    with patch.object(logging_obj, "get_combined_callback_list", return_value=[]):
        # Call async_success_handler
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )

    # Verify that standard_logging_object was set
    assert "standard_logging_object" in logging_obj.model_call_details, (
        "standard_logging_object should be set for pass-through endpoints "
        "even when complete_streaming_response is None"
    )
    assert logging_obj.model_call_details["standard_logging_object"] is not None, (
        "standard_logging_object should not be None for pass-through endpoints"
    )

    # Verify that async_complete_streaming_response was set to prevent re-processing
    # This is consistent with the existing code pattern for regular streaming
    assert "async_complete_streaming_response" in logging_obj.model_call_details, (
        "async_complete_streaming_response should be set to prevent re-processing, "
        "consistent with the existing code pattern"
    )
    assert logging_obj.model_call_details["async_complete_streaming_response"] is result, (
        "async_complete_streaming_response should be set to the result"
    )

    # Verify that response_cost is set to None (cost calculation not possible for pass-through)
    # This is consistent with the error handling in the non-pass-through code path
    assert "response_cost" in logging_obj.model_call_details, (
        "response_cost should be set for pass-through endpoints"
    )
    assert logging_obj.model_call_details["response_cost"] is None, (
        "response_cost should be None for pass-through endpoints since "
        "StandardPassThroughResponseObject doesn't have standard usage info"
    )


@pytest.mark.asyncio
async def test_async_success_handler_prevents_reprocessing_for_pass_through_endpoints():
    """
    Test that async_success_handler prevents re-processing for pass-through endpoints
    by setting async_complete_streaming_response, consistent with the existing code pattern.

    This ensures that if async_success_handler is called multiple times (e.g., during
    streaming), it won't re-process the response after the first complete call.
    """
    from datetime import datetime
    from unittest.mock import patch

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import StandardPassThroughResponseObject

    # Create a logging object for a pass-through endpoint
    logging_obj = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="test-call-id-reprocess",
        function_id="test-function-id-reprocess",
    )

    # Set up model_call_details with required fields
    logging_obj.model_call_details = {
        "litellm_params": {
            "metadata": {},
            "proxy_server_request": {},
        },
        "litellm_call_id": "test-call-id-reprocess",
    }

    result = StandardPassThroughResponseObject(response='{"status": "success"}')
    start_time = datetime.now()
    end_time = datetime.now()

    # Mock the callbacks to avoid actual logging
    with patch.object(logging_obj, "get_combined_callback_list", return_value=[]):
        # First call - should process and set standard_logging_object
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )

    # Verify first call set the values
    assert "standard_logging_object" in logging_obj.model_call_details
    assert "async_complete_streaming_response" in logging_obj.model_call_details
    first_standard_logging_object = logging_obj.model_call_details["standard_logging_object"]

    # Second call - should return early due to async_complete_streaming_response guard
    with patch.object(logging_obj, "get_combined_callback_list", return_value=[]) as mock_callbacks:
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )
        # The guard should cause early return, so get_combined_callback_list should not be called
        mock_callbacks.assert_not_called()

    # Verify standard_logging_object wasn't modified by second call
    assert logging_obj.model_call_details["standard_logging_object"] is first_standard_logging_object, (
        "standard_logging_object should not be modified on re-processing"
    )


@pytest.mark.asyncio
async def test_async_success_handler_sets_standard_logging_object_for_streaming_pass_through():
    """
    Test that async_success_handler sets standard_logging_object for streaming
    pass-through endpoints when the response cannot be parsed into a ModelResponse.

    This covers the case where streaming pass-through endpoints for unknown providers
    return a StandardPassThroughResponseObject instead of a ModelResponse.
    """
    from datetime import datetime
    from unittest.mock import patch

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import StandardPassThroughResponseObject

    # Create a logging object for a streaming pass-through endpoint
    logging_obj = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "test"}],
        stream=True,  # Streaming request
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="test-call-id-streaming",
        function_id="test-function-id-streaming",
    )

    # Set up model_call_details with required fields
    logging_obj.model_call_details = {
        "litellm_params": {
            "metadata": {},
            "proxy_server_request": {},
        },
        "litellm_call_id": "test-call-id-streaming",
    }

    # Create a pass-through response object (simulating unparseable streaming response)
    result = StandardPassThroughResponseObject(
        response='data: {"chunk": 1}\ndata: {"chunk": 2}\ndata: [DONE]'
    )

    start_time = datetime.now()
    end_time = datetime.now()

    # Mock the callbacks to avoid actual logging
    with patch.object(logging_obj, "get_combined_callback_list", return_value=[]):
        # Call async_success_handler
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )

    # Verify that standard_logging_object was set
    assert "standard_logging_object" in logging_obj.model_call_details, (
        "standard_logging_object should be set for streaming pass-through endpoints "
        "even when the response cannot be parsed into a ModelResponse"
    )
    assert logging_obj.model_call_details["standard_logging_object"] is not None, (
        "standard_logging_object should not be None for streaming pass-through endpoints"
    )
def test_get_error_information_error_code_priority():
    """
    Test get_error_information prioritizes 'code' attribute over 'status_code' attribute
    and handles edge cases like empty strings and "None" string values.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Test case 1: Exception with 'code' attribute (ProxyException style)
    class ProxyException(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message
            super().__init__(message)

    proxy_exception = ProxyException(code="500", message="Internal Server Error")
    result = StandardLoggingPayloadSetup.get_error_information(proxy_exception)
    assert result["error_code"] == "500"
    assert result["error_class"] == "ProxyException"

    # Test case 2: Exception with 'status_code' attribute (LiteLLM style)
    class LiteLLMException(Exception):
        def __init__(self, status_code, message):
            self.status_code = status_code
            self.message = message
            super().__init__(message)

    litellm_exception = LiteLLMException(status_code=429, message="Rate limit exceeded")
    result = StandardLoggingPayloadSetup.get_error_information(litellm_exception)
    assert result["error_code"] == "429"
    assert result["error_class"] == "LiteLLMException"

    # Test case 3: Exception with both 'code' and 'status_code' - should prefer 'code'
    class BothAttributesException(Exception):
        def __init__(self, code, status_code, message):
            self.code = code
            self.status_code = status_code
            self.message = message
            super().__init__(message)

    both_exception = BothAttributesException(
        code="400", status_code=500, message="Bad Request"
    )
    result = StandardLoggingPayloadSetup.get_error_information(both_exception)
    assert result["error_code"] == "400"  # Should prefer 'code' over 'status_code'

    # Test case 4: Exception with 'code' as empty string - should fall back to 'status_code'
    empty_code_exception = BothAttributesException(
        code="", status_code=404, message="Not Found"
    )
    result = StandardLoggingPayloadSetup.get_error_information(empty_code_exception)
    assert result["error_code"] == "404"  # Should fall back to status_code

    # Test case 5: Exception with 'code' as "None" string - should fall back to 'status_code'
    none_string_exception = BothAttributesException(
        code="None", status_code=503, message="Service Unavailable"
    )
    result = StandardLoggingPayloadSetup.get_error_information(none_string_exception)
    assert result["error_code"] == "503"  # Should fall back to status_code

    # Test case 6: Exception with 'code' as None - should fall back to 'status_code'
    none_code_exception = BothAttributesException(
        code=None, status_code=401, message="Unauthorized"
    )
    result = StandardLoggingPayloadSetup.get_error_information(none_code_exception)
    assert result["error_code"] == "401"  # Should fall back to status_code

    # Test case 7: Exception with neither 'code' nor 'status_code' - should return empty string
    class NoCodeException(Exception):
        def __init__(self, message):
            self.message = message
            super().__init__(message)

    no_code_exception = NoCodeException(message="Generic error")
    result = StandardLoggingPayloadSetup.get_error_information(no_code_exception)
    assert result["error_code"] == ""
    assert result["error_class"] == "NoCodeException"
