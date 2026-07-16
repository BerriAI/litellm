import os
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import time

import litellm
from litellm.constants import SENTRY_DENYLIST, SENTRY_PII_DENYLIST
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.litellm_core_utils.litellm_logging import set_callbacks
from litellm.types.utils import ModelResponse, TextCompletionResponse


@pytest.fixture
def logging_obj():
    return LitellmLogging(
        model="bedrock/claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    )


def test_get_combined_callback_list_preserves_insertion_order(logging_obj):
    assert logging_obj.get_combined_callback_list(
        dynamic_success_callbacks=["prometheus", "langfuse", "datadog", "otel", "s3"],
        global_callbacks=["langfuse", "gcs_bucket", "arize", "logfire"],
    ) == ["prometheus", "langfuse", "datadog", "otel", "s3", "gcs_bucket", "arize", "logfire"]


def test_get_masked_api_base(logging_obj):
    api_base = "https://api.openai.com/v1"
    masked_api_base = logging_obj._get_masked_api_base(api_base)
    assert masked_api_base == "https://api.openai.com/v1"
    assert type(masked_api_base) == str


def test_post_call_serializes_dict_with_datetime(logging_obj):
    import datetime

    response = {
        "status": "InProgress",
        "submitTime": datetime.datetime(2026, 5, 11, 23, 49, 13, 132000),
    }
    logging_obj.post_call(original_response=response)
    serialized = logging_obj.model_call_details["original_response"]
    assert isinstance(serialized, str)
    assert "2026-05-11" in serialized


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


def test_use_custom_pricing_for_model_via_litellm_metadata():
    """Pricing in litellm_metadata.model_info must be detected.

    Generic API call routes (/messages, /responses) store model_info
    under litellm_metadata, not metadata. Regression test for #23185.
    """
    from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model

    litellm_params = {
        "litellm_metadata": {
            "model_info": {
                "id": "claude-sonnet-4-custom",
                "input_cost_per_token": 0.0003,
                "output_cost_per_token": 0.0015,
            },
        },
    }
    assert use_custom_pricing_for_model(litellm_params) is True


def test_use_custom_pricing_not_detected_litellm_metadata_no_pricing():
    """Should return False when litellm_metadata.model_info has no pricing keys."""
    from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model

    litellm_params = {
        "litellm_metadata": {
            "model_info": {"id": "some-id", "db_model": False},
        },
    }
    assert use_custom_pricing_for_model(litellm_params) is False


def test_response_cost_calculator_uses_router_model_id_from_litellm_metadata():
    """_response_cost_calculator should extract router_model_id from
    litellm_params.litellm_metadata.model_info.id when the result object
    does not carry _hidden_params (e.g. ResponsesAPIResponse from /v1/responses
    streaming). Regression test for custom pricing on streaming responses."""
    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import ResponsesAPIResponse

    custom_model_id = "gpt-5-custom-pricing"
    custom_input_cost = 125.0
    custom_output_cost = 10.0

    litellm.register_model(
        model_cost={
            custom_model_id: {
                "input_cost_per_token": custom_input_cost,
                "output_cost_per_token": custom_output_cost,
                "max_tokens": 128000,
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "litellm_provider": "openai",
            }
        }
    )

    try:
        logging_obj = LiteLLMLoggingObj(
            model="gpt-5",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            call_type="aresponses",
            start_time=time.time(),
            litellm_call_id="test-123",
            function_id="test-fn",
        )

        logging_obj.update_environment_variables(
            model="gpt-5",
            user="",
            optional_params={},
            litellm_params={
                "api_base": "",
                "litellm_metadata": {
                    "model_info": {
                        "id": custom_model_id,
                        "input_cost_per_token": custom_input_cost,
                        "output_cost_per_token": custom_output_cost,
                    },
                },
            },
        )

        response_obj = ResponsesAPIResponse(
            id="resp_abc",
            created_at=1234567890,
            model="gpt-5",
            output=[],
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )

        cost = logging_obj._response_cost_calculator(result=response_obj)

        assert cost is not None, "Cost should not be None"
        expected_cost = (10 * custom_input_cost) + (5 * custom_output_cost)
        assert cost == pytest.approx(
            expected_cost
        ), f"Expected {expected_cost}, got {cost}"
    finally:
        litellm.model_cost.pop(custom_model_id, None)


class TestGetRouterModelId:
    """Tests for the get_router_model_id helper method."""

    def test_returns_id_from_litellm_metadata(self, logging_obj):
        """Should extract model_info.id from litellm_metadata."""
        logging_obj.litellm_params = {
            "litellm_metadata": {
                "model_info": {"id": "custom-deploy-1"},
            },
        }
        assert logging_obj.get_router_model_id() == "custom-deploy-1"

    def test_returns_id_from_metadata(self, logging_obj):
        """Should fall back to metadata when litellm_metadata has no model_info."""
        logging_obj.litellm_params = {
            "metadata": {
                "model_info": {"id": "custom-deploy-2"},
            },
        }
        assert logging_obj.get_router_model_id() == "custom-deploy-2"

    def test_prefers_litellm_metadata_over_metadata(self, logging_obj):
        """litellm_metadata should take priority over metadata."""
        logging_obj.litellm_params = {
            "litellm_metadata": {
                "model_info": {"id": "from-litellm-meta"},
            },
            "metadata": {
                "model_info": {"id": "from-meta"},
            },
        }
        assert logging_obj.get_router_model_id() == "from-litellm-meta"

    def test_returns_none_when_no_model_info(self, logging_obj):
        """Should return None when no model_info is present."""
        logging_obj.litellm_params = {"api_base": ""}
        assert logging_obj.get_router_model_id() is None

    def test_returns_none_when_no_litellm_params(self):
        """Should return None when litellm_params is not set."""
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        obj = LiteLLMLoggingObj(
            model="test",
            messages=[],
            stream=False,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="x",
            function_id="x",
        )
        # litellm_params exists but is empty by default
        assert obj.get_router_model_id() is None


class TestAnthropicPassthroughCustomPricing:
    """Verify the Anthropic pass-through handler forwards custom pricing."""

    def test_completion_cost_receives_custom_pricing_args(self):
        """_create_anthropic_response_logging_payload should pass
        custom_pricing and router_model_id to litellm.completion_cost
        when the logging object carries custom pricing in model_info."""
        from unittest.mock import patch

        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
            AnthropicPassthroughLoggingHandler,
        )

        logging_obj = LiteLLMLoggingObj(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            call_type="anthropic_messages",
            start_time=time.time(),
            litellm_call_id="test-456",
            function_id="test-fn",
        )
        logging_obj.update_environment_variables(
            model="claude-sonnet-4-20250514",
            user="",
            optional_params={},
            litellm_params={
                "api_base": "",
                "litellm_metadata": {
                    "model_info": {
                        "id": "claude-custom-pricing",
                        "input_cost_per_token": 0.5,
                        "output_cost_per_token": 1.5,
                    },
                },
            },
        )
        logging_obj.model_call_details["custom_llm_provider"] = "anthropic"

        mock_response = ModelResponse()
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5}  # type: ignore

        with patch("litellm.completion_cost", return_value=42.0) as mock_cost:
            AnthropicPassthroughLoggingHandler._create_anthropic_response_logging_payload(
                litellm_model_response=mock_response,
                model="claude-sonnet-4-20250514",
                kwargs={},
                start_time=time.time(),
                end_time=time.time(),
                logging_obj=logging_obj,
            )

            mock_cost.assert_called_once()
            call_kwargs = mock_cost.call_args
            assert call_kwargs.kwargs.get("custom_pricing") is True
            assert call_kwargs.kwargs.get("router_model_id") == "claude-custom-pricing"


class TestUpdateFromKwargs:
    """Tests for the update_from_kwargs convenience wrapper."""

    def test_extracts_metadata_from_kwargs(self, logging_obj):
        metadata = {"user_api_key": "sk-test", "model_info": {"id": "abc"}}
        kwargs = {"metadata": metadata, "other_key": "ignored"}

        logging_obj.update_from_kwargs(
            kwargs=kwargs,
            litellm_params={"litellm_call_id": "call-1"},
        )

        assert logging_obj.litellm_params["metadata"] == metadata
        assert logging_obj.litellm_params["litellm_call_id"] == "call-1"

    def test_extracts_litellm_metadata_from_kwargs(self, logging_obj):
        lm_meta = {
            "model_info": {
                "id": "deploy-1",
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            }
        }
        kwargs = {"litellm_metadata": lm_meta}

        logging_obj.update_from_kwargs(
            kwargs=kwargs,
            litellm_params={"litellm_call_id": "call-2"},
        )

        assert logging_obj.litellm_params["litellm_metadata"] == lm_meta
        assert logging_obj.litellm_params["litellm_call_id"] == "call-2"

    def test_backfills_metadata_from_litellm_metadata(self, logging_obj):
        """When only litellm_metadata is present, metadata should be backfilled."""
        lm_meta = {"model_info": {"id": "deploy-1"}}
        kwargs = {"litellm_metadata": lm_meta}

        logging_obj.update_from_kwargs(kwargs=kwargs)

        assert logging_obj.litellm_params["metadata"] == lm_meta

    def test_no_backfill_when_metadata_already_present(self, logging_obj):
        metadata = {"user_api_key": "sk-real"}
        lm_meta = {"model_info": {"id": "deploy-1"}}
        kwargs = {"metadata": metadata, "litellm_metadata": lm_meta}

        logging_obj.update_from_kwargs(kwargs=kwargs)

        assert logging_obj.litellm_params["metadata"] == metadata
        assert logging_obj.litellm_params["litellm_metadata"] == lm_meta

    def test_caller_litellm_params_win_over_kwargs(self, logging_obj):
        """Explicit litellm_params metadata merges into kwargs metadata without overwriting."""
        kwargs = {"metadata": {"from_kwargs": True}}

        logging_obj.update_from_kwargs(
            kwargs=kwargs,
            litellm_params={"metadata": {"from_caller": True}, "litellm_call_id": "x"},
        )

        # kwargs metadata is preserved, caller metadata is merged in
        assert logging_obj.litellm_params["metadata"] == {
            "from_kwargs": True,
            "from_caller": True,
        }

    def test_kwargs_metadata_wins_over_caller_metadata_in_conflict(self, logging_obj):
        """kwargs metadata takes precedence; caller litellm_params metadata is merged without overwriting."""
        kwargs = {"metadata": {"from_kwargs": True, "shared_key": "kwargs_value"}}

        logging_obj.update_from_kwargs(
            kwargs=kwargs,
            litellm_params={
                "metadata": {"from_caller": True, "shared_key": "caller_value"},
                "litellm_call_id": "x",
            },
        )

        # kwargs metadata is preserved (shared_key keeps the kwargs value), caller-only keys are added
        assert logging_obj.litellm_params["metadata"] == {
            "from_kwargs": True,
            "from_caller": True,
            "shared_key": "kwargs_value",  # kwargs wins on conflict
        }

    def test_custom_pricing_detected_via_litellm_metadata(self, logging_obj):
        """Custom pricing in litellm_metadata.model_info should set custom_pricing flag."""
        from litellm.litellm_core_utils.litellm_logging import (
            use_custom_pricing_for_model,
        )

        lm_meta = {
            "model_info": {
                "id": "deploy-custom",
                "input_cost_per_token": 0.005,
                "output_cost_per_token": 0.015,
            }
        }
        kwargs = {"litellm_metadata": lm_meta}

        logging_obj.update_from_kwargs(kwargs=kwargs)

        assert use_custom_pricing_for_model(logging_obj.litellm_params) is True

    def test_additional_params_forwarded(self, logging_obj):
        kwargs = {"metadata": {}}
        logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model="gpt-5",
            user="test-user",
            optional_params={"temperature": 0.7},
            custom_llm_provider="openai",
        )

        assert logging_obj.model == "gpt-5"
        assert logging_obj.user == "test-user"
        assert logging_obj.model_call_details["custom_llm_provider"] == "openai"

    def test_empty_kwargs_no_error(self, logging_obj):
        logging_obj.update_from_kwargs(
            kwargs={},
            litellm_params={"litellm_call_id": "call-empty"},
        )
        assert logging_obj.litellm_params["litellm_call_id"] == "call-empty"


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

    from litellm.integrations.datadog.datadog import DataDogLogger
    from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
    from litellm.litellm_core_utils import litellm_logging as logging_module

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
        assert any(
            isinstance(cb, DataDogLLMObsLogger)
            for cb in logging_module._in_memory_loggers
        )
        assert any(
            type(cb) is DataDogLogger for cb in logging_module._in_memory_loggers
        )
    finally:
        logging_module._in_memory_loggers.clear()


@pytest.mark.asyncio
async def test_logfire_logger_accepts_env_vars_for_base_url(monkeypatch):
    """Ensure Logfire logger uses LOGFIRE_BASE_URL to build the OTLP HTTP endpoint (/v1/traces)."""

    # Required env vars for Logfire integration
    monkeypatch.setenv("LOGFIRE_TOKEN", "test-token")
    monkeypatch.setenv(
        "LOGFIRE_BASE_URL", "https://logfire-api-custom.pydantic.dev"
    )  # no trailing slash on purpose

    # Import after env vars are set (important if module-level caching exists)
    from litellm.integrations.opentelemetry import OpenTelemetry  # logger class
    from litellm.litellm_core_utils import litellm_logging as logging_module

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
        assert any(
            type(cb) is OpenTelemetry for cb in logging_module._in_memory_loggers
        )

        # Core regression check: base URL env var should influence the exporter endpoint.
        #
        # OpenTelemetry integration has historically stored config on the instance.
        # We defensively check a few common attribute names to avoid brittle coupling.
        cfg = (
            getattr(logger, "otel_config", None)
            or getattr(logger, "config", None)
            or getattr(logger, "_otel_config", None)
        )
        assert (
            cfg is not None
        ), "Expected OpenTelemetry logger to keep an otel config on the instance"

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
        assert mock_should_run_logging.call_count == 1


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


@pytest.mark.parametrize(
    "async_flag", ["acompletion", "aresponses", "allm_passthrough_route"]
)
def test_success_handler_skips_sync_callbacks_for_async_requests(
    logging_obj, async_flag
):
    """Ensure sync success callbacks are skipped when async call type flags are set."""
    from litellm.integrations.custom_logger import CustomLogger

    class DummyLogger(CustomLogger):
        pass

    logging_obj.stream = (
        False  # simulate non-streaming request where sync callbacks would normally run
    )
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


def test_is_sync_litellm_request(logging_obj):
    assert logging_obj.is_async_entrypoint is None
    assert logging_obj._is_sync_litellm_request({}) is True
    assert logging_obj._is_sync_litellm_request({"acompletion": True}) is False
    assert logging_obj._is_sync_litellm_request({"allm_passthrough_route": True}) is False

    logging_obj.is_async_entrypoint = True
    assert logging_obj._is_sync_litellm_request({}) is False

    logging_obj.is_async_entrypoint = False
    assert logging_obj._is_sync_litellm_request({"acompletion": True}) is True


@pytest.mark.asyncio
async def test_anthropic_messages_success_logs_custom_logger_exactly_once(logging_obj):
    """A request stamped async by the @client wrapper must reach a CustomLogger exactly
    once, via the async hook only; the sync success_handler skips CustomLogger hooks.
    Regression guard for LIT-4447."""
    from litellm.integrations.custom_logger import CustomLogger

    class DummyLogger(CustomLogger):
        pass

    logging_obj.stream = False
    logging_obj.call_type = "anthropic_messages"
    logging_obj.is_async_entrypoint = True
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.litellm_params = {}

    dummy_logger = DummyLogger()

    model_response = ModelResponse(
        id="resp-anthropic-123",
        model="claude-sonnet-5",
        choices=[
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    with (
        patch.object(dummy_logger, "async_log_success_event", new_callable=AsyncMock) as mock_async_log,
        patch.object(dummy_logger, "log_success_event") as mock_sync_log,
        patch.object(
            logging_obj,
            "get_combined_callback_list",
            return_value=[dummy_logger],
        ),
    ):
        await logging_obj.async_success_handler(result=model_response)
        logging_obj.success_handler(result=model_response)

    mock_async_log.assert_awaited_once()
    mock_sync_log.assert_not_called()


@pytest.mark.asyncio
async def test_anthropic_messages_failure_logs_custom_logger_exactly_once(logging_obj):
    """A request stamped async by the @client wrapper must reach a CustomLogger exactly
    once on failure, via the async hook only; the sync failure_handler skips CustomLogger
    hooks. Regression guard for LIT-4447."""
    from litellm.integrations.custom_logger import CustomLogger

    class DummyLogger(CustomLogger):
        pass

    logging_obj.stream = False
    logging_obj.call_type = "anthropic_messages"
    logging_obj.is_async_entrypoint = True
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.litellm_params = {}

    dummy_logger = DummyLogger()
    exception = ValueError("provider blew up")

    with (
        patch.object(dummy_logger, "async_log_failure_event", new_callable=AsyncMock) as mock_async_log,
        patch.object(dummy_logger, "log_failure_event") as mock_sync_log,
        patch.object(
            logging_obj,
            "get_combined_callback_list",
            return_value=[dummy_logger],
        ),
    ):
        await logging_obj.async_failure_handler(exception, "traceback")
        logging_obj.failure_handler(exception, "traceback")

    mock_async_log.assert_awaited_once()
    mock_sync_log.assert_not_called()


def test_custom_logger_sync_only_hook_detection():
    """The fallback helpers key off class-level overrides: only a logger that overrides a
    sync hook without the matching async hook qualifies for sync delivery on async requests."""
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.litellm_logging import (
        _custom_logger_has_only_sync_failure_hook,
        _custom_logger_has_only_sync_success_hooks,
    )

    class SyncOnlyLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            pass

    class BothHooksLogger(SyncOnlyLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            pass

    class AsyncOnlyLogger(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            pass

    assert _custom_logger_has_only_sync_success_hooks(SyncOnlyLogger()) is True
    assert _custom_logger_has_only_sync_failure_hook(SyncOnlyLogger()) is True
    assert _custom_logger_has_only_sync_success_hooks(BothHooksLogger()) is False
    assert _custom_logger_has_only_sync_failure_hook(BothHooksLogger()) is False
    assert _custom_logger_has_only_sync_success_hooks(AsyncOnlyLogger()) is False
    assert _custom_logger_has_only_sync_failure_hook(AsyncOnlyLogger()) is False
    assert _custom_logger_has_only_sync_success_hooks(CustomLogger()) is False
    assert _custom_logger_has_only_sync_failure_hook(CustomLogger()) is False


@pytest.mark.asyncio
async def test_sync_only_custom_logger_still_logs_async_request_exactly_once(logging_obj):
    """A CustomLogger that overrides only the sync hooks must keep receiving events for
    async requests via the sync dispatch fallback, exactly once. Companion to the LIT-4447
    dedupe: fixing the double-log must not silence sync-only integrations entirely."""
    from litellm.integrations.custom_logger import CustomLogger

    events = []

    class SyncOnlyLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("sync_success")

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            events.append("sync_failure")

    logging_obj.stream = False
    logging_obj.call_type = "anthropic_messages"
    logging_obj.is_async_entrypoint = True
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.litellm_params = {}

    sync_only_logger = SyncOnlyLogger()

    model_response = ModelResponse(
        id="resp-sync-only-123",
        model="claude-sonnet-5",
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
        return_value=[sync_only_logger],
    ):
        await logging_obj.async_success_handler(result=model_response)
        logging_obj.success_handler(result=model_response)

    assert events == ["sync_success"]

    events.clear()
    exception = ValueError("provider blew up")

    with patch.object(
        logging_obj,
        "get_combined_callback_list",
        return_value=[sync_only_logger],
    ):
        await logging_obj.async_failure_handler(exception, "traceback")
        logging_obj.failure_handler(exception, "traceback")

    assert events == ["sync_failure"]


def test_sync_dispatch_gate_opens_for_sync_only_custom_logger(logging_obj):
    """The executor gate that decides whether sync callbacks run for async requests must
    count a sync-only CustomLogger, since the sync dispatch is its only delivery path;
    loggers with async hooks must not open it."""
    from litellm.integrations.custom_logger import CustomLogger

    class SyncOnlyLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

    class BothHooksLogger(SyncOnlyLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

    logging_obj.dynamic_success_callbacks = None

    with patch.object(litellm, "success_callback", [SyncOnlyLogger()]):
        assert logging_obj._should_run_sync_callbacks_for_async_calls() is True

    with patch.object(litellm, "success_callback", [BothHooksLogger()]):
        assert logging_obj._should_run_sync_callbacks_for_async_calls() is False

    with patch.object(litellm, "success_callback", []):
        assert logging_obj._should_run_sync_callbacks_for_async_calls() is False


@pytest.mark.asyncio
async def test_sync_only_logger_receives_async_event_through_dispatch_gate(logging_obj):
    """End-to-end through the executor dispatch: a sync-only CustomLogger registered with
    no other callbacks must receive exactly one success event for an async request; the
    gate itself must let it through rather than relying on an unrelated callback to open it."""
    import datetime as dt

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.thread_pool_executor import executor

    events = []

    class SyncOnlyLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("sync_success")

    logging_obj.stream = False
    logging_obj.call_type = "anthropic_messages"
    logging_obj.is_async_entrypoint = True
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.litellm_params = {}
    logging_obj.dynamic_success_callbacks = None

    sync_only_logger = SyncOnlyLogger()
    now = dt.datetime.now()

    with patch.object(litellm, "success_callback", [sync_only_logger]):
        model_response = ModelResponse(
            id="resp-dispatch-123",
            model="claude-sonnet-5",
            choices=[
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        await logging_obj.async_success_handler(result=model_response, start_time=now, end_time=now)
        logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=model_response, start_time=now, end_time=now
        )
        executor.submit(lambda: None).result()

    assert events == ["sync_success"]


@pytest.mark.asyncio
async def test_async_client_entrypoint_stamps_and_dedupes_flagless_call_type():
    """End-to-end through the real @client wrapper: litellm.anthropic_messages sets no
    `a*` flag in litellm_params, so only the wrapper-stamped is_async_entrypoint marks
    the request async. With the executor sync dispatch open (a surviving plain-callable
    callback, same mechanism as a legacy string callback), the CustomLogger must fire
    via the async hook exactly once. Regression guard for LIT-4447 and LIT-4475."""

    events = []

    class Rec(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append(("sync", kwargs.get("call_type")))

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append(("async", kwargs.get("call_type")))

    def _gate_opener(kwargs, completion_response, start_time, end_time):
        pass

    rec = Rec()
    original_success = litellm.success_callback
    original_async = litellm._async_success_callback
    litellm.success_callback = [rec, _gate_opener]
    litellm._async_success_callback = [rec]
    try:
        await litellm.anthropic_messages(
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-5",
            custom_llm_provider="anthropic",
            api_key="sk-ant-dummy",
            mock_response="hello",
        )
        for _ in range(50):
            if events:
                break
            await asyncio.sleep(0.1)
        from litellm.litellm_core_utils.thread_pool_executor import executor

        executor.submit(lambda: None).result()
    finally:
        litellm.success_callback = original_success
        litellm._async_success_callback = original_async

    assert events == [("async", "anthropic_messages")]


@pytest.mark.asyncio
async def test_client_wrapper_stamp_is_first_wins_across_nested_calls():
    """An async entrypoint that internally invokes a sync @client function with the
    shared logging object (e.g. agenerate_content delegating to generate_content) must
    keep is_async_entrypoint=True; the inner sync wrapper must not overwrite the
    entrypoint's stamp, and the nested request must reach a CustomLogger exactly once,
    via the async hook. Regression guard for LIT-4475 (gemini /generate_content kept
    double-logging because the inner sync wrapper flipped the bit back to sync)."""
    from litellm.litellm_core_utils.thread_pool_executor import executor
    from litellm.utils import client

    observed = {}
    events = []

    class Rec(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("sync")

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            events.append("async")

    def _gate_opener(kwargs, completion_response, start_time, end_time):
        pass

    @client
    def fake_inner_sync(model: str, messages=None, **kwargs):
        observed["inner_bit"] = kwargs["litellm_logging_obj"].is_async_entrypoint
        return ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop", "index": 0}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    @client
    async def fake_outer_async(model: str, messages=None, **kwargs):
        result = fake_inner_sync(model=model, messages=messages, **kwargs)
        observed["outer_bit_after_inner"] = kwargs["litellm_logging_obj"].is_async_entrypoint
        return result

    rec = Rec()
    original_success = litellm.success_callback
    original_async = litellm._async_success_callback
    litellm.success_callback = [rec, _gate_opener]
    litellm._async_success_callback = [rec]
    try:
        await fake_outer_async(model="gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}])
        for _ in range(50):
            if events:
                break
            await asyncio.sleep(0.1)
        executor.submit(lambda: None).result()
    finally:
        litellm.success_callback = original_success
        litellm._async_success_callback = original_async

    assert observed == {"inner_bit": True, "outer_bit_after_inner": True}
    assert events == ["async"]

    fake_inner_sync(model="gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}])

    assert observed["inner_bit"] is False


def test_get_litellm_params_propagates_allm_passthrough_route(logging_obj):
    """`allm_passthrough_route=True` set on kwargs by the async passthrough entrypoint
    must land in `litellm_params` so `_is_sync_litellm_request` sees it and the
    request is classified as async. Regression guard for LIT-4192."""
    from litellm.litellm_core_utils.get_litellm_params import get_litellm_params

    params = get_litellm_params(allm_passthrough_route=True)
    assert params.get("allm_passthrough_route") is True
    assert logging_obj._is_sync_litellm_request(params) is False


@pytest.mark.asyncio
async def test_dispatch_success_handlers_invokes_callbacks_once_for_final_stream(
    logging_obj,
):
    """Second final-stream dispatch must not re-export (CSW + deferred guardrail paths)."""
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class MockCallback(CustomLogger):
        pass

    mock_callback = MockCallback()
    original_async_callbacks = list(litellm._async_success_callback or [])
    litellm._async_success_callback = [mock_callback]

    result = ModelResponse(
        id="resp-dedupe",
        model="gpt-4o-mini",
        choices=[
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    try:
        logging_obj.stream = True
        logging_obj.model_call_details["litellm_params"] = {"acompletion": True}

        with (
            patch.object(
                mock_callback, "async_log_success_event", new_callable=AsyncMock
            ) as mock_async_log,
            patch.object(mock_callback, "log_success_event") as mock_sync_log,
            patch.object(
                logging_obj,
                "_success_handler_helper_fn",
                return_value=(time.time(), time.time(), result),
            ),
            patch.object(
                logging_obj,
                "_get_assembled_streaming_response",
                return_value=result,
            ),
            patch.object(
                logging_obj,
                "_should_run_sync_callbacks_for_async_calls",
                return_value=True,
            ),
        ):
            await logging_obj.dispatch_success_handlers(result=result)
            await logging_obj.dispatch_success_handlers(result=result)

        mock_async_log.assert_awaited_once()
        mock_sync_log.assert_not_called()
    finally:
        litellm._async_success_callback = original_async_callbacks


@pytest.mark.asyncio
async def test_dispatch_success_handlers_sync_path_invokes_callback_once_for_final_stream(
    logging_obj,
):
    """Sync dispatch path must also dedupe when dispatch is called twice."""
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class MockCallback(CustomLogger):
        pass

    mock_callback = MockCallback()
    original_success_callbacks = list(litellm.success_callback or [])
    litellm.success_callback = [mock_callback]

    result = ModelResponse(
        id="resp-sync-dedupe",
        model="gpt-4o-mini",
        choices=[
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    try:
        logging_obj.stream = True
        logging_obj.model_call_details["litellm_params"] = {}

        with (
            patch.object(mock_callback, "log_success_event") as mock_sync_log,
            patch.object(
                mock_callback, "async_log_success_event", new_callable=AsyncMock
            ) as mock_async_log,
            patch.object(
                logging_obj,
                "_success_handler_helper_fn",
                return_value=(time.time(), time.time(), result),
            ),
            patch.object(
                logging_obj,
                "_get_assembled_streaming_response",
                return_value=result,
            ),
        ):
            await logging_obj.dispatch_success_handlers(result=result)
            await logging_obj.dispatch_success_handlers(result=result)

        mock_sync_log.assert_called_once()
        mock_async_log.assert_not_awaited()
    finally:
        litellm.success_callback = original_success_callbacks


@pytest.mark.asyncio
async def test_dispatch_prefer_async_handlers_runs_legacy_callbacks(
    logging_obj,
):
    """``prefer_async_handlers`` must not skip executor.submit for string callbacks."""
    result = ModelResponse(
        id="resp-prefer-async",
        model="gpt-4o-mini",
        choices=[
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    )

    logging_obj.stream = True
    logging_obj.model_call_details["litellm_params"] = {}

    with (
        patch.object(
            logging_obj, "async_success_handler", new_callable=AsyncMock
        ) as mock_async,
        patch.object(
            logging_obj, "success_handler", new_callable=MagicMock
        ) as mock_sync,
        patch.object(
            logging_obj,
            "_should_run_sync_callbacks_for_async_calls",
            return_value=True,
        ),
        patch(
            "litellm.litellm_core_utils.litellm_logging.executor.submit"
        ) as mock_submit,
    ):
        await logging_obj.dispatch_success_handlers(
            result=result,
            prefer_async_handlers=True,
        )

    mock_async.assert_awaited_once()
    mock_sync.assert_not_called()
    mock_submit.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_success_handlers_invokes_async_callback_for_pass_through(
    logging_obj,
):
    """Pass-through must use async_success_handler (CustomLogger skips sync success_handler)."""
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.types.utils import CallTypes

    class MockCallback(CustomLogger):
        pass

    mock_callback = MockCallback()
    original_async_callbacks = list(litellm._async_success_callback or [])
    litellm._async_success_callback = [mock_callback]

    logging_obj.call_type = CallTypes.pass_through.value
    logging_obj.stream = False
    logging_obj.model_call_details["litellm_params"] = {}

    try:
        with (
            patch.object(
                mock_callback, "async_log_success_event", new_callable=AsyncMock
            ) as mock_async_log,
            patch.object(mock_callback, "log_success_event") as mock_sync_log,
        ):
            await logging_obj.dispatch_success_handlers(result={"id": "pt-1"})

        mock_async_log.assert_awaited_once()
        mock_sync_log.assert_not_called()
    finally:
        litellm._async_success_callback = original_async_callbacks


def test_success_handler_skips_guardrail_logging_hook_when_disabled(logging_obj):
    """Ensure CustomGuardrail logging_hook is skipped when should_run_guardrail is False."""
    import datetime

    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.types.guardrails import GuardrailEventHooks

    class DummyGuardrail(CustomGuardrail):
        pass

    class DummyLogger(CustomLogger):
        pass

    logging_obj.stream = False

    model_response = ModelResponse(
        id="resp-guardrail-skip",
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

    guardrail = DummyGuardrail(
        guardrail_name="dummy-guardrail",
        event_hook=GuardrailEventHooks.logging_only,
    )
    guardrail.should_run_guardrail = MagicMock(return_value=False)
    guardrail.logging_hook = MagicMock(
        return_value=(logging_obj.model_call_details, model_response)
    )

    dummy_logger = DummyLogger()
    dummy_logger.logging_hook = MagicMock(
        return_value=(logging_obj.model_call_details, model_response)
    )

    with patch.object(
        logging_obj,
        "get_combined_callback_list",
        return_value=[guardrail, dummy_logger],
    ):
        logging_obj.success_handler(
            result=model_response,
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
            cache_hit=False,
        )

    guardrail.should_run_guardrail.assert_called_once()
    guardrail_call_kwargs = guardrail.should_run_guardrail.call_args.kwargs
    assert guardrail_call_kwargs["event_type"] == GuardrailEventHooks.logging_only
    guardrail.logging_hook.assert_not_called()
    dummy_logger.logging_hook.assert_called_once()


def test_success_handler_runs_guardrail_logging_hook_when_enabled(logging_obj):
    """Ensure CustomGuardrail logging_hook runs when should_run_guardrail is True."""
    import datetime

    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

    class DummyGuardrail(CustomGuardrail):
        pass

    logging_obj.stream = False

    model_response = ModelResponse(
        id="resp-guardrail-run",
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

    guardrail = DummyGuardrail(
        guardrail_name="dummy-guardrail",
        event_hook=GuardrailEventHooks.logging_only,
    )
    guardrail.should_run_guardrail = MagicMock(return_value=True)

    def _guardrail_logging_hook(kwargs, result, call_type):
        updated_kwargs = dict(kwargs)
        updated_kwargs["guardrail_hook_ran"] = True
        return updated_kwargs, result

    guardrail.logging_hook = MagicMock(side_effect=_guardrail_logging_hook)

    with patch.object(
        logging_obj,
        "get_combined_callback_list",
        return_value=[guardrail],
    ):
        logging_obj.success_handler(
            result=model_response,
            start_time=datetime.datetime.now(),
            end_time=datetime.datetime.now(),
            cache_hit=False,
        )

    guardrail.should_run_guardrail.assert_called_once()
    guardrail_call_kwargs = guardrail.should_run_guardrail.call_args.kwargs
    assert guardrail_call_kwargs["event_type"] == GuardrailEventHooks.logging_only
    guardrail.logging_hook.assert_called_once()
    assert logging_obj.model_call_details.get("guardrail_hook_ran") is True


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
    assert original_tags == [
        "custom-tag-1",
        "custom-tag-2",
    ], f"Original tags list was mutated: {original_tags}"
    assert metadata["tags"] == [
        "custom-tag-1",
        "custom-tag-2",
    ], f"metadata['tags'] was mutated: {metadata['tags']}"

    # Verify each returned list has exactly 2 User-Agent tags (not duplicated)
    user_agent_count_1 = len([t for t in tags1 if t.startswith("User-Agent:")])
    user_agent_count_2 = len([t for t in tags2 if t.startswith("User-Agent:")])
    user_agent_count_3 = len([t for t in tags3 if t.startswith("User-Agent:")])

    assert (
        user_agent_count_1 == 2
    ), f"Expected 2 User-Agent tags, got {user_agent_count_1}"
    assert (
        user_agent_count_2 == 2
    ), f"Expected 2 User-Agent tags, got {user_agent_count_2}"
    assert (
        user_agent_count_3 == 2
    ), f"Expected 2 User-Agent tags, got {user_agent_count_3}"

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


def test_response_cost_calculator_native_generate_content_body_uses_usage_metadata():
    """
    Regression for LIT-4076: a native Google :generateContent body reports tokens
    under ``usageMetadata`` rather than ``usage``, so the cost calculator read 0
    tokens and returned 0.0 synchronously. The calculator now transforms the native
    body (as the async logging path does) so the cost is the real non-zero amount.
    """
    from litellm.types.llms.vertex_ai import GenerateContentResponseBody
    from litellm.types.utils import ModelResponse, Usage

    logging_obj = LitellmLogging(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hey"}],
        stream=False,
        call_type="agenerate_content",
        start_time=time.time(),
        litellm_call_id="lit4076",
        function_id="lit4076",
    )
    logging_obj.model_call_details["custom_llm_provider"] = "gemini"
    logging_obj.optional_params = {}

    native_body = GenerateContentResponseBody(
        candidates=[{"content": {"parts": [{"text": "hi"}], "role": "model"}, "finishReason": "STOP"}],
        usageMetadata={
            "promptTokenCount": 1000,
            "candidatesTokenCount": 500,
            "totalTokenCount": 1500,
        },
    )

    expected_cost = litellm.completion_cost(
        completion_response=ModelResponse(
            model="gemini-2.5-flash",
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        ),
        model="gemini-2.5-flash",
        custom_llm_provider="gemini",
    )
    assert expected_cost > 0

    cost = logging_obj._response_cost_calculator(result=native_body)
    assert cost == pytest.approx(expected_cost)


def test_response_cost_calculator_does_not_transform_non_generate_content_dict():
    """The native-body transform must only run for generate_content call types, so a
    plain dict on a chat completion call is left untouched (no spurious Gemini cost)."""
    logging_obj = LitellmLogging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hey"}],
        stream=False,
        call_type="acompletion",
        start_time=time.time(),
        litellm_call_id="lit4076-2",
        function_id="lit4076-2",
    )
    logging_obj.optional_params = {}

    cost = logging_obj._response_cost_calculator(
        result={"usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 500}}
    )
    assert not cost


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

    with (
        patch("litellm.cold_storage_custom_logger", return_value="s3"),
        patch("litellm.integrations.s3.get_s3_object_key") as mock_get_s3_key,
    ):
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
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create test data
    start_time = datetime(2025, 1, 15, 10, 30, 45, 123456, timezone.utc)
    response_id = "chatcmpl-test-12345"

    # Create mock custom logger with s3_path
    mock_custom_logger = MagicMock()
    mock_custom_logger.s3_path = "storage"

    with (
        patch("litellm.cold_storage_custom_logger", "s3_v2"),
        patch(
            "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name"
        ) as mock_get_logger,
        patch("litellm.integrations.s3.get_s3_object_key") as mock_get_s3_key,
    ):
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
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Create test data
    start_time = datetime(2025, 1, 15, 10, 30, 45, 123456, timezone.utc)
    response_id = "chatcmpl-test-12345"

    # Create mock custom logger without s3_path
    mock_custom_logger = MagicMock()
    mock_custom_logger.s3_path = None  # or could be missing attribute

    with (
        patch("litellm.cold_storage_custom_logger", "s3_v2"),
        patch(
            "litellm.logging_callback_manager.get_active_custom_logger_for_callback_name"
        ) as mock_get_logger,
        patch("litellm.integrations.s3.get_s3_object_key") as mock_get_s3_key,
    ):
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


def test_get_usage_as_dict():
    """
    Test get_usage_as_dict returns usage as plain dict from response_obj or combined_usage_object.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
    from litellm.types.utils import Usage

    # Test case 1: None response_obj returns empty usage dict
    result = StandardLoggingPayloadSetup.get_usage_as_dict(response_obj=None)
    assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Test case 2: Empty response_obj returns empty usage dict
    result = StandardLoggingPayloadSetup.get_usage_as_dict(response_obj={})
    assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Test case 3: combined_usage_object takes priority
    combined = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    result = StandardLoggingPayloadSetup.get_usage_as_dict(
        response_obj={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        combined_usage_object=combined,
    )
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 5
    assert result["total_tokens"] == 15

    # Test case 4: response_obj with usage dict
    result = StandardLoggingPayloadSetup.get_usage_as_dict(
        response_obj={"usage": {"prompt_tokens": 20, "completion_tokens": 30}}
    )
    assert result == {"prompt_tokens": 20, "completion_tokens": 30}

    # Test case 5: response_obj with no usage key returns empty
    result = StandardLoggingPayloadSetup.get_usage_as_dict(
        response_obj={"id": "resp-1", "choices": []}
    )
    assert result == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


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
    assert (
        logging_obj.model_call_details["standard_logging_object"] is not None
    ), "standard_logging_object should not be None for pass-through endpoints"

    # Verify that async_complete_streaming_response was set to prevent re-processing
    # This is consistent with the existing code pattern for regular streaming
    assert "async_complete_streaming_response" in logging_obj.model_call_details, (
        "async_complete_streaming_response should be set to prevent re-processing, "
        "consistent with the existing code pattern"
    )
    assert (
        logging_obj.model_call_details["async_complete_streaming_response"] is result
    ), "async_complete_streaming_response should be set to the result"

    # Verify that response_cost is set to None (cost calculation not possible for pass-through)
    # This is consistent with the error handling in the non-pass-through code path
    assert (
        "response_cost" in logging_obj.model_call_details
    ), "response_cost should be set for pass-through endpoints"
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
    first_standard_logging_object = logging_obj.model_call_details[
        "standard_logging_object"
    ]

    # Second call - should return early due to async_complete_streaming_response guard
    with patch.object(
        logging_obj, "get_combined_callback_list", return_value=[]
    ) as mock_callbacks:
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )
        # The guard should cause early return, so get_combined_callback_list should not be called
        mock_callbacks.assert_not_called()

    # Verify standard_logging_object wasn't modified by second call
    assert (
        logging_obj.model_call_details["standard_logging_object"]
        is first_standard_logging_object
    ), "standard_logging_object should not be modified on re-processing"


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
    assert (
        logging_obj.model_call_details["standard_logging_object"] is not None
    ), "standard_logging_object should not be None for streaming pass-through endpoints"


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


def test_get_error_information_prefers_message_attribute_over_str():
    """
    Regression for empty-error_message-in-spend-logs.

    ProxyException sets `self.message` but does NOT call
    `super().__init__(message)` nor define `__str__`, so `str(exc)`
    returns the empty string. Before the fix, get_error_information
    used `str(original_exception)` and silently stripped the
    human-readable message from spend_logs.metadata.error_information,
    making dashboard "LLM Failure" rows un-triagable.

    Asserts the `.message` attribute is consulted first.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    # Simulate a ProxyException-shaped exception: .message set, but
    # super().__init__() NOT called and no __str__ override.
    class ProxyExceptionLike(Exception):
        def __init__(self, message, code):
            self.message = str(message)
            self.code = str(code)
            # NOTE: deliberately NOT calling super().__init__(message)

    msg = "Authentication Error, Invalid proxy server token passed. key=..."
    exc = ProxyExceptionLike(message=msg, code=401)

    # Sanity check: this exception type's str() really is empty
    assert str(exc) == "", (
        "Test premise broken — bare-base Exception now returns message; "
        "review whether ProxyException fix landed at the class level instead"
    )

    result = StandardLoggingPayloadSetup.get_error_information(exc)
    assert (
        result["error_message"] == msg
    ), f"expected message from .message attribute, got {result['error_message']!r}"
    assert result["error_code"] == "401"
    assert result["error_class"] == "ProxyExceptionLike"


def test_get_error_information_preserves_explicit_empty_message():
    """
    An exception that deliberately sets `.message = ""` must surface
    the empty string verbatim, not fall through to `str(exc)`.

    Regression for greptile P2 finding on PR #30381: a truthiness
    check (`if message_attr:`) would silently mask an explicit empty
    message and substitute `str(original_exception)` — which for
    ProxyException-shaped objects is also empty, but for plain
    `Exception("boom")` would inject the wrong string and corrupt
    the error_information signal.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    class ProxyExceptionLike(Exception):
        def __init__(self, message, code):
            self.message = message
            self.code = str(code)
            super().__init__("unrelated-args-summary")

    exc = ProxyExceptionLike(message="", code=500)
    result = StandardLoggingPayloadSetup.get_error_information(exc)
    assert result["error_message"] == "", (
        "explicit empty .message must survive verbatim; got "
        f"{result['error_message']!r}"
    )


def test_get_error_information_falls_back_to_str_when_no_message_attr():
    """
    Plain Exception (no `.message` attr) must still produce a useful
    error_message via str(exc), preserving prior behavior for
    non-litellm exception types.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    exc = ValueError("boom")
    result = StandardLoggingPayloadSetup.get_error_information(exc)
    assert result["error_message"] == "boom"
    assert result["error_class"] == "ValueError"


# ──────────────────────────────────────────────────────────────────────
# Tests for _get_assembled_streaming_response non-streaming early return
# ──────────────────────────────────────────────────────────────────────


def _make_logging_obj(stream: bool) -> LitellmLogging:
    return LitellmLogging(
        model="openai/codex-mini-latest",
        messages=[{"role": "user", "content": "Hey"}],
        stream=stream,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="test-123",
        function_id="test-fn",
    )


def test_get_assembled_streaming_response_returns_none_for_non_streaming():
    """Non-streaming requests should return None so the streaming block is skipped."""
    import datetime

    logging_obj = _make_logging_obj(stream=False)
    result = ModelResponse(id="resp-1", choices=[], model="test")
    assembled = logging_obj._get_assembled_streaming_response(
        result=result,
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now(),
        is_async=True,
        streaming_chunks=[],
    )
    assert assembled is None


def test_get_assembled_streaming_response_returns_result_for_streaming():
    """Streaming requests should return the ModelResponse for further processing."""
    import datetime

    logging_obj = _make_logging_obj(stream=True)
    result = ModelResponse(id="resp-1", choices=[], model="test")
    assembled = logging_obj._get_assembled_streaming_response(
        result=result,
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now(),
        is_async=True,
        streaming_chunks=[],
    )
    assert assembled is result


def test_streaming_success_handler_includes_vertex_ai_metadata_in_standard_logging():
    """Assembled streaming responses should include Vertex AI metadata in logging payload."""
    import datetime

    from litellm.types.utils import Choices, Message

    logging_obj = _make_logging_obj(stream=True)
    grounding_metadata = [{"webSearchQueries": ["weather in SF"]}]
    url_context_metadata = [{"urlMetadata": [{"retrievedUrl": "https://example.com"}]}]
    result = ModelResponse(
        id="resp-1",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        model="gemini-2.5-flash",
    )
    setattr(result, "vertex_ai_grounding_metadata", grounding_metadata)
    setattr(result, "vertex_ai_url_context_metadata", url_context_metadata)
    result._hidden_params["vertex_ai_grounding_metadata"] = grounding_metadata
    result._hidden_params["vertex_ai_url_context_metadata"] = url_context_metadata

    start = datetime.datetime.now()
    end = datetime.datetime.now()
    logging_obj.success_handler(result=result, start_time=start, end_time=end)

    payload = logging_obj.model_call_details.get("standard_logging_object")
    assert payload is not None
    assert payload["response"]["vertex_ai_grounding_metadata"] == grounding_metadata
    assert payload["response"]["vertex_ai_url_context_metadata"] == url_context_metadata


def test_get_assembled_streaming_response_returns_none_for_non_streaming_text_completion():
    """Non-streaming TextCompletionResponse should also return None."""
    import datetime

    logging_obj = _make_logging_obj(stream=False)
    result = TextCompletionResponse(id="resp-1", choices=[], model="test")
    assembled = logging_obj._get_assembled_streaming_response(
        result=result,
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now(),
        is_async=True,
        streaming_chunks=[],
    )
    assert assembled is None


@pytest.mark.asyncio
async def test_non_streaming_computes_standard_logging_object_once():
    """
    Non-streaming acompletion should call get_standard_logging_object_payload
    exactly once, not twice.
    """
    import asyncio

    import litellm

    with patch.object(
        litellm.litellm_core_utils.litellm_logging,
        "get_standard_logging_object_payload",
    ) as mock_payload:
        await litellm.acompletion(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hey"}],
            model="openai/codex-mini-latest",
            mock_response="Hello, world!",
        )
        await asyncio.sleep(1)
        assert mock_payload.call_count == 1


@pytest.mark.asyncio
async def test_emit_standard_logging_payload_called_for_non_streaming():
    """
    emit_standard_logging_payload should still be called for non-streaming
    requests (moved from the streaming block to _process_hidden_params_and_response_cost).
    """
    import asyncio

    import litellm

    with patch.object(
        litellm.litellm_core_utils.litellm_logging,
        "emit_standard_logging_payload",
    ) as mock_emit:
        await litellm.acompletion(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hey"}],
            model="openai/codex-mini-latest",
            mock_response="Hello, world!",
        )
        await asyncio.sleep(1)
        assert mock_emit.call_count >= 1


@pytest.mark.asyncio
async def test_async_success_handler_preserves_response_cost_for_pass_through_endpoints():
    """Regression test: PR #19887 added a pass-through branch in async_success_handler
    that unconditionally set response_cost=None, overwriting costs already calculated
    by pass-through handlers (Gemini/Vertex)."""
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse, Usage

    logging_obj = LiteLLMLoggingObj(
        model="gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="test-call-id-cost",
        function_id="test-function-id-cost",
    )

    # Simulate what pass-through handlers do: pre-calculate response_cost
    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {}, "proxy_server_request": {}},
        "litellm_call_id": "test-call-id-cost",
        "response_cost": 0.0000047,  # Pre-calculated by pass-through handler
        "custom_llm_provider": "gemini",
    }

    result = ModelResponse(
        id="test-response",
        model="gemini-2.5-flash-lite",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    start_time = datetime.now()
    end_time = datetime.now()

    with patch.object(logging_obj, "get_combined_callback_list", return_value=[]):
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )

    # response_cost must be preserved, not overwritten to None
    assert logging_obj.model_call_details.get("response_cost") is not None
    assert logging_obj.model_call_details["response_cost"] > 0

    # standard_logging_object should also have the cost
    slo = logging_obj.model_call_details.get("standard_logging_object")
    assert slo is not None
    assert slo["response_cost"] > 0


def test_process_hidden_params_recalculates_cost_after_failure_handler_zero():
    """
    Regression: PR #21844 preserved response_cost=0 set by failure_handler on failed
    router retry attempts, so a later successful response with usage logged $0 spend.
    """
    from datetime import datetime

    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse, Usage

    logging_obj = LiteLLMLoggingObj(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="test-retry-zero-cost",
        function_id="test-retry-zero-cost",
    )
    logging_obj.model_call_details["litellm_params"] = {"model": "openai/gpt-4o-mini"}
    logging_obj.optional_params = {}

    err = litellm.RateLimitError(
        message="rate limit",
        llm_provider="openai",
        model="openai/gpt-4o-mini",
    )
    for _ in range(2):
        logging_obj._failure_handler_helper_fn(
            exception=err,
            traceback_exception="",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
    assert logging_obj.model_call_details.get("response_cost") == 0

    result = ModelResponse(
        id="success",
        choices=[{"message": {"role": "assistant", "content": "ok"}}],
        usage=Usage(prompt_tokens=9698, completion_tokens=30, total_tokens=9728),
    )
    logging_obj._process_hidden_params_and_response_cost(
        result, datetime.now(), datetime.now()
    )

    cost = logging_obj.model_call_details.get("response_cost")
    assert cost is not None and cost > 0
    slo = logging_obj.model_call_details.get("standard_logging_object") or {}
    assert slo.get("response_cost", 0) > 0


def test_process_hidden_params_preserves_zero_cost_in_hidden_params():
    """Pass-through handlers often set response_cost on result._hidden_params (including 0)."""
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse, Usage

    logging_obj = LiteLLMLoggingObj(
        model="gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="test-hidden-zero-cost",
        function_id="test-hidden-zero-cost",
    )
    logging_obj.model_call_details["litellm_params"] = {
        "model": "gemini-2.5-flash-lite"
    }
    logging_obj.optional_params = {}

    result = ModelResponse(
        id="batch-pending",
        choices=[{"message": {"role": "assistant", "content": "pending"}}],
        usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )
    result._hidden_params = {"response_cost": 0.0}

    logging_obj._process_hidden_params_and_response_cost(
        result, datetime.now(), datetime.now()
    )

    assert logging_obj.model_call_details.get("response_cost") == 0.0
    slo = logging_obj.model_call_details.get("standard_logging_object") or {}
    assert slo.get("response_cost") == 0.0


def test_process_hidden_params_uses_hidden_params_cost_after_failure_handler_zero():
    """After retry failures pin model_call_details to 0, success cost on _hidden_params wins."""
    from datetime import datetime

    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse, Usage

    logging_obj = LiteLLMLoggingObj(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="test-retry-hidden-cost",
        function_id="test-retry-hidden-cost",
    )
    logging_obj.model_call_details["litellm_params"] = {"model": "openai/gpt-4o-mini"}
    logging_obj.optional_params = {}

    err = litellm.RateLimitError(
        message="rate limit",
        llm_provider="openai",
        model="openai/gpt-4o-mini",
    )
    for _ in range(2):
        logging_obj._failure_handler_helper_fn(
            exception=err,
            traceback_exception="",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
    assert logging_obj.model_call_details.get("response_cost") == 0

    passthrough_cost = 0.00042
    result = ModelResponse(
        id="success",
        choices=[{"message": {"role": "assistant", "content": "ok"}}],
        usage=Usage(prompt_tokens=9698, completion_tokens=30, total_tokens=9728),
    )
    result._hidden_params = {"response_cost": passthrough_cost}

    logging_obj._process_hidden_params_and_response_cost(
        result, datetime.now(), datetime.now()
    )

    assert logging_obj.model_call_details.get("response_cost") == passthrough_cost
    slo = logging_obj.model_call_details.get("standard_logging_object") or {}
    assert slo.get("response_cost") == passthrough_cost


def test_function_setup_litellm_metadata_populates_metadata():
    """
    Test that function_setup() properly handles litellm_metadata (used by /v1/messages,
    /batches, /responses, /files endpoints) and populates litellm_params["metadata"]
    so callbacks like Langfuse can read API key fields.

    This is the root cause of: Claude Code requests missing user_api_key_hash in Langfuse.
    """
    import litellm

    test_api_key_hash = "sk-hashed-1234567890abcdef"
    test_team_id = "team-test-123"
    test_key_alias = "my-test-key"

    # Simulate what happens for /v1/messages: metadata is in "litellm_metadata", not "metadata"
    kwargs = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "test-call-id-123",
        "litellm_metadata": {
            "user_api_key_hash": test_api_key_hash,
            "user_api_key_alias": test_key_alias,
            "user_api_key_team_id": test_team_id,
            "user_api_key_user_id": "user-123",
            "user_api_key": test_api_key_hash,
        },
    }

    logging_obj, returned_kwargs = litellm.utils.function_setup(
        original_function="anthropic_messages",
        rules_obj=litellm.utils.Rules(),
        start_time=time.time(),
        **kwargs,
    )

    # litellm_params["metadata"] must contain the API key fields
    litellm_params = logging_obj.model_call_details.get("litellm_params", {})
    metadata = litellm_params.get("metadata")
    assert metadata is not None, "litellm_params['metadata'] should not be None"
    assert isinstance(metadata, dict), "litellm_params['metadata'] should be a dict"
    assert metadata.get("user_api_key_hash") == test_api_key_hash
    assert metadata.get("user_api_key_alias") == test_key_alias
    assert metadata.get("user_api_key_team_id") == test_team_id

    # litellm_metadata should also be preserved
    litellm_metadata = litellm_params.get("litellm_metadata")
    assert litellm_metadata is not None
    assert litellm_metadata.get("user_api_key_hash") == test_api_key_hash

    # metadata should be a COPY, not an alias — mutating one must not affect the other
    assert (
        metadata is not litellm_metadata
    ), "litellm_params['metadata'] should be a copy, not the same object"


def test_function_setup_metadata_takes_precedence_over_litellm_metadata():
    """
    Test that when BOTH metadata and litellm_metadata are present (e.g., user sets
    Anthropic API metadata AND proxy adds litellm_metadata), metadata is used as
    litellm_params["metadata"] and litellm_metadata is stored separately.
    """
    import litellm

    kwargs = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "test-call-id-456",
        "metadata": {
            "user_id": "anthropic-user-id",
        },
        "litellm_metadata": {
            "user_api_key_hash": "sk-hashed-xyz",
            "user_api_key_team_id": "team-xyz",
        },
    }

    logging_obj, _ = litellm.utils.function_setup(
        original_function="anthropic_messages",
        rules_obj=litellm.utils.Rules(),
        start_time=time.time(),
        **kwargs,
    )

    litellm_params = logging_obj.model_call_details.get("litellm_params", {})

    # When both are present, metadata should be the explicit "metadata" dict
    metadata = litellm_params.get("metadata")
    assert metadata is not None
    assert metadata.get("user_id") == "anthropic-user-id"

    # litellm_metadata should be preserved separately for merge_litellm_metadata()
    litellm_metadata = litellm_params.get("litellm_metadata")
    assert litellm_metadata is not None
    assert litellm_metadata.get("user_api_key_hash") == "sk-hashed-xyz"


def test_update_from_kwargs_litellm_params_metadata_does_not_overwrite_proxy_fields():
    """
    Test the exact bug: when update_from_kwargs is called with litellm_params
    containing a 'metadata' key (e.g. Anthropic's native metadata with user_id),
    it must NOT overwrite proxy key-auth fields already merged from litellm_metadata.

    This is the anthropic_messages code path where async_anthropic_messages_handler
    passes anthropic_messages_optional_request_params (which includes metadata)
    as litellm_params to update_from_kwargs.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj = Logging(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type="anthropic_messages",
        start_time=time.time(),
        litellm_call_id="test-overwrite-bug",
        function_id="test-function-id",
    )

    kwargs = {
        "litellm_metadata": {
            "user_api_key_hash": "sk-hashed-proxy",
            "user_api_key_alias": "claude-api",
            "user_api_key_team_id": "team-zurich",
        },
    }

    # Simulate what async_anthropic_messages_handler does:
    # passes Anthropic's native metadata in litellm_params
    logging_obj.update_from_kwargs(
        kwargs=kwargs,
        litellm_params={
            "preset_cache_key": None,
            "stream_response": {},
            "metadata": {"user_id": "anthropic-device-id"},  # Anthropic native metadata
        },
    )

    litellm_params = logging_obj.model_call_details.get("litellm_params", {})
    metadata = litellm_params.get("metadata")

    assert metadata is not None
    # Proxy key-auth fields must survive the litellm_params.update()
    assert metadata.get("user_api_key_hash") == "sk-hashed-proxy"
    assert metadata.get("user_api_key_alias") == "claude-api"
    assert metadata.get("user_api_key_team_id") == "team-zurich"
    # Anthropic native metadata must also be present
    assert metadata.get("user_id") == "anthropic-device-id"


def test_function_setup_empty_metadata_falls_back_to_litellm_metadata():
    """
    Test that when metadata is explicitly set to {} (empty dict), litellm_metadata
    is still used to populate litellm_params["metadata"] so API key fields are visible.
    """
    import litellm

    kwargs = {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
        "litellm_call_id": "test-call-id-789",
        "metadata": {},
        "litellm_metadata": {
            "user_api_key_hash": "sk-hashed-empty-test",
            "user_api_key_team_id": "team-empty-test",
        },
    }

    logging_obj, _ = litellm.utils.function_setup(
        original_function="anthropic_messages",
        rules_obj=litellm.utils.Rules(),
        start_time=time.time(),
        **kwargs,
    )

    litellm_params = logging_obj.model_call_details.get("litellm_params", {})
    metadata = litellm_params.get("metadata")
    assert metadata is not None
    assert metadata.get("user_api_key_hash") == "sk-hashed-empty-test"
    assert metadata.get("user_api_key_team_id") == "team-empty-test"


def test_failure_handler_skips_sync_callbacks_for_pass_through_requests(logging_obj):
    """Ensure sync failure callbacks are skipped for pass-through endpoint requests.

    Regression test for duplicate Datadog/Arize logs on pass-through endpoint failures.
    The async_failure_handler fires async_log_failure_event; the sync failure_handler
    must NOT also fire log_failure_event for pass-through requests.
    """
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.types.utils import CallTypes

    class DummyLogger(CustomLogger):
        pass

    logging_obj.call_type = CallTypes.pass_through.value
    logging_obj.stream = False
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.litellm_params = {}

    dummy_logger = DummyLogger()
    dummy_logger.log_failure_event = MagicMock()

    with patch.object(
        logging_obj,
        "get_combined_callback_list",
        return_value=[dummy_logger],
    ):
        logging_obj.failure_handler(
            exception=Exception("test error"),
            traceback_exception="",
        )

    dummy_logger.log_failure_event.assert_not_called()


@pytest.mark.parametrize("call_type", ["completion", "acompletion"])
def test_failure_handler_runs_sync_callbacks_for_non_pass_through_requests(
    logging_obj, call_type
):
    """Ensure sync failure callbacks still fire for normal (non-pass-through) requests."""
    from litellm.integrations.custom_logger import CustomLogger

    class DummyLogger(CustomLogger):
        pass

    logging_obj.call_type = call_type
    logging_obj.stream = False
    logging_obj.model_call_details["litellm_params"] = {}
    logging_obj.litellm_params = {}

    dummy_logger = DummyLogger()
    dummy_logger.log_failure_event = MagicMock()

    with patch.object(
        logging_obj,
        "get_combined_callback_list",
        return_value=[dummy_logger],
    ):
        logging_obj.failure_handler(
            exception=Exception("test error"),
            traceback_exception="",
        )

    dummy_logger.log_failure_event.assert_called_once()


def test_merge_hidden_params_from_response_into_metadata_populates_metadata():
    """Streaming completion path should mirror non-stream: metadata.hidden_params from response."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=time.time(),
        litellm_call_id="merge-hp-test",
        function_id="merge-hp-fn",
    )
    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {}},
    }

    class _Resp:
        _hidden_params = {"response_cost": 0.001, "model_id": "mid-test"}

    logging_obj._merge_hidden_params_from_response_into_metadata(_Resp())
    meta = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert meta["hidden_params"]["response_cost"] == 0.001
    assert meta["hidden_params"]["model_id"] == "mid-test"


def test_merge_hidden_params_from_response_into_metadata_backfills_response_cost():
    """Streaming metadata should include the already-calculated response cost."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=time.time(),
        litellm_call_id="merge-hp-cost-test",
        function_id="merge-hp-cost-fn",
    )
    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {}},
        "response_cost": 0.002,
    }

    class _Resp:
        _hidden_params = {"response_cost": None, "model_id": "mid-test"}

    response = _Resp()
    logging_obj._merge_hidden_params_from_response_into_metadata(response)
    meta = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert meta["hidden_params"]["response_cost"] == 0.002
    assert meta["hidden_params"]["model_id"] == "mid-test"
    assert response._hidden_params["response_cost"] is None


def test_standard_logging_hidden_params_backfills_response_cost_without_mutating_response():
    """Streaming standard logging payload should expose the calculated response cost."""
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import Usage

    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=time.time(),
        litellm_call_id="standard-hp-cost-test",
        function_id="standard-hp-cost-fn",
    )
    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {}, "proxy_server_request": {}},
        "litellm_call_id": "standard-hp-cost-test",
        "call_type": "acompletion",
        "stream": True,
        "model": "gpt-4o-mini",
        "custom_llm_provider": "openai",
        "optional_params": {"stream": True},
        "response_cost": 0.002,
    }
    response = ModelResponse(
        id="standard-hp-cost-response",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    response._hidden_params = {"response_cost": None, "model_id": "mid-test"}

    payload = logging_obj._build_standard_logging_payload(
        response, datetime.now(), datetime.now()
    )

    assert payload is not None
    assert payload["hidden_params"]["response_cost"] == 0.002
    assert response._hidden_params["response_cost"] is None


def test_merge_hidden_params_from_response_into_metadata_preserves_response_cost():
    """Do not overwrite provider-supplied response cost when it already exists."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=time.time(),
        litellm_call_id="merge-hp-preserve-cost-test",
        function_id="merge-hp-preserve-cost-fn",
    )
    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {}},
        "response_cost": 0.002,
    }

    class _Resp:
        _hidden_params = {"response_cost": 0.001, "model_id": "mid-test"}

    logging_obj._merge_hidden_params_from_response_into_metadata(_Resp())
    meta = logging_obj.model_call_details["litellm_params"]["metadata"]
    assert meta["hidden_params"]["response_cost"] == 0.001
    assert meta["hidden_params"]["model_id"] == "mid-test"


def test_merge_hidden_params_from_response_into_metadata_no_op_when_empty():
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=time.time(),
        litellm_call_id="merge-hp-empty",
        function_id="merge-hp-empty-fn",
    )
    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {"existing": True}},
    }

    class _NoHp:
        _hidden_params = {}

    logging_obj._merge_hidden_params_from_response_into_metadata(_NoHp())
    assert (
        "hidden_params"
        not in logging_obj.model_call_details["litellm_params"]["metadata"]
    )


# ── StandardLoggingPayloadSetup.get_additional_headers ───────────────────────


def test_get_additional_headers_preserves_provider_request_id():
    """llm_provider-x-request-id must survive the get_additional_headers filter."""
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    raw = {
        "x-ratelimit-remaining-requests": "29999",
        "x-ratelimit-remaining-tokens": "149999970",
        "llm_provider-x-request-id": "req_85f49b546c7b4d3180755621f36631a1",
        "llm_provider-openai-organization": "my-org",
        "llm_provider-openai-processing-ms": "649",
    }

    result = StandardLoggingPayloadSetup.get_additional_headers(raw)

    assert result is not None
    # well-known fields parsed as ints
    assert result["x_ratelimit_remaining_requests"] == 29999  # type: ignore
    assert result["x_ratelimit_remaining_tokens"] == 149999970  # type: ignore
    # provider-specific headers must be preserved verbatim
    assert result["llm_provider-x-request-id"] == "req_85f49b546c7b4d3180755621f36631a1"  # type: ignore
    assert result["llm_provider-openai-organization"] == "my-org"  # type: ignore
    assert result["llm_provider-openai-processing-ms"] == "649"  # type: ignore


def test_get_additional_headers_returns_none_for_none_input():
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    assert StandardLoggingPayloadSetup.get_additional_headers(None) is None


def test_get_additional_headers_reset_fields_preserved():
    """x-ratelimit-reset-* fields (added to the TypedDict) must be captured."""
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    raw = {
        "x-ratelimit-reset-requests": "1s",
        "x-ratelimit-reset-tokens": "100ms",
    }

    result = StandardLoggingPayloadSetup.get_additional_headers(raw)

    assert result is not None
    assert result["x_ratelimit_reset_requests"] == "1s"  # type: ignore
    assert result["x_ratelimit_reset_tokens"] == "100ms"  # type: ignore


# ── litellm_call_id propagation ───────────────────────────────────────────────


def test_get_standard_logging_object_payload_includes_litellm_call_id(logging_obj):
    """litellm_call_id from kwargs must appear in the returned StandardLoggingPayload."""
    import datetime

    from litellm.litellm_core_utils.litellm_logging import (
        get_standard_logging_object_payload,
    )

    call_id = "test-call-id-abc-123"
    now = datetime.datetime.now()
    payload = get_standard_logging_object_payload(
        kwargs={"litellm_call_id": call_id, "model": "gpt-4o", "messages": []},
        init_response_obj={},
        start_time=now,
        end_time=now,
        logging_obj=logging_obj,
        status="success",
    )

    assert payload is not None
    assert payload["litellm_call_id"] == call_id


def _make_dict_logging_obj():
    """Build a Logging instance configured for a non-streaming dict result."""
    obj = LitellmLogging(
        model="claude-haiku-4-5@20251001",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="test-call-id",
        start_time=time.time(),
        function_id="test-fn",
    )
    obj.model_call_details = {
        "model": "claude-haiku-4-5@20251001",
        "custom_llm_provider": "vertex_ai",
        "litellm_params": {"metadata": {}},
        "response_cost": None,
    }
    return obj


def test_success_handler_computes_cost_for_dict_response():
    """Non-streaming dict responses run through the cost calculator."""
    logging_obj = _make_dict_logging_obj()
    expected_cost = 0.42
    with (
        patch.object(
            logging_obj,
            "_response_cost_calculator",
            return_value=expected_cost,
        ) as mock_calc,
        patch.object(
            logging_obj,
            "_build_standard_logging_payload",
            return_value={"response_cost": expected_cost},
        ),
        patch(
            "litellm.litellm_core_utils.litellm_logging.emit_standard_logging_payload"
        ),
        patch.object(
            logging_obj,
            "_is_recognized_call_type_for_logging",
            return_value=False,
        ),
        patch.object(
            logging_obj,
            "_transform_usage_objects",
            side_effect=lambda result: result,
        ),
    ):
        logging_obj.success_handler(
            result={"id": "msg_1"},
            start_time=time.time(),
            end_time=time.time(),
        )
        mock_calc.assert_called_once()
        assert logging_obj.model_call_details["response_cost"] == expected_cost


def test_success_handler_preserves_precomputed_cost_for_dict_response():
    """Precomputed response_cost on model_call_details must not be overwritten."""
    logging_obj = _make_dict_logging_obj()
    precomputed_cost = 1.23
    logging_obj.model_call_details["response_cost"] = precomputed_cost
    with (
        patch.object(
            logging_obj,
            "_response_cost_calculator",
            return_value=9.99,
        ) as mock_calc,
        patch.object(
            logging_obj,
            "_build_standard_logging_payload",
            return_value={"response_cost": precomputed_cost},
        ),
        patch(
            "litellm.litellm_core_utils.litellm_logging.emit_standard_logging_payload"
        ),
        patch.object(
            logging_obj,
            "_is_recognized_call_type_for_logging",
            return_value=False,
        ),
        patch.object(
            logging_obj,
            "_transform_usage_objects",
            side_effect=lambda result: result,
        ),
    ):
        logging_obj.success_handler(
            result={"id": "msg_2"},
            start_time=time.time(),
            end_time=time.time(),
        )
        mock_calc.assert_not_called()
        assert logging_obj.model_call_details["response_cost"] == precomputed_cost


def test_success_handler_unified_helper_runs_for_typed_results():
    """Recognized typed responses still flow through the unified helper."""
    logging_obj = _make_dict_logging_obj()
    expected_cost = 0.10
    typed_result = MagicMock()
    typed_result._hidden_params = {}

    with (
        patch.object(
            logging_obj,
            "_response_cost_calculator",
            return_value=expected_cost,
        ) as mock_calc,
        patch.object(
            logging_obj,
            "_build_standard_logging_payload",
            return_value={"response_cost": expected_cost},
        ),
        patch(
            "litellm.litellm_core_utils.litellm_logging.emit_standard_logging_payload"
        ),
        patch.object(
            logging_obj,
            "_is_recognized_call_type_for_logging",
            return_value=True,
        ),
        patch.object(
            logging_obj,
            "_transform_usage_objects",
            side_effect=lambda result: result,
        ),
    ):
        logging_obj.success_handler(
            result=typed_result,
            start_time=time.time(),
            end_time=time.time(),
        )
        mock_calc.assert_called_once()
        assert logging_obj.model_call_details["response_cost"] == expected_cost


class TestFirstApiCallStartTimeSetOnce:
    """first_api_call_start_time pins the FIRST provider handoff so
    preprocessing latency excludes retries/backoff (api_call_start_time is
    overwritten on every attempt). It is set ONLY on the logging object's
    model_call_details. It must never be written into
    litellm_params["metadata"] — that is the caller's request metadata,
    echoed back into provider request bodies, spend logs, and batch
    objects (typed Dict[str, str]); a datetime there breaks them. The
    proxy failure path lifts it off the logging object into request_data
    separately (see proxy/utils.py), not via this dict.
    """

    def _logging_obj(self):
        obj = LitellmLogging(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="set-once-1",
            function_id="f1",
        )
        obj.model_call_details["litellm_params"] = {"metadata": {}}
        return obj

    def test_set_once_survives_retry_and_never_touches_user_metadata(self):
        obj = self._logging_obj()
        user_meta = obj.model_call_details["litellm_params"]["metadata"]

        obj.pre_call(input="hi", api_key="sk-test")
        first = obj.model_call_details["first_api_call_start_time"]
        assert first == obj.model_call_details["api_call_start_time"]
        # Set on the logging object only — user metadata untouched.
        assert user_meta == {}
        assert (
            "first_api_call_start_time" not in obj.model_call_details["litellm_params"]
        )

        time.sleep(0.002)  # ensure a distinct retry timestamp
        obj.pre_call(input="hi", api_key="sk-test")

        # retry advanced api_call_start_time but NOT first_api_call_start_time
        assert obj.model_call_details["api_call_start_time"] > first
        assert obj.model_call_details["first_api_call_start_time"] == first
        assert user_meta == {}


def test_get_error_information_for_logging_payload_ignores_spoofed_disconnect_without_flag():
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    baseline = StandardLoggingPayloadSetup.get_error_information(
        original_exception=ValueError("provider failure"),
    )
    error_information, error_str = (
        StandardLoggingPayloadSetup.get_error_information_for_logging_payload(
            metadata={
                "error_information": {
                    "error_code": "499",
                    "error_message": "Client disconnected the request",
                    "error_class": "ClientDisconnected",
                }
            },
            original_exception=ValueError("provider failure"),
            error_str="provider failure",
        )
    )
    assert error_information == baseline
    assert error_str == "provider failure"


def test_get_error_information_for_logging_payload_client_disconnect():
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    custom_error = {
        "error_code": "499",
        "error_message": "Client disconnected the request",
        "error_class": "ClientDisconnected",
    }
    error_information, error_str = (
        StandardLoggingPayloadSetup.get_error_information_for_logging_payload(
            metadata={"client_disconnected": True, "error_information": custom_error},
            original_exception=None,
            error_str=None,
        )
    )
    assert error_information == custom_error
    assert error_str == "Client disconnected the request"

    error_information, error_str = (
        StandardLoggingPayloadSetup.get_error_information_for_logging_payload(
            metadata={"client_disconnected": True},
            original_exception=None,
            error_str="existing error",
        )
    )
    assert error_information["error_code"] == "499"
    assert error_str == "existing error"

    baseline = StandardLoggingPayloadSetup.get_error_information(
        original_exception=None,
    )
    error_information, error_str = (
        StandardLoggingPayloadSetup.get_error_information_for_logging_payload(
            metadata={},
            original_exception=None,
            error_str=None,
        )
    )
    assert error_information == baseline
    assert error_str is None


def test_get_error_information_proxy_exception_preserves_message():
    """ProxyException keeps its text in ``.message`` (str() was empty pre-fix),
    so error_information must still surface the message and code."""
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
    from litellm.proxy._types import ProxyException

    msg = "Authentication Error, Invalid proxy server token passed."
    exc = ProxyException(message=msg, type="auth_error", param="key", code=401)

    info = StandardLoggingPayloadSetup.get_error_information(original_exception=exc)
    assert info["error_message"] == msg
    assert info["error_class"] == "ProxyException"
    assert info["error_code"] == "401"


def test_get_error_information_prefers_message_attribute_over_empty_str():
    """error_message must come from a populated ``.message`` even when the
    exception's __str__ is empty — guards classes that store the text on
    ``.message`` without forwarding it to ``Exception.__init__``."""
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    class _SilentExc(Exception):
        def __init__(self):
            self.message = "real failure detail"
            self.code = 401

        def __str__(self):
            return ""

    info = StandardLoggingPayloadSetup.get_error_information(
        original_exception=_SilentExc()
    )
    assert info["error_message"] == "real failure detail"
    assert info["error_code"] == "401"


def _anthropic_messages_logging_obj():
    return LitellmLogging(
        model="openai/my-local",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=time.time(),
        litellm_call_id="28595",
        function_id="28595",
    )


def _responses_api_response_with_text(text="hello world"):
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

    return ResponsesAPIResponse(
        id="resp-28595",
        created_at=1700000000,
        output=[
            ResponseOutputMessage(
                id="msg-1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(annotations=[], text=text, type="output_text")
                ],
            )
        ],
        usage=ResponseAPIUsage(input_tokens=11, output_tokens=7, total_tokens=18),
    )


@pytest.mark.parametrize(
    "event_cls, event_type",
    [
        ("ResponseCompletedEvent", "response.completed"),
        ("ResponseIncompleteEvent", "response.incomplete"),
        ("ResponseFailedEvent", "response.failed"),
    ],
)
def test_handle_anthropic_messages_response_logging_translates_terminal_responses_api_event(
    event_cls, event_type
):
    """Regression for #28595 / #28943. When anthropic_messages routes to the OpenAI
    Responses backend and stream=True, success_handler receives a terminal Responses
    API event. The handler must translate it to a ModelResponse whose choices carry
    the assistant text, so the proxy UI Logs tab (which reads response.choices[0])
    renders the response content instead of "No response data available"."""
    import importlib

    openai_types = importlib.import_module("litellm.types.llms.openai")
    EventClass = getattr(openai_types, event_cls)

    logging_obj = _anthropic_messages_logging_obj()
    inner_response = _responses_api_response_with_text("hello world")
    event = EventClass(type=event_type, response=inner_response)

    result = logging_obj._handle_anthropic_messages_response_logging(result=event)

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hello world"  # type: ignore[union-attr]
    assert result.usage.prompt_tokens == 11  # type: ignore[attr-defined]
    assert result.usage.completion_tokens == 7  # type: ignore[attr-defined]


def test_handle_anthropic_messages_response_logging_translates_bare_responses_api_response():
    """Non-streaming bridge path: result is a bare ResponsesAPIResponse (no event wrap)."""
    logging_obj = _anthropic_messages_logging_obj()
    result = logging_obj._handle_anthropic_messages_response_logging(
        result=_responses_api_response_with_text("hi there")
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hi there"  # type: ignore[union-attr]
    assert result.usage.total_tokens == 18  # type: ignore[attr-defined]


def test_handle_anthropic_messages_response_logging_passes_model_response_through():
    """Anthropic-native path already yields a ModelResponse; it must be returned unchanged."""
    logging_obj = _anthropic_messages_logging_obj()
    model_response = ModelResponse()
    assert (
        logging_obj._handle_anthropic_messages_response_logging(result=model_response)
        is model_response
    )


def test_handle_anthropic_messages_response_logging_degrades_on_unparseable_responses_payload():
    """If the Responses translation raises (eg. empty output on an incomplete response),
    the row must still land: a minimal ModelResponse with model + usage is returned."""
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

    logging_obj = _anthropic_messages_logging_obj()
    empty = ResponsesAPIResponse(
        id="resp-empty",
        created_at=1700000000,
        output=[],
        usage=ResponseAPIUsage(input_tokens=4, output_tokens=0, total_tokens=4),
    )

    result = logging_obj._handle_anthropic_messages_response_logging(result=empty)

    assert isinstance(result, ModelResponse)
    assert result.model == "openai/my-local"
    assert result.usage.prompt_tokens == 4  # type: ignore[attr-defined]


class _SuccessCapturingLogger(CustomLogger):
    """Records the success payload. success_payload is populated only in
    async_log_success_event, so it stays None when the buggy no-op
    async_log_stream_event path runs for streaming."""

    def __init__(self):
        super().__init__()
        self.success_payload = None
        self.success_calls = 0
        self.stream_event_calls = 0

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.success_calls += 1
        self.success_payload = kwargs.get("standard_logging_object")

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        self.stream_event_calls += 1


def _responses_stream_sse_bytes():
    """A full Responses stream: an opened message item, two text deltas, then the
    terminal response.completed carrying usage. Exercises mid-stream delta handling
    in addition to end-of-stream success logging."""
    import json

    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg-1",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello ",
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "world",
            "sequence_number": 3,
        },
        {
            "type": "response.completed",
            "sequence_number": 4,
            "response": _responses_api_response_with_text("hello world").model_dump(),
        },
    ]
    return [f"data: {json.dumps(e)}\n\n".encode("utf-8") for e in events]


def _fake_streaming_responses_http_response():
    sse_chunks = _responses_stream_sse_bytes()

    async def aiter_bytes(*args, **kwargs):
        for chunk in sse_chunks:
            yield chunk

    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.aiter_bytes = aiter_bytes
    return resp


def _chunk_text(chunk):
    if isinstance(chunk, (bytes, bytearray)):
        return chunk.decode("utf-8", "ignore")
    return str(chunk)


async def _drain_until_logged(logger, max_iter=30):
    for _ in range(max_iter):
        if logger.success_payload is not None:
            break
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_streaming_anthropic_messages_openai_bridge_fires_success_logging(
    monkeypatch,
):
    """Regression for #28595 / #28943. The existing tests above call
    _handle_anthropic_messages_response_logging directly; they do not cover the
    streaming wiring that originally broke. Drive a real streaming
    anthropic_messages call routed to the OpenAI Responses backend (upstream SSE
    mocked) and assert the bridge surfaces delta chunks and fires success logging
    exactly once with real cost. On the broken version the stream ran but only the
    no-op async_log_stream_event was called, so success_payload stayed None and the
    SpendLogs row never landed."""
    logger = _SuccessCapturingLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])

    chunks = []
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new=AsyncMock(return_value=_fake_streaming_responses_http_response()),
    ):
        stream = await litellm.anthropic_messages(
            model="openai/gpt-4o",
            api_key="sk-test-28595",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
            stream=True,
        )
        async for chunk in stream:  # logging fires on stream end; must drain fully
            chunks.append(chunk)

    await _drain_until_logged(logger)

    assert chunks, "stream yielded no chunks"
    assert any("content_block_delta" in _chunk_text(c) for c in chunks), (
        "no delta chunks surfaced; the streaming text deltas were not forwarded"
    )
    assert logger.success_payload is not None, (
        "async_log_success_event never fired for streaming /v1/messages -> openai "
        "Responses bridge; the no-op stream path dropped the spend row"
    )
    assert logger.success_calls == 1, "bridge call must log success exactly once"
    assert logger.success_payload["response_cost"] > 0
    assert logger.success_payload["call_type"] == "anthropic_messages"


def test_failure_handler_records_recovered_partial_spend(logging_obj):
    """A stream interrupted mid-flight still billed the provider for the chunks
    already delivered. When the router stashes that recovered usage as
    ``combined_usage_object`` and pre-computes ``response_cost``, the failure
    handler must preserve them so the failure row carries the real partial
    spend instead of zero.
    """
    from litellm.types.utils import Usage

    logging_obj.model_call_details["combined_usage_object"] = Usage(
        prompt_tokens=17, completion_tokens=9, total_tokens=26
    )
    logging_obj.model_call_details["response_cost"] = 0.00012

    logging_obj._failure_handler_helper_fn(
        exception=Exception("Connection lost"),
        traceback_exception="Traceback ...",
    )

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["status"] == "failure"
    assert payload["response_cost"] == 0.00012
    assert payload["prompt_tokens"] == 17
    assert payload["completion_tokens"] == 9
    assert payload["total_tokens"] == 26


def test_failure_handler_zeroes_spend_without_recovered_usage(logging_obj):
    """A failure with no recovered partial usage keeps the existing behavior of
    recording zero spend, so the partial-spend preservation does not leak into
    ordinary failures.
    """
    logging_obj._failure_handler_helper_fn(
        exception=Exception("boom"),
        traceback_exception="Traceback ...",
    )

    payload = logging_obj.model_call_details["standard_logging_object"]
    assert payload["status"] == "failure"
    assert payload["response_cost"] == 0
    assert payload["total_tokens"] == 0


def test_set_cost_breakdown_stores_reasoning_cost():
    """reasoning_cost is stored only when positive, mirroring the cache-cost fields."""
    from datetime import datetime

    logging_obj = LitellmLogging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="reasoning-cost-set",
        function_id="f",
    )
    logging_obj.set_cost_breakdown(
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.003,
        cost_for_built_in_tools_cost_usd_dollar=0.0,
        reasoning_cost=0.0005,
    )
    assert logging_obj.cost_breakdown["reasoning_cost"] == 0.0005

    no_reasoning = LitellmLogging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="reasoning-cost-absent",
        function_id="f",
    )
    no_reasoning.set_cost_breakdown(
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.003,
        cost_for_built_in_tools_cost_usd_dollar=0.0,
    )
    assert "reasoning_cost" not in no_reasoning.cost_breakdown


def _build_payload_for_media_response(logging_obj, init_response_obj, kwargs=None):
    import datetime

    from litellm.litellm_core_utils.litellm_logging import (
        get_standard_logging_object_payload,
    )

    now = datetime.datetime.now()
    return get_standard_logging_object_payload(
        kwargs=kwargs or {"litellm_call_id": "media-call-id", "model": "test-model", "messages": []},
        init_response_obj=init_response_obj,
        start_time=now,
        end_time=now,
        logging_obj=logging_obj,
        status="success",
    )


def test_image_response_sets_output_image_count_on_usage_object(logging_obj):
    """Generated-image count must land on metadata.usage_object for callbacks (e.g. Prometheus)."""
    from litellm.types.utils import ImageResponse

    response = ImageResponse(created=1, data=[{"url": "https://img/1"}, {"url": "https://img/2"}])

    payload = _build_payload_for_media_response(logging_obj, response)

    assert payload is not None
    assert payload["metadata"]["usage_object"]["output_image_count"] == 2


def test_output_image_count_survives_message_redaction(logging_obj, monkeypatch):
    """Redaction replaces the ImageResponse body, so the count must be captured pre-redaction."""
    import litellm
    from litellm.types.utils import ImageResponse

    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    response = ImageResponse(created=1, data=[{"url": "https://img/1"}])

    payload = _build_payload_for_media_response(logging_obj, response)

    assert payload is not None
    assert payload["response"] == {"text": "redacted-by-litellm"}
    assert payload["metadata"]["usage_object"]["output_image_count"] == 1


def test_non_image_response_has_no_output_image_count(logging_obj):
    payload = _build_payload_for_media_response(
        logging_obj, {"id": "chatcmpl-1", "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    )

    assert payload is not None
    assert "output_image_count" not in payload["metadata"]["usage_object"]


def test_zero_token_video_usage_preserves_duration_seconds(logging_obj):
    """Video usage bills by duration; the payload must keep duration_seconds even with zero tokens."""
    payload = _build_payload_for_media_response(
        logging_obj, {"id": "video-1", "usage": {"duration_seconds": 4.0}}
    )

    assert payload is not None
    assert payload["metadata"]["usage_object"]["duration_seconds"] == 4.0
    assert payload["total_tokens"] == 0
    assert payload["completion_tokens"] == 0


def test_pre_call_does_not_pin_request_in_module_state(logging_obj):
    """
    pre_call/post_call must not stash their locals (full messages, the Logging
    object, complete_input_dict) into module-level state. That pinned the most
    recent request's entire payload in memory for the life of the worker,
    which with multi-hundred-KB requests is a permanent per-worker leak.
    """
    litellm.error_logs.clear()
    big_input = [{"role": "user", "content": "x" * 10_000}]

    logging_obj.pre_call(input=big_input, api_key="sk-test")
    logging_obj.post_call(original_response='{"ok": true}', input=big_input, api_key="sk-test")

    assert litellm.error_logs == {}
