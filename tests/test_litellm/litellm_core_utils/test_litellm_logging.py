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
        mock_async_log_success_event.assert_called_once()
        assert mock_async_log_success_event.call_count == 1
        print(
            "mock_async_log_success_event.call_args.kwargs",
            mock_async_log_success_event.call_args.kwargs,
        )
        standard_logging_object = mock_async_log_success_event.call_args.kwargs[
            "kwargs"
        ]["standard_logging_object"]
        assert standard_logging_object["stream"] is not True


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
    }
    masked_values = _get_masked_values(
        sensitive_object, unmasked_length=4, number_of_asterisks=4
    )
    assert masked_values["presidio_anonymizer_api_base"] is None


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
    
    with patch("litellm.cold_storage_custom_logger", return_value="s3"), \
         patch("litellm.integrations.s3.get_s3_object_key") as mock_get_s3_key:
        
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
    
    with patch("litellm.cold_storage_custom_logger", "s3_v2"), \
         patch("litellm.logging_callback_manager.get_active_custom_logger_for_callback_name") as mock_get_logger, \
         patch("litellm.integrations.s3.get_s3_object_key") as mock_get_s3_key:
        
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
    
    with patch("litellm.cold_storage_custom_logger", "s3_v2"), \
         patch("litellm.logging_callback_manager.get_active_custom_logger_for_callback_name") as mock_get_logger, \
         patch("litellm.integrations.s3.get_s3_object_key") as mock_get_s3_key:
        
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
    with patch.object(litellm, 'cold_storage_custom_logger', None):
        # Call the function
        result = StandardLoggingPayloadSetup._generate_cold_storage_object_key(
            start_time=start_time, response_id=response_id, team_alias=team_alias
        )

    # Verify the result is None when cold storage is not configured
    assert result is None
