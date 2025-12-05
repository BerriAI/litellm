import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath("../.."))

import pytest
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging


class TestLogRawRequestResponse:
    """
    Test that log_raw_request_response properly persists raw_request to metadata

    This tests the fix for the bug where raw_request was added to a temporary
    local dictionary when metadata was None or empty, preventing it from being
    passed to callbacks like Langfuse.
    """

    def test_raw_request_persists_when_metadata_is_none(self):
        """Test that raw_request is persisted when metadata is None"""
        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            litellm_call_id="test-call-id",
            start_time=None,
            function_id="test-function-id",
            log_raw_request_response=True,
        )

        logging_obj.model_call_details = {
            "litellm_params": {
                "metadata": None
            }
        }

        additional_args = {
            "api_base": "https://api.openai.com/v1",
            "headers": {"Authorization": "Bearer test"},
            "complete_input_dict": {"model": "gpt-3.5-turbo", "messages": []}
        }

        logging_obj.pre_call(
            input=[{"role": "user", "content": "test"}],
            api_key="test-key",
            model="gpt-3.5-turbo",
            additional_args=additional_args
        )

        metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
        assert metadata is not None, "Metadata should be initialized"
        assert "raw_request" in metadata, "raw_request should be in metadata"
        assert metadata["raw_request"] != "", "raw_request should not be empty"

    def test_raw_request_persists_when_metadata_is_empty_dict(self):
        """Test that raw_request is persisted when metadata is an empty dict"""
        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            litellm_call_id="test-call-id",
            start_time=None,
            function_id="test-function-id",
            log_raw_request_response=True,
        )

        logging_obj.model_call_details = {
            "litellm_params": {
                "metadata": {}
            }
        }

        additional_args = {
            "api_base": "https://api.openai.com/v1",
            "headers": {"Authorization": "Bearer test"},
            "complete_input_dict": {"model": "gpt-3.5-turbo", "messages": []}
        }

        logging_obj.pre_call(
            input=[{"role": "user", "content": "test"}],
            api_key="test-key",
            model="gpt-3.5-turbo",
            additional_args=additional_args
        )

        metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
        assert "raw_request" in metadata, "raw_request should be in metadata"
        assert metadata["raw_request"] != "", "raw_request should not be empty"

    def test_raw_request_persists_with_existing_metadata(self):
        """Test that raw_request is persisted when metadata has existing content"""
        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            litellm_call_id="test-call-id",
            start_time=None,
            function_id="test-function-id",
            log_raw_request_response=True,
        )

        logging_obj.model_call_details = {
            "litellm_params": {
                "metadata": {"existing_key": "existing_value"}
            }
        }

        additional_args = {
            "api_base": "https://api.openai.com/v1",
            "headers": {"Authorization": "Bearer test"},
            "complete_input_dict": {"model": "gpt-3.5-turbo", "messages": []}
        }

        logging_obj.pre_call(
            input=[{"role": "user", "content": "test"}],
            api_key="test-key",
            model="gpt-3.5-turbo",
            additional_args=additional_args
        )

        metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
        assert "raw_request" in metadata, "raw_request should be in metadata"
        assert metadata["raw_request"] != "", "raw_request should not be empty"
        assert metadata["existing_key"] == "existing_value", "Existing metadata should be preserved"

    def test_raw_request_redacted_when_turn_off_message_logging(self):
        """Test that raw_request is redacted when turn_off_message_logging is True"""
        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            litellm_call_id="test-call-id",
            start_time=None,
            function_id="test-function-id",
            log_raw_request_response=True,
        )

        logging_obj.model_call_details = {
            "litellm_params": {
                "metadata": {}
            }
        }

        additional_args = {
            "api_base": "https://api.openai.com/v1",
            "headers": {"Authorization": "Bearer test"},
            "complete_input_dict": {"model": "gpt-3.5-turbo", "messages": []}
        }

        with patch("litellm.litellm_core_utils.litellm_logging.turn_off_message_logging", True):
            logging_obj.pre_call(
                input=[{"role": "user", "content": "test"}],
                api_key="test-key",
                model="gpt-3.5-turbo",
                additional_args=additional_args
            )

        metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
        assert "raw_request" in metadata, "raw_request should be in metadata"
        assert "redacted" in metadata["raw_request"].lower(), "raw_request should be redacted"

    @pytest.mark.asyncio
    async def test_raw_request_in_langfuse_callback(self):
        """Test that raw_request is actually passed to Langfuse callback"""
        from litellm.integrations.langfuse.langfuse import LangFuseLogger
        from litellm.types.utils import ModelResponse, Choices, Message, Usage

        langfuse_logger = LangFuseLogger(
            langfuse_public_key="test_key",
            langfuse_secret="test_secret",
            langfuse_host="https://test.langfuse.com"
        )

        kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_params": {
                "metadata": {
                    "raw_request": "curl -X POST https://api.openai.com/v1/chat/completions"
                }
            },
            "call_type": "completion",
            "litellm_call_id": "test-call-id",
        }

        response_obj = ModelResponse(
            id="test-id",
            choices=[Choices(
                message=Message(role="assistant", content="test response"),
                finish_reason="stop",
                index=0
            )],
            model="gpt-3.5-turbo",
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        )

        with patch.object(langfuse_logger.Langfuse, 'trace') as mock_trace:
            trace_client = Mock()
            trace_client.generation.return_value = Mock(trace_id="test-trace", generation_id="test-gen")
            mock_trace.return_value = trace_client

            result = langfuse_logger.log_event_on_langfuse(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=None,
                end_time=None,
                user_id=None
            )

            mock_trace.assert_called_once()
            assert trace_client.generation.call_args is not None, "Langfuse generation should be invoked"
            gen_args, gen_kwargs = trace_client.generation.call_args
            if gen_kwargs:
                generation_metadata = gen_kwargs["metadata"]
            else:
                generation_metadata = getattr(gen_args[0], "metadata", {})

            assert generation_metadata
            assert generation_metadata["raw_request"] == kwargs["litellm_params"]["metadata"]["raw_request"]

    def test_raw_request_not_added_when_flag_disabled(self):
        """Test that raw_request is not added when log_raw_request_response is False"""
        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            litellm_call_id="test-call-id",
            start_time=None,
            function_id="test-function-id",
            log_raw_request_response=False,
        )

        logging_obj.model_call_details = {
            "litellm_params": {
                "metadata": {}
            }
        }

        additional_args = {
            "api_base": "https://api.openai.com/v1",
            "headers": {"Authorization": "Bearer test"},
            "complete_input_dict": {"model": "gpt-3.5-turbo", "messages": []}
        }

        logging_obj.pre_call(
            input=[{"role": "user", "content": "test"}],
            api_key="test-key",
            model="gpt-3.5-turbo",
            additional_args=additional_args
        )

        metadata = logging_obj.model_call_details["litellm_params"]["metadata"]
        assert "raw_request" not in metadata, "raw_request should not be in metadata when flag is disabled"
