import json
import os
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.weave.weave_otel import WeaveOtelLogger
from litellm.types.integrations.weave import WeaveOtelConfig


class TestWeaveOtelIntegration:

    def test_get_weave_otel_config_with_required_env_vars(self):
        """Test that config is created correctly with required environment variables."""
        env_vars_to_clean = [
            "WEAVE_HOST",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_HEADERS",
        ]
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
            },
            clear=False,
        ):
            # Remove any existing Weave variables
            for var in env_vars_to_clean:
                if var in os.environ:
                    del os.environ[var]

            config = WeaveOtelLogger.get_weave_otel_config()

            assert isinstance(config, WeaveOtelConfig)
            assert config.protocol == "otlp_http"
            assert config.project_id == "test-entity/test-project"
            assert "Authorization=Basic" in config.otlp_auth_headers
            assert "project_id=test-entity/test-project" in config.otlp_auth_headers
            # Check that environment variables are set correctly (cloud default)
            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://trace.wandb.ai/otel/v1/traces"
            )
            assert "Authorization=Basic" in os.environ.get(
                "OTEL_EXPORTER_OTLP_HEADERS", ""
            )

    def test_get_weave_otel_config_missing_api_key(self):
        """Test that ValueError is raised when WANDB_API_KEY is missing."""
        with patch.dict(
            os.environ, {"WEAVE_PROJECT_ID": "test-entity/test-project"}, clear=True
        ):
            with pytest.raises(
                ValueError, match="WANDB_API_KEY must be set for Weave OpenTelemetry"
            ):
                WeaveOtelLogger.get_weave_otel_config()

    def test_get_weave_otel_config_missing_project_id(self):
        """Test that ValueError is raised when WEAVE_PROJECT_ID is missing."""
        with patch.dict(os.environ, {"WANDB_API_KEY": "test_api_key"}, clear=True):
            with pytest.raises(
                ValueError, match="WEAVE_PROJECT_ID must be set for Weave OpenTelemetry"
            ):
                WeaveOtelLogger.get_weave_otel_config()

    def test_get_weave_otel_config_with_custom_host(self):
        """Test config with custom host."""
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
                "WEAVE_HOST": "https://my-weave.wandb.io",
            },
            clear=False,
        ):
            config = WeaveOtelLogger.get_weave_otel_config()

            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://my-weave.wandb.io/traces/otel/v1/traces"
            )

    def test_get_weave_otel_config_with_host_no_protocol(self):
        """Test config with custom host without protocol."""
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
                "WEAVE_HOST": "my-weave.wandb.io",
            },
            clear=False,
        ):
            config = WeaveOtelLogger.get_weave_otel_config()

            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://my-weave.wandb.io/traces/otel/v1/traces"
            )

    def test_get_weave_otel_config_with_otel_host_priority(self):
        """WEAVE_OTEL_HOST should take priority over WEAVE_HOST."""
        with patch.dict(
            os.environ,
            {
                "WANDB_API_KEY": "test_api_key",
                "WEAVE_PROJECT_ID": "test-entity/test-project",
                "WEAVE_HOST": "https://should-not-be-used.com",
                "WEAVE_OTEL_HOST": "https://otel-host.wandb.io",
            },
            clear=False,
        ):
            _ = WeaveOtelLogger.get_weave_otel_config()

            assert (
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                == "https://otel-host.wandb.io/traces/otel/v1/traces"
            )

    def test_set_weave_otel_attributes(self):
        """Test that set_weave_otel_attributes calls the Arize utils function."""
        from litellm.integrations.weave.weave_otel_attributes import (
            WeaveLLMObsOTELAttributes,
        )

        mock_span = MagicMock()
        mock_kwargs = {"test": "kwargs"}
        mock_response = {"test": "response"}

        with patch(
            "litellm.integrations.arize._utils.set_attributes"
        ) as mock_set_attributes:
            WeaveOtelLogger.set_weave_otel_attributes(
                mock_span, mock_kwargs, mock_response
            )

            mock_set_attributes.assert_called_once_with(
                mock_span, mock_kwargs, mock_response, WeaveLLMObsOTELAttributes
            )

    def test_extract_weave_metadata_basic(self):
        """Ensure metadata is correctly pulled from litellm_params."""
        metadata_in = {"generation_name": "my-gen", "custom": "data"}
        kwargs = {"litellm_params": {"metadata": metadata_in}}
        extracted = WeaveOtelLogger._extract_weave_metadata(kwargs)
        assert extracted == metadata_in

    def test_set_weave_specific_attributes_metadata(self):
        """Verify supported metadata keys map to the correct OTEL attributes."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        metadata = {
            "thread_id": "thread-123",
            "is_turn": True,
            "trace_user_id": "user-123",
            "user_id": "user-456",  # Alternative key for user.id
            "session_id": "sess-456",
            "display_name": "My Custom Name",
        }
        kwargs = {"litellm_params": {"metadata": metadata}}

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(MagicMock(), kwargs, None)

            # Flatten the actual calls into {key: value}
            actual = {
                call.args[1]: call.args[2]  # (span, key, value)
                for call in mock_safe_set_attribute.call_args_list
            }

            # Check Weave-specific attributes
            assert actual.get(WeaveSpanAttributes.THREAD_ID.value) == "thread-123"
            assert actual.get(WeaveSpanAttributes.IS_TURN.value) is True
            assert actual.get(WeaveSpanAttributes.DISPLAY_NAME.value) == "My Custom Name"

            # Check standard attributes (user_id takes precedence as it comes after trace_user_id in mapping)
            assert actual.get(WeaveSpanAttributes.TRACE_USER_ID.value) in ["user-123", "user-456"]
            assert actual.get(WeaveSpanAttributes.SESSION_ID.value) == "sess-456"

            # Check that metadata is set as JSON
            assert WeaveSpanAttributes.METADATA.value in actual

    def test_set_weave_specific_attributes_with_model_and_provider(self):
        """Test that model and provider are correctly set."""
        kwargs = {
            "model": "gpt-4o-mini",
            "litellm_params": {"custom_llm_provider": "openai", "metadata": {}},
            "optional_params": {"temperature": 0.7, "max_tokens": 100},
        }

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(MagicMock(), kwargs, None)

            actual = {
                call.args[1]: call.args[2]
                for call in mock_safe_set_attribute.call_args_list
            }

            # Check model attributes (Weave uses llm.model_name and gen_ai.response.model)
            assert actual.get("llm.model_name") == "gpt-4o-mini"
            assert actual.get("gen_ai.response.model") == "gpt-4o-mini"

            # Check provider
            assert actual.get("llm.provider") == "openai"

            # Check span kind
            assert actual.get("openinference.span.kind") == "LLM"

            # Check display name is auto-generated
            assert actual.get("wandb.display_name") == "openai/gpt-4o-mini"

            # Check model parameters are set
            assert "llm.invocation_parameters" in actual

    def test_set_weave_specific_attributes_with_response(self):
        """Test that output.value is set to the full response object."""
        from litellm.types.utils import Choices, ModelResponse, Usage

        response_obj = ModelResponse(
            id="chatcmpl-test",
            model="gpt-4o",
            choices=[
                Choices(
                    finish_reason="stop",
                    message={
                        "role": "assistant",
                        "content": "The weather in Tokyo is sunny.",
                    },
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18),
        )

        kwargs = {"litellm_params": {"metadata": {}}}

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(
                MagicMock(), kwargs, response_obj
            )

            actual = {
                call.args[1]: call.args[2]
                for call in mock_safe_set_attribute.call_args_list
            }

            # Check that output.value is set (should be full response as JSON)
            assert "output.value" in actual
            output_value = json.loads(actual["output.value"])
            assert output_value["id"] == "chatcmpl-test"
            assert output_value["model"] == "gpt-4o"
            assert len(output_value["choices"]) == 1

    def test_set_token_usage(self):
        """Test that token usage attributes are correctly set."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        response_obj = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }

        mock_span = MagicMock()

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_token_usage(mock_span, response_obj)

            actual = {
                call.args[1]: call.args[2]
                for call in mock_safe_set_attribute.call_args_list
            }

            assert actual.get(WeaveSpanAttributes.LLM_TOKEN_COUNT_PROMPT.value) == 100
            assert actual.get(WeaveSpanAttributes.LLM_TOKEN_COUNT_COMPLETION.value) == 50
            assert actual.get(WeaveSpanAttributes.LLM_TOKEN_COUNT_TOTAL.value) == 150

    def test_construct_dynamic_otel_headers_with_weave_keys(self):
        """Test that construct_dynamic_otel_headers creates proper auth headers."""
        from litellm.types.utils import StandardCallbackDynamicParams

        dynamic_params = StandardCallbackDynamicParams(
            wandb_api_key="test_api_key", weave_project_id="test-entity/test-project"
        )

        logger = WeaveOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)

        assert result is not None
        assert "Authorization" in result
        assert "project_id" in result

        auth_header = result["Authorization"]
        assert auth_header.startswith("Basic ")

        import base64

        base64_part = auth_header.replace("Basic ", "")
        decoded = base64.b64decode(base64_part).decode()

        assert decoded == "api:test_api_key"
        assert result["project_id"] == "test-entity/test-project"

    def test_construct_dynamic_otel_headers_empty_params(self):
        """Test that construct_dynamic_otel_headers returns None when no keys provided."""
        from litellm.types.utils import StandardCallbackDynamicParams

        dynamic_params = StandardCallbackDynamicParams()

        logger = WeaveOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)

        assert result is None

    def test_weave_authorization_header_format(self):
        """Test that Weave auth header uses correct api:<key> format."""
        import base64

        auth_header = WeaveOtelLogger._get_weave_authorization_header("my_api_key")

        assert auth_header.startswith("Basic ")
        base64_part = auth_header.replace("Basic ", "")
        decoded = base64.b64decode(base64_part).decode()

        # Weave uses api:<api_key> format (not user:key like Langfuse)
        assert decoded == "api:my_api_key"

    def test_maybe_log_raw_request_is_disabled(self):
        """Test that _maybe_log_raw_request does nothing (no child spans)."""
        logger = WeaveOtelLogger()

        # Should not raise and should do nothing
        result = logger._maybe_log_raw_request(
            kwargs={},
            response_obj={},
            start_time=None,
            end_time=None,
            parent_span=MagicMock()
        )

        # Should return None (pass)
        assert result is None


class TestWeaveOtelResponsesAPI:
    """Test suite for Weave OTEL integration with ResponsesAPI"""

    def test_weave_otel_with_responses_api(self):
        """Test that Weave OTEL logger works with ResponsesAPI responses."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="response-123",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Hello from responses API"}],
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            top_p=1.0,
        )

        test_metadata = {
            "thread_id": "thread123",
            "session_id": "abc456",
            "custom_field": "test_value",
        }

        kwargs = {
            "call_type": "responses",
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
            "optional_params": {},
            "litellm_params": {"metadata": test_metadata},
        }

        mock_span = MagicMock()

        from litellm.integrations.weave.weave_otel_attributes import (
            WeaveLLMObsOTELAttributes,
        )

        with patch(
            "litellm.integrations.arize._utils.set_attributes"
        ) as mock_set_attributes:
            with patch(
                "litellm.integrations.arize._utils.safe_set_attribute"
            ) as mock_safe_set_attribute:
                logger = WeaveOtelLogger()
                logger.set_weave_otel_attributes(mock_span, kwargs, mock_response)

                mock_set_attributes.assert_called_once_with(
                    mock_span, kwargs, mock_response, WeaveLLMObsOTELAttributes
                )


class TestWeaveThreadOrganization:
    """Test suite for Weave's thread organization features."""

    def test_thread_id_attribute(self):
        """Test that thread_id is correctly set for trace organization."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        metadata = {"thread_id": "conversation-123"}
        kwargs = {"litellm_params": {"metadata": metadata}}

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(MagicMock(), kwargs, None)

            actual = {
                call.args[1]: call.args[2]
                for call in mock_safe_set_attribute.call_args_list
            }

            assert actual.get(WeaveSpanAttributes.THREAD_ID.value) == "conversation-123"

    def test_is_turn_attribute(self):
        """Test that is_turn is correctly set for conversation turns."""
        from litellm.types.integrations.weave import WeaveSpanAttributes

        metadata = {"thread_id": "conversation-123", "is_turn": True}
        kwargs = {"litellm_params": {"metadata": metadata}}

        with patch(
            "litellm.integrations.arize._utils.safe_set_attribute"
        ) as mock_safe_set_attribute:
            WeaveOtelLogger._set_weave_specific_attributes(MagicMock(), kwargs, None)

            actual = {
                call.args[1]: call.args[2]
                for call in mock_safe_set_attribute.call_args_list
            }

            assert actual.get(WeaveSpanAttributes.THREAD_ID.value) == "conversation-123"
            assert actual.get(WeaveSpanAttributes.IS_TURN.value) is True


class TestWeaveOtelAttributesClass:
    """Test suite for WeaveLLMObsOTELAttributes"""

    def test_set_messages(self):
        """Test that set_messages sets input.value correctly."""
        from litellm.integrations.weave.weave_otel_attributes import (
            WeaveLLMObsOTELAttributes,
        )
        from litellm.types.integrations.weave import WeaveSpanAttributes

        mock_span = MagicMock()
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            "optional_params": {},
        }

        WeaveLLMObsOTELAttributes.set_messages(mock_span, kwargs)

        # Check that set_attribute was called
        calls = mock_span.set_attribute.call_args_list

        # Find the input.value call
        input_calls = [c for c in calls if c.args[0] == WeaveSpanAttributes.INPUT_VALUE.value]
        assert len(input_calls) == 1

        input_value = json.loads(input_calls[0].args[1])
        assert "messages" in input_value
        assert len(input_value["messages"]) == 2

    def test_set_response_output_messages(self):
        """Test that set_response_output_messages sets output.value correctly."""
        from litellm.integrations.weave.weave_otel_attributes import (
            WeaveLLMObsOTELAttributes,
        )
        from litellm.types.integrations.weave import WeaveSpanAttributes
        from litellm.types.utils import Choices, ModelResponse

        mock_span = MagicMock()
        response_obj = ModelResponse(
            id="test-id",
            model="gpt-4",
            choices=[
                Choices(
                    finish_reason="stop",
                    message={"role": "assistant", "content": "Hello back!"},
                )
            ],
        )

        WeaveLLMObsOTELAttributes.set_response_output_messages(mock_span, response_obj)

        calls = mock_span.set_attribute.call_args_list

        # Find the output.value call
        output_calls = [c for c in calls if c.args[0] == WeaveSpanAttributes.OUTPUT_VALUE.value]
        assert len(output_calls) == 1

        # Output should be JSON string of the response
        output_value = output_calls[0].args[1]
        assert isinstance(output_value, str)


if __name__ == "__main__":
    pytest.main([__file__])
