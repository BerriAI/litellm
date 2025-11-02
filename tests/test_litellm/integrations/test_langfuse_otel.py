import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
from litellm.types.integrations.langfuse_otel import LangfuseOtelConfig
from litellm.types.llms.openai import ResponsesAPIResponse


class TestLangfuseOtelIntegration:
    
    def test_get_langfuse_otel_config_with_required_env_vars(self):
        """Test that config is created correctly with required environment variables."""
        # Clean environment of any Langfuse-related variables
        env_vars_to_clean = ['LANGFUSE_HOST', 'OTEL_EXPORTER_OTLP_ENDPOINT', 'OTEL_EXPORTER_OTLP_HEADERS']
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key'
        }, clear=False):
            # Remove any existing Langfuse variables
            for var in env_vars_to_clean:
                if var in os.environ:
                    del os.environ[var]
                    
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert isinstance(config, LangfuseOtelConfig)
            assert config.protocol == "otlp_http"
            assert "Authorization=Basic" in config.otlp_auth_headers
            # Check that environment variables are set correctly (US default)
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://us.cloud.langfuse.com/api/public/otel"
            assert "Authorization=Basic" in os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
    
    def test_get_langfuse_otel_config_missing_keys(self):
        """Test that ValueError is raised when required keys are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set"):
                LangfuseOtelLogger.get_langfuse_otel_config()
    
    def test_get_langfuse_otel_config_with_eu_host(self):
        """Test config with EU host."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'https://cloud.langfuse.com'
        }, clear=False):
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://cloud.langfuse.com/api/public/otel"
    
    def test_get_langfuse_otel_config_with_custom_host(self):
        """Test config with custom host."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'https://my-langfuse.com'
        }, clear=False):
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://my-langfuse.com/api/public/otel"
    
    def test_get_langfuse_otel_config_with_host_no_protocol(self):
        """Test config with custom host without protocol."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'my-langfuse.com'
        }, clear=False):
            config = LangfuseOtelLogger.get_langfuse_otel_config()
            
            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://my-langfuse.com/api/public/otel"
    
    def test_set_langfuse_otel_attributes(self):
        """Test that set_langfuse_otel_attributes calls the Arize utils function."""
        from litellm.integrations.langfuse.langfuse_otel_attributes import (
            LangfuseLLMObsOTELAttributes,
        )
        
        mock_span = MagicMock()
        mock_kwargs = {"test": "kwargs"}
        mock_response = {"test": "response"}
        
        with patch('litellm.integrations.arize._utils.set_attributes') as mock_set_attributes:
            LangfuseOtelLogger.set_langfuse_otel_attributes(mock_span, mock_kwargs, mock_response)
            
            mock_set_attributes.assert_called_once_with(mock_span, mock_kwargs, mock_response, LangfuseLLMObsOTELAttributes)

    def test_set_langfuse_environment_attribute(self):
        """Test that Langfuse environment is set correctly when environment variable is present."""
        mock_span = MagicMock()
        mock_kwargs = {"test": "kwargs"}
        test_env = "staging"

        with patch.dict(os.environ, {'LANGFUSE_TRACING_ENVIRONMENT': test_env}):
            with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
                LangfuseOtelLogger._set_langfuse_specific_attributes(mock_span, mock_kwargs, {})
                
                # safe_set_attribute(span, key, value) → positional args
                mock_safe_set_attribute.assert_called_once_with(
                    mock_span,
                    "langfuse.environment",
                    test_env
                )

    def test_extract_langfuse_metadata_basic(self):
        """Ensure metadata is correctly pulled from litellm_params."""
        metadata_in = {"generation_name": "my-gen", "custom": "data"}
        kwargs = {"litellm_params": {"metadata": metadata_in}}
        extracted = LangfuseOtelLogger._extract_langfuse_metadata(kwargs)
        assert extracted == metadata_in

    def test_extract_langfuse_metadata_with_header_enrichment(self, monkeypatch):
        """_extract_langfuse_metadata should call LangFuseLogger.add_metadata_from_header when available."""
        import sys
        import types

        # Build a stub module + class on-the-fly
        stub_module = types.ModuleType("litellm.integrations.langfuse.langfuse")
        class StubLFLogger:
            @staticmethod
            def add_metadata_from_header(litellm_params, metadata):
                # Echo back existing metadata plus a marker
                return {**metadata, "enriched": True}
        stub_module.LangFuseLogger = StubLFLogger  # type: ignore

        # Register stub in sys.modules so import inside method succeeds
        sys.modules["litellm.integrations.langfuse.langfuse"] = stub_module  # type: ignore

        kwargs = {"litellm_params": {"metadata": {"foo": "bar"}}}
        extracted = LangfuseOtelLogger._extract_langfuse_metadata(kwargs)
        assert extracted.get("foo") == "bar"
        assert extracted.get("enriched") is True

    def test_set_langfuse_specific_attributes_metadata(self):
        """Verify every supported metadata key maps to the correct OTEL attribute and complex types are JSON-serialised."""
        # Build a sample metadata payload covering all mappings
        metadata = {
            "generation_name": "gen-name",
            "generation_id": "gen-id",
            "parent_observation_id": "parent-id",
            "version": "v1",
            "mask_input": True,
            "mask_output": False,
            "trace_user_id": "user-123",
            "session_id": "sess-456",
            "tags": ["tagA", "tagB"],
            "trace_name": "trace-name",
            "trace_id": "trace-id",
            "trace_metadata": {"k": "v"},
            "trace_version": "t-ver",
            "trace_release": "rel-1",
            "existing_trace_id": "existing-id",
            "update_trace_keys": ["key1", "key2"],
            "debug_langfuse": True,
        }
        kwargs = {"litellm_params": {"metadata": metadata}}

        # Capture calls to safe_set_attribute
        with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
            LangfuseOtelLogger._set_langfuse_specific_attributes(MagicMock(), kwargs, None)

            # Build expected calls manually for clarity
            from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
            expected = {
                LangfuseSpanAttributes.GENERATION_NAME.value: "gen-name",
                LangfuseSpanAttributes.GENERATION_ID.value: "gen-id",
                LangfuseSpanAttributes.PARENT_OBSERVATION_ID.value: "parent-id",
                LangfuseSpanAttributes.GENERATION_VERSION.value: "v1",
                LangfuseSpanAttributes.MASK_INPUT.value: True,
                LangfuseSpanAttributes.MASK_OUTPUT.value: False,
                LangfuseSpanAttributes.TRACE_USER_ID.value: "user-123",
                LangfuseSpanAttributes.SESSION_ID.value: "sess-456",
                # Lists / dicts should be JSON strings
                LangfuseSpanAttributes.TAGS.value: json.dumps(["tagA", "tagB"]),
                LangfuseSpanAttributes.TRACE_NAME.value: "trace-name",
                LangfuseSpanAttributes.TRACE_ID.value: "trace-id",
                LangfuseSpanAttributes.TRACE_METADATA.value: json.dumps({"k": "v"}),
                LangfuseSpanAttributes.TRACE_VERSION.value: "t-ver",
                LangfuseSpanAttributes.TRACE_RELEASE.value: "rel-1",
                LangfuseSpanAttributes.EXISTING_TRACE_ID.value: "existing-id",
                LangfuseSpanAttributes.UPDATE_TRACE_KEYS.value: json.dumps(["key1", "key2"]),
                LangfuseSpanAttributes.DEBUG_LANGFUSE.value: True,
            }

            # Flatten the actual calls into {key: value}
            actual = {
                call.args[1]: call.args[2]  # (span, key, value)
                for call in mock_safe_set_attribute.call_args_list
            }

            assert actual == expected, "Mismatch between expected and actual OTEL attribute mapping."

    def test_set_langfuse_specific_attributes_with_content(self):
        """Test that _set_langfuse_specific_attributes correctly sets observation.output with regular content response."""
        from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
        from litellm.types.utils import Choices, ModelResponse

        # Create response with content
        response_obj = ModelResponse(
            id='chatcmpl-test',
            model='gpt-4o',
            choices=[
                Choices(
                    finish_reason='stop',
                    message={
                        "role": "assistant",
                        "content": "The weather in Tokyo is sunny."
                    }
                )
            ],
        )

        kwargs = {
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        }

        with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
            LangfuseOtelLogger._set_langfuse_specific_attributes(MagicMock(), kwargs, response_obj)

            expect_output = {
                LangfuseSpanAttributes.OBSERVATION_INPUT.value: [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo?"
                    }
                ],
                LangfuseSpanAttributes.OBSERVATION_OUTPUT.value: {
                    "role": "assistant",
                    "content": "The weather in Tokyo is sunny."
                }
            }

            # Flatten the actual calls into {key: value}
            actual = {
                call.args[1]: json.loads(call.args[2])
                for call in mock_safe_set_attribute.call_args_list
            }

            assert actual == expect_output, "Mismatch in observation input/output OTEL attributes."


    def test_set_langfuse_specific_attributes_with_tool_calls(self):
        """Test that _set_langfuse_specific_attributes correctly sets observation.output with tool calls in Langfuse format."""
        from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
        from litellm.types.utils import (
            ChatCompletionMessageToolCall,
            Choices,
            Function,
            ModelResponse,
        )

        # Create response with tool calls
        response_obj = ModelResponse(
            id='chatcmpl-test',
            model='gpt-4o',
            choices=[
                Choices(
                    finish_reason='tool_calls',
                    message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            ChatCompletionMessageToolCall(
                                function=Function(
                                    arguments='{"location":"Tokyo"}',
                                    name='get_weather'
                                ),
                                id='call_123',
                                type='function'
                            )
                        ]
                    }
                )
            ],
        )

        with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
            LangfuseOtelLogger._set_langfuse_specific_attributes(MagicMock(), {}, 
            response_obj)

            expected = {
                LangfuseSpanAttributes.OBSERVATION_OUTPUT.value: [
                        {
                            "id": "chatcmpl-test",
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo"},
                            "call_id": "call_123",
                            "type": "function_call"
                        }
                ]
            }

            # Flatten the actual calls into {key: value}
            actual = {
                call.args[1]: json.loads(call.args[2])
                for call in mock_safe_set_attribute.call_args_list
            }
            assert actual == expected, "Mismatch in observation output OTEL attribute for tool calls."


    def test_construct_dynamic_otel_headers_with_langfuse_keys(self):
        """Test that construct_dynamic_otel_headers creates proper auth headers when langfuse keys are provided."""
        from litellm.types.utils import StandardCallbackDynamicParams

        # Create dynamic params with langfuse keys
        dynamic_params = StandardCallbackDynamicParams(
            langfuse_public_key="test_public_key",
            langfuse_secret_key="test_secret_key"
        )
        
        logger = LangfuseOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)
        
        # Should return a dict with otlp_auth_headers
        assert result is not None
        assert "Authorization" in result
        
        # The auth header should contain the basic auth format
        auth_header = result["Authorization"]
        assert auth_header.startswith("Basic ")
        
        # Verify the header format by decoding
        import base64

        # Extract the base64 part from "Authorization=Basic <base64>"
        base64_part = auth_header.replace("Basic ", "")
        decoded = base64.b64decode(base64_part).decode()
        
        assert decoded == "test_public_key:test_secret_key"

    def test_construct_dynamic_otel_headers_empty_params(self):
        """Test that construct_dynamic_otel_headers returns empty dict when no langfuse keys are provided."""
        from litellm.types.utils import StandardCallbackDynamicParams

        # Create dynamic params without langfuse keys
        dynamic_params = StandardCallbackDynamicParams()
        
        logger = LangfuseOtelLogger()
        result = logger.construct_dynamic_otel_headers(dynamic_params)
        
        # Should return an empty dict
        assert result == {}
    
    def test_get_langfuse_otel_config_with_otel_host_priority(self):
        """LANGFUSE_OTEL_HOST should take priority over LANGFUSE_HOST."""
        with patch.dict(os.environ, {
            'LANGFUSE_PUBLIC_KEY': 'test_public_key',
            'LANGFUSE_SECRET_KEY': 'test_secret_key',
            'LANGFUSE_HOST': 'https://should-not-be-used.com',
            'LANGFUSE_OTEL_HOST': 'https://otel-host.com'
        }, clear=False):
            _ = LangfuseOtelLogger.get_langfuse_otel_config()

            assert os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "https://otel-host.com/api/public/otel"


class TestLangfuseOtelResponsesAPI:
    """Test suite for Langfuse OTEL integration with ResponsesAPI"""

    def test_langfuse_otel_with_responses_api(self):
        """Test that Langfuse OTEL logger works with ResponsesAPI responses and logs metadata."""
        # Create a mock ResponsesAPIResponse
        mock_response = ResponsesAPIResponse(
            id="response-123",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Hello from responses API"}]
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            top_p=1.0
        )
        
        # Create kwargs with metadata that should be logged
        test_metadata = {
            "user_id": "test123", 
            "session_id": "abc456", 
            "custom_field": "test_value",
            "generation_name": "responses_test_generation",
            "trace_name": "responses_api_trace"
        }
        
        kwargs = {
            "call_type": "responses",
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
            "optional_params": {},
            "litellm_params": {"metadata": test_metadata}
        }
        
        mock_span = MagicMock()
        
        from litellm.integrations.langfuse.langfuse_otel_attributes import (
            LangfuseLLMObsOTELAttributes,
        )
        
        with patch('litellm.integrations.arize._utils.set_attributes') as mock_set_attributes:
            with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
                logger = LangfuseOtelLogger()
                logger.set_langfuse_otel_attributes(mock_span, kwargs, mock_response)
                
                # Verify that set_attributes was called for general attributes
                mock_set_attributes.assert_called_once_with(mock_span, kwargs, mock_response, LangfuseLLMObsOTELAttributes)
                
                # Verify that Langfuse-specific attributes were set
                mock_safe_set_attribute.assert_any_call(
                    mock_span, "langfuse.generation.name", "responses_test_generation"
                )
                mock_safe_set_attribute.assert_any_call(
                    mock_span, "langfuse.trace.name", "responses_api_trace"
                )

    def test_responses_api_metadata_extraction(self):
        """Test that metadata is correctly extracted from ResponsesAPI kwargs."""
        # Clean up any existing module mocks
        import sys
        if "litellm.integrations.langfuse.langfuse" in sys.modules:
            original_module = sys.modules["litellm.integrations.langfuse.langfuse"]
        
        test_metadata = {
            "user_id": "responses_user_123",
            "session_id": "responses_session_456", 
            "custom_metadata": {"key": "value"},
            "generation_name": "responses_generation",
            "trace_id": "custom_trace_id"
        }
        
        kwargs = {
            "call_type": "responses",
            "model": "gpt-4o",
            "litellm_params": {"metadata": test_metadata}
        }
        
        extracted_metadata = LangfuseOtelLogger._extract_langfuse_metadata(kwargs)
        
        # Verify all expected metadata was extracted (may have additional fields from header enrichment)
        for key, value in test_metadata.items():
            assert extracted_metadata[key] == value
            
        assert extracted_metadata["user_id"] == "responses_user_123"
        assert extracted_metadata["generation_name"] == "responses_generation"
        assert extracted_metadata["trace_id"] == "custom_trace_id"

    def test_responses_api_langfuse_specific_attributes(self):
        """Test that ResponsesAPI metadata maps correctly to Langfuse OTEL attributes."""
        metadata = {
            "generation_name": "responses_gen",
            "generation_id": "resp_gen_123",
            "trace_name": "responses_trace",
            "trace_user_id": "resp_user_456",
            "session_id": "resp_session_789",
            "tags": ["responses", "api", "test"],
            "trace_metadata": {"source": "responses_api", "version": "1.0"}
        }
        
        kwargs = {
            "call_type": "responses",
            "litellm_params": {"metadata": metadata}
        }
        
        mock_span = MagicMock()
        
        with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
            LangfuseOtelLogger._set_langfuse_specific_attributes(mock_span, kwargs, {})
            
            # Verify specific attributes were set
            from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
            
            expected_calls = [
                (mock_span, LangfuseSpanAttributes.GENERATION_NAME.value, "responses_gen"),
                (mock_span, LangfuseSpanAttributes.GENERATION_ID.value, "resp_gen_123"),
                (mock_span, LangfuseSpanAttributes.TRACE_NAME.value, "responses_trace"),
                (mock_span, LangfuseSpanAttributes.TRACE_USER_ID.value, "resp_user_456"),
                (mock_span, LangfuseSpanAttributes.SESSION_ID.value, "resp_session_789"),
                (mock_span, LangfuseSpanAttributes.TAGS.value, json.dumps(["responses", "api", "test"])),
                (mock_span, LangfuseSpanAttributes.TRACE_METADATA.value, 
                 json.dumps({"source": "responses_api", "version": "1.0"}))
            ]
            
            for expected_call in expected_calls:
                mock_safe_set_attribute.assert_any_call(*expected_call)

    def test_responses_api_with_output(self):
        """Test Langfuse OTEL logger with Responses API output (reasoning + message)."""
        from openai.types.responses import ResponseReasoningItem, ResponseOutputMessage, ResponseOutputText
        from openai.types.responses.response_reasoning_item import Summary
        from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes

        # Create Responses API response with reasoning and message
        response_obj = ResponsesAPIResponse(
            id="response-456",
            created_at=1625247600,
            output=[
                ResponseReasoningItem(
                    id="reasoning-001",
                    type="reasoning",
                    summary=[
                        Summary(
                            text="Let me analyze this problem step by step...",
                            type="summary_text"
                        )
                    ]
                ),
                ResponseOutputMessage(
                    id="msg-001",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="The weather in San Francisco is sunny, 20°C.",
                            type="output_text",
                        )
                    ]
                )
            ]
        )

        kwargs = {
            "call_type": "responses",
            "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
            "model": "gpt-4o",
            "optional_params": {},
        }

        mock_span = MagicMock()

        with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
            LangfuseOtelLogger._set_langfuse_specific_attributes(mock_span, kwargs, response_obj)

            # Verify observation output was set
            output_calls = [
                call for call in mock_safe_set_attribute.call_args_list
                if call.args[1] == LangfuseSpanAttributes.OBSERVATION_OUTPUT.value
            ]

            assert len(output_calls) > 0, "observation.output should be set"
            output_json = output_calls[0].args[2]
            output_data = json.loads(output_json)

            # Verify output contains reasoning and message
            assert isinstance(output_data, list)
            assert len(output_data) == 2

            # Verify reasoning summary
            assert output_data[0]["role"] == "reasoning_summary"
            assert output_data[0]["content"] == "Let me analyze this problem step by step..."

            # Verify message
            assert output_data[1]["role"] == "assistant"
            assert output_data[1]["content"] == "The weather in San Francisco is sunny, 20°C."

    def test_responses_api_with_function_calls(self):
        """Test Langfuse OTEL logger with Responses API function_call output."""
        from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes
        from openai.types.responses import ResponseFunctionToolCall

        # Create Responses API response with function call
        response_obj = ResponsesAPIResponse(
            id="response-789",
            created_at=1625247700,
            output=[
                ResponseFunctionToolCall(
                    id="fc-123",
                    type="function_call",
                    name="get_weather",
                    call_id="call-abc",
                    arguments='{"location": "San Francisco", "unit": "celsius"}',
                    status="completed"
                )
            ]
        )

        kwargs = {
            "call_type": "responses",
            "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
            "model": "gpt-4o",
            "optional_params": {},
        }

        mock_span = MagicMock()

        with patch('litellm.integrations.arize._utils.safe_set_attribute') as mock_safe_set_attribute:
            LangfuseOtelLogger._set_langfuse_specific_attributes(mock_span, kwargs, response_obj)

            # Verify observation output was set
            output_calls = [
                call for call in mock_safe_set_attribute.call_args_list
                if call.args[1] == LangfuseSpanAttributes.OBSERVATION_OUTPUT.value
            ]

            assert len(output_calls) > 0, "observation.output should be set"
            output_json = output_calls[0].args[2]
            output_data = json.loads(output_json)

            # Verify output contains function call
            assert isinstance(output_data, list)
            assert len(output_data) == 1

            # Verify function call details
            assert output_data[0]["type"] == "function_call"
            assert output_data[0]["id"] == "fc-123"
            assert output_data[0]["name"] == "get_weather"
            assert output_data[0]["call_id"] == "call-abc"
            assert output_data[0]["arguments"]["location"] == "San Francisco"
            assert output_data[0]["arguments"]["unit"] == "celsius"


if __name__ == "__main__":
    pytest.main([__file__]) 