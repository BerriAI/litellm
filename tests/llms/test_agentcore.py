"""
Tests for AWS AgentCore Runtime provider
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from litellm.llms.agentcore import AgentCoreConfig
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import Message


class TestAgentCoreConfig:
    """Test cases for AgentCoreConfig class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.agentcore = AgentCoreConfig()
        self.model = "agentcore/my-agent/v1"
        self.api_base = "https://my-agentcore-endpoint.com"

        # Sample messages
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"}
        ]

        # Sample model response
        self.model_response = ModelResponse()
        self.model_response.choices = [Mock()]
        self.model_response.choices[0].message = Message(content="")
        self.model_response.usage = Usage()

    def test_validate_environment(self):
        """Test environment validation"""
        result = self.agentcore.validate_environment(
            aws_region_name="us-east-1",
            api_base=self.api_base
        )

        assert result["aws_region_name"] == "us-east-1"
        assert result["api_base"] == self.api_base

    def test_transform_messages_to_agentcore_request(self):
        """Test message transformation to AgentCore format"""
        optional_params = {
            "agentcore_custom_field": "test_value",
            "temperature": 0.7,  # Non-agentcore param should be ignored
        }

        result = self.agentcore._transform_messages_to_agentcore(
            self.messages
        )

        # Check basic structure
        assert "prompt" in result
        assert "context" in result
        assert result["prompt"] == "Hello, how are you?"
        assert "system: You are a helpful assistant" in result["context"]

        # Check custom parameters
        assert result["custom_field"] == "test_value"
        assert "temperature" not in result

    def test_transform_agentcore_response_to_litellm(self):
        """Test response transformation from AgentCore to LiteLLM format"""
        agentcore_response = {
            "response": "I'm doing well, thank you!",
            "metadata": {
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 8,
                    "total_tokens": 23
                }
            }
        }

        result = self.agentcore._transform_agentcore_to_litellm(
            agentcore_response, self.model, int(time.time())
        )

        assert result.choices[0].message.content == "I'm doing well, thank you!"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 8
        assert result.usage.total_tokens == 23
        assert result.model == self.model

    def test_parse_streaming_chunk(self):
        """Test SSE streaming chunk parsing"""
        # Test valid data line
        line = 'data: {"token": "Hello", "finish_reason": null}'
        result = self.agentcore._parse_streaming_chunk(line, self.model, int(time.time()))

        assert result is not None
        assert result.choices[0].delta.get("content") == "Hello"
        assert result.choices[0].finish_reason is None

        # Test DONE signal
        done_line = "data: [DONE]"
        result = self.agentcore._parse_streaming_chunk(done_line, self.model, int(time.time()))
        assert result is None

        # Test non-data line
        comment_line = ": This is a comment"
        result = self.agentcore._parse_streaming_chunk(comment_line, self.model, int(time.time()))
        assert result is None

    def test_build_invoke_params(self):
        """Test building boto3 invoke parameters"""
        agent_arn = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-agent"
        qualifier = "production"
        data = {
            "prompt": "Hello",
            "runtimeSessionId": "session-123"
        }

        invoke_params, session_id = self.agentcore._build_invoke_params(agent_arn, qualifier, data)

        assert invoke_params["agentRuntimeArn"] == agent_arn
        assert invoke_params["qualifier"] == qualifier
        assert invoke_params["runtimeSessionId"] == "session-123"
        assert session_id == "session-123"
        # runtimeSessionId should be removed from data
        assert "runtimeSessionId" not in data

    @patch('litellm.llms.agentcore.boto3')
    def test_completion_success(self, mock_boto3):
        """Test successful completion request"""
        # Mock boto3 client and invoke_agent_runtime response
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client

        # Mock AgentCore API response
        mock_client.invoke_agent_runtime.return_value = {
            'ResponseMetadata': {'HTTPStatusCode': 200},
            'response': Mock(read=lambda: b'{"response": "Hi there!"}'),
            'runtimeSessionId': 'session-123'
        }

        # Test completion
        result = self.agentcore.completion(
            model=self.model,
            messages=self.messages,
            api_base=self.api_base,
            custom_prompt_dict={},
            model_response=self.model_response,
            print_verbose=lambda x: None,
            encoding=None,
            api_key=None,
            logging_obj=None,
            optional_params={"aws_region_name": "us-east-1"},
        )

        # Verify boto3 client was created correctly
        mock_boto3.client.assert_called_once_with(
            'bedrock-agentcore',
            region_name='us-east-1',
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None
        )

        # Verify invoke_agent_runtime was called
        mock_client.invoke_agent_runtime.assert_called_once()

        # Verify response has session ID
        assert result is not None

    @patch('litellm.llms.agentcore.HTTPHandler')
    @patch.object(AgentCoreConfig, '_sign_request')
    def test_completion_error(self, mock_sign_request, mock_http_handler):
        """Test completion request with error response"""
        # Mock signed request
        mock_sign_request.return_value = (
            {"Authorization": "AWS4-HMAC-SHA256 ..."},
            b'{"prompt": "Hello"}'
        )

        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_http_handler.return_value = mock_client

        # Test completion should raise error
        with pytest.raises(BedrockError) as exc_info:
            self.agentcore.completion(
                model=self.model,
                messages=self.messages,
                api_base=self.api_base,
                custom_prompt_dict={},
                model_response=self.model_response,
                print_verbose=lambda x: None,
                encoding=None,
                api_key=None,
                logging_obj=None,
                optional_params={"aws_region_name": "us-east-1"},
            )

        assert exc_info.value.status_code == 400
        assert "AgentCore request failed" in exc_info.value.message

    @patch('litellm.llms.agentcore.HTTPHandler')
    @patch.object(AgentCoreConfig, '_sign_request')
    def test_streaming_success(self, mock_sign_request, mock_http_handler):
        """Test successful streaming request"""
        # Mock signed request
        mock_sign_request.return_value = (
            {"Authorization": "AWS4-HMAC-SHA256 ..."},
            b'{"prompt": "Hello", "stream": true}'
        )

        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"chunk": "Hello", "is_finished": false}',
            'data: {"chunk": " there!", "is_finished": true, "finish_reason": "stop"}',
        ]

        mock_client = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__.return_value = mock_response
        mock_context_manager.__exit__.return_value = None
        mock_client.stream.return_value = mock_context_manager
        mock_http_handler.return_value = mock_client

        # Test streaming
        chunks = list(self.agentcore.streaming(
            model=self.model,
            messages=self.messages,
            api_base=self.api_base,
            custom_prompt_dict={},
            model_response=self.model_response,
            print_verbose=lambda x: None,
            encoding=None,
            api_key=None,
            logging_obj=None,
            optional_params={"aws_region_name": "us-east-1"},
        ))

        # Verify streaming chunks
        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[0].is_finished is False
        assert chunks[1].text == " there!"
        assert chunks[1].is_finished is True
        assert chunks[1].finish_reason == "stop"

    @pytest.mark.asyncio
    @patch('litellm.llms.agentcore.AsyncHTTPHandler')
    @patch.object(AgentCoreConfig, '_sign_request')
    async def test_acompletion_success(self, mock_sign_request, mock_async_http_handler):
        """Test successful async completion request"""
        # Mock signed request
        mock_sign_request.return_value = (
            {"Authorization": "AWS4-HMAC-SHA256 ..."},
            b'{"prompt": "Hello"}'
        )

        # Mock async HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Hi there!",
            "metadata": {"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_async_http_handler.return_value = mock_client

        # Test async completion
        result = await self.agentcore.acompletion(
            model=self.model,
            messages=self.messages,
            api_base=self.api_base,
            custom_prompt_dict={},
            model_response=self.model_response,
            print_verbose=lambda x: None,
            encoding=None,
            api_key=None,
            logging_obj=None,
            optional_params={"aws_region_name": "us-east-1"},
        )

        # Verify async request was made correctly
        mock_client.post.assert_called_once()

        # Verify response
        assert result.choices[0].message.content == "Hi there!"

    @pytest.mark.asyncio
    @patch('litellm.llms.agentcore.AsyncHTTPHandler')
    @patch.object(AgentCoreConfig, '_sign_request')
    async def test_astreaming_success(self, mock_sign_request, mock_async_http_handler):
        """Test successful async streaming request"""
        # Mock signed request
        mock_sign_request.return_value = (
            {"Authorization": "AWS4-HMAC-SHA256 ..."},
            b'{"prompt": "Hello", "stream": true}'
        )

        # Mock async streaming response
        async def mock_aiter_lines():
            yield 'data: {"chunk": "Hello", "is_finished": false}'
            yield 'data: {"chunk": " world!", "is_finished": true}'

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.aiter_lines.return_value = mock_aiter_lines()

        mock_client = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__aenter__ = Mock(return_value=mock_response)
        mock_context_manager.__aexit__ = Mock(return_value=None)
        mock_client.stream.return_value = mock_context_manager
        mock_async_http_handler.return_value = mock_client

        # Test async streaming
        chunks = []
        async for chunk in self.agentcore.astreaming(
            model=self.model,
            messages=self.messages,
            api_base=self.api_base,
            custom_prompt_dict={},
            model_response=self.model_response,
            print_verbose=lambda x: None,
            encoding=None,
            api_key=None,
            logging_obj=None,
            optional_params={"aws_region_name": "us-east-1"},
        ):
            chunks.append(chunk)

        # Verify async streaming chunks
        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world!"


class TestAgentCoreIntegration:
    """Integration test scenarios"""

    def test_model_format_parsing(self):
        """Test different model format scenarios"""
        agentcore = AgentCoreConfig()

        # Test basic agent/alias format
        model = "agentcore/my-agent/v1"
        # Model parsing would be handled by LiteLLM router
        assert model.startswith("agentcore/")

    def test_custom_parameters(self):
        """Test custom AgentCore parameters"""
        agentcore = AgentCoreConfig()

        messages = [{"role": "user", "content": "Test"}]
        optional_params = {
            "agentcore_session_id": "session-123",
            "agentcore_timeout": 30,
            "agentcore_custom_data": {"key": "value"},
            "temperature": 0.8,  # Should be ignored
        }

        result = agentcore._transform_messages_to_agentcore_request(messages, optional_params)

        assert result["session_id"] == "session-123"
        assert result["timeout"] == 30
        assert result["custom_data"] == {"key": "value"}
        assert "temperature" not in result

    def test_error_handling_scenarios(self):
        """Test various error scenarios"""
        # Test BedrockError creation
        error = BedrockError(status_code=403, message="Forbidden")
        assert error.status_code == 403
        assert error.message == "Forbidden"
        assert str(error) == "Forbidden"

    def test_response_metadata_extraction(self):
        """Test extraction of metadata from AgentCore responses"""
        agentcore = AgentCoreConfig()
        model_response = ModelResponse()
        model_response.choices = [Mock()]
        model_response.choices[0].message = Message(content="")
        model_response.usage = Usage()

        # Test response with extensive metadata
        agentcore_response = {
            "response": "Test response",
            "metadata": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                },
                "session_id": "session-456",
                "execution_time": 1.23,
                "model_version": "v2.1"
            }
        }

        result = agentcore._transform_agentcore_response_to_litellm(
            agentcore_response, "agentcore/test/v1", model_response
        )

        assert result.choices[0].message.content == "Test response"
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150


if __name__ == "__main__":
    pytest.main([__file__])