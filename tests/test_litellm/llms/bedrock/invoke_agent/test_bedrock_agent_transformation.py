import base64
import json
import os
import sys
from litellm._uuid import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.chat.invoke_agent.transformation import (
    AmazonInvokeAgentConfig,
)
from litellm.types.llms.bedrock_invoke_agents import (
    InvokeAgentEvent,
    InvokeAgentEventHeaders,
    InvokeAgentUsage,
)
from litellm.types.utils import Message, ModelResponse, Usage


class TestAmazonInvokeAgentConfig:
    """Test suite for AmazonInvokeAgentConfig methods"""

    @pytest.fixture
    def config(self):
        """Create a test instance of AmazonInvokeAgentConfig"""
        return AmazonInvokeAgentConfig()

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing"""
        return [
            {"role": "user", "content": "Hello, how can you help me?"},
            {"role": "assistant", "content": "I can help with various tasks."},
            {"role": "user", "content": "What is the weather like?"},
        ]

    @pytest.fixture
    def sample_events(self):
        """Sample events for testing event parsing"""
        return [
            {
                "headers": {"event_type": "chunk"},
                "payload": {
                    "bytes": base64.b64encode("Hello ".encode("utf-8")).decode("utf-8")
                },
            },
            {
                "headers": {"event_type": "chunk"},
                "payload": {
                    "bytes": base64.b64encode("world!".encode("utf-8")).decode("utf-8")
                },
            },
            {
                "headers": {"event_type": "trace"},
                "payload": {
                    "trace": {
                        "preProcessingTrace": {
                            "modelInvocationOutput": {
                                "metadata": {
                                    "usage": {"inputTokens": 10, "outputTokens": 20}
                                }
                            }
                        },
                        "orchestrationTrace": {
                            "modelInvocationInput": {
                                "foundationModel": "anthropic.claude-v2"
                            }
                        },
                    }
                },
            },
        ]

    def test_get_agent_id_and_alias_id_valid(self, config):
        """Test parsing valid agent model string"""
        model = "agent/L1RT58GYRW/MFPSBCXYTW"
        agent_id, agent_alias_id = config._get_agent_id_and_alias_id(model)

        assert agent_id == "L1RT58GYRW"
        assert agent_alias_id == "MFPSBCXYTW"

    def test_get_agent_id_and_alias_id_invalid_format(self, config):
        """Test parsing invalid agent model string"""
        invalid_models = [
            "invalid/L1RT58GYRW/MFPSBCXYTW",  # Wrong prefix
            "agent/L1RT58GYRW",  # Missing alias
            "agent/L1RT58GYRW/MFPSBCXYTW/extra",  # Too many parts
            "L1RT58GYRW/MFPSBCXYTW",  # Missing prefix
        ]

        for invalid_model in invalid_models:
            with pytest.raises(ValueError, match="Invalid model format"):
                config._get_agent_id_and_alias_id(invalid_model)

    @patch(
        "litellm.llms.bedrock.chat.invoke_agent.transformation.convert_content_list_to_str"
    )
    def test_transform_request(self, mock_convert, config, sample_messages):
        """Test transform_request method"""
        mock_convert.return_value = "What is the weather like?"

        model = "agent/TEST123/ALIAS456"
        optional_params = {}
        litellm_params = {}
        headers = {}

        result = config.transform_request(
            model, sample_messages, optional_params, litellm_params, headers
        )

        expected = {
            "inputText": "What is the weather like?",
            "enableTrace": True,
        }
        assert result == expected
        mock_convert.assert_called_once_with(sample_messages[-1])

    def test_extract_response_content(self, config, sample_events):
        """Test _extract_response_content method"""
        result = config._extract_response_content(sample_events)
        assert result == "Hello world!"

    def test_extract_response_content_empty_events(self, config):
        """Test _extract_response_content with empty events"""
        result = config._extract_response_content([])
        assert result == ""

    def test_extract_response_content_no_chunk_events(self, config):
        """Test _extract_response_content with no chunk events"""
        events = [{"headers": {"event_type": "trace"}, "payload": {"some": "data"}}]
        result = config._extract_response_content(events)
        assert result == ""

    def test_is_trace_event(self, config):
        """Test _is_trace_event method"""
        trace_event = {"headers": {"event_type": "trace"}, "payload": {"some": "data"}}
        chunk_event = {"headers": {"event_type": "chunk"}, "payload": {"bytes": "data"}}
        invalid_event = {"headers": {"event_type": "trace"}, "payload": None}

        assert config._is_trace_event(trace_event) is True
        assert config._is_trace_event(chunk_event) is False
        assert config._is_trace_event(invalid_event) is False

    def test_get_trace_data(self, config):
        """Test _get_trace_data method"""
        event = {"payload": {"trace": {"preProcessingTrace": {"some": "data"}}}}
        result = config._get_trace_data(event)
        assert result == {"preProcessingTrace": {"some": "data"}}

    def test_get_trace_data_no_payload(self, config):
        """Test _get_trace_data with no payload"""
        event = {"payload": None}
        result = config._get_trace_data(event)
        assert result is None

    def test_extract_usage_info(self, config, sample_events):
        """Test _extract_usage_info method"""
        result = config._extract_usage_info(sample_events)

        assert result["inputTokens"] == 10
        assert result["outputTokens"] == 20
        assert result["model"] == "anthropic.claude-v2"

    def test_extract_usage_info_empty_events(self, config):
        """Test _extract_usage_info with empty events"""
        result = config._extract_usage_info([])

        assert result["inputTokens"] == 0
        assert result["outputTokens"] == 0
        assert result["model"] is None

    def test_extract_and_update_preprocessing_usage(self, config):
        """Test _extract_and_update_preprocessing_usage method"""
        trace_data = {
            "preProcessingTrace": {
                "modelInvocationOutput": {
                    "metadata": {"usage": {"inputTokens": 15, "outputTokens": 25}}
                }
            }
        }
        usage_info = {"inputTokens": 5, "outputTokens": 10, "model": None}

        config._extract_and_update_preprocessing_usage(trace_data, usage_info)

        assert usage_info["inputTokens"] == 20  # 5 + 15
        assert usage_info["outputTokens"] == 35  # 10 + 25

    def test_extract_and_update_preprocessing_usage_no_data(self, config):
        """Test _extract_and_update_preprocessing_usage with missing data"""
        trace_data = {}
        usage_info = {"inputTokens": 5, "outputTokens": 10, "model": None}

        config._extract_and_update_preprocessing_usage(trace_data, usage_info)

        # Should remain unchanged
        assert usage_info["inputTokens"] == 5
        assert usage_info["outputTokens"] == 10

    def test_extract_orchestration_model(self, config):
        """Test _extract_orchestration_model method"""
        trace_data = {
            "orchestrationTrace": {
                "modelInvocationInput": {"foundationModel": "anthropic.claude-v2"}
            }
        }
        result = config._extract_orchestration_model(trace_data)
        assert result == "anthropic.claude-v2"

    def test_extract_orchestration_model_no_data(self, config):
        """Test _extract_orchestration_model with missing data"""
        trace_data = {}
        result = config._extract_orchestration_model(trace_data)
        assert result is None

    def test_build_model_response(self, config):
        """Test _build_model_response method"""
        content = "Hello, world!"
        model = "agent/TEST123/ALIAS456"
        usage_info = {
            "inputTokens": 10,
            "outputTokens": 20,
            "model": "anthropic.claude-v2",
        }
        model_response = ModelResponse()

        result = config._build_model_response(
            content, model, usage_info, model_response
        )

        assert len(result.choices) == 1
        assert result.choices[0].message.content == content
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.model == "anthropic.claude-v2"
        assert hasattr(result, "usage")
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    @patch(
        "litellm.llms.bedrock.chat.invoke_agent.transformation.convert_content_list_to_str"
    )
    @patch.object(AmazonInvokeAgentConfig, "get_runtime_endpoint")
    @patch.object(AmazonInvokeAgentConfig, "_get_aws_region_name")
    def test_get_complete_url(self, mock_region, mock_endpoint, mock_convert, config):
        """Test get_complete_url method"""
        mock_endpoint.return_value = (
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            None,
        )
        mock_region.return_value = "us-east-1"

        api_base = None
        api_key = None
        model = "agent/L1RT58GYRW/MFPSBCXYTW"
        optional_params = {}
        litellm_params = {}

        result = config.get_complete_url(
            api_base, api_key, model, optional_params, litellm_params
        )

        assert (
            "https://bedrock-runtime.us-east-1.amazonaws.com/agents/L1RT58GYRW/agentAliases/MFPSBCXYTW/sessions"
            in result
        )
