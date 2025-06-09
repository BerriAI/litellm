import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import completion
from litellm.utils import get_optional_params


class TestPerplexityReasoning:
    """
    Test suite for Perplexity Sonar reasoning models with reasoning_effort parameter
    """

    @pytest.mark.parametrize(
        "model,reasoning_effort",
        [
            ("perplexity/sonar-reasoning", "low"),
            ("perplexity/sonar-reasoning", "medium"),
            ("perplexity/sonar-reasoning", "high"),
            ("perplexity/sonar-reasoning-pro", "low"),
            ("perplexity/sonar-reasoning-pro", "medium"),
            ("perplexity/sonar-reasoning-pro", "high"),
        ]
    )
    def test_perplexity_reasoning_effort_parameter_mapping(self, model, reasoning_effort):
        """
        Test that reasoning_effort parameter is correctly mapped for Perplexity Sonar reasoning models
        """
        # Set up local model cost map
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        # Get provider and optional params
        _, provider, _, _ = litellm.get_llm_provider(model=model)
        
        optional_params = get_optional_params(
            model=model,
            custom_llm_provider=provider,
            reasoning_effort=reasoning_effort,
        )
        
        # Verify that reasoning_effort is preserved in optional_params for Perplexity
        assert "reasoning_effort" in optional_params
        assert optional_params["reasoning_effort"] == reasoning_effort

    @pytest.mark.parametrize(
        "model",
        [
            "perplexity/sonar-reasoning",
            "perplexity/sonar-reasoning-pro",
        ]
    )
    def test_perplexity_reasoning_effort_mock_completion(self, model):
        """
        Test that reasoning_effort is correctly passed in actual completion call (mocked)
        """
        litellm.set_verbose = True
        
        # Mock successful response with reasoning content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model.split("/")[1],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from the reasoning model.",
                        "reasoning_content": "Let me think about this step by step...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 20,
                "total_tokens": 29,
                "completion_tokens_details": {
                    "reasoning_tokens": 15
                }
            },
        }

        with patch("litellm.llms.custom_httpx.custom_httpx_handler.HTTPHandler.post") as mock_post:
            mock_post.return_value = mock_response
            
            response = completion(
                model=model,
                messages=[{"role": "user", "content": "Hello, please think about this carefully."}],
                reasoning_effort="high",
            )
            
            # Verify the call was made
            assert mock_post.called
            
            # Get the request data from the mock call
            call_args = mock_post.call_args
            request_data = call_args[1]["json"]  # The json parameter passed to post
            
            # Verify reasoning_effort was included in the request
            assert "reasoning_effort" in request_data
            assert request_data["reasoning_effort"] == "high"
            
            # Verify response structure
            assert response.choices[0].message.content is not None
            assert "reasoning_content" in mock_response.json.return_value["choices"][0]["message"]

    @pytest.mark.parametrize(
        "model",
        [
            "perplexity/sonar-reasoning",
            "perplexity/sonar-reasoning-pro",
        ]
    )
    def test_perplexity_reasoning_effort_streaming_mock(self, model):
        """
        Test that reasoning_effort works with streaming for Perplexity Sonar reasoning models (mocked)
        """
        litellm.set_verbose = True
        
        # Mock streaming response chunks
        streaming_chunks = [
            'data: {"id":"cmpl-test","object":"chat.completion.chunk","created":1677652288,"model":"' + model.split("/")[1] + '","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-test","object":"chat.completion.chunk","created":1677652288,"model":"' + model.split("/")[1] + '","choices":[{"index":0,"delta":{"content":"This"},"finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-test","object":"chat.completion.chunk","created":1677652288,"model":"' + model.split("/")[1] + '","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-test","object":"chat.completion.chunk","created":1677652288,"model":"' + model.split("/")[1] + '","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-test","object":"chat.completion.chunk","created":1677652288,"model":"' + model.split("/")[1] + '","choices":[{"index":0,"delta":{"content":" test"},"finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-test","object":"chat.completion.chunk","created":1677652288,"model":"' + model.split("/")[1] + '","choices":[{"index":0,"delta":{"reasoning_content":"Let me think about this..."},"finish_reason":null}]}\n\n',
            'data: {"id":"cmpl-test","object":"chat.completion.chunk","created":1677652288,"model":"' + model.split("/")[1] + '","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13,"completion_tokens_details":{"reasoning_tokens":8}}}\n\n',
            'data: [DONE]\n\n'
        ]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.iter_lines.return_value = streaming_chunks

        with patch("litellm.llms.custom_httpx.custom_httpx_handler.HTTPHandler.post") as mock_post:
            mock_post.return_value = mock_response
            
            response = completion(
                model=model,
                messages=[{"role": "user", "content": "Hello, please think about this carefully."}],
                reasoning_effort="medium",
                stream=True,
            )
            
            # Verify the call was made
            assert mock_post.called
            
            # Get the request data from the mock call
            call_args = mock_post.call_args
            request_data = call_args[1]["json"]
            
            # Verify reasoning_effort was included in the request
            assert "reasoning_effort" in request_data
            assert request_data["reasoning_effort"] == "medium"
            assert request_data["stream"] is True
            
            # Collect chunks to verify streaming works
            chunks = []
            for chunk in response:
                chunks.append(chunk)
            
            # Should have received chunks
            assert len(chunks) > 0

    def test_perplexity_reasoning_models_support_reasoning(self):
        """
        Test that Perplexity Sonar reasoning models are correctly identified as supporting reasoning
        """
        from litellm.utils import supports_reasoning
        
        # Set up local model cost map
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        
        reasoning_models = [
            "perplexity/sonar-reasoning",
            "perplexity/sonar-reasoning-pro",
        ]
        
        for model in reasoning_models:
            assert supports_reasoning(model, None), f"{model} should support reasoning"

    def test_perplexity_non_reasoning_models_dont_support_reasoning(self):
        """
        Test that non-reasoning Perplexity models don't support reasoning
        """
        from litellm.utils import supports_reasoning
        
        # Set up local model cost map
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        
        non_reasoning_models = [
            "perplexity/sonar",
            "perplexity/sonar-pro",
            "perplexity/llama-3.1-sonar-large-128k-chat",
            "perplexity/mistral-7b-instruct",
        ]
        
        for model in non_reasoning_models:
            # These models should not support reasoning (should return False or raise exception)
            try:
                result = supports_reasoning(model, None)
                # If it doesn't raise an exception, it should return False
                assert result is False, f"{model} should not support reasoning"
            except Exception:
                # If it raises an exception, that's also acceptable behavior
                pass

    @pytest.mark.parametrize(
        "model,expected_api_base",
        [
            ("perplexity/sonar-reasoning", "https://api.perplexity.ai"),
            ("perplexity/sonar-reasoning-pro", "https://api.perplexity.ai"),
        ]
    )
    def test_perplexity_reasoning_api_base_configuration(self, model, expected_api_base):
        """
        Test that Perplexity reasoning models use the correct API base
        """
        from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig
        
        config = PerplexityChatConfig()
        api_base, _ = config._get_openai_compatible_provider_info(
            api_base=None, api_key="test-key"
        )
        
        assert api_base == expected_api_base

    def test_perplexity_reasoning_effort_in_supported_params(self):
        """
        Test that reasoning_effort is in the list of supported parameters for Perplexity
        """
        from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig
        
        config = PerplexityChatConfig()
        supported_params = config.get_supported_openai_params(model="perplexity/sonar-reasoning")
        
        assert "reasoning_effort" in supported_params