import pytest
from unittest.mock import MagicMock, patch

import litellm

def test_vllm():
    litellm.set_verbose = True
    
    with patch("litellm.llms.vllm.completion.handler.validate_environment") as mock_client:
        mock_client.return_value = MagicMock(), MagicMock()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        
        response = litellm.completion(
            model="vllm/facebook/opt-125m",
            messages=messages
        )
        
        # Verify the request was made
        mock_client.assert_called_once()
        
        # Check the request body
        request_body = mock_client.call_args.kwargs
        
        assert request_body["model"] == "facebook/opt-125m"
        assert request_body["vllm_params"] is not None
        assert request_body["vllm_params"]["quantization"] is None
        
        
def test_vllm_quantized():
    litellm.set_verbose = True
    
    with patch("litellm.llms.vllm.completion.handler.validate_environment") as mock_client:
        mock_client.return_value = MagicMock(), MagicMock()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        
        response = litellm.completion(
            model="vllm/facebook/opt-125m",
            messages=messages,        
            dtype="auto",
            quantization="bitsandbytes",
            load_format="bitsandbytes"
        )
        
        # Verify the request was made
        mock_client.assert_called_once()
        
        # Check the request body
        request_body = mock_client.call_args.kwargs
        
        assert request_body["model"] == "facebook/opt-125m"
        assert request_body["vllm_params"] is not None
        assert request_body["vllm_params"]["quantization"] == "bitsandbytes"
        assert request_body["vllm_params"]["dtype"] == "auto"
        assert request_body["vllm_params"]["load_format"] == "bitsandbytes"