"""
Tests for Wandb Hub chat completion handler
Maps to: litellm/llms/wandb_hub/chat/handler.py
"""
import asyncio
from unittest.mock import MagicMock, patch
import pytest

from litellm.llms.wandb_hub.chat.handler import WandbHubChatHandler
from litellm.types.utils import ModelResponse


class TestWandbHubChatHandler:
    """Test WandbHubChatHandler functionality"""
    
    def test_wandb_hub_handler_initialization(self):
        """Test WandbHubChatHandler initializes correctly"""
        handler = WandbHubChatHandler()
        assert isinstance(handler, WandbHubChatHandler)

    @patch('litellm.llms.wandb_hub.chat.handler.HTTPHandler')
    def test_sync_completion_headers(self, mock_http_handler):
        """Test that headers are set correctly in sync completion"""
        handler = WandbHubChatHandler()
        
        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "moonshotai/Kimi-K2-Instruct",
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_http_handler.return_value = mock_client
        
        # Mock other dependencies
        model_response = ModelResponse()
        logging_obj = MagicMock()
        
        result = handler.completion(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://api.inference.wandb.ai/v1",
            custom_llm_provider="wandb_hub",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=lambda x: None,
            encoding=None,
            api_key="test-key",
            logging_obj=logging_obj,
            optional_params={"project_id": "test-project"}
        )
        
        # Verify that the HTTP client was called with correct headers
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        headers = call_args.kwargs['headers']
        
        # Check that OpenAI-Project header was set
        assert headers["OpenAI-Project"] == "test-project"

    @patch('litellm.llms.wandb_hub.chat.handler.make_sync_call')
    def test_sync_streaming_headers(self, mock_make_sync_call):
        """Test that headers are set correctly in sync streaming"""
        handler = WandbHubChatHandler()
        
        # Mock the streaming response
        mock_stream = MagicMock()
        mock_make_sync_call.return_value = mock_stream
        
        model_response = ModelResponse()
        logging_obj = MagicMock()
        
        result = handler.completion(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://api.inference.wandb.ai/v1",
            custom_llm_provider="wandb_hub",
            custom_prompt_dict={},
            model_response=model_response,
            print_verbose=lambda x: None,
            encoding=None,
            api_key="test-key",
            logging_obj=logging_obj,
            optional_params={"project": "test-project-2", "stream": True}
        )
        
        # Verify that make_sync_call was called with correct headers
        mock_make_sync_call.assert_called_once()
        call_args = mock_make_sync_call.call_args
        headers = call_args.kwargs['headers']
        
        # Check that OpenAI-Project header was set (using "project" field)
        assert headers["OpenAI-Project"] == "test-project-2"

    @pytest.mark.asyncio
    async def test_async_completion_headers(self):
        """Test that headers are set correctly in async completion"""
        handler = WandbHubChatHandler()
        
        # Mock the async HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "test-id",
            "model": "moonshotai/Kimi-K2-Instruct",
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        }
        mock_response.raise_for_status.return_value = None
        # Make the post method return a future that resolves to the mock response
        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.post.return_value = future
        
        model_response = ModelResponse()
        logging_obj = MagicMock()
        
        result = await handler.acompletion_function(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            api_base="https://api.inference.wandb.ai/v1",
            custom_prompt_dict={},
            model_response=model_response,
            custom_llm_provider="wandb_hub",
            print_verbose=lambda x: None,
            client=mock_client,
            encoding=None,
            api_key="test-key",
            logging_obj=logging_obj,
            stream=False,
            data={"model": "moonshotai/Kimi-K2-Instruct", "messages": [{"role": "user", "content": "Hello"}]},
            base_model=None,
            optional_params={"project_id": "async-project"}
        )
        
        # Verify that the async client was called with correct headers
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        headers = call_args.kwargs['headers']
        
        # Check that OpenAI-Project header was set
        assert headers["OpenAI-Project"] == "async-project"

    @pytest.mark.asyncio
    async def test_async_streaming_headers(self):
        """Test that headers are set correctly in async streaming"""
        handler = WandbHubChatHandler()
        
        # Mock make_call
        with patch('litellm.llms.wandb_hub.chat.handler.make_call') as mock_make_call:
            mock_stream = MagicMock()
            mock_make_call.return_value = mock_stream
            
            model_response = ModelResponse()
            logging_obj = MagicMock()
            
            result = await handler.acompletion_stream_function(
                model="moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": "Hello"}],
                custom_llm_provider="wandb_hub",
                api_base="https://api.inference.wandb.ai/v1",
                custom_prompt_dict={},
                model_response=model_response,
                print_verbose=lambda x: None,
                encoding=None,
                api_key="test-key",
                logging_obj=logging_obj,
                stream=True,
                data={"model": "moonshotai/Kimi-K2-Instruct", "messages": [{"role": "user", "content": "Hello"}]},
                optional_params={"project": "async-stream-project"},
                headers={}
            )
            
            # Verify that make_call was called with correct headers
            mock_make_call.assert_called_once()
            call_args = mock_make_call.call_args
            headers = call_args.kwargs['headers']
            
            # Check that OpenAI-Project header was set
            assert headers["OpenAI-Project"] == "async-stream-project"

    def test_error_handling_sync(self):
        """Test error handling in sync completion"""
        handler = WandbHubChatHandler()
        
        # Mock HTTP client that raises an exception
        with patch('litellm.llms.wandb_hub.chat.handler.HTTPHandler') as mock_http_handler:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            
            from litellm.llms.openai_like.common_utils import OpenAILikeError
            from httpx import HTTPStatusError, Request, Response
            
            # Create a proper HTTPStatusError
            request = Request("POST", "https://api.inference.wandb.ai/v1")
            response = Response(401, content="Unauthorized", request=request)
            mock_client.post.side_effect = HTTPStatusError("Unauthorized", request=request, response=response)
            mock_http_handler.return_value = mock_client
            
            model_response = ModelResponse()
            logging_obj = MagicMock()
            
            with pytest.raises(OpenAILikeError) as exc_info:
                handler.completion(
                    model="moonshotai/Kimi-K2-Instruct",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_base="https://api.inference.wandb.ai/v1",
                    custom_llm_provider="wandb_hub",
                    custom_prompt_dict={},
                    model_response=model_response,
                    print_verbose=lambda x: None,
                    encoding=None,
                    api_key="invalid-key",
                    logging_obj=logging_obj,
                    optional_params={"project_id": "test-project"}
                )
            
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_error_handling_async(self):
        """Test error handling in async completion"""
        handler = WandbHubChatHandler()
        
        # Mock async client that raises an exception
        mock_client = MagicMock()
        from httpx import HTTPStatusError, Request, Response
        from litellm.llms.openai_like.common_utils import OpenAILikeError
        
        # Create a proper HTTPStatusError
        request = Request("POST", "https://api.inference.wandb.ai/v1")
        response = Response(429, content="Rate limited", request=request)
        
        # Make post return a future that raises
        future = asyncio.Future()
        future.set_exception(HTTPStatusError("Rate limited", request=request, response=response))
        mock_client.post.return_value = future
        
        model_response = ModelResponse()
        logging_obj = MagicMock()
        
        with pytest.raises(OpenAILikeError) as exc_info:
            await handler.acompletion_function(
                model="moonshotai/Kimi-K2-Instruct",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="https://api.inference.wandb.ai/v1",
                custom_prompt_dict={},
                model_response=model_response,
                custom_llm_provider="wandb_hub",
                print_verbose=lambda x: None,
                client=mock_client,
                encoding=None,
                api_key="test-key",
                logging_obj=logging_obj,
                stream=False,
                data={"model": "moonshotai/Kimi-K2-Instruct", "messages": [{"role": "user", "content": "Hello"}]},
                base_model=None,
                optional_params={"project_id": "test-project"}
            )
        
        assert exc_info.value.status_code == 429