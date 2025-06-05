"""
Test HuggingFace LLM
"""

from re import M

import httpx
from base_llm_unit_tests import BaseLLMChatTest
from base_embedding_unit_tests import BaseLLMEmbeddingTest
import json
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
import pytest
from litellm.types.utils import ModelResponseStream, ModelResponse
from respx import MockRouter

MOCK_COMPLETION_RESPONSE = {
        "id": "9115d3daeab10608",
        "object": "chat.completion",
        "created": 11111,
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "prompt": [],
        "choices": [
            {
                "finish_reason": "stop",
                "seed": 3629048360264764400,
                "logprobs": None,
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the mocked HuggingFace API.",
                    "tool_calls": []
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

MOCK_STREAMING_CHUNKS = [
        {"id": "id1", "object": "chat.completion.chunk", "created": 1111, 
         "choices": [{"index": 0, "text": "Deep", "logprobs": None, "finish_reason": None, "seed": None, 
                     "delta": {"token_id": 34564, "role": "assistant", "content": "Deep", "tool_calls": None}}], 
         "model": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo", "usage": None},
        
        {"id": "id2", "object": "chat.completion.chunk", "created": 1111, 
         "choices": [{"index": 0, "text": " learning", "logprobs": None, "finish_reason": None, "seed": None, 
                     "delta": {"token_id": 6975, "role": "assistant", "content": " learning", "tool_calls": None}}], 
         "model": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo", "usage": None},
        
        {"id": "id3", "object": "chat.completion.chunk", "created": 1111, 
         "choices": [{"index": 0, "text": " is", "logprobs": None, "finish_reason": None, "seed": None, 
                     "delta": {"token_id": 374, "role": "assistant", "content": " is", "tool_calls": None}}], 
         "model": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo", "usage": None},
        
        {"id": "sid4", "object": "chat.completion.chunk", "created": 1111, 
         "choices": [{"index": 0, "text": " response", "logprobs": None, "finish_reason": "length", "seed": 2853637492034609700, 
                     "delta": {"token_id": 323, "role": "assistant", "content": " response", "tool_calls": None}}], 
         "model": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo", 
         "usage": {"prompt_tokens": 26, "completion_tokens": 20, "total_tokens": 46}}
    ]


PROVIDER_MAPPING_RESPONSE = {
    "fireworks-ai": {
      "status": "live",
      "providerId": "accounts/fireworks/models/llama-v3-8b-instruct",
      "task": "conversational"
    },
    "together": {
      "status": "live",
      "providerId": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
      "task": "conversational"
    },
    "hf-inference": {
      "status": "live",
      "providerId": "meta-llama/Meta-Llama-3-8B-Instruct",
      "task": "conversational"
    },
}

@pytest.fixture
def mock_provider_mapping():
    with patch("litellm.llms.huggingface.chat.transformation._fetch_inference_provider_mapping") as mock:
        mock.return_value = PROVIDER_MAPPING_RESPONSE
        yield mock
        
@pytest.fixture(autouse=True)
def clear_lru_cache():
    from litellm.llms.huggingface.common_utils import _fetch_inference_provider_mapping

    _fetch_inference_provider_mapping.cache_clear()
    yield
    _fetch_inference_provider_mapping.cache_clear()
    
@pytest.fixture
def mock_http_handler():
    """Fixture to mock the HTTP handler"""
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock:
        print(f"Creating mock HTTP handler: {mock}")
        
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        
        def mock_side_effect(*args, **kwargs):
            if kwargs.get("stream", True):
                mock_response.iter_lines.return_value = iter([
                    f"data: {json.dumps(chunk)}".encode('utf-8')
                    for chunk in MOCK_STREAMING_CHUNKS
                ] + [b'data: [DONE]'])
            else:
                mock_response.json.return_value = MOCK_COMPLETION_RESPONSE
            return mock_response
            
        mock.side_effect = mock_side_effect
        yield mock
        
@pytest.fixture
def mock_http_async_handler():
    """Fixture to mock the async HTTP handler"""
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock 
    ) as mock:
        print(f"Creating mock async HTTP handler: {mock}")
        
        mock_response = MagicMock()  
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        
        mock_response.json.return_value = MOCK_COMPLETION_RESPONSE
        mock_response.text = json.dumps(MOCK_COMPLETION_RESPONSE)
        
        async def mock_side_effect(*args, **kwargs):
            if kwargs.get("stream", True):
                async def mock_aiter():
                    for chunk in MOCK_STREAMING_CHUNKS:
                        yield f"data: {json.dumps(chunk)}".encode('utf-8')
                    yield b"data: [DONE]"
                
                mock_response.aiter_lines = mock_aiter
            return mock_response
            
        mock.side_effect = mock_side_effect
        yield mock  

class TestHuggingFace(BaseLLMChatTest):

    @pytest.fixture(autouse=True)
    def setup(self, mock_provider_mapping, mock_http_handler, mock_http_async_handler):

        self.mock_provider_mapping = mock_provider_mapping
        self.mock_http = mock_http_handler
        self.mock_http_async = mock_http_async_handler
        self.model = "huggingface/together/meta-llama/Meta-Llama-3-8B-Instruct"
        litellm.set_verbose = False
    def get_base_completion_call_args(self) -> dict:
        """Implementation of abstract method from BaseLLMChatTest"""
        return {"model": self.model}
    
    def test_completion_non_streaming(self):
        messages = [{"role": "user", "content": "This is a dummy message"}]
        
        response = litellm.completion(
            model=self.model,
            messages=messages,
            stream=False
        )
        assert isinstance(response, ModelResponse)
        assert response.choices[0].message.content == "This is a test response from the mocked HuggingFace API."
        assert response.usage is not None
        assert response.model == self.model.split("/",2)[2]
        
    def test_completion_streaming(self):
        messages = [{"role": "user", "content": "This is a dummy message"}]
        
        response = litellm.completion(
            model=self.model,
            messages=messages,
            stream=True
        )
        
        chunks = list(response)
        assert len(chunks) > 0
        
        assert self.mock_http.called
        call_args = self.mock_http.call_args
        assert call_args is not None
        
        kwargs = call_args[1]
        data = json.loads(kwargs["data"])
        assert data["stream"] is True
        assert data["messages"] == messages
        
        assert isinstance(chunks, list)
        assert isinstance(chunks[0], ModelResponseStream)
        assert isinstance(chunks[0].id, str)
        assert chunks[0].model == self.model.split("/",1)[1]
            
    @pytest.mark.asyncio
    async def test_async_completion_streaming(self):
        """Test async streaming completion"""
        messages = [{"role": "user", "content": "This is a dummy message"}]
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=True
        )

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        assert self.mock_http_async.called
        assert len(chunks) > 0
        assert isinstance(chunks[0], ModelResponseStream)
        assert isinstance(chunks[0].id, str)
        assert chunks[0].model == self.model.split("/",1)[1]
            
    @pytest.mark.asyncio
    async def test_async_completion_non_streaming(self):
        """Test async non-streaming completion"""
        messages = [{"role": "user", "content": "This is a dummy message"}]
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            stream=False
        )

        assert self.mock_http_async.called
        assert isinstance(response, ModelResponse)
        assert response.choices[0].message.content == "This is a test response from the mocked HuggingFace API."
        assert response.usage is not None
        assert response.model == self.model.split("/",2)[2]
    
    def test_tool_call_no_arguments(self, tool_call_no_arguments):
     
        mock_tool_response = {
            **MOCK_COMPLETION_RESPONSE,
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": tool_call_no_arguments  
            }]
        }
        
        with patch.object(self.mock_http, "side_effect", lambda *args, **kwargs: MagicMock(
            status_code=200,
            json=lambda: mock_tool_response,
            raise_for_status=lambda: None
        )):
            messages = [{"role": "user", "content": "Get the FAQ"}]
            tools = [{
                "type": "function",
                "function": {
                    "name": "Get-FAQ",
                    "description": "Get FAQ information",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }]
            
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            assert response.choices[0].message.tool_calls is not None
            assert len(response.choices[0].message.tool_calls) == 1
            assert response.choices[0].message.tool_calls[0].function.name == tool_call_no_arguments["tool_calls"][0]["function"]["name"]
            assert response.choices[0].message.tool_calls[0].function.arguments == tool_call_no_arguments["tool_calls"][0]["function"]["arguments"]

    @pytest.mark.parametrize(
        "model, provider, expected_url",
        [
            ("meta-llama/Llama-3-8B-Instruct", None, "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3-8B-Instruct/v1/chat/completions"),
            ("together/meta-llama/Llama-3-8B-Instruct", None, "https://router.huggingface.co/together/v1/chat/completions"),
            ("novita/meta-llama/Llama-3-8B-Instruct", None, "https://router.huggingface.co/novita/chat/completions"),
            ("http://custom-endpoint.com/v1/chat/completions", None, "http://custom-endpoint.com/v1/chat/completions"),
        ],
    )
    def test_get_complete_url(self, model, provider, expected_url):
        """Test that the complete URL is constructed correctly for different providers"""
        from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

        config = HuggingFaceChatConfig()
        url = config.get_complete_url(
            api_base=None,
            model=model,
            optional_params={},
            stream=False,
            api_key="test_api_key",
            litellm_params={}
        )
        assert url == expected_url

    def test_validate_environment(self):
        """Test that the environment is validated correctly"""
        from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig

        config = HuggingFaceChatConfig()
        
        headers = config.validate_environment(
            headers={},
            model="huggingface/fireworks-ai/meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            api_key="test_api_key",
            litellm_params={}
        )
        
        assert headers["Authorization"] == "Bearer test_api_key"
        assert headers["content-type"] == "application/json"

    @pytest.mark.parametrize(
        "model, expected_model",
        [
            ("together/meta-llama/Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct-Turbo"),
            ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
        ],
    )
    def test_transform_request(self, model, expected_model):
        from litellm.llms.huggingface.chat.transformation import HuggingFaceChatConfig
    
        config = HuggingFaceChatConfig()
        messages = [{"role": "user", "content": "Hello"}]
        
        transformed_request = config.transform_request(
            model=model,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={}
        )
        
        assert transformed_request["model"] == expected_model
        assert transformed_request["messages"] == messages

    @pytest.mark.asyncio
    async def test_completion_cost(self):
        pass


MOCK_EMBEDDING_RESPONSE = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Raw HuggingFace format: array of embeddings

@pytest.fixture
def mock_embedding_http_handler():
    """Fixture to mock the HTTP handler for embedding tests"""
    with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_EMBEDDING_RESPONSE
        mock_post.return_value = mock_response
        yield mock_post

@pytest.fixture
def mock_embedding_async_http_handler():
    """Fixture to mock the async HTTP handler for embedding tests"""
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = MOCK_EMBEDDING_RESPONSE
        mock_post.return_value = mock_response
        yield mock_post

class TestHuggingFaceEmbedding(BaseLLMEmbeddingTest):

    @pytest.fixture(autouse=True)
    def setup(self, mock_embedding_http_handler, mock_embedding_async_http_handler):
        # Mock the task detection function to avoid HTTP calls to HuggingFace model info
        self.mock_get_task_patcher = patch("litellm.llms.huggingface.embedding.handler.get_hf_task_embedding_for_model")
        self.mock_get_task = self.mock_get_task_patcher.start()

        # Make the mock behave like the real function: return task_type if provided, otherwise default
        def mock_get_task_side_effect(model, task_type, api_base):
            if task_type is not None:
                return task_type  # Return the task_type if explicitly provided
            return "sentence-similarity"  # Default task type when no task_type provided

        self.mock_get_task.side_effect = mock_get_task_side_effect

        self.model = "huggingface/BAAI/bge-m3"
        self.mock_http = mock_embedding_http_handler
        self.mock_async_http = mock_embedding_async_http_handler
        litellm.set_verbose = False

        yield

        # Clean up the patch
        self.mock_get_task_patcher.stop()

    def get_base_embedding_call_args(self) -> dict:
        """Implementation of abstract method from BaseLLMEmbeddingTest"""
        return {"model": self.model}

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        """Implementation of abstract method from BaseLLMEmbeddingTest"""
        return litellm.LlmProviders.HUGGINGFACE

    def test_input_type_preserved_in_optional_params(self):
        """Test that input_type is preserved in optional_params and not removed by pop"""

        input_text = ["hello world"]

        response = litellm.embedding(
            model=self.model,
            input=input_text,
            input_type="embed",
        )

        # Verify the mock was called
        self.mock_http.assert_called_once()
        post_call_args = self.mock_http.call_args
        request_data = json.loads(post_call_args[1]["data"])

        # When input_type="embed", it should use simple format {"inputs": [...]}
        # NOT the sentence-similarity format which would require 2+ sentences
        assert "inputs" in request_data
        assert request_data["inputs"] == input_text

        # Should NOT have sentence-similarity format
        assert "source_sentence" not in str(request_data)
        assert "sentences" not in str(request_data)

    def test_embedding_with_sentence_similarity_task(self):
        """Test embedding when task type is sentence-similarity (requires 2+ sentences)"""

        similarity_response = {
            "similarities": [[0, 0.9], [1, 0.8]]
        }

        # Update the mock response for this test
        self.mock_http.return_value.json.return_value = similarity_response

        # Test with 2+ sentences (required for sentence-similarity)
        input_text = ["This is the source sentence", "This is sentence one", "This is sentence two"]

        response = litellm.embedding(
            model=self.model,
            input=input_text,
            # Use the model's natural task type (sentence-similarity)
        )

        self.mock_http.assert_called_once()
        post_call_args = self.mock_http.call_args
        request_data = json.loads(post_call_args[1]["data"])

        assert "inputs" in request_data
        assert "source_sentence" in request_data["inputs"]
        assert "sentences" in request_data["inputs"]
        assert request_data["inputs"]["source_sentence"] == input_text[0]
        assert request_data["inputs"]["sentences"] == input_text[1:]