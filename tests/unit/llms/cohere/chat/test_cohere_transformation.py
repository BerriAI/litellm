import json
from collections.abc import Callable, Iterator
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from unittest.mock import AsyncMock, patch

COHERE_V1_CHAT_URL: Final = "https://api.cohere.ai/v1/chat"
COHERE_V2_CHAT_URL: Final = "https://api.cohere.com/v2/chat"


@pytest.fixture
def _cohere_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "sync_transport", None, raising=False)
    yield
    client_cache.flush_cache()


def _mock_post_response(mock_post: MagicMock, response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    def _respond(request: httpx.Request) -> httpx.Response:
        mock_post(request)
        return response

    return _respond


class TestCohereTransform:
    def setup_method(self):
        self.config = CohereChatConfig()
        self.model = "command-r-plus-latest"
        self.logging_obj = MagicMock()

    def test_map_cohere_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_cohere_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_tokens": 200}


class TestCohereV2Transform:
    def setup_method(self):
        self.config = CohereV2ChatConfig()
        self.model = "command-r"

    def test_v2_supports_max_completion_tokens(self):
        """max_completion_tokens must be advertised so get_optional_params does not reject it"""
        assert "max_completion_tokens" in self.config.get_supported_openai_params(self.model)

    def test_v2_max_tokens_only_still_maps(self):
        """max_tokens alone maps to cohere max_tokens when max_completion_tokens is absent"""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 200}

    def test_v2_map_max_completion_tokens_overrides_max_tokens(self):
        """max_completion_tokens maps to cohere max_tokens and overrides max_tokens, matching v1"""
        result = self.config.map_openai_params(
            non_default_params={
                "temperature": 0.7,
                "max_tokens": 200,
                "max_completion_tokens": 256,
            },
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_v2_max_completion_tokens_precedence_is_order_independent(self):
        """max_completion_tokens wins over max_tokens regardless of dict ordering"""
        max_tokens_first = self.config.map_openai_params(
            non_default_params={"max_tokens": 200, "max_completion_tokens": 256},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        max_completion_first = self.config.map_openai_params(
            non_default_params={"max_completion_tokens": 256, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert max_tokens_first == {"max_tokens": 256}
        assert max_completion_first == {"max_tokens": 256}

    def test_v2_default_route_accepts_max_completion_tokens(self):
        """The default cohere_chat route resolves to v2; max_completion_tokens must not raise"""
        optional_params = litellm.get_optional_params(
            model=self.model,
            custom_llm_provider="cohere_chat",
            max_completion_tokens=256,
        )

        assert optional_params["max_tokens"] == 256


@pytest.mark.asyncio
async def test_cohere_request_body_with_allowed_params():
    """
    Test to validate that when allowed_openai_params is provided, the request body contains
    the correct response_format and reasoning_effort values.
    """
    # Define test parameters
    test_response_format = {"type": "json"}
    test_reasoning_effort = "low"
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "I am Command, a language model developed by Cohere.",
        "generation_id": "mock-generation-id",
        "finish_reason": "COMPLETE",
    }

    # Mock the AsyncHTTPHandler.post method at the module level
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        try:
            await litellm.acompletion(
                model="cohere/v1/command",
                messages=[{"content": "what llm are you", "role": "user"}],
                allowed_openai_params=["tools", "response_format", "reasoning_effort"],
                response_format=test_response_format,
                reasoning_effort=test_reasoning_effort,
                tools=test_tools,
            )
        except Exception:
            pass  # We only care about the request body validation

        # Verify the API call was made
        mock_post.assert_called_once()

        # Get and parse the request body
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        print(f"request_data: {request_data}")

        # Validate request contains our specified parameters
        assert "allowed_openai_params" not in request_data
        assert request_data["response_format"] == test_response_format
        assert request_data["reasoning_effort"] == test_reasoning_effort


@pytest.mark.asyncio
async def test_cohere_documents_options_in_request_body():
    """
    Test that documents parameters is properly included
    in the request body after transformation (sent via extra_body).
    """
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Test response with citations",
        "generation_id": "mock-generation-id",
        "finish_reason": "COMPLETE",
    }

    # Mock the AsyncHTTPHandler.post method
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        try:
            # Test documents and citation_options parameters
            test_documents = [
                {
                    "data": {
                        "title": "Test Document 1",
                        "snippet": "This is test content 1",
                    }
                },
                {
                    "data": {
                        "title": "Test Document 2",
                        "snippet": "This is test content 2",
                    }
                },
            ]
            await litellm.acompletion(
                model="cohere_chat/command-a-03-2025",
                messages=[{"role": "user", "content": "Test message"}],
                documents=test_documents,
            )
        except Exception:
            pass  # We only care about the request body validation

        # Verify the API call was made
        mock_post.assert_called_once()

        # Get and parse the request body
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        print(f"Request body: {request_data}")

        # Validate that documents and citation_options are in the request body
        assert "documents" in request_data
        assert request_data["documents"] == test_documents
