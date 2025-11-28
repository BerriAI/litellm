import asyncio
import time
import unittest
from unittest.mock import patch, AsyncMock

import httpx

import litellm
from litellm.llms.custom_httpx import http_handler


class TestSharedSession(unittest.TestCase):
    """
    Tests the shared session behavior for various LiteLLM endpoints.

    This test suite verifies two primary scenarios:
    1. LiteLLM-Managed Session: Ensures that when no session is provided, LiteLLM
       internally creates and reuses a single HTTP client for subsequent calls
       to the same endpoint.
    2. User-Provided Session: Ensures that when a user provides their own `httpx.AsyncClient`
       instance via the `shared_session` parameter, LiteLLM correctly uses that
       specific session for making API calls.

    The tests work by mocking the transport creation layer and asserting that it is
    only called once across multiple API calls, proving that the connection pool
    is being reused.
    """

    # Constants for test configuration
    DUMMY_API_KEY = "dummy_key"
    MOCK_MODEL_GPT = "gpt-3.5-turbo"
    MOCK_PROVIDER_OPENAI = "openai"
    MOCK_RESPONSE_ID = "resp_123"

    def setUp(self):
        """Set up the test environment before each test."""
        # Force litellm to use httpx transport instead of aiohttp, as our mocks
        # are specific to the httpx handler.
        self.should_use_aiohttp_patcher = patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler._should_use_aiohttp_transport",
            return_value=False,
        )
        self.should_use_aiohttp_patcher.start()

        # Store the original transport creation method to be used by our mock's side_effect.
        self.original_create_transport = (
            http_handler.AsyncHTTPHandler._create_async_transport
        )

    def tearDown(self):
        """Clean up the test environment after each test."""
        self.should_use_aiohttp_patcher.stop()
        # Clear the client cache to ensure tests are isolated.
        litellm.in_memory_llm_clients_cache.cache_dict.clear()
        litellm.in_memory_llm_clients_cache.ttl_dict.clear()
        litellm.in_memory_llm_clients_cache.expiration_heap.clear()

    async def _test_endpoint(
        self, endpoint_func, mock_response_payload, endpoint_kwargs, use_shared_session=False
    ):
        """
        A generic helper method to test an endpoint for shared session behavior.

        Args:
            endpoint_func (callable): The LiteLLM endpoint function to test (e.g., `litellm.acompletion`).
            mock_response_payload (dict): The JSON payload for the mock HTTP response.
            endpoint_kwargs (dict): The keyword arguments to pass to the endpoint function.
            use_shared_session (bool): If True, tests with a user-provided session.
                                       If False, tests LiteLLM's internal session management.
        """
        mock_response = httpx.Response(200, json=mock_response_payload)
        user_session = httpx.AsyncClient() if use_shared_session else None

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler._create_async_transport"
        ) as mock_create_transport, patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.delete",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            mock_create_transport.side_effect = self.original_create_transport

            if user_session:
                endpoint_kwargs["shared_session"] = user_session

            # First call
            await endpoint_func(**endpoint_kwargs)
            
            mock_create_transport.assert_called_once()

            if use_shared_session:
                # Assert that the transport was created with the session we provided.
                # This is the key check that would have caught the bug.
                call_kwargs = mock_create_transport.call_args.kwargs
                passed_session = call_kwargs.get("shared_session")
                self.assertIs(
                    passed_session,
                    user_session,
                    f"Endpoint '{endpoint_func.__name__}' failed to pass the shared_session to the HTTP handler. "
                    f"Expected session object {user_session}, but got {passed_session}."
                )

            # Second call
            await endpoint_func(**endpoint_kwargs)
            self.assertEqual(
                mock_create_transport.call_count,
                1,
                f"Transport should be reused on the second call for {endpoint_func.__name__}",
            )

    def test_acompletion(self):
        """Tests shared session behavior for the `litellm.acompletion` endpoint."""
        endpoint_kwargs = {
            "model": self.MOCK_MODEL_GPT,
            "messages": [{"role": "user", "content": "hello"}],
            "api_key": self.DUMMY_API_KEY,
        }
        mock_payload = {
            "choices": [{"message": {"role": "assistant", "content": "mocked response"}}]
        }

        for use_session in [True, False]:
            with self.subTest(use_shared_session=use_session):
                asyncio.run(
                    self._test_endpoint(
                        endpoint_func=litellm.acompletion,
                        mock_response_payload=mock_payload,
                        endpoint_kwargs=endpoint_kwargs.copy(),
                        use_shared_session=use_session,
                    )
                )

    def test_aresponses(self):
        """Tests shared session behavior for the `litellm.aresponses` endpoint."""
        endpoint_kwargs = {
            "model": self.MOCK_MODEL_GPT,
            "input": "hello",
            "api_key": self.DUMMY_API_KEY,
        }
        mock_payload = {
            "id": self.MOCK_RESPONSE_ID,
            "object": "response",
            "created_at": int(time.time()),
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

        for use_session in [True, False]:
            with self.subTest(use_shared_session=use_session):
                asyncio.run(
                    self._test_endpoint(
                        endpoint_func=litellm.aresponses,
                        mock_response_payload=mock_payload,
                        endpoint_kwargs=endpoint_kwargs.copy(),
                        use_shared_session=use_session,
                    )
                )

    def test_adelete_responses(self):
        """Tests shared session behavior for the `litellm.adelete_responses` endpoint."""
        endpoint_kwargs = {
            "response_id": self.MOCK_RESPONSE_ID,
            "api_key": self.DUMMY_API_KEY,
            "custom_llm_provider": self.MOCK_PROVIDER_OPENAI,
        }
        mock_payload = {"id": self.MOCK_RESPONSE_ID, "object": "response", "deleted": True}

        for use_session in [True, False]:
            with self.subTest(use_shared_session=use_session):
                asyncio.run(
                    self._test_endpoint(
                        endpoint_func=litellm.adelete_responses,
                        mock_response_payload=mock_payload,
                        endpoint_kwargs=endpoint_kwargs.copy(),
                        use_shared_session=use_session,
                    )
                )

    def test_aget_responses(self):
        """Tests shared session behavior for the `litellm.aget_responses` endpoint."""
        endpoint_kwargs = {
            "response_id": self.MOCK_RESPONSE_ID,
            "api_key": self.DUMMY_API_KEY,
            "custom_llm_provider": self.MOCK_PROVIDER_OPENAI,
        }
        mock_payload = {
            "id": self.MOCK_RESPONSE_ID,
            "object": "response",
            "created_at": int(time.time()),
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

        for use_session in [True, False]:
            with self.subTest(use_shared_session=use_session):
                asyncio.run(
                    self._test_endpoint(
                        endpoint_func=litellm.aget_responses,
                        mock_response_payload=mock_payload,
                        endpoint_kwargs=endpoint_kwargs.copy(),
                        use_shared_session=use_session,
                    )
                )
    
    def test_alist_input_items(self):
        """Tests shared session behavior for the `litellm.alist_input_items` endpoint."""
        endpoint_kwargs = {
            "response_id": self.MOCK_RESPONSE_ID,
            "api_key": self.DUMMY_API_KEY,
            "custom_llm_provider": self.MOCK_PROVIDER_OPENAI,
        }
        mock_payload = {"object": "list", "data": []}

        for use_session in [True, False]:
            with self.subTest(use_shared_session=use_session):
                asyncio.run(
                    self._test_endpoint(
                        endpoint_func=litellm.alist_input_items,
                        mock_response_payload=mock_payload,
                        endpoint_kwargs=endpoint_kwargs.copy(),
                        use_shared_session=use_session,
                    )
                )

    def test_acancel_responses(self):
        """Tests shared session behavior for the `litellm.acancel_responses` endpoint."""
        endpoint_kwargs = {
            "response_id": self.MOCK_RESPONSE_ID,
            "api_key": self.DUMMY_API_KEY,
            "custom_llm_provider": self.MOCK_PROVIDER_OPENAI,
        }
        mock_payload = {
            "id": self.MOCK_RESPONSE_ID,
            "object": "response",
            "status": "cancelled",
            "created_at": int(time.time()),
            "output": [],
        }

        for use_session in [True, False]:
            with self.subTest(use_shared_session=use_session):
                asyncio.run(
                    self._test_endpoint(
                        endpoint_func=litellm.acancel_responses,
                        mock_response_payload=mock_payload,
                        endpoint_kwargs=endpoint_kwargs.copy(),
                        use_shared_session=use_session,
                    )
                )

    @patch(
        "litellm.llms.openai.openai.OpenAI.make_openai_embedding_request",
        new_callable=AsyncMock,
    )
    def test_aembedding(self, mock_make_request):
        """Tests shared session behavior for the `litellm.aembedding` endpoint."""
        from openai.types import CreateEmbeddingResponse, Embedding
        from openai.types.embedding_usage import EmbeddingUsage

        mock_openai_response = CreateEmbeddingResponse(
            object="list",
            data=[
                Embedding(
                    object="embedding",
                    embedding=[0.1, 0.2, 0.3],
                    index=0,
                )
            ],
            model="text-embedding-ada-002",
            usage=EmbeddingUsage(prompt_tokens=2, total_tokens=2),
        )
        mock_make_request.return_value = ({}, mock_openai_response)

        endpoint_kwargs = {
            "model": "text-embedding-ada-002",
            "input": ["hello world"],
            "api_key": self.DUMMY_API_KEY,
        }

        async def run_test(use_shared_session):
            """Helper coroutine to run the actual test logic."""
            test_kwargs = endpoint_kwargs.copy()
            user_session = httpx.AsyncClient() if use_shared_session else None
            if user_session:
                test_kwargs["shared_session"] = user_session

            # First call
            await litellm.aembedding(**test_kwargs)
            # Second call
            await litellm.aembedding(**test_kwargs)

            # Assert that the underlying request method was called twice.
            self.assertEqual(mock_make_request.call_count, 2)

            # Get the arguments from the last call.
            last_call_kwargs = mock_make_request.call_args.kwargs

            if use_shared_session:
                # Assert that the shared session was passed through correctly.
                passed_session = last_call_kwargs.get("shared_session")
                self.assertIs(
                    passed_session,
                    user_session,
                    "The user-provided shared_session was not passed to the request handler.",
                )
            else:
                # Assert that no session was passed when not provided.
                self.assertIsNone(
                    last_call_kwargs.get("shared_session"),
                    "A session was passed to the request handler even though none was provided.",
                )

        for use_session in [True, False]:
            with self.subTest(use_shared_session=use_session):
                # Reset mock for each sub-test to ensure call counts are isolated.
                mock_make_request.reset_mock()
                asyncio.run(run_test(use_session))


if __name__ == "__main__":
    unittest.main()
