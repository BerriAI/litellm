"""
Integration test for Ollama Turbo via LiteLLM and native Ollama client.

This test requires a valid Ollama Turbo API key to run.
Set the OLLAMA_API_KEY environment variable before running.

Run with: pytest tests/local_testing/test_ollama_turbo_integration.py -v -s

Tests:
1. Auth header format (without Bearer prefix for ollama.com)
2. Environment variable pickup (OLLAMA_API_KEY)
3. Real API calls to Ollama Turbo
"""
import os
import sys

import pytest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm import completion

# Skip integration tests if no API key is provided
integration_tests = pytest.mark.skipif(
    not os.getenv("OLLAMA_API_KEY"), reason="OLLAMA_API_KEY not set"
)


class TestOllamaTurboAuthHeaders:
    """Test auth header formatting for Ollama Turbo."""

    def test_ollama_turbo_auth_header_without_bearer(self):
        """Test that ollama.com URLs get auth header without Bearer prefix."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()

        with mock.patch.object(client, "post") as mock_post:
            try:
                completion(
                    model="ollama_chat/gpt-oss:120b",
                    messages=[{"role": "user", "content": "test"}],
                    api_base="https://ollama.com",
                    api_key="test_key_123",
                    client=client,
                )
            except Exception:
                pass

            mock_post.assert_called()

            # Check the headers
            headers = mock_post.call_args.kwargs.get("headers", {})

            # Should NOT have Bearer prefix for ollama.com
            assert headers.get("Authorization") == "test_key_123"
            assert "Bearer" not in headers.get("Authorization", "")

    def test_ollama_env_var_pickup(self):
        """Test that OLLAMA_API_KEY environment variable is picked up."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        # Set the environment variable
        test_api_key = "env_test_key_789"
        original_key = os.environ.get("OLLAMA_API_KEY")
        os.environ["OLLAMA_API_KEY"] = test_api_key

        try:
            client = HTTPHandler()

            with mock.patch.object(client, "post") as mock_post:
                try:
                    # Don't pass api_key explicitly - it should pick up from env
                    completion(
                        model="ollama_chat/gpt-oss:120b",
                        messages=[{"role": "user", "content": "test"}],
                        api_base="https://ollama.com",
                        client=client,
                    )
                except Exception:
                    pass

                mock_post.assert_called()

                # Check the headers
                headers = mock_post.call_args.kwargs.get("headers", {})

                # Should have the env var value without Bearer prefix
                assert headers.get("Authorization") == test_api_key
                assert "Bearer" not in headers.get("Authorization", "")

        finally:
            # Restore original value
            if original_key is not None:
                os.environ["OLLAMA_API_KEY"] = original_key
            else:
                del os.environ["OLLAMA_API_KEY"]

    def test_ollama_turbo_embedding_auth_header(self):
        """Test that embedding calls to ollama.com get correct auth header."""
        import litellm

        with mock.patch.object(litellm.module_level_client, "post") as mock_post:
            # Mock the embedding response
            mock_response = mock.Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
            mock_post.return_value = mock_response

            try:
                from litellm import embedding

                embedding(
                    model="ollama/nomic-embed-text",
                    input=["test embedding"],
                    api_base="https://ollama.com",
                    api_key="test_embed_key",
                )
            except Exception:
                pass

            mock_post.assert_called()

            # Check the headers for embedding call
            headers = mock_post.call_args.kwargs.get("headers", {})

            # Should NOT have Bearer prefix for ollama.com
            assert headers.get("Authorization") == "test_embed_key"
            assert "Bearer" not in headers.get("Authorization", "")

    def test_ollama_turbo_vision_auth_header(self):
        """Test that vision/multimodal calls to ollama.com get correct auth header."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()

        # Mock the image conversion to avoid PIL dependency
        with mock.patch(
            "litellm.llms.ollama.common_utils._convert_image"
        ) as mock_convert:
            mock_convert.return_value = "base64_image_data"

            with mock.patch.object(client, "post") as mock_post:
                try:
                    completion(
                        model="ollama/llama3.2-vision:11b",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What's in this image?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "https://dummyimage.com/100/100/fff&text=Test+image"
                                        },
                                    },
                                ],
                            }
                        ],
                        api_base="https://ollama.com",
                        api_key="test_vision_key",
                        client=client,
                    )
                except Exception:
                    pass

                mock_post.assert_called()

                # Check the headers for vision call
                headers = mock_post.call_args.kwargs.get("headers", {})

                # Should NOT have Bearer prefix for ollama.com
                assert headers.get("Authorization") == "test_vision_key"
                assert "Bearer" not in headers.get("Authorization", "")


class TestOllamaTurboIntegration:
    """Integration tests for Ollama Turbo service."""

    @integration_tests
    def test_litellm_ollama_turbo_completion(self):
        """Test Ollama Turbo via LiteLLM completion."""
        api_key = os.getenv("OLLAMA_API_KEY")

        response = completion(
            model="ollama/gpt-oss:120b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Say 'Hello from Ollama Turbo via LiteLLM' and nothing else.",
                },
            ],
            api_base="https://ollama.com",
            api_key=api_key,
            max_tokens=50,
        )

        assert response.choices[0].message.content
        assert (
            "Ollama Turbo" in response.choices[0].message.content
            or "Hello" in response.choices[0].message.content
        )

    @integration_tests
    @pytest.mark.skipif(
        not os.path.exists(
            os.path.expanduser(
                "~/.local/share/pipx/venvs/poetry/lib/python3.12/site-packages/ollama"
            )
        ),
        reason="ollama package not installed",
    )
    def test_native_ollama_turbo_completion(self):
        """Test Ollama Turbo via native Ollama client."""
        try:
            from ollama import Client
        except ImportError:
            pytest.skip("ollama package not installed")

        api_key = os.getenv("OLLAMA_API_KEY")

        client = Client(
            host="https://ollama.com", headers={"Authorization": f"{api_key}"}
        )

        messages = [
            {
                "role": "user",
                "content": 'Say "Hello from Ollama Turbo native client" and nothing else.',
            },
        ]

        response_parts = []
        for part in client.chat("gpt-oss:120b", messages=messages, stream=True):
            if "message" in part and "content" in part["message"]:
                response_parts.append(part["message"]["content"])

        full_response = "".join(response_parts)
        assert full_response
        assert "Ollama Turbo" in full_response or "Hello" in full_response

    @integration_tests
    @pytest.mark.asyncio
    async def test_litellm_ollama_turbo_async_streaming(self):
        """Test async streaming with Ollama Turbo via LiteLLM."""
        from litellm import acompletion

        api_key = os.getenv("OLLAMA_API_KEY")

        response_parts = []
        async for chunk in await acompletion(
            model="ollama/gpt-oss:120b",
            messages=[{"role": "user", "content": "Count from 1 to 3"}],
            api_base="https://ollama.com",
            api_key=api_key,
            stream=True,
            max_tokens=50,
        ):
            if chunk.choices[0].delta.content:
                response_parts.append(chunk.choices[0].delta.content)

        full_response = "".join(response_parts)
        assert full_response
