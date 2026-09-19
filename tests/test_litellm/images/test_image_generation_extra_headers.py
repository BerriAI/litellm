"""
Unit test for https://github.com/BerriAI/litellm/issues/22285

Verifies that extra_headers passed to image_generation() are forwarded
to the OpenAI SDK on the openai/litellm_proxy/openai_compatible_providers
code paths.
"""

import json
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.images.main import image_generation


class TestImageGenerationExtraHeaders:
    """Test that extra_headers are forwarded on the OpenAI code path."""

    @patch("litellm.images.main.openai_chat_completions")
    def test_extra_headers_forwarded_to_openai_image_generation(self, mock_openai_chat_completions):
        """
        extra_headers passed to image_generation() should appear in
        optional_params["extra_headers"] when the provider is openai.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        extra_headers = {"traceparent": "00-abc123-def456-01", "X-Custom": "value"}

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
            extra_headers=extra_headers,
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get("optional_params", call_kwargs[1].get("optional_params", {}))

        assert "extra_headers" in optional_params
        assert optional_params["extra_headers"] == extra_headers

    @patch("litellm.images.main.openai_chat_completions")
    def test_no_extra_headers_when_not_provided(self, mock_openai_chat_completions):
        """
        When extra_headers is not passed, optional_params should not
        contain extra_headers.
        """
        mock_image_response = litellm.utils.ImageResponse(
            created=1234567890,
            data=[{"url": "https://example.com/image.png"}],
        )
        mock_openai_chat_completions.image_generation.return_value = mock_image_response

        image_generation(
            model="openai/dall-e-3",
            prompt="A red circle",
        )

        mock_openai_chat_completions.image_generation.assert_called_once()
        call_kwargs = mock_openai_chat_completions.image_generation.call_args
        optional_params = call_kwargs.kwargs.get("optional_params", call_kwargs[1].get("optional_params", {}))

        assert "extra_headers" not in optional_params

    def test_openai_image_generation_excludes_extra_headers_from_body(self):
        captured: Final[list[httpx.Request]] = []  # mutable-ok: test request capture log

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"data": [{"url": "https://example.com/image.png"}]})

        transport: Final = httpx.MockTransport(handle_request)
        client: Final = litellm.OpenAI(api_key="fake-key", http_client=httpx.Client(transport=transport))

        image_generation(
            model="gpt-image-2",
            prompt="test prompt",
            client=client,
            extra_headers={"cf-aig-auth": "secret-123"},
        )

        assert len(captured) == 1
        req: Final = captured[0]
        assert req.headers.get("cf-aig-auth") == "secret-123"
        body: Final = json.loads(req.read())
        assert "extra_headers" not in body
        assert body == {"prompt": "test prompt", "model": "gpt-image-2"}

    @pytest.mark.asyncio
    async def test_openai_aimage_generation_excludes_extra_headers_from_body(self):
        captured: Final[list[httpx.Request]] = []  # mutable-ok: test request capture log

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"data": [{"url": "https://example.com/image.png"}]})

        transport: Final = httpx.MockTransport(handle_request)
        client: Final = litellm.AsyncOpenAI(api_key="fake-key", http_client=httpx.AsyncClient(transport=transport))

        await litellm.aimage_generation(
            model="gpt-image-2",
            prompt="async test prompt",
            client=client,
            extra_headers={"cf-aig-auth": "async-secret-123"},
        )

        assert len(captured) == 1
        req: Final = captured[0]
        assert req.headers.get("cf-aig-auth") == "async-secret-123"
        body: Final = json.loads(req.read())
        assert "extra_headers" not in body
        assert body == {"prompt": "async test prompt", "model": "gpt-image-2"}

    def test_openai_image_generation_with_headers_excludes_from_body(self):
        captured: Final[list[httpx.Request]] = []  # mutable-ok: test request capture log

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"data": [{"url": "https://example.com/image.png"}]})

        transport: Final = httpx.MockTransport(handle_request)
        client: Final = litellm.OpenAI(api_key="fake-key", http_client=httpx.Client(transport=transport))

        image_generation(
            model="gpt-image-2",
            prompt="headers test",
            client=client,
            headers={"custom-header": "custom-val"},
        )

        assert len(captured) == 1
        req: Final = captured[0]
        assert req.headers.get("custom-header") == "custom-val"
        body: Final = json.loads(req.read())
        assert "headers" not in body
        assert "extra_headers" not in body
        assert body == {"prompt": "headers test", "model": "gpt-image-2"}

    def test_openai_image_generation_sanitizes_extra_body_and_preserves_siblings(self):
        captured: Final[list[httpx.Request]] = []  # mutable-ok: test request capture log

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"data": [{"url": "https://example.com/image.png"}]})

        transport: Final = httpx.MockTransport(handle_request)
        client: Final = litellm.OpenAI(api_key="fake-key", http_client=httpx.Client(transport=transport))

        image_generation(
            model="gpt-image-2",
            prompt="test prompt",
            client=client,
            extra_body={"extra_headers": {"cf-aig-auth": "secret-123"}, "custom_sibling": "sibling_val"},
        )

        assert len(captured) == 1
        req: Final = captured[0]
        body: Final = json.loads(req.read())
        assert "extra_headers" not in body
        assert body.get("custom_sibling") == "sibling_val"
        assert body == {"prompt": "test prompt", "model": "gpt-image-2", "custom_sibling": "sibling_val"}

    @pytest.mark.asyncio
    async def test_openai_aimage_generation_sanitizes_extra_body_and_preserves_siblings(self):
        captured: Final[list[httpx.Request]] = []  # mutable-ok: test request capture log

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"data": [{"url": "https://example.com/image.png"}]})

        transport: Final = httpx.MockTransport(handle_request)
        client: Final = litellm.AsyncOpenAI(api_key="fake-key", http_client=httpx.AsyncClient(transport=transport))

        await litellm.aimage_generation(
            model="gpt-image-2",
            prompt="async test prompt",
            client=client,
            extra_body={"extra_headers": {"cf-aig-auth": "secret-123"}, "custom_sibling": "async_sibling_val"},
        )

        assert len(captured) == 1
        req: Final = captured[0]
        body: Final = json.loads(req.read())
        assert "extra_headers" not in body
        assert body.get("custom_sibling") == "async_sibling_val"
        assert body == {"prompt": "async test prompt", "model": "gpt-image-2", "custom_sibling": "async_sibling_val"}

    def test_openai_image_generation_extra_body_only_headers_pops_empty_extra_body(self):
        captured: Final[list[httpx.Request]] = []  # mutable-ok: test request capture log

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"data": [{"url": "https://example.com/image.png"}]})

        transport: Final = httpx.MockTransport(handle_request)
        client: Final = litellm.OpenAI(api_key="fake-key", http_client=httpx.Client(transport=transport))

        image_generation(
            model="gpt-image-2",
            prompt="test prompt",
            client=client,
            extra_body={"extra_headers": {"cf-aig-auth": "secret-123"}},
        )

        assert len(captured) == 1
        req: Final = captured[0]
        body: Final = json.loads(req.read())
        assert "extra_headers" not in body
        assert "extra_body" not in body
        assert body == {"prompt": "test prompt", "model": "gpt-image-2"}

    @pytest.mark.asyncio
    async def test_openai_aimage_generation_extra_body_only_headers_pops_empty_extra_body(self):
        captured: Final[list[httpx.Request]] = []  # mutable-ok: test request capture log

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"data": [{"url": "https://example.com/image.png"}]})

        transport: Final = httpx.MockTransport(handle_request)
        client: Final = litellm.AsyncOpenAI(api_key="fake-key", http_client=httpx.AsyncClient(transport=transport))

        await litellm.aimage_generation(
            model="gpt-image-2",
            prompt="async test prompt",
            client=client,
            extra_body={"extra_headers": {"cf-aig-auth": "secret-123"}},
        )

        assert len(captured) == 1
        req: Final = captured[0]
        body: Final = json.loads(req.read())
        assert "extra_headers" not in body
        assert "extra_body" not in body
        assert body == {"prompt": "async test prompt", "model": "gpt-image-2"}
