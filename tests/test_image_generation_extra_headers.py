"""
Tests for OpenAI image generation extra_headers handling.

Ensures extra_headers are used as HTTP headers only and never leak into the
JSON request body sent to the upstream API.

Ref: https://github.com/BerriAI/litellm/issues/40628
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestOpenAIImageGenerationExtraHeaders:
    """Verify extra_headers stays on the HTTP layer, not the JSON body."""

    def test_extra_headers_not_in_request_body(self):
        """
        When optional_params contains extra_headers (e.g. from credential config),
        the OpenAI images.generate() call must NOT receive extra_headers as a
        JSON body field.
        """
        from litellm.llms.openai.openai import OpenAIHandler

        handler = OpenAIHandler(api_key="sk-test", api_base="https://example.com/v1")

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "data": [{"url": "https://example.com/img.png"}]
        }

        mock_client = MagicMock()
        mock_client.images.generate.return_value = mock_response
        mock_client.api_key = "sk-test"
        mock_client._base_url = MagicMock()
        mock_client._base_url._uri_reference = "https://example.com/v1/"

        logging_obj = MagicMock()

        optional_params = {
            "n": 1,
            "size": "1024x1024",
            "extra_headers": {"cf-aig-authorization": "Bearer cfut_123"},
        }

        result = handler.image_generation(
            model="gpt-image-2",
            prompt="a red circle",
            timeout=60,
            optional_params=optional_params,
            logging_obj=logging_obj,
            api_key="sk-test",
            api_base="https://example.com/v1",
            model_response=MagicMock(),
            client=mock_client,
            aimg_generation=False,
        )

        # Verify the call was made
        assert mock_client.images.generate.called

        # Inspect the kwargs passed to images.generate
        call_kwargs = mock_client.images.generate.call_args
        # extra_headers should NOT be in the body parameters
        body_params = call_kwargs.kwargs
        assert "extra_headers" not in body_params, (
            "extra_headers leaked into the JSON body params! "
            f"Got params: {list(body_params.keys())}"
        )

    def test_extra_headers_passed_as_http_headers(self):
        """
        When headers are explicitly provided, they should be passed via
        extra_headers keyword to the SDK (which sends them as HTTP headers).
        """
        from litellm.llms.openai.openai import OpenAIHandler

        handler = OpenAIHandler(api_key="sk-test", api_base="https://example.com/v1")

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "data": [{"url": "https://example.com/img.png"}]
        }

        mock_client = MagicMock()
        mock_client.images.generate.return_value = mock_response
        mock_client.api_key = "sk-test"
        mock_client._base_url = MagicMock()
        mock_client._base_url._uri_reference = "https://example.com/v1/"

        logging_obj = MagicMock()

        optional_params = {"n": 1, "size": "1024x1024"}
        headers = {"cf-aig-authorization": "Bearer cfut_123"}

        result = handler.image_generation(
            model="gpt-image-2",
            prompt="a red circle",
            timeout=60,
            optional_params=optional_params,
            logging_obj=logging_obj,
            api_key="sk-test",
            api_base="https://example.com/v1",
            model_response=MagicMock(),
            client=mock_client,
            aimg_generation=False,
            headers=headers,
        )

        # Verify the call was made
        assert mock_client.images.generate.called

        call_kwargs = mock_client.images.generate.call_args.kwargs
        # extra_headers should be passed as the SDK keyword for HTTP headers
        assert "extra_headers" in call_kwargs, (
            "headers should be passed as extra_headers to the SDK"
        )
        assert call_kwargs["extra_headers"]["cf-aig-authorization"] == "Bearer cfut_123"
        # Body params should NOT contain extra_headers
        body_keys = [k for k in call_kwargs if k != "extra_headers"]
        assert "extra_headers" not in body_keys

    def test_image_generation_entry_strips_extra_headers_from_optional_params(self):
        """
        The image_generation() entry point in litellm/images/main.py should
        strip extra_headers from optional_params before passing to OpenAI handler.
        """
        # Mock the components to test the flow
        with patch("litellm.images.main.get_llm_provider", return_value=("gpt-image-2", "openai", "sk-test", "https://api.openai.com/v1")):
            with patch("litellm.images.main.get_optional_params_image_gen") as mock_get_params:
                with patch("litellm.images.main.openai_chat_completions") as mock_openai:
                    mock_get_params.return_value = {
                        "n": 1,
                        "extra_headers": {"cf-aig-authorization": "Bearer cfut_123"},
                    }
                    mock_openai.image_generation.return_value = MagicMock()

                    import litellm.images.main as img_main

                    img_main.image_generation(
                        prompt="a red circle",
                        model="gpt-image-2",
                        n=1,
                        extra_headers={"cf-aig-authorization": "Bearer cfut_123"},
                    )

                    # Verify the OpenAI handler was called
                    assert mock_openai.image_generation.called
                    call_kwargs = mock_openai.image_generation.call_args.kwargs
                    optional_params = call_kwargs.get("optional_params", {})
                    # extra_headers should have been stripped
                    assert "extra_headers" not in optional_params, (
                        f"extra_headers leaked into optional_params: {optional_params}"
                    )
