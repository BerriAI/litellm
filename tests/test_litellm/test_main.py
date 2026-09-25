import json
import os

import pytest


from unittest.mock import MagicMock, patch

import litellm


async def _async_fake_bedrock_image_details(image_url):
    return "ZmFrZS1pbWFnZQ==", "image/png"


@pytest.fixture(autouse=True)
def clear_client_cache():
    """
    Clear the HTTP client cache before each test to ensure mocks are used.
    This prevents cached real clients from being reused across tests.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    if cache is not None:
        cache.flush_cache()


@pytest.fixture(autouse=True)
def add_api_keys_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-1234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-api03-1234567890")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "my-fake-aws-access-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "my-fake-aws-secret-access-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    # Keep these transformation tests on the simple access-key path. A leaked
    # session token or role/web-identity env var pushes Bedrock auth down a
    # different branch and fails before the mocked HTTP client is exercised.
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-1.5-flash",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic/claude-3-5-sonnet",
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_url_with_format_param(model, sync_mode, monkeypatch):
    from litellm import acompletion, completion
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.litellm_core_utils.prompt_templates import factory as prompt_factory

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    # This test is about request shaping, not live image downloads. Stub the
    # URL->image conversion helpers so suite-level network/client state from
    # earlier tests cannot prevent the mocked provider client from being hit.
    fake_base64_image = "data:image/png;base64,ZmFrZS1pbWFnZQ=="
    monkeypatch.setattr(
        prompt_factory, "convert_url_to_base64", lambda url: fake_base64_image
    )
    monkeypatch.setattr(
        prompt_factory.BedrockImageProcessor,
        "get_image_details",
        staticmethod(lambda image_url: ("ZmFrZS1pbWFnZQ==", "image/png")),
    )
    monkeypatch.setattr(
        prompt_factory.BedrockImageProcessor,
        "get_image_details_async",
        staticmethod(_async_fake_bedrock_image_details),
    )

    args = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png",
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    if model.startswith("gemini/"):
        args["api_key"] = "test-api-key"
    with patch.object(client, "post", new=MagicMock()) as mock_client:
        try:
            if sync_mode:
                response = completion(**args, client=client)
            else:
                response = await acompletion(**args, client=client)
            print(response)
        except Exception as e:
            pass

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        if "data" in mock_client.call_args.kwargs:
            json_str = mock_client.call_args.kwargs["data"]
        else:
            json_str = json.dumps(mock_client.call_args.kwargs["json"])

        if isinstance(json_str, bytes):
            json_str = json_str.decode("utf-8")

        print(f"type of json_str: {type(json_str)}")

        # Bedrock models convert URLs to base64, while direct Anthropic models support URLs
        # bedrock/invoke models use Anthropic messages API which supports URLs
        if model.startswith("bedrock/invoke/"):
            # bedrock/invoke should convert URLs to base64 (doesn't support URL references)
            # URL should NOT be in the JSON (it should be converted to base64)
            assert "https://awsmp-logos.s3.amazonaws.com" not in json_str
            # Should have base64 data in the source (type="base64", not type="url")
            assert '"type":"base64"' in json_str or '"type": "base64"' in json_str
            # Should have "data" field containing base64 content
            assert '"data"' in json_str
        elif model.startswith("bedrock/"):
            # Regular Bedrock models should convert URLs to base64 (uses "bytes" field)
            # URL should NOT be in the JSON (it should be converted to base64)
            assert "https://awsmp-logos.s3.amazonaws.com" not in json_str
            # Should have "bytes" field (Bedrock uses "bytes" not "base64" in the field name)
            assert '"bytes"' in json_str or '"bytes":' in json_str
        elif model.startswith("anthropic/"):
            # Direct Anthropic models should pass HTTPS URLs directly (HTTP URLs are converted to base64)
            # Since we're using HTTPS URL, it should be passed as-is
            assert "https://awsmp-logos.s3.amazonaws.com" in json_str
            # For Anthropic, URL references use "url" type, not base64
            assert '"type":"url"' in json_str or '"type": "url"' in json_str
        else:
            # For other models, check format parameter is respected
            assert "png" in json_str
            assert "jpeg" not in json_str


@pytest.fixture(autouse=True)
def set_openrouter_api_key():
    original_api_key = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = "fake-key-for-testing"
    yield
    if original_api_key is not None:
        os.environ["OPENROUTER_API_KEY"] = original_api_key
    else:
        del os.environ["OPENROUTER_API_KEY"]
