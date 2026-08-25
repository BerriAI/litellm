"""
Regression test for issue #38178.

``ssl_verify`` is a LiteLLM-level TLS setting, not a provider body param. It has to
configure the httpx client backing the OpenAI SDK client, and it must never be swept
into ``extra_body``: OpenAI-compatible endpoints reject unknown body fields with a 400.
"""

from pathlib import Path
from unittest.mock import MagicMock

import certifi
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import get_ssl_configuration
from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.types.utils import all_litellm_params
from litellm.utils import get_non_default_completion_params


@pytest.fixture
def ca_bundle(tmp_path: Path) -> str:
    """A real, loadable CA bundle at a path that is not certifi's default."""
    bundle = tmp_path / "corporate-ca.crt"
    bundle.write_bytes(Path(certifi.where()).read_bytes())
    return str(bundle)


def test_ssl_verify_is_a_known_litellm_param():
    assert "ssl_verify" in all_litellm_params


def test_ssl_verify_not_forwarded_as_provider_param(ca_bundle: str):
    forwarded = get_non_default_completion_params({"ssl_verify": ca_bundle, "temperature": 0.5})
    assert "ssl_verify" not in forwarded


def test_completion_does_not_leak_ssl_verify_into_provider_request_body(ca_bundle: str):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {}
    mock_raw_response.parse.return_value = mock_response

    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create.return_value = mock_raw_response

    litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        ssl_verify=ca_bundle,
        api_key="sk-test",
        client=mock_client,
    )

    create_kwargs = mock_client.chat.completions.with_raw_response.create.call_args.kwargs
    assert "ssl_verify" not in create_kwargs
    assert "ssl_verify" not in (create_kwargs.get("extra_body") or {})


def test_sync_openai_client_uses_ssl_verify_ca_bundle(ca_bundle: str):
    handler = OpenAIChatCompletion()
    client = handler._get_openai_client(
        is_async=False,
        api_key="sk-test",
        api_base="https://private.example.com/v1",
        max_retries=0,
        ssl_verify=ca_bundle,
    )
    assert client is not None
    pool = client._client._transport._pool
    assert pool._ssl_context is get_ssl_configuration(ssl_verify=ca_bundle)
    assert pool._ssl_context is not get_ssl_configuration()


@pytest.mark.asyncio
async def test_async_openai_client_uses_ssl_verify_ca_bundle(ca_bundle: str):
    handler = OpenAIChatCompletion()
    client = handler._get_openai_client(
        is_async=True,
        api_key="sk-test",
        api_base="https://private.example.com/v1",
        max_retries=0,
        ssl_verify=ca_bundle,
    )
    assert client is not None
    session = client._client._transport._client_factory()
    try:
        assert session.connector._ssl is get_ssl_configuration(ssl_verify=ca_bundle)
        assert session.connector._ssl is not get_ssl_configuration()
    finally:
        await session.close()


def test_openai_client_cache_is_keyed_on_ssl_verify(ca_bundle: str):
    handler = OpenAIChatCompletion()
    shared_args = {
        "is_async": False,
        "api_key": "sk-test",
        "api_base": "https://private.example.com/v1",
        "max_retries": 0,
    }
    with_ca = handler._get_openai_client(**shared_args, ssl_verify=ca_bundle)
    without_ca = handler._get_openai_client(**shared_args)
    assert with_ca is not without_ca
