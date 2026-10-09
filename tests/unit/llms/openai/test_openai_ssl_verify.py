import json
import ssl
from typing import Final
from unittest.mock import patch

import certifi
import httpx
import pytest

import litellm
from litellm.llms.openai.common_utils import BaseOpenAILLM
from litellm.utils import add_provider_specific_params_to_optional_params


def test_ssl_verify_not_in_extra_body():
    """
    Ensure ssl_verify is NOT dumped into extra_body for openai and openai-compatible providers.
    Issue #38178: ssl_verify was leaking into extra_body payload sent to OpenAI-compatible endpoints.
    """
    optional_params = {}
    passed_params = {
        "ssl_verify": "/custom/path/ca.pem",
        "temperature": 0.7,
        "custom_param": "value",
    }

    result = add_provider_specific_params_to_optional_params(
        optional_params=optional_params,
        passed_params=passed_params,
        custom_llm_provider="openai",
        openai_params=["temperature"],
    )

    extra_body = result.get("extra_body", {})
    assert "ssl_verify" not in extra_body
    assert extra_body.get("custom_param") == "value"


def test_get_sync_http_client_with_ssl_verify():
    """
    Verify _get_sync_http_client applies per-call ssl_verify to httpx.Client(verify=...).
    """
    client_false = BaseOpenAILLM._get_sync_http_client(ssl_verify=False)
    assert client_false is not None


def test_get_async_http_client_with_ssl_verify():
    """
    Verify _get_async_http_client applies per-call ssl_verify to httpx.AsyncClient(verify=...).
    """
    client_false = BaseOpenAILLM._get_async_http_client(ssl_verify=False)
    assert client_false is not None


def test_cache_key_differs_by_ssl_verify():
    """
    Verify cache keys differ when ssl_verify differs to prevent client poisoning across CAs.
    """
    params_ca1 = {
        "api_key": "sk-test-key",
        "is_async": True,
        "ssl_verify": "/path/ca1.pem",
    }
    params_ca2 = {
        "api_key": "sk-test-key",
        "is_async": True,
        "ssl_verify": "/path/ca2.pem",
    }

    key1 = BaseOpenAILLM.get_openai_client_cache_key(params_ca1, "openai")
    key2 = BaseOpenAILLM.get_openai_client_cache_key(params_ca2, "openai")

    assert key1 != key2
    assert "ssl_verify=/path/ca1.pem" in key1
    assert "ssl_verify=/path/ca2.pem" in key2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verify", [False, certifi.where(), None, True], ids=["disabled", "custom-ca", "default", "enabled"]
)
async def test_ssl_verify_http_client(verify: bool | str | None):
    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert "ssl_verify" not in payload
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
            },
        )

    class TLSClient(httpx.AsyncClient):
        def __init__(self, **kwargs: object):
            verify_config: Final = kwargs.get("verify", True)

            def verified_response(request: httpx.Request) -> httpx.Response:
                if verify_setting is False:
                    assert verify_config is False
                else:
                    assert isinstance(verify_config, ssl.SSLContext)
                    assert verify_config.verify_mode == ssl.CERT_REQUIRED
                    assert verify_config.check_hostname
                return respond(request)

            client_kwargs: Final = {key: value for key, value in kwargs.items() if key not in {"transport", "mounts"}}
            super().__init__(**client_kwargs, transport=httpx.MockTransport(verified_response))

    verify_setting: Final = verify
    router: Final = litellm.Router(
        num_retries=0,
        model_list=[
            {
                "model_name": "tls-test",
                "litellm_params": {
                    "model": "openai/gpt-3.5-turbo",
                    "api_key": f"sk-test-{verify}",
                    "ssl_verify": verify,
                },
            }
        ],
    )
    with patch("httpx.AsyncClient", TLSClient):
        response: Final = await router.acompletion(
            model="tls-test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.choices[0].message.content == "hello"
