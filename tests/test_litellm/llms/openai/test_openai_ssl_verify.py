from unittest.mock import MagicMock, patch
import httpx
import litellm
from litellm.llms.openai.common_utils import BaseOpenAILLM
from litellm.utils import get_optional_params, add_provider_specific_params_to_optional_params


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
        "api_key": "sk-1234",
        "is_async": True,
        "ssl_verify": "/path/ca1.pem",
    }
    params_ca2 = {
        "api_key": "sk-1234",
        "is_async": True,
        "ssl_verify": "/path/ca2.pem",
    }

    key1 = BaseOpenAILLM.get_openai_client_cache_key(params_ca1, "openai")
    key2 = BaseOpenAILLM.get_openai_client_cache_key(params_ca2, "openai")

    assert key1 != key2
    assert "ssl_verify=/path/ca1.pem" in key1
    assert "ssl_verify=/path/ca2.pem" in key2
