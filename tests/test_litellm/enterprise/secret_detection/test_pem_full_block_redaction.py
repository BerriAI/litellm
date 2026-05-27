"""
Tests for full PEM private-key block redaction in
``_ENTERPRISE_SecretDetection``.

Regression test for: PEM private keys were only partially redacted because
``detect-secrets`` PrivateKeyDetector returns the matched header line as the
secret value; the existing ``str.replace(secret_value, "[REDACTED]")`` call
sites therefore left the base64 body and END footer in the forwarded payload.
"""
import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm_enterprise.enterprise_callbacks.secret_detection import (
    PRIVATE_KEY_BLOCK_PATTERN,
    _ENTERPRISE_SecretDetection,
    _redact_pem_blocks,
)


PEM_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA0Z3VQ7l7yz0i9wPnZvJk6F6q3rZ5w9p2y7eVwXyT3Q1aN8mB\n"
    "c4dKjY5wM8sVbJ6yT9xV5dN3yK0X1q2W8R7p4g5cVf1Q3hZ7yJ8mD9pV0aB4nL5x\n"
    "KZyW8wN0pHvT1cQ3gR9xJ2bF4mY5aN8sV3iL7tP6q2W9R0p4g5cVfX1Q3hZ7yJ8m\n"
    "D9pV0aB4nL5xKZyW8wN0pHvT1cQ3gR9xJ2bF4mY5aN8sV3iL7tP6q2W9R0p4g5cV\n"
    "fX1Q3hZ7yJ8mD9pV0aB4nL5xKZyW8wN0pHvT1cQ3gR9xJ2bF4mY5aN8sV3iL7tP6\n"
    "q2W9R0p4g5cVfX1Q3hZ7yJ8mD9pV0aB4nL5xKZQIDAQABAoIBAGTest=\n"
    "-----END RSA PRIVATE KEY-----"
)
BASE64_BODY_FRAGMENT = "MIIEowIBAAKCAQEA0Z3VQ7l7"
END_FOOTER = "-----END RSA PRIVATE KEY-----"
REDACTED_TOKEN = "[" + "REDACTED" + "]"


def test_redact_pem_blocks_replaces_full_block():
    text = f"prefix\n{PEM_BLOCK}\nsuffix"
    out = _redact_pem_blocks(text)
    assert PEM_BLOCK not in out
    assert BASE64_BODY_FRAGMENT not in out
    assert END_FOOTER not in out
    assert REDACTED_TOKEN in out
    assert out.startswith("prefix\n")
    assert out.endswith("\nsuffix")


def test_redact_pem_blocks_handles_multiple_blocks():
    text = f"a\n{PEM_BLOCK}\nb\n{PEM_BLOCK}\nc"
    out = _redact_pem_blocks(text)
    assert BASE64_BODY_FRAGMENT not in out
    assert out.count(REDACTED_TOKEN) == 2


def test_redact_pem_blocks_handles_ec_private_key_label():
    block = PEM_BLOCK.replace("RSA PRIVATE KEY", "EC PRIVATE KEY")
    out = _redact_pem_blocks(f"prefix\n{block}\nsuffix")
    assert "EC PRIVATE KEY" not in out
    assert BASE64_BODY_FRAGMENT not in out
    assert REDACTED_TOKEN in out


def test_redact_pem_blocks_noop_when_no_pem():
    text = "no key here, just plain text"
    assert _redact_pem_blocks(text) == text


def test_redact_pem_blocks_does_not_match_unterminated_block():
    text = f"prefix\n-----BEGIN RSA PRIVATE KEY-----\n{BASE64_BODY_FRAGMENT}\nno end footer"
    out = _redact_pem_blocks(text)
    assert out == text


def test_private_key_block_pattern_is_non_greedy():
    text = f"{PEM_BLOCK}\n\n{PEM_BLOCK}"
    matches = PRIVATE_KEY_BLOCK_PATTERN.findall(text)
    assert len(matches) == 2


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_in_message_content():
    guardrail = _ENTERPRISE_SecretDetection()
    user_msg = f"Here is my key, do not log it:\n{PEM_BLOCK}\nThanks."
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": user_msg}],
    }
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="u"),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    forwarded = data["messages"][0]["content"]
    assert BASE64_BODY_FRAGMENT not in forwarded
    assert END_FOOTER not in forwarded
    assert REDACTED_TOKEN in forwarded


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_in_prompt_string():
    guardrail = _ENTERPRISE_SecretDetection()
    data = {
        "model": "text-davinci-003",
        "prompt": f"please summarise:\n{PEM_BLOCK}\nend",
    }
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="u"),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert BASE64_BODY_FRAGMENT not in data["prompt"]
    assert END_FOOTER not in data["prompt"]
    assert REDACTED_TOKEN in data["prompt"]


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_in_prompt_list():
    guardrail = _ENTERPRISE_SecretDetection()
    data = {
        "model": "text-davinci-003",
        "prompt": ["nothing here", f"key:\n{PEM_BLOCK}\nend"],
    }
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="u"),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert BASE64_BODY_FRAGMENT not in data["prompt"][1]
    assert END_FOOTER not in data["prompt"][1]
    assert REDACTED_TOKEN in data["prompt"][1]


@pytest.mark.asyncio
async def test_async_pre_call_hook_passes_through_when_no_secret():
    guardrail = _ENTERPRISE_SecretDetection()
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello there"}],
    }
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="u"),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert data["messages"][0]["content"] == "Hello there"
