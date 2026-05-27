"""Unit tests for full-block PEM private-key redaction in the hide_secrets
guardrail.

Regression coverage for the LIT-3292 bug where ``detect-secrets`` (which
scans line-by-line) returned only the ``-----BEGIN ... PRIVATE KEY-----``
armor header as the secret value, and the subsequent ``str.replace`` call
sites left the base64 body and ``END`` footer in the forwarded request.

The fix (``_expand_private_key_values``) post-processes the detected secrets
list and promotes every ``Private Key`` entry from the header-only value to
the full PEM block found in the source text.
"""

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import hash_token
from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
    _PEM_BLOCK_RE,
    _expand_private_key_values,
)

# ---------------------------------------------------------------------------
# Fixtures - bytes are placeholders, NOT a real key.
# ---------------------------------------------------------------------------

_PKCS8_BEGIN = "-----BEGIN PRIVATE KEY-----"

_SAMPLE_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n"
    "MzEfYyjiWA4R4/M2bS1GB4t7NXp98C3SC6dVMvDuictGeurT8jNbvJZHtCSuYEvu\n"
    "-----END PRIVATE KEY-----"
)

_SAMPLE_RSA_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29\n"
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
    "-----END RSA PRIVATE KEY-----"
)


# ---------------------------------------------------------------------------
# _expand_private_key_values - direct unit coverage
# ---------------------------------------------------------------------------


def test_expand_promotes_header_to_full_block():
    """The BEGIN armor header is expanded to the whole PEM block."""
    detected = [{"type": "Private Key", "value": _PKCS8_BEGIN}]
    text = f"Here is a key:\n{_SAMPLE_PEM}\nEnd of message."

    result = _expand_private_key_values(detected, text)

    assert len(result) == 1
    assert result[0]["value"] == _SAMPLE_PEM


def test_expand_handles_rsa_private_key():
    """Works for RSA PRIVATE KEY blocks, not just PKCS#8."""
    header = "-----BEGIN RSA PRIVATE KEY-----"
    detected = [{"type": "Private Key", "value": header}]
    text = f"Key:\n{_SAMPLE_RSA_PEM}\n"

    result = _expand_private_key_values(detected, text)

    assert result[0]["value"] == _SAMPLE_RSA_PEM


def test_expand_preserves_non_private_key_secrets():
    """Non-PEM secret types are returned unchanged."""
    detected = [
        {"type": "AWS Access Key", "value": "AKIAIOSFODNN7EXAMPLE"},
        {"type": "Private Key", "value": _PKCS8_BEGIN},
    ]
    text = f"key={_SAMPLE_PEM}"

    result = _expand_private_key_values(detected, text)

    assert result[0] == {"type": "AWS Access Key", "value": "AKIAIOSFODNN7EXAMPLE"}
    assert result[1]["value"] == _SAMPLE_PEM


def test_expand_no_match_leaves_secret_unchanged():
    """If no enclosing PEM block is found, the entry is kept as-is."""
    detected = [{"type": "Private Key", "value": _PKCS8_BEGIN}]
    text = "no key here"

    result = _expand_private_key_values(detected, text)

    assert result == detected


def test_expand_multiple_keys_each_claim_a_distinct_block():
    """Two PEM keys in one message get two distinct full-block values."""
    detected = [
        {"type": "Private Key", "value": _PKCS8_BEGIN},
        {"type": "Private Key", "value": _PKCS8_BEGIN},
    ]
    text = f"first:\n{_SAMPLE_PEM}\nsecond:\n{_SAMPLE_PEM}\n"

    result = _expand_private_key_values(detected, text)

    assert result[0]["value"] == _SAMPLE_PEM
    assert result[1]["value"] == _SAMPLE_PEM


def test_pem_block_regex_spans_newlines():
    r"""The regex matches across newlines via the ``[\s\S]`` class."""
    text = f"prefix {_SAMPLE_PEM} suffix"
    m = _PEM_BLOCK_RE.search(text)

    assert m is not None
    assert m.group(0) == _SAMPLE_PEM


# ---------------------------------------------------------------------------
# End-to-end via the real async_pre_call_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_hook_redacts_full_pem_block_in_chat_message():
    """The proxy pre_call hook redacts the whole PEM block, not just the
    BEGIN header line that detect-secrets returns."""
    inst = _ENTERPRISE_SecretDetection()
    data = {
        "messages": [
            {
                "role": "user",
                "content": f"Here is my key, please ignore:\n{_SAMPLE_PEM}\nThanks",
            }
        ],
        "model": "gpt-3.5-turbo",
    }

    await inst.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-12345")),
        call_type="completion",
    )

    out = data["messages"][0]["content"]
    assert "-----BEGIN PRIVATE KEY-----" not in out
    assert "-----END PRIVATE KEY-----" not in out
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKc" not in out
    assert "[REDACTED]" in out


@pytest.mark.asyncio
async def test_pre_call_hook_redacts_full_pem_block_in_string_prompt():
    """Same end-to-end check via the ``prompt`` shape (text-completion)."""
    inst = _ENTERPRISE_SecretDetection()
    data = {
        "prompt": f"key={_SAMPLE_PEM}",
        "model": "gpt-3.5-turbo",
    }

    await inst.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-12345")),
        call_type="completion",
    )

    out = data["prompt"]
    assert "-----BEGIN PRIVATE KEY-----" not in out
    assert "-----END PRIVATE KEY-----" not in out
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKc" not in out
    assert "[REDACTED]" in out


@pytest.mark.asyncio
async def test_pre_call_hook_non_pem_secrets_still_masked():
    """Non-PEM secret types continue to be redacted; expansion does not
    regress the existing detector behaviour."""
    inst = _ENTERPRISE_SecretDetection()
    data = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "AWS_ACCESS_KEY = 'AKIAIOSFODNN7EXAMPLE' and secret = "
                    "'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'"
                ),
            }
        ],
        "model": "gpt-3.5-turbo",
    }

    await inst.async_pre_call_hook(
        cache=DualCache(),
        data=data,
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-12345")),
        call_type="completion",
    )

    out = data["messages"][0]["content"]
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[REDACTED]" in out
