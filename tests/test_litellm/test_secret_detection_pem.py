"""
Unit tests for PEM private-key full-block redaction in the hide_secrets guardrail.

Regression coverage for the bug where detect-secrets (line-based) returned only
the BEGIN armor header as the secret value, leaving the base64 body and END
footer forwarded to the downstream LLM.

The fix (_expand_private_key_values) post-processes the detected secrets list
and promotes any 'Private Key' entry from the header-only value to the full
PEM block found in the original text.
"""

import pytest

from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _expand_private_key_values,
    _PEM_BLOCK_RE,
    _ENTERPRISE_SecretDetection,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_PKCS8_BEGIN = "-----BEGIN PRIVATE KEY-----"
_PKCS8_END = "-----END PRIVATE KEY-----"

_SAMPLE_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
    "-----END PRIVATE KEY-----"
)

_SAMPLE_RSA_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29\n"
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
    "-----END RSA PRIVATE KEY-----"
)


# ---------------------------------------------------------------------------
# _expand_private_key_values unit tests
# ---------------------------------------------------------------------------


def test_expand_promotes_header_to_full_block():
    """The BEGIN header is expanded to the whole PEM block."""
    detected = [{"type": "Private Key", "value": _PKCS8_BEGIN}]
    text = f"Here is a key:\n{_SAMPLE_PEM}\nEnd of message."

    result = _expand_private_key_values(detected, text)

    assert len(result) == 1
    assert result[0]["value"] == _SAMPLE_PEM


def test_expand_rsa_private_key():
    """Works for RSA PRIVATE KEY blocks (not just PKCS#8)."""
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
    """If the header isn't found in the text, the original entry is kept."""
    detected = [{"type": "Private Key", "value": _PKCS8_BEGIN}]
    text = "No PEM block in here at all."

    result = _expand_private_key_values(detected, text)

    # Falls back to the original (header-only) value rather than crashing.
    assert result[0]["value"] == _PKCS8_BEGIN


def test_expand_empty_list():
    result = _expand_private_key_values([], _SAMPLE_PEM)
    assert result == []


# ---------------------------------------------------------------------------
# Integration: scan_message_for_secrets returns the full block
# ---------------------------------------------------------------------------


def test_scan_message_returns_full_pem_block():
    """scan_message_for_secrets exposes the full PEM block, not just the header."""
    guard = _ENTERPRISE_SecretDetection()
    text = f"Please review:\n{_SAMPLE_PEM}\nthanks."

    secrets = guard.scan_message_for_secrets(text)

    private_key_secrets = [s for s in secrets if s["type"] == "Private Key"]
    assert private_key_secrets, "Expected at least one Private Key detection"
    for s in private_key_secrets:
        assert (
            _PKCS8_END in s["value"]
        ), f"Expected full PEM block but got only: {s['value']!r}"
        assert "MIIEvQIBADANBgkq" in s["value"], "Base64 body must be in the value"


# ---------------------------------------------------------------------------
# End-to-end: async_pre_call_hook redacts the entire block in messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_hook_redacts_full_pem_in_message():
    """The full PEM block (header + body + footer) is replaced with [REDACTED]."""
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    guard = _ENTERPRISE_SecretDetection()
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
    local_cache = DualCache()

    message_with_pem = f"Here is a private key:\n{_SAMPLE_PEM}\nPlease review it."

    data = {
        "messages": [{"role": "user", "content": message_with_pem}],
        "model": "gpt-4",
    }

    await guard.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    redacted_content = data["messages"][0]["content"]

    # The entire PEM block must be gone — not just the header.
    assert (
        "-----BEGIN PRIVATE KEY-----" not in redacted_content
    ), "BEGIN header was not redacted"
    assert (
        "MIIEvQIBADANBgkq" not in redacted_content
    ), "Base64 key body was not redacted"
    assert (
        "-----END PRIVATE KEY-----" not in redacted_content
    ), "END footer was not redacted"
    assert "[REDACTED]" in redacted_content


@pytest.mark.asyncio
async def test_pre_call_hook_redacts_full_pem_in_prompt():
    """Full PEM block in a text-completion prompt is fully redacted."""
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    guard = _ENTERPRISE_SecretDetection()
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
    local_cache = DualCache()

    data = {
        "prompt": f"key:\n{_SAMPLE_PEM}\nend",
        "model": "gpt-3.5-turbo-instruct",
    }

    await guard.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data=data,
        call_type="completion",
    )

    assert "-----BEGIN PRIVATE KEY-----" not in data["prompt"]
    assert "MIIEvQIBADANBgkq" not in data["prompt"]
    assert "-----END PRIVATE KEY-----" not in data["prompt"]
    assert "[REDACTED]" in data["prompt"]
