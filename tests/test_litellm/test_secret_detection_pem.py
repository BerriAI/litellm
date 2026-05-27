"""
Unit tests for PEM private-key full-block redaction in the hide_secrets guardrail.

Regression coverage for LIT-3292: detect-secrets (line-based) returned only the
BEGIN armor header as the secret value, leaving the base64 body and END footer
forwarded to the downstream LLM.
"""

import asyncio

import pytest

from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _PEM_BLOCK_RE,
    _ENTERPRISE_SecretDetection,
    _expand_private_key_values,
    _redact_full_pem_blocks,
)

_PKCS8_BEGIN = "-----BEGIN PRIVATE KEY-----"

_SAMPLE_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
    "-----END PRIVATE KEY-----"
)

_SAMPLE_RSA_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29\n"
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
    "-----END RSA PRIVATE KEY-----"
)


def test_expand_promotes_header_to_full_block():
    detected = [{"type": "Private Key", "value": _PKCS8_BEGIN}]
    text = f"Here is a key:\n{_SAMPLE_PEM}\nEnd of message."
    result = _expand_private_key_values(detected, text)
    assert len(result) == 1
    assert result[0]["value"] == _SAMPLE_PEM


def test_expand_rsa_private_key():
    header = "-----BEGIN RSA PRIVATE KEY-----"
    detected = [{"type": "Private Key", "value": header}]
    text = f"Key:\n{_SAMPLE_RSA_PEM}\n"
    result = _expand_private_key_values(detected, text)
    assert result[0]["value"] == _SAMPLE_RSA_PEM


def test_expand_preserves_non_private_key_secrets():
    detected = [
        {"type": "AWS Access Key", "value": "AKIAIOSFODNN7EXAMPLE"},
        {"type": "Private Key", "value": _PKCS8_BEGIN},
    ]
    text = f"key={_SAMPLE_PEM}"
    result = _expand_private_key_values(detected, text)
    assert result[0] == {"type": "AWS Access Key", "value": "AKIAIOSFODNN7EXAMPLE"}
    assert result[1]["value"] == _SAMPLE_PEM


def test_expand_no_match_leaves_secret_unchanged():
    detected = [{"type": "Private Key", "value": _PKCS8_BEGIN}]
    text = "No PEM block in here at all."
    result = _expand_private_key_values(detected, text)
    assert result[0]["value"] == _PKCS8_BEGIN


def test_expand_empty_list():
    assert _expand_private_key_values([], _SAMPLE_PEM) == []


def test_expand_two_same_type_pem_blocks_each_get_own_block():
    """Greptile P1: two same-type PEM blocks both get full-block expansion."""
    key1 = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        "-----END RSA PRIVATE KEY-----"
    )
    key2 = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\n"
        "-----END RSA PRIVATE KEY-----"
    )
    text = f"first:\n{key1}\n\nsecond:\n{key2}"
    header = "-----BEGIN RSA PRIVATE KEY-----"
    detected = [
        {"type": "Private Key", "value": header},
        {"type": "Private Key", "value": header},
    ]
    result = _expand_private_key_values(detected, text)
    values = {r["value"] for r in result}
    assert values == {key1, key2}
    assert result[0]["value"] != result[1]["value"]


def test_redact_full_pem_blocks_single_key():
    text = f"prefix\n{_SAMPLE_PEM}\nsuffix"
    out = _redact_full_pem_blocks(text)
    assert out == "prefix\n[REDACTED]\nsuffix"


def test_redact_full_pem_blocks_multiple_keys():
    text = f"k1:{_SAMPLE_PEM} k2:{_SAMPLE_RSA_PEM}"
    out = _redact_full_pem_blocks(text)
    assert out.count("[REDACTED]") == 2
    assert "MIIEvQ" not in out
    assert "MIIEow" not in out


def test_redact_full_pem_blocks_no_pem_is_passthrough():
    text = "just some normal text without any keys"
    assert _redact_full_pem_blocks(text) == text


def test_pem_block_re_matches_variants():
    for kind in ("EC", "DSA", "OPENSSH"):
        block = (
            f"-----BEGIN {kind} PRIVATE KEY-----\n"
            "abc123\n"
            f"-----END {kind} PRIVATE KEY-----"
        )
        assert _PEM_BLOCK_RE.search(block) is not None


def _detect_secrets_available():
    try:
        import detect_secrets  # noqa: F401
        return True
    except Exception:
        return False


_skip_no_ds = pytest.mark.skipif(
    not _detect_secrets_available(),
    reason="detect-secrets not installed in this test env",
)


@_skip_no_ds
def test_scan_message_returns_full_pem_block():
    guard = _ENTERPRISE_SecretDetection()
    text = f"prefix\n{_SAMPLE_RSA_PEM}\nsuffix"
    detected = guard.scan_message_for_secrets(text)
    pem_secrets = [d for d in detected if d.get("type") == "Private Key"]
    assert len(pem_secrets) == 1
    assert pem_secrets[0]["value"] == _SAMPLE_RSA_PEM


@_skip_no_ds
def test_pre_call_hook_redacts_full_pem_in_message():
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    guard = _ENTERPRISE_SecretDetection()
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"please review:\n{_SAMPLE_RSA_PEM}"}]}
        ],
    }
    asyncio.run(
        guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(),
            data=data, call_type="completion",
        )
    )
    out = data["messages"][0]["content"][0]["text"]
    assert "[REDACTED]" in out
    assert "MIIEow" not in out, "base64 body leaked"
    assert "-----END RSA PRIVATE KEY-----" not in out, "END footer leaked"


@_skip_no_ds
def test_pre_call_hook_redacts_full_pem_in_prompt_string():
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    guard = _ENTERPRISE_SecretDetection()
    data = {"model": "text-davinci-003", "prompt": f"check:\n{_SAMPLE_PEM}"}
    asyncio.run(
        guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(),
            data=data, call_type="completion",
        )
    )
    assert "[REDACTED]" in data["prompt"]
    assert "MIIEvQ" not in data["prompt"]
    assert "-----END PRIVATE KEY-----" not in data["prompt"]


@_skip_no_ds
def test_pre_call_hook_redacts_full_pem_in_prompt_list():
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    guard = _ENTERPRISE_SecretDetection()
    data = {
        "model": "text-davinci-003",
        "prompt": [f"first:\n{_SAMPLE_PEM}", "no key here"],
    }
    asyncio.run(
        guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(),
            data=data, call_type="completion",
        )
    )
    assert "[REDACTED]" in data["prompt"][0]
    assert "MIIEvQ" not in data["prompt"][0]
    assert data["prompt"][1] == "no key here"


@_skip_no_ds
def test_pre_call_hook_redacts_two_same_type_pem_blocks():
    """Greptile P1 end-to-end."""
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    key1 = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        "-----END RSA PRIVATE KEY-----"
    )
    key2 = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\n"
        "-----END RSA PRIVATE KEY-----"
    )
    guard = _ENTERPRISE_SecretDetection()
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"first:\n{key1}\nsecond:\n{key2}"}]}
        ],
    }
    asyncio.run(
        guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(),
            data=data, call_type="completion",
        )
    )
    out = data["messages"][0]["content"][0]["text"]
    assert "AAAAAAAAAAAA" not in out, "first key body leaked"
    assert "ZZZZZZZZZZZZ" not in out, "second key body leaked"
    assert out.count("[REDACTED]") >= 2


# ---------------------------------------------------------------------------
# Audit logging: _redact_full_pem_blocks emits a warning when it actually fires
# ---------------------------------------------------------------------------


def test_redact_full_pem_blocks_logs_when_it_fires(caplog):
    """When the defensive sweep actually replaces a block (i.e. detect-secrets
    didn't flag it), a warning is emitted so the audit trail matches the rest
    of the guardrail. Silent divergence from the configured detectors would
    be a security surprise (Greptile feedback)."""
    import logging
    caplog.set_level(logging.WARNING)
    out = _redact_full_pem_blocks(f"hi\n{_SAMPLE_PEM}\nbye")
    assert "[REDACTED]" in out
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("defensive PEM-block sweep" in m for m in msgs), msgs


def test_redact_full_pem_blocks_no_log_when_passthrough(caplog):
    """When the sweep doesn't replace anything, no warning is emitted."""
    import logging
    caplog.set_level(logging.WARNING)
    out = _redact_full_pem_blocks("nothing to redact here")
    assert out == "nothing to redact here"
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("defensive PEM-block sweep" in m for m in msgs)
