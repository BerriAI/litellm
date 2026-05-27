"""Tests for full-PEM-block redaction in the hide_secrets guardrail.

Regression coverage for the bug where ``detect-secrets`` only flags the
``-----BEGIN ... PRIVATE KEY-----`` armor header as the matched secret value,
which previously caused the downstream ``str.replace`` redaction loop to leave
the base64 body and ``-----END`` footer in the forwarded request.

The fix expands every ``Private Key`` entry returned by
``scan_message_for_secrets`` to the full PEM block found in the original
message text, so the existing replace loop strikes the entire block without
any call-site changes.
"""
import asyncio
import importlib
import os
import sys

import pytest

ENTERPRISE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "enterprise")
)
if ENTERPRISE_ROOT not in sys.path:
    sys.path.insert(0, ENTERPRISE_ROOT)

secret_detection = importlib.import_module(
    "litellm_enterprise.enterprise_callbacks.secret_detection"
)

_PEM_BLOCK_RE = secret_detection._PEM_BLOCK_RE
_expand_private_key_values = secret_detection._expand_private_key_values


RSA_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEpAIBAAKCAQEAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
    "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n"
    "-----END RSA PRIVATE KEY-----"
)
PKCS8_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAabcdefghijklmnopqr\n"
    "stuvwxyz0123456789abcdefghijklmnop==\n"
    "-----END PRIVATE KEY-----"
)


# --- Pure-logic tests (no detect-secrets required) ---

def test_pem_regex_matches_rsa_block():
    matches = _PEM_BLOCK_RE.findall(f"hello {RSA_PEM} world")
    assert matches == [RSA_PEM]


def test_pem_regex_matches_pkcs8_block():
    matches = _PEM_BLOCK_RE.findall(f"prefix {PKCS8_PEM} suffix")
    assert matches == [PKCS8_PEM]


def test_pem_regex_matches_multiple_blocks():
    text = f"key1: {RSA_PEM}\nkey2: {PKCS8_PEM}"
    matches = _PEM_BLOCK_RE.findall(text)
    assert matches == [RSA_PEM, PKCS8_PEM]


def test_expand_replaces_header_only_value_with_full_block():
    text = f"please summarize: {RSA_PEM}"
    detected = [{"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"}]
    expanded = _expand_private_key_values(text, detected)
    assert expanded == [{"type": "Private Key", "value": RSA_PEM}]


def test_expand_emits_one_entry_per_pem_block():
    text = f"two keys: {RSA_PEM} and {PKCS8_PEM}"
    detected = [{"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"}]
    expanded = _expand_private_key_values(text, detected)
    assert {"type": "Private Key", "value": RSA_PEM} in expanded
    assert {"type": "Private Key", "value": PKCS8_PEM} in expanded
    assert len(expanded) == 2


def test_expand_preserves_non_private_key_entries():
    text = f"summarize: {RSA_PEM} and an aws key"
    detected = [
        {"type": "AWS Access Key", "value": "AKIATESTTESTTESTTEST"},
        {"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"},
        {"type": "Stripe Key", "value": "sk_test_xyz"},
    ]
    expanded = _expand_private_key_values(text, detected)
    types = [e["type"] for e in expanded]
    assert "AWS Access Key" in types
    assert "Stripe Key" in types
    pk_entries = [e for e in expanded if e["type"] == "Private Key"]
    assert len(pk_entries) == 1
    assert pk_entries[0]["value"] == RSA_PEM


def test_expand_no_op_when_no_private_key_detected():
    detected = [{"type": "AWS Access Key", "value": "AKIATESTTESTTESTTEST"}]
    assert _expand_private_key_values("any text", detected) == detected


def test_expand_no_op_when_detector_flagged_header_but_no_full_block():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEp...truncated"
    detected = [{"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"}]
    assert _expand_private_key_values(text, detected) == detected


def test_expand_empty_input():
    assert _expand_private_key_values("anything", []) == []


def test_expand_collapses_duplicate_private_key_entries():
    text = f"a: {RSA_PEM}\nb: {RSA_PEM}"
    detected = [
        {"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"},
        {"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"},
    ]
    expanded = _expand_private_key_values(text, detected)
    assert len(expanded) == 2
    assert all(e["value"] == RSA_PEM for e in expanded)


# --- End-to-end tests (detect-secrets required) ---

def _require_detect_secrets():
    try:
        import detect_secrets  # noqa: F401
    except ImportError:
        pytest.skip("detect-secrets not installed")


def test_scan_message_returns_full_pem_block_not_just_header():
    _require_detect_secrets()
    guard = secret_detection._ENTERPRISE_SecretDetection(
        detect_secrets_config={"plugins_used": [{"name": "PrivateKeyDetector"}]}
    )
    text = f"please redact: {RSA_PEM}"
    detected = guard.scan_message_for_secrets(text)
    assert any(d["value"] == RSA_PEM for d in detected), (
        f"expected full PEM block in detected values, got: {detected!r}"
    )


def test_scan_message_redact_loop_strikes_entire_block():
    _require_detect_secrets()
    guard = secret_detection._ENTERPRISE_SecretDetection(
        detect_secrets_config={"plugins_used": [{"name": "PrivateKeyDetector"}]}
    )
    text = f"please redact: {RSA_PEM}"
    detected = guard.scan_message_for_secrets(text)
    redacted = text
    for secret in detected:
        redacted = redacted.replace(secret["value"], "[REDACTED]")
    assert "MIIEpAIBAAKCAQEA" not in redacted
    assert "-----END RSA PRIVATE KEY-----" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_in_chat_messages():
    _require_detect_secrets()
    from litellm.proxy._types import UserAPIKeyAuth

    guard = secret_detection._ENTERPRISE_SecretDetection(
        detect_secrets_config={"plugins_used": [{"name": "PrivateKeyDetector"}]}
    )
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize this key:\n{RSA_PEM}"},
        ],
    }
    await guard.async_pre_call_hook(UserAPIKeyAuth(), None, data, "completion")
    out = data["messages"][1]["content"]
    assert "MIIEpAIBAAKCAQEA" not in out
    assert "-----END RSA PRIVATE KEY-----" not in out
    assert "[REDACTED]" in out


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_in_prompt():
    _require_detect_secrets()
    from litellm.proxy._types import UserAPIKeyAuth

    guard = secret_detection._ENTERPRISE_SecretDetection(
        detect_secrets_config={"plugins_used": [{"name": "PrivateKeyDetector"}]}
    )
    data = {"prompt": f"complete: {PKCS8_PEM}"}
    await guard.async_pre_call_hook(UserAPIKeyAuth(), None, data, "completion")
    assert "MIIBIjANBgkqhkiG" not in data["prompt"]
    assert "-----END PRIVATE KEY-----" not in data["prompt"]
    assert "[REDACTED]" in data["prompt"]


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_in_prompt_list():
    _require_detect_secrets()
    from litellm.proxy._types import UserAPIKeyAuth

    guard = secret_detection._ENTERPRISE_SecretDetection(
        detect_secrets_config={"plugins_used": [{"name": "PrivateKeyDetector"}]}
    )
    data = {"prompt": ["safe text", f"key here: {RSA_PEM}"]}
    await guard.async_pre_call_hook(UserAPIKeyAuth(), None, data, "completion")
    assert data["prompt"][0] == "safe text"
    assert "MIIEpAIBAAKCAQEA" not in data["prompt"][1]
    assert "[REDACTED]" in data["prompt"][1]
