"""Regression tests for LIT-3292: ``hide_secrets`` must redact the FULL PEM
private-key block (header + base64 body + footer), not only the BEGIN-armor
line that ``detect-secrets`` exposes as the matched secret value.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
ENT = os.path.join(ROOT, "enterprise")
if ENT not in sys.path:
    sys.path.insert(0, ENT)

from litellm_enterprise.enterprise_callbacks.secret_detection import (  # noqa: E402
    _ENTERPRISE_SecretDetection,
    _PEM_BLOCK_RE,
    _expand_private_key_values,
)


PEM_BLOCK = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
    "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n"
    "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n"
    "-----END RSA PRIVATE KEY-----"
)


def _apply_naive_replace(text, secrets):
    out = text
    for s in secrets:
        out = out.replace(s["value"], "[REDACTED]")
    return out


def test_pem_block_regex_matches_full_block():
    text = "prefix\n" + PEM_BLOCK + "\nsuffix"
    matches = _PEM_BLOCK_RE.findall(text)
    assert len(matches) == 1
    assert matches[0].startswith("-----BEGIN RSA PRIVATE KEY-----")
    assert matches[0].endswith("-----END RSA PRIVATE KEY-----")


def test_pem_block_regex_matches_multiple_blocks():
    text = PEM_BLOCK + "\nmiddle\n" + PEM_BLOCK
    matches = _PEM_BLOCK_RE.findall(text)
    assert len(matches) == 2


def test_expand_private_key_values_replaces_header_with_block():
    text = "Key:\n" + PEM_BLOCK + "\nbye"
    detected = [{"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"}]
    expanded = _expand_private_key_values(text, detected)
    assert len(expanded) == 1
    assert expanded[0]["type"] == "Private Key"
    assert expanded[0]["value"] == PEM_BLOCK


def test_expand_private_key_values_emits_one_entry_per_block():
    text = PEM_BLOCK + "\n---\n" + PEM_BLOCK
    detected = [{"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"}]
    expanded = _expand_private_key_values(text, detected)
    assert len(expanded) == 2
    assert all(e["value"] == PEM_BLOCK for e in expanded)


def test_expand_private_key_values_preserves_other_secrets():
    text = "key " + PEM_BLOCK + " aws=AKIAIOSFODNN7EXAMPLE"
    detected = [
        {"type": "AWS Access Key", "value": "AKIAIOSFODNN7EXAMPLE"},
        {"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"},
    ]
    expanded = _expand_private_key_values(text, detected)
    types = sorted(e["type"] for e in expanded)
    assert types == ["AWS Access Key", "Private Key"]
    assert any(e["value"] == PEM_BLOCK for e in expanded)
    assert any(e["value"] == "AKIAIOSFODNN7EXAMPLE" for e in expanded)


def test_expand_private_key_values_passthrough_when_no_pem_in_text():
    detected = [{"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"}]
    expanded = _expand_private_key_values("no key here at all", detected)
    assert expanded == detected


def test_expand_private_key_values_passthrough_when_no_private_key():
    detected = [{"type": "AWS Access Key", "value": "AKIA..."}]
    expanded = _expand_private_key_values("something", detected)
    assert expanded == detected


def test_scan_message_for_secrets_redacts_full_pem_block():
    inst = _ENTERPRISE_SecretDetection()
    msg = "Here is my key:\n" + PEM_BLOCK + "\nplease keep it safe."
    secrets = inst.scan_message_for_secrets(msg)
    pk_entries = [s for s in secrets if s["type"] == "Private Key"]
    assert pk_entries, "PrivateKeyDetector did not fire"
    assert any(e["value"] == PEM_BLOCK for e in pk_entries)
    redacted = _apply_naive_replace(msg, secrets)
    assert "-----BEGIN RSA PRIVATE KEY-----" not in redacted
    assert "-----END RSA PRIVATE KEY-----" not in redacted
    assert "MIIEowIBAAKCAQEAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" not in redacted
    assert "[REDACTED]" in redacted


def test_scan_handles_multiple_pem_blocks():
    inst = _ENTERPRISE_SecretDetection()
    msg = "first:\n" + PEM_BLOCK + "\nsecond:\n" + PEM_BLOCK + "\nend."
    secrets = inst.scan_message_for_secrets(msg)
    pk_entries = [s for s in secrets if s["type"] == "Private Key"]
    assert len(pk_entries) >= 2
    redacted = _apply_naive_replace(msg, secrets)
    assert "MIIEowIBAAKCAQEA" not in redacted
    assert "-----END RSA PRIVATE KEY-----" not in redacted
