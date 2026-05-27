"""
Unit tests for full PEM private-key redaction in the ``hide_secrets`` guardrail.

Regression coverage for the bug where ``detect-secrets`` (line-based) returned
only the ``-----BEGIN ... PRIVATE KEY-----`` armor header as the secret value,
leaving the base64 body and ``-----END`` footer forwarded to the downstream
LLM. The fix (``_expand_private_key_values``) post-processes the detected
secrets list and promotes any ``Private Key`` entry from the header-only value
to the full PEM block found in the original text.
"""

import pytest

from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
    _PEM_BLOCK_RE,
    _expand_private_key_values,
)


_PKCS8_BEGIN = "-----BEGIN PRIVATE KEY-----"

_SAMPLE_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
    "-----END PRIVATE KEY-----"
)

_SAMPLE_RSA_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29\n"
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB\n"
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
    text = "No PEM block anywhere in this string."
    result = _expand_private_key_values(detected, text)
    assert result == detected


def test_expand_backfills_when_detect_secrets_dedupes_same_header_blocks():
    """detect-secrets dedupes by secret value, so two PEM blocks with the same
    BEGIN header collapse to one entry. The backfill adds the missing block."""
    second_pem = _SAMPLE_PEM.replace("AAAA", "ZZZZ")
    text = f"first:\n{_SAMPLE_PEM}\n---\nsecond:\n{second_pem}\n"
    detected = [{"type": "Private Key", "value": _PKCS8_BEGIN}]

    result = _expand_private_key_values(detected, text)

    values = [s["value"] for s in result if s["type"] == "Private Key"]
    assert _SAMPLE_PEM in values
    assert second_pem in values


def test_expand_distinct_inputs_each_get_distinct_block():
    """If detect-secrets does emit two entries, each pairs to its own block."""
    second_pem = _SAMPLE_PEM.replace("AAAA", "ZZZZ")
    text = f"first:\n{_SAMPLE_PEM}\n---\nsecond:\n{second_pem}\n"
    detected = [
        {"type": "Private Key", "value": _PKCS8_BEGIN},
        {"type": "Private Key", "value": _PKCS8_BEGIN},
    ]
    result = _expand_private_key_values(detected, text)
    pks = [s for s in result if s["type"] == "Private Key"]
    assert pks[0]["value"] == _SAMPLE_PEM
    assert pks[1]["value"] == second_pem


def test_expand_mixed_pem_types_in_same_message():
    text = f"a:\n{_SAMPLE_RSA_PEM}\nb:\n{_SAMPLE_PEM}\n"
    detected = [
        {"type": "Private Key", "value": "-----BEGIN RSA PRIVATE KEY-----"},
        {"type": "Private Key", "value": _PKCS8_BEGIN},
    ]
    result = _expand_private_key_values(detected, text)
    pks = [s["value"] for s in result if s["type"] == "Private Key"]
    assert _SAMPLE_RSA_PEM in pks
    assert _SAMPLE_PEM in pks


def test_expand_missing_value_key_passes_through():
    detected = [{"type": "Private Key"}]
    result = _expand_private_key_values(detected, _SAMPLE_PEM)
    assert {"type": "Private Key"} in result


def test_pem_block_regex_matches_pkcs8_and_rsa():
    assert _PEM_BLOCK_RE.search(_SAMPLE_PEM)
    assert _PEM_BLOCK_RE.search(_SAMPLE_RSA_PEM)


# End-to-end through scan_message_for_secrets


def test_scan_message_for_secrets_returns_full_pem_block():
    text = f"Here is the key the user sent:\n{_SAMPLE_PEM}\nThanks!"
    det = _ENTERPRISE_SecretDetection()
    secrets = det.scan_message_for_secrets(text)
    private_keys = [s for s in secrets if s["type"] == "Private Key"]
    assert private_keys
    assert private_keys[0]["value"] == _SAMPLE_PEM


def test_redaction_strips_full_pem_block():
    text = f"Here is my key:\n{_SAMPLE_PEM}\nplease analyze."
    det = _ENTERPRISE_SecretDetection()
    secrets = det.scan_message_for_secrets(text)
    redacted = text
    for s in secrets:
        redacted = redacted.replace(s["value"], "[REDACTED]")
    assert "[REDACTED]" in redacted
    for marker in ("MIIEvQIBADANBgkqhkiG", "AAAAAAAAAAAAAAAAAAAAA", "BBBBBBBBBBBBBBBBBBBB"):
        assert marker not in redacted
    assert "-----END PRIVATE KEY-----" not in redacted


def test_redaction_handles_two_distinct_pem_keys_in_message():
    """Two PEM keys with the same BEGIN header but different bodies both get
    fully redacted, even though detect-secrets dedupes them."""
    second_pem = _SAMPLE_PEM.replace("AAAA", "ZZZZ")
    text = f"first:\n{_SAMPLE_PEM}\n---\nsecond:\n{second_pem}\nthanks"
    det = _ENTERPRISE_SecretDetection()
    secrets = det.scan_message_for_secrets(text)
    redacted = text
    for s in secrets:
        redacted = redacted.replace(s["value"], "[REDACTED]")
    for marker in ("AAAAAAAAAAAAAAAAAAAA", "ZZZZZZZZZZZZZZZZZZZZ"):
        assert marker not in redacted
    assert "-----END PRIVATE KEY-----" not in redacted



# ---------------------------------------------------------------------------
# Tests using the *actual* no-dashes header value emitted by detect-secrets,
# plus coverage for the regex-matched-but-undetected backfill path.
# ---------------------------------------------------------------------------
#
# detect-secrets' PrivateKeyDetector reports a value like "BEGIN PRIVATE KEY"
# (no dashes, len 17), not the full "-----BEGIN PRIVATE KEY-----" line.  The
# unit tests above mostly use the dashed form; these tests pin the contract
# against the real header form so a future change to the matching logic
# can't pass unit tests while regressing the real path.

_DETECT_SECRETS_RAW_HEADER = "BEGIN PRIVATE KEY"
_DETECT_SECRETS_RAW_RSA_HEADER = "BEGIN RSA PRIVATE KEY"
_DETECT_SECRETS_RAW_OPENSSH_HEADER = "BEGIN OPENSSH PRIVATE KEY"

_SAMPLE_OPENSSH_PEM = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAA\n"
    "QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ\n"
    "-----END OPENSSH PRIVATE KEY-----"
)


def test_expand_promotes_no_dashes_header_value_to_full_block():
    """Matches the *actual* value detect-secrets emits ('BEGIN PRIVATE KEY')."""
    detected = [{"type": "Private Key", "value": _DETECT_SECRETS_RAW_HEADER}]
    text = f"Here is a key:\n{_SAMPLE_PEM}\nEnd."
    result = _expand_private_key_values(detected, text)
    assert result[0]["value"] == _SAMPLE_PEM


def test_expand_promotes_no_dashes_rsa_header():
    detected = [{"type": "Private Key", "value": _DETECT_SECRETS_RAW_RSA_HEADER}]
    text = f"Key:\n{_SAMPLE_RSA_PEM}\n"
    result = _expand_private_key_values(detected, text)
    assert result[0]["value"] == _SAMPLE_RSA_PEM


def test_expand_promotes_no_dashes_openssh_header():
    detected = [{"type": "Private Key", "value": _DETECT_SECRETS_RAW_OPENSSH_HEADER}]
    text = f"Key:\n{_SAMPLE_OPENSSH_PEM}\n"
    result = _expand_private_key_values(detected, text)
    assert result[0]["value"] == _SAMPLE_OPENSSH_PEM


def test_backfill_redacts_pem_block_even_when_no_detection_present():
    """If ``_PEM_BLOCK_RE`` matches a block but detect-secrets emitted nothing
    for it (e.g. an obscure key format only our regex catches), the backfill
    still synthesizes a ``Private Key`` entry so the block gets redacted."""
    text = f"Surprise key:\n{_SAMPLE_PEM}\n"
    result = _expand_private_key_values([], text)
    pks = [s for s in result if s["type"] == "Private Key"]
    assert any(p["value"] == _SAMPLE_PEM for p in pks)


def test_backfill_redacts_block_when_detect_secrets_emits_unrelated_header():
    """A detected entry for one header type plus a different-header block both
    get handled: the matched entry is expanded, the other-header block is
    backfilled regardless of whether its header was seen in detected_secrets."""
    text = (
        f"first:\n{_SAMPLE_PEM}\n"
        f"second:\n{_SAMPLE_OPENSSH_PEM}\n"
    )
    detected = [{"type": "Private Key", "value": _DETECT_SECRETS_RAW_HEADER}]
    result = _expand_private_key_values(detected, text)
    pks = [s["value"] for s in result if s["type"] == "Private Key"]
    assert _SAMPLE_PEM in pks
    assert _SAMPLE_OPENSSH_PEM in pks
