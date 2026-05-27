"""Tests for hide_secrets guardrail full PEM block redaction (LIT-3292).

Regression for the bug where `detect-secrets` only returned the
`-----BEGIN ... PRIVATE KEY-----` armor line as the `secret_value`, causing
`str.replace(secret_value, "[REDACTED]")` at the redact call site to strike
only the header — leaving the base64 body and `-----END` footer to be
forwarded downstream.
"""
import pytest


@pytest.fixture
def secret_detector():
    from litellm_enterprise.enterprise_callbacks.secret_detection import (
        _ENTERPRISE_SecretDetection,
    )

    return _ENTERPRISE_SecretDetection()


PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDCx7VbZ8t8Wx9k\n"
    "2T9wK6gT9X+0lO0fmTfH3Y4n8oA9oC+sZ5dQ1RVrtKfVQK0J5MJoF7+P3vBn3Lck\n"
    "yYz5p+wK0ZJ7XlqVmM9I4kZbA8Ks5XfV6mEAaH1eK1f8XjF0WJK1RZ9KqcAGm7zP\n"
    "xZpQK2tH3rGqHvFvW2vG0K6Wp8B7uW8E8lhQ4G7jLkP1aB9C0a2D7XYz1Q6S8ZHC\n"
    "HVw0kT5BAgMBAAECggEAJ8M0e8q3D5Pq6Gv1QH9q5fY7w1xT2nKjV3sLkN0X8L9I\n"
    "-----END PRIVATE KEY-----"
)

EC_PEM = (
    "-----BEGIN EC PRIVATE KEY-----\n"
    "MHcCAQEEIK1xPnZb7N5xtv5K1u8j5h3iN2x8E9q7BfP3Q5R4yT0FoAoGCCqGSM49\n"
    "AwEHoUQDQgAE4Hk6lTPjMo3W8gqkc3hZpY6lNcA1qy4xZJ8mEf1Db0kI9pK7s4y2\n"
    "B5z1Dy8w6+0FjFsZRqV+w7yMz3GgPpZ8Lw==\n"
    "-----END EC PRIVATE KEY-----"
)


def test_scan_returns_full_pem_block(secret_detector):
    text = "context line\n" + PEM + "\ntrailing line"
    found = secret_detector.scan_message_for_secrets(text)
    assert len(found) == 1
    assert found[0]["type"] == "Private Key"
    assert found[0]["value"] == PEM


def test_replace_redacts_entire_block(secret_detector):
    text = "before\n" + PEM + "\nafter"
    found = secret_detector.scan_message_for_secrets(text)
    redacted = text
    for s in found:
        redacted = redacted.replace(s["value"], "[REDACTED]")
    assert "MIIEvQIBADANBgkqhkiG" not in redacted
    assert "END PRIVATE KEY" not in redacted
    assert "BEGIN PRIVATE KEY" not in redacted
    assert "[REDACTED]" in redacted
    assert redacted == "before\n[REDACTED]\nafter"


def test_ec_private_key_also_redacted(secret_detector):
    text = "leading\n" + EC_PEM + "\ntrailing"
    found = secret_detector.scan_message_for_secrets(text)
    assert len(found) == 1
    assert found[0]["type"] == "Private Key"
    assert found[0]["value"] == EC_PEM
    redacted = text.replace(found[0]["value"], "[REDACTED]")
    assert "EC PRIVATE KEY" not in redacted
    assert "MHcCAQEEI" not in redacted


def test_multiple_pem_blocks(secret_detector):
    text = "k1:\n" + PEM + "\nk2:\n" + EC_PEM + "\ndone"
    found = secret_detector.scan_message_for_secrets(text)
    assert len(found) >= 2
    private_key_values = [s["value"] for s in found if s["type"] == "Private Key"]
    assert PEM in private_key_values
    assert EC_PEM in private_key_values
    redacted = text
    for s in found:
        redacted = redacted.replace(s["value"], "[REDACTED]")
    assert "MIIEvQIBADANBgkqhkiG" not in redacted
    assert "MHcCAQEEI" not in redacted
    assert "PRIVATE KEY" not in redacted


def test_no_pem_block_passes_through(secret_detector):
    text = "Hello world, no secrets here."
    found = secret_detector.scan_message_for_secrets(text)
    assert found == []
