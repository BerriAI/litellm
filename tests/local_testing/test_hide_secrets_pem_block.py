"""Unit tests for full-PEM-block redaction in the hide_secrets guardrail.

These tests cover the bug where ``detect-secrets``'s line-based
``PrivateKeyDetector`` only flags the ``-----BEGIN ... PRIVATE KEY-----`` armor
header. The previous ``str.replace(secret["value"], "[REDACTED]")`` call sites
therefore only struck the header line and forwarded the base64 body + END
footer downstream (LIT-3292).
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
    _PEM_BLOCK_RE,
    _expand_private_key_values,
)


# A throwaway, syntactically-valid PEM body. Not a real key; just bytes that
# look like base64 so the regex anchors match.
_PEM_BODY = (
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n"
    "MzEfYyjiWA4R4/M2bS1GB4t7NXp98C3SC6dVMvDuictGeurT8jNbvJZHtCSuYEvu\n"
    "VuNd9tybAgMBAAECggEBAKTmjaS6tkK8BlPXClTQ2vpz/N6uxDeS35mXpqasqskV"
)
_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    f"{_PEM_BODY}\n"
    "-----END PRIVATE KEY-----"
)


def test_pem_block_regex_matches_standard_armor():
    """The block regex must match the standard PEM armor variants."""
    for header, footer in [
        ("BEGIN PRIVATE KEY", "END PRIVATE KEY"),
        ("BEGIN RSA PRIVATE KEY", "END RSA PRIVATE KEY"),
        ("BEGIN EC PRIVATE KEY", "END EC PRIVATE KEY"),
        ("BEGIN DSA PRIVATE KEY", "END DSA PRIVATE KEY"),
        ("BEGIN ENCRYPTED PRIVATE KEY", "END ENCRYPTED PRIVATE KEY"),
        ("BEGIN OPENSSH PRIVATE KEY", "END OPENSSH PRIVATE KEY"),
    ]:
        text = f"-----{header}-----\nBODY-{header}\n-----{footer}-----"
        matches = _PEM_BLOCK_RE.findall(text)
        assert len(matches) == 1, f"expected 1 match for {header}, got {matches}"


def test_expand_private_key_values_expands_header_to_full_block():
    """A header-only ``Private Key`` value gets replaced with the full PEM block."""
    detected = [{"type": "Private Key", "value": "BEGIN PRIVATE KEY"}]
    expanded = _expand_private_key_values(_PEM, detected)
    assert len(expanded) == 1
    assert expanded[0]["type"] == "Private Key"
    assert expanded[0]["value"] == _PEM


def test_expand_private_key_values_emits_one_entry_per_block():
    """Two PEM blocks in one message produce two ``Private Key`` entries."""
    second = _PEM.replace("PRIVATE KEY", "RSA PRIVATE KEY")
    message = f"first:\n{_PEM}\nsecond:\n{second}\n"
    detected = [{"type": "Private Key", "value": "BEGIN PRIVATE KEY"}]
    expanded = _expand_private_key_values(message, detected)
    pk_values = [e["value"] for e in expanded if e["type"] == "Private Key"]
    assert _PEM in pk_values
    assert second in pk_values
    assert len(pk_values) == 2


def test_expand_private_key_values_preserves_other_secret_types():
    """Non-``Private Key`` secrets must pass through untouched and in order."""
    detected = [
        {"type": "OpenAI Token", "value": "sk-AAAA"},
        {"type": "Private Key", "value": "BEGIN PRIVATE KEY"},
        {"type": "Stripe Access Key", "value": "sk_live_BBBB"},
    ]
    expanded = _expand_private_key_values(_PEM, detected)
    types = [e["type"] for e in expanded]
    assert "OpenAI Token" in types
    assert "Stripe Access Key" in types
    # OpenAI Token came first, so it must still come first
    assert types[0] == "OpenAI Token"


def test_expand_private_key_values_no_block_falls_back_to_header():
    """If no full PEM block is present (truncated input), keep the original entry
    so the existing header-line redaction still runs."""
    detected = [{"type": "Private Key", "value": "BEGIN PRIVATE KEY"}]
    message = "Truncated:\n-----BEGIN PRIVATE KEY-----\nMIIE\n(no end marker)"
    expanded = _expand_private_key_values(message, detected)
    assert len(expanded) == 1
    assert expanded[0]["value"] == "BEGIN PRIVATE KEY"


def test_expand_private_key_values_passthrough_when_no_private_key():
    """If no ``Private Key`` entry is present, the list is returned unchanged."""
    detected = [{"type": "OpenAI Token", "value": "sk-AAAA"}]
    expanded = _expand_private_key_values("any message", detected)
    assert expanded == detected


def test_expand_private_key_values_handles_empty_list():
    """Empty input must return empty output."""
    assert _expand_private_key_values("anything", []) == []


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_block_from_messages():
    """End-to-end: the proxy pre-call hook redacts header + body + footer from
    the chat ``messages`` payload, not just the BEGIN line."""
    cfg = {"plugins_used": [{"name": "PrivateKeyDetector"}]}
    g = _ENTERPRISE_SecretDetection(detect_secrets_config=cfg)
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": f"please deploy with key:\n{_PEM}\nthanks"},
        ],
    }
    auth = UserAPIKeyAuth()
    await g.async_pre_call_hook(auth, DualCache(), data, "completion")
    forwarded = data["messages"][0]["content"]
    assert "[REDACTED]" in forwarded
    assert _PEM_BODY not in forwarded, "base64 body must not survive redaction"
    assert "-----END PRIVATE KEY-----" not in forwarded, "END footer must not survive"
    assert "-----BEGIN PRIVATE KEY-----" not in forwarded, "BEGIN header must not survive"


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_block_from_prompt_string():
    """The same fix applies when the payload uses the legacy ``prompt`` field."""
    cfg = {"plugins_used": [{"name": "PrivateKeyDetector"}]}
    g = _ENTERPRISE_SecretDetection(detect_secrets_config=cfg)
    data = {
        "model": "text-davinci-003",
        "prompt": f"deploy this:\n{_PEM}\nthanks",
    }
    auth = UserAPIKeyAuth()
    await g.async_pre_call_hook(auth, DualCache(), data, "completion")
    assert "[REDACTED]" in data["prompt"]
    assert _PEM_BODY not in data["prompt"]
    assert "-----END PRIVATE KEY-----" not in data["prompt"]


@pytest.mark.asyncio
async def test_async_pre_call_hook_redacts_full_pem_block_from_prompt_list():
    """The same fix applies when the payload uses a list ``prompt``."""
    cfg = {"plugins_used": [{"name": "PrivateKeyDetector"}]}
    g = _ENTERPRISE_SecretDetection(detect_secrets_config=cfg)
    data = {
        "model": "text-davinci-003",
        "prompt": ["hello", f"deploy: {_PEM}", "thanks"],
    }
    auth = UserAPIKeyAuth()
    await g.async_pre_call_hook(auth, DualCache(), data, "completion")
    joined = " ".join(data["prompt"])
    assert "[REDACTED]" in joined
    assert _PEM_BODY not in joined
    assert "-----END PRIVATE KEY-----" not in joined
