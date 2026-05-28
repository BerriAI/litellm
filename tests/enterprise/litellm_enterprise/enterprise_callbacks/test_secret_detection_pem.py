"""
Regression tests for LIT-3292: hide_secrets only redacts PEM header line.

The ``hide_secrets`` guardrail wraps ``detect-secrets`` which is line-based:
``PrivateKeyDetector`` returns just the ``BEGIN ... PRIVATE KEY`` header line
as ``secret_value``. The pre-fix code did
``text.replace(secret["value"], "[REDACTED]")`` which only replaced that single
line — leaving the base64 key body and ``-----END ... PRIVATE KEY-----`` footer
in the payload forwarded to the upstream LLM.

These tests pin the fixed behaviour: a private key embedded in a chat /
completion request must be redacted in its entirety, regardless of PEM
variant (RSA, EC, OpenSSH, generic, encrypted-with-headers, truncated).
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "enterprise")))

from litellm.caching.caching import DualCache  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm_enterprise.enterprise_callbacks.secret_detection import (  # noqa: E402
    _ENTERPRISE_SecretDetection,
    _redact_text_with_detected_secrets,
)


# Synthetic PEM material used in tests; not a real key.
_RSA_PEM = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAabcdEF1234567890+/abcdef0123456789ABCDEF
ABCDabcdEF1234567890+/abcdef0123456789ABCDEF==
-----END RSA PRIVATE KEY-----"""

_EC_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIPbGm5d/abcDEF+1234567890==
oAoGCCqGSM49AwEHoUQDQgAEabcdef
-----END EC PRIVATE KEY-----"""

_OPENSSH_PEM = """-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAACmFlczI1Ni1jdHIAAAAGYmNyeXB0AAAAGAAAABCsalkdjflkajsdflkj
KSjwZmpsfnsm1234567890==
-----END OPENSSH PRIVATE KEY-----"""

_ENCRYPTED_PEM = """-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,FEEDFACECAFEBEEF

abc123XYZ+++/ab1234567890ABCDEFabcdef0123456789
-----END RSA PRIVATE KEY-----"""

_TRUNCATED_PEM = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAabcdEF1234567890+/abcdef0123456789ABCDEF=="""

_BODY_MARKERS = {
    "_RSA_PEM": "MIIEpAIBAAKCAQEAabcdEF",
    "_EC_PEM": "MHcCAQEEIPbGm5d",
    "_OPENSSH_PEM": "b3BlbnNzaC1rZXktdjEAAAAACmFlczI1Ni1jdHIAAAAGYm",
    "_ENCRYPTED_PEM": "abc123XYZ+++",
    "_TRUNCATED_PEM": "MIIEpAIBAAKCAQEAabcdEF",
}


def _hook():
    return _ENTERPRISE_SecretDetection()


def _user_auth():
    return UserAPIKeyAuth(api_key="sk-test", permissions={})


@pytest.mark.parametrize(
    "name,pem,body_marker,expect_end",
    [
        ("rsa", _RSA_PEM, _BODY_MARKERS["_RSA_PEM"], True),
        ("ec", _EC_PEM, _BODY_MARKERS["_EC_PEM"], True),
        ("openssh", _OPENSSH_PEM, _BODY_MARKERS["_OPENSSH_PEM"], True),
        ("encrypted", _ENCRYPTED_PEM, _BODY_MARKERS["_ENCRYPTED_PEM"], True),
        ("truncated", _TRUNCATED_PEM, _BODY_MARKERS["_TRUNCATED_PEM"], False),
    ],
)
def test_full_pem_block_redacted_in_chat_messages(name, pem, body_marker, expect_end):
    """End-to-end: pre_call_hook on chat messages strips body + END footer."""
    data = {
        "model": "echo-model",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "secret follows:\n" + pem + "\nplease parse."},
        ],
    }
    asyncio.run(
        _hook().async_pre_call_hook(
            user_api_key_dict=_user_auth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    )
    forwarded = data["messages"][1]["content"]
    assert body_marker not in forwarded, f"PEM body leaked for {name}: {forwarded!r}"
    if expect_end:
        assert "-----END" not in forwarded, f"PEM END footer leaked for {name}: {forwarded!r}"
    assert "[REDACTED]" in forwarded


def test_full_pem_block_redacted_in_completion_prompt_str():
    """Pre-fix the `data['prompt']` string branch had the same line-only bug."""
    data = {
        "model": "echo-model",
        "prompt": "look at this:\n" + _RSA_PEM + "\nthanks",
    }
    asyncio.run(
        _hook().async_pre_call_hook(
            user_api_key_dict=_user_auth(),
            cache=DualCache(),
            data=data,
            call_type="text_completion",
        )
    )
    assert _BODY_MARKERS["_RSA_PEM"] not in data["prompt"]
    assert "-----END" not in data["prompt"]
    assert "[REDACTED]" in data["prompt"]


def test_full_pem_block_redacted_in_completion_prompt_list():
    """Same for the `data['prompt']` list-of-strings branch."""
    data = {
        "model": "echo-model",
        "prompt": [
            "plain question",
            "leaked here:\n" + _RSA_PEM,
            "another one " + _EC_PEM,
        ],
    }
    asyncio.run(
        _hook().async_pre_call_hook(
            user_api_key_dict=_user_auth(),
            cache=DualCache(),
            data=data,
            call_type="text_completion",
        )
    )
    joined = "\n".join(data["prompt"])
    assert _BODY_MARKERS["_RSA_PEM"] not in joined
    assert _BODY_MARKERS["_EC_PEM"] not in joined
    assert "-----END" not in joined
    assert joined.count("[REDACTED]") >= 2


def test_pem_plus_non_pem_secret_both_redacted():
    """A request containing an AWS access key alongside a PEM block should
    redact both — non-PEM secrets must keep working after the fix."""
    aws = "AKIAIOSFODNN7EXAMPLE"
    data = {
        "model": "echo-model",
        "messages": [
            {"role": "user", "content": f"creds {aws} and key:\n{_RSA_PEM}"},
        ],
    }
    asyncio.run(
        _hook().async_pre_call_hook(
            user_api_key_dict=_user_auth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    )
    content = data["messages"][0]["content"]
    assert aws not in content
    assert _BODY_MARKERS["_RSA_PEM"] not in content
    assert "-----END" not in content


def test_no_secret_no_change():
    """Negative case: a plain message must not be mangled by the new helper."""
    data = {
        "model": "echo-model",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }
    asyncio.run(
        _hook().async_pre_call_hook(
            user_api_key_dict=_user_auth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    )
    assert data["messages"][0]["content"] == "What is the capital of France?"


def test_helper_handles_empty_detected_secrets():
    """Direct unit test: helper with empty `detected_secrets` is identity."""
    text = "hello world"
    assert _redact_text_with_detected_secrets(text, []) == text


def test_helper_handles_non_pem_only():
    """Direct unit test: helper falls back to `str.replace` for non-PEM types."""
    text = "key=sk-abc-123 here"
    out = _redact_text_with_detected_secrets(text, [{"type": "OpenAI Token", "value": "sk-abc-123"}])
    assert "sk-abc-123" not in out
    assert "[REDACTED]" in out


def test_helper_skips_private_key_str_value_match():
    """Direct unit test: ``"Private Key"`` entries are *not* used for plain
    str.replace (they only drive the PEM-block regex expansion). This makes
    the helper robust against fake/header-only `secret_value`s that detect-
    secrets might emit and ensures we never accidentally fall back to the
    line-only behaviour for the PEM case."""
    text = "no key here"
    out = _redact_text_with_detected_secrets(text, [{"type": "Private Key", "value": "BEGIN RSA PRIVATE KEY"}])
    # Helper should not crash and should not substitute when no BEGIN..END
    # spans match (and we deliberately don't substitute the literal header).
    assert out == "no key here"


_TRUNCATED_ENCRYPTED_PEM = """-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-EDE3-CBC,FEEDFACECAFEBEEF

abc123XYZ+++/ab1234567890ABCDEFabcdef0123456789"""


def test_truncated_encrypted_pem_blank_line_separator_does_not_leak_body():
    """LIT-3292 / Veria PR review #29149: encrypted PEM blocks have a blank
    line between the ``Proc-Type``/``DEK-Info`` headers and the base64 body.
    If the END footer is missing (truncated paste), an earlier two-regex
    formulation stopped consuming at the blank line and let the body through.
    The consolidated single-span fallback must consume from BEGIN through
    end-of-string so the body cannot leak."""
    data = {
        "model": "echo-model",
        "messages": [
            {"role": "user", "content": "partial:\n" + _TRUNCATED_ENCRYPTED_PEM},
        ],
    }
    asyncio.run(
        _hook().async_pre_call_hook(
            user_api_key_dict=_user_auth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    )
    content = data["messages"][0]["content"]
    # Body marker (after the blank line) MUST be gone.
    assert "abc123XYZ+++" not in content, content
    # And nothing of the encrypted-headers section either.
    assert "Proc-Type:" not in content
    assert "DEK-Info:" not in content
    assert "[REDACTED]" in content

