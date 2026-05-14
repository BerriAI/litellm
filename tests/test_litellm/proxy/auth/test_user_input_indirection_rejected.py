"""
``os.environ/<NAME>`` and ``oidc/<...>`` are operator-config indirections
that ``get_secret`` resolves server-side. They must never come from a
request body — a user-controlled value would let an attacker name the
env var or OIDC source the proxy reads on their behalf, enabling
exfiltration of arbitrary server-process state via provider auth flows.

Two layers of defense:
1. ``is_request_body_safe`` rejects the indirection prefix in any string
   value walked at the request boundary.
2. ``get_secret`` denylists sensitive env vars at the ``os.environ/`` /
   ``oidc/env/`` resolvers regardless of how they were framed.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.proxy.auth.auth_utils import (  # noqa: E402
    _check_no_user_supplied_indirection,
    is_request_body_safe,
)
from litellm.secret_managers.main import (  # noqa: E402
    _RESOLVER_INDIRECTION_DENYLIST,
    get_secret,
)


# ---------------------------------------------------------------------
# Layer 1: boundary rejection in is_request_body_safe
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "os.environ/LITELLM_MASTER_KEY",
        "os.environ/PATH",
        "oidc/env/LITELLM_MASTER_KEY",
        "oidc/file//etc/passwd",
        "oidc/google/example.com",
    ],
)
def test_check_no_user_supplied_indirection_rejects(value):
    # Any field carrying an indirection prefix is rejected — the key
    # doesn't have to be in the banned list.
    with pytest.raises(ValueError, match="indirection"):
        _check_no_user_supplied_indirection({"some_field": value})


@pytest.mark.parametrize(
    "value",
    [
        "plain-string",
        "sk-1234",
        "https://api.openai.com/v1",  # not an indirection
        "",
        "os.environ",  # exact, no trailing slash — not the indirection
    ],
)
def test_check_no_user_supplied_indirection_allows_normal_values(value):
    _check_no_user_supplied_indirection({"some_field": value})


def test_check_no_user_supplied_indirection_ignores_non_strings():
    # Lists, dicts, ints don't carry the prefix syntax.
    _check_no_user_supplied_indirection(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "n": 1,
            "stream": True,
        }
    )


def test_is_request_body_safe_rejects_indirection_via_top_level():
    body = {
        "model": "bedrock/anthropic.claude-v2",
        "aws_session_name": "os.environ/LITELLM_MASTER_KEY",
    }
    with pytest.raises(ValueError, match="indirection"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="bedrock/anthropic.claude-v2",
        )


def test_is_request_body_safe_rejects_indirection_via_nested_metadata():
    # Use a non-banned metadata key so the indirection check is what
    # fires (banned-keys check would shadow it otherwise).
    body = {
        "model": "openai/gpt-4",
        "metadata": {
            "x-attacker-marker": "oidc/env/LITELLM_MASTER_KEY",
        },
    }
    with pytest.raises(ValueError, match="indirection"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="openai/gpt-4",
        )


def test_is_request_body_safe_allows_clean_body():
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "tell me about os.environ"}],
        "max_tokens": 100,
    }
    # Chat content mentioning "os.environ" must not trigger the check —
    # only TOP-LEVEL keys' string values are walked.
    assert is_request_body_safe(
        request_body=body,
        general_settings={},
        llm_router=None,
        model="openai/gpt-4",
    )


# ---------------------------------------------------------------------
# Layer 2: resolver denylist in get_secret
# ---------------------------------------------------------------------


@pytest.mark.parametrize("var", sorted(_RESOLVER_INDIRECTION_DENYLIST))
def test_get_secret_refuses_os_environ_prefix_for_denied_var(var, monkeypatch):
    monkeypatch.setenv(var, "the-secret")
    with pytest.raises(ValueError, match="not exposable"):
        get_secret(f"os.environ/{var}")


@pytest.mark.parametrize("var", sorted(_RESOLVER_INDIRECTION_DENYLIST))
def test_get_secret_refuses_oidc_env_for_denied_var(var, monkeypatch):
    monkeypatch.setenv(var, "the-secret")
    with pytest.raises(ValueError, match="not exposable"):
        get_secret(f"oidc/env/{var}")


def test_get_secret_refuses_oidc_env_path_for_denied_var(monkeypatch):
    monkeypatch.setenv("LITELLM_MASTER_KEY", "/etc/passwd")
    with pytest.raises(ValueError, match="not exposable"):
        get_secret("oidc/env_path/LITELLM_MASTER_KEY")


def test_get_secret_allows_non_denied_var_via_os_environ_prefix(monkeypatch):
    # The fix targets ONLY the listed sensitive vars; other env-var
    # indirections continue to resolve.
    monkeypatch.setenv("MY_CUSTOM_VAR", "my-value")
    assert get_secret("os.environ/MY_CUSTOM_VAR") == "my-value"


def test_get_secret_allows_bare_name_for_denied_var(monkeypatch):
    # Server-side callers (``get_secret_str("LITELLM_MASTER_KEY")``) use
    # the bare name — no prefix — and must continue to work. The
    # denylist only fires on the prefix-driven (request-body)
    # resolution paths.
    monkeypatch.setenv("LITELLM_MASTER_KEY", "the-key")
    assert get_secret("LITELLM_MASTER_KEY") == "the-key"
