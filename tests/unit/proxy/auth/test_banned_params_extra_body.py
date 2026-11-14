"""
``extra_body`` is the OpenAI-SDK passthrough container — provider modules
pull provider-auth fields out of it without re-validating. Without
descending into it, the banned-param boundary check is bypassed by
nesting the same fields under ``extra_body``.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.proxy.auth.auth_utils import is_request_body_safe  # noqa: E402


@pytest.mark.parametrize(
    "banned_param",
    [
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_role_name",
        "aws_profile_name",
        "api_base",
        "base_url",
        "vertex_credentials",
        "azure_ad_token",
    ],
)
def test_banned_param_under_extra_body_is_rejected(banned_param):
    body = {
        "model": "bedrock/anthropic.claude-v2",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {banned_param: "anything-attacker-chose"},
    }
    with pytest.raises(ValueError, match="not allowed in request body"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="bedrock/anthropic.claude-v2",
        )


def test_extra_body_with_safe_fields_is_allowed():
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"reasoning_effort": "low", "seed": 42},
    }
    assert is_request_body_safe(
        request_body=body,
        general_settings={},
        llm_router=None,
        model="openai/gpt-4",
    )


def test_admin_opt_in_still_permits_extra_body_credentials():
    # ``allow_client_side_credentials`` is the admin escape; descending
    # into ``extra_body`` must preserve it.
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"api_base": "https://my-private-openai.internal"},
    }
    assert is_request_body_safe(
        request_body=body,
        general_settings={"allow_client_side_credentials": True},
        llm_router=None,
        model="openai/gpt-4",
    )


def test_banned_param_under_stringified_extra_body_is_rejected():
    # Raw-HTTP and multipart/form-data clients can send ``extra_body`` as
    # a JSON-encoded string rather than an object. An ``isinstance(...,
    # dict)`` guard on the nested descent would skip such payloads,
    # leaving the banned-key check bypassed. Coercion via
    # ``_coerce_metadata_to_dict`` closes that variant.
    import json

    body = {
        "model": "bedrock/anthropic.claude-v2",
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": json.dumps({"aws_web_identity_token": "anything"}),
    }
    with pytest.raises(ValueError, match="not allowed in request body"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="bedrock/anthropic.claude-v2",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"fallbacks": ["openai/gpt-4o-mini"]},
        {"extra_body": {"fallbacks": ["openai/gpt-4o-mini"]}},
        {"router_settings_override": {"fallbacks": ["openai/gpt-4o-mini"]}},
        {"metadata": {"fallbacks": ["openai/gpt-4o-mini"]}},
    ],
    ids=["root", "extra_body", "router_settings_override", "metadata"],
)
def test_configured_banned_request_body_params_rejects_param_across_containers(payload):
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        **payload,
    }
    with pytest.raises(ValueError, match="fallbacks is not allowed in request body"):
        is_request_body_safe(
            request_body=body,
            general_settings={"banned_request_body_params": ["fallbacks", "num_retries"]},
            llm_router=None,
            model="openai/gpt-4",
        )


def test_configured_banned_request_body_params_preserves_builtin_denylist():
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        "api_base": "https://attacker.invalid",
    }
    with pytest.raises(ValueError, match="api_base is not allowed in request body"):
        is_request_body_safe(
            request_body=body,
            general_settings={"banned_request_body_params": ["fallbacks"]},
            llm_router=None,
            model="openai/gpt-4",
        )


def test_configured_banned_request_body_params_respects_allow_client_side_credentials():
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        "fallbacks": ["openai/gpt-4o-mini"],
    }
    assert is_request_body_safe(
        request_body=body,
        general_settings={
            "banned_request_body_params": ["fallbacks"],
            "allow_client_side_credentials": True,
        },
        llm_router=None,
        model="openai/gpt-4",
    )


def test_configured_banned_request_body_params_respects_deployment_opt_in():
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "sk-test",
                    "configurable_clientside_auth_params": ["num_retries"],
                },
            }
        ]
    )
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        "num_retries": 2,
    }
    assert is_request_body_safe(
        request_body=body,
        general_settings={"banned_request_body_params": ["num_retries", "fallbacks"]},
        llm_router=router,
        model="gpt-4",
    )


@pytest.mark.parametrize(
    "invalid_setting",
    ["fallbacks", 123, ["fallbacks", 7]],
    ids=["bare-string", "integer", "mixed-list"],
)
def test_malformed_banned_request_body_params_fails_closed(invalid_setting):
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
    }
    with pytest.raises(ValueError, match="Invalid general_settings.banned_request_body_params"):
        is_request_body_safe(
            request_body=body,
            general_settings={"banned_request_body_params": invalid_setting},
            llm_router=None,
            model="openai/gpt-4",
        )
