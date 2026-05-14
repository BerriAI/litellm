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

from litellm.proxy.auth.auth_utils import (  # noqa: E402
    _BANNED_REQUEST_BODY_PARAMS,
    is_request_body_safe,
)


@pytest.mark.parametrize(
    "banned_param",
    [
        "aws_web_identity_token",
        "aws_sts_endpoint",
        "aws_role_name",
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


def test_azure_ad_token_is_in_banned_list():
    assert "azure_ad_token" in _BANNED_REQUEST_BODY_PARAMS


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
    # ``general_settings.allow_client_side_credentials`` is the documented
    # admin escape for clientside-credential passthrough. Walking
    # ``extra_body`` for banned params must not break the escape.
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
