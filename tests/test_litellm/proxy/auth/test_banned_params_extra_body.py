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


def test_admin_opt_in_still_permits_oci_routing_params():
    # The OCI serving-mode/endpoint/compartment selectors are banned by default
    # but, like every other credential/routing field, remain available behind the
    # ``allow_client_side_credentials`` admin opt-in.
    body = {
        "model": "oci/some-model",
        "messages": [{"role": "user", "content": "x"}],
        "oci_serving_mode": "DEDICATED",
        "oci_endpoint_id": "ocid1.generativeaiendpoint...",
    }
    assert is_request_body_safe(
        request_body=body,
        general_settings={"allow_client_side_credentials": True},
        llm_router=None,
        model="oci/some-model",
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


# --- Cluster: nested-container + alias + regex-anchor hardening ---

from litellm.proxy.auth.auth_utils import check_regex_or_str_match  # noqa: E402


@pytest.mark.parametrize(
    "banned_param",
    [
        "model_list",  # deployment-dict smuggling past model authz
        "vertex_ai_credentials",  # alias of vertex_credentials (executable-cred RCE sink)
        "vertex_project",  # cross-tenant project impersonation
        "litellm_credential_name",  # globally-stored credential by name, no ownership
        "aws_profile_name",  # server-side AWS profile selection
        "base_model",  # cost-lookup model spoofing
        "oci_key_file",  # arbitrary server file path read (DoS)
        "oci_serving_mode",  # flip ON_DEMAND->DEDICATED to retarget the model
        "oci_endpoint_id",  # retarget the OCI dedicated endpoint
        "oci_compartment_id",  # retarget the OCI compartment
    ],
)
def test_newly_banned_root_param_is_rejected(banned_param):
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        banned_param: "attacker-chosen",
    }
    with pytest.raises(ValueError, match="not allowed in request body"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="openai/gpt-4",
        )


def test_banned_param_under_litellm_params_template_is_rejected():
    # The Gemini/Vertex Agents endpoints spread litellm_params_template into the
    # outbound request, so an api_base smuggled there must be caught.
    body = {
        "model": "gemini/gemini-1.5-pro",
        "messages": [{"role": "user", "content": "x"}],
        "litellm_params_template": {
            "api_base": "http://169.254.169.254/",
            "api_key": "dummy",
        },
    }
    with pytest.raises(ValueError, match="not allowed in request body"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="gemini/gemini-1.5-pro",
        )


def test_banned_param_in_a_tool_object_is_rejected():
    # The Advisor orchestration tool reads api_base straight off the tool dict;
    # an attacker endpoint there exfiltrates the proxy's provider key.
    body = {
        "model": "anthropic/claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "advisor_20260301", "api_base": "https://attacker.example"}],
    }
    with pytest.raises(ValueError, match="not allowed in request body"):
        is_request_body_safe(
            request_body=body,
            general_settings={},
            llm_router=None,
            model="anthropic/claude-3-5-sonnet",
        )


def test_legit_function_tool_with_api_base_param_is_allowed():
    # A real function tool whose JSON-schema parameters happen to define an
    # "api_base" property is data, not a top-level override — must NOT be rejected.
    body = {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "fetch",
                    "parameters": {
                        "type": "object",
                        "properties": {"api_base": {"type": "string"}},
                    },
                },
            }
        ],
    }
    assert is_request_body_safe(
        request_body=body,
        general_settings={},
        llm_router=None,
        model="openai/gpt-4",
    )


def test_clientside_api_base_regex_is_fully_anchored():
    # An allowlist regex must match the WHOLE value: a prefix-only match would
    # accept a look-alike host with an attacker suffix.
    allow = r"https://api\.openai\.com"
    assert check_regex_or_str_match("https://api.openai.com", allow) is True
    assert (
        check_regex_or_str_match("https://api.openai.com.attacker.com/v1", allow)
        is False
    )
