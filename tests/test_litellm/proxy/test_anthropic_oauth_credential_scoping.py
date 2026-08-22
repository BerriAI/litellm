"""A client-supplied Anthropic OAuth credential must only ever reach Anthropic.

The proxy forwards a caller's ``Authorization: Bearer sk-ant-oat...`` upstream so an
Anthropic subscription keeps working through LiteLLM. That credential is meaningless to
AWS Bedrock and Google Vertex AI, and sending it there both breaks the request and hands
a third-party cloud a credential it has no business holding. These tests pin the scope of
that credential from the proxy pre-call path all the way into the headers each provider
actually signs and sends.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest
from botocore.credentials import Credentials

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.get_provider_specific_headers import (
    ProviderSpecificHeaderUtils,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.proxy.litellm_pre_call_utils import (
    add_provider_specific_headers_to_request,
)

OAUTH_TOKEN = "Bearer sk-ant-oat01-fake-subscription-token-for-testing-0123456789"
GOOGLE_ACCESS_TOKEN = "Bearer ya29.fake-google-access-token-for-testing"
BEDROCK_API_KEY = "ABSKQmVkcm9ja0FQSUtleUZvclRlc3Rpbmc="
CROSS_ACCOUNT_AUTHORIZATION = "Bearer deliberately-configured-pass-through-token"

SIGV4_PREFIX = "AWS4-HMAC-SHA256"
AUTHORIZATION_HEADER_CASINGS = ["authorization", "Authorization", "AUTHORIZATION"]
LEAK_TARGET_PROVIDERS = ["bedrock", "bedrock_converse", "vertex_ai"]

BEDROCK_ENDPOINT = (
    "https://bedrock-runtime.us-west-2.amazonaws.com"
    "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke"
)
BEDROCK_REGION = "us-west-2"
BEDROCK_REQUEST_DATA = {"messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 32}
SIGV4_OPTIONAL_PARAMS = {
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "aws_region_name": BEDROCK_REGION,
}


def _client_headers(authorization_header_name: str | None = "authorization") -> dict:
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "user-agent": "claude-cli/2.1.239",
    }
    if authorization_header_name is not None:
        headers[authorization_header_name] = OAUTH_TOKEN
    return headers


def _headers_forwarded_to(client_headers: dict, custom_llm_provider: str) -> dict:
    data: dict = {}
    add_provider_specific_headers_to_request(data=data, headers=client_headers)
    return ProviderSpecificHeaderUtils.get_provider_specific_headers(
        provider_specific_header=data.get("provider_specific_header"),
        custom_llm_provider=custom_llm_provider,
    )


def _authorization_values(headers) -> list:
    return [value for name, value in headers.items() if name.lower() == "authorization"]


def _signed_headers_for_bedrock(request_headers: dict, api_key: str | None = None) -> dict:
    with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": ""}):
        signed_headers, _ = BaseAWSLLM()._sign_request(
            service_name="bedrock",
            headers=request_headers,
            optional_params=SIGV4_OPTIONAL_PARAMS,
            request_data=BEDROCK_REQUEST_DATA,
            api_base=BEDROCK_ENDPOINT,
            api_key=api_key,
        )
    return signed_headers


def _signed_headers_component(signature: str, component: str) -> str:
    for part in signature.removeprefix(SIGV4_PREFIX).split(","):
        name, _, value = part.strip().partition("=")
        if name == component:
            return value
    raise AssertionError(f"{component} missing from {signature}")


@pytest.mark.parametrize("authorization_header_name", AUTHORIZATION_HEADER_CASINGS)
@pytest.mark.parametrize("custom_llm_provider", LEAK_TARGET_PROVIDERS)
def test_oauth_credential_is_never_forwarded_to_bedrock_or_vertex(
    authorization_header_name, custom_llm_provider
):
    forwarded = _headers_forwarded_to(_client_headers(authorization_header_name), custom_llm_provider)

    assert _authorization_values(forwarded) == []
    assert OAUTH_TOKEN not in forwarded.values()


@pytest.mark.parametrize("authorization_header_name", AUTHORIZATION_HEADER_CASINGS)
def test_oauth_credential_still_reaches_anthropic_unchanged(authorization_header_name):
    forwarded = _headers_forwarded_to(_client_headers(authorization_header_name), "anthropic")

    assert forwarded[authorization_header_name] == OAUTH_TOKEN
    assert _authorization_values(forwarded) == [OAUTH_TOKEN]


def test_oauth_credential_entry_is_scoped_to_anthropic_alone():
    data: dict = {}
    add_provider_specific_headers_to_request(data=data, headers=_client_headers())

    scoped_headers = data["provider_specific_header"]
    if not isinstance(scoped_headers, list):
        scoped_headers = [scoped_headers]

    credential_entries = [
        entry for entry in scoped_headers if OAUTH_TOKEN in entry["extra_headers"].values()
    ]
    assert [entry["custom_llm_provider"] for entry in credential_entries] == ["anthropic"]


def test_no_provider_specific_header_when_client_sends_nothing_anthropic():
    data: dict = {}
    add_provider_specific_headers_to_request(
        data=data, headers={"content-type": "application/json", "authorization": "Bearer sk-a-normal-key"}
    )

    assert "provider_specific_header" not in data


def test_bedrock_sigv4_signature_survives_a_client_oauth_header():
    forwarded = _headers_forwarded_to(_client_headers(), "bedrock")

    signed = _signed_headers_for_bedrock({"Content-Type": "application/json", **forwarded})

    authorizations = _authorization_values(signed)
    assert len(authorizations) == 1
    assert authorizations[0].startswith(SIGV4_PREFIX)
    assert signed["X-Amz-Date"]


def test_bedrock_sigv4_signing_is_unchanged_by_the_client_oauth_header():
    without_oauth = _signed_headers_for_bedrock(
        {"Content-Type": "application/json", **_headers_forwarded_to(_client_headers(None), "bedrock")}
    )
    with_oauth = _signed_headers_for_bedrock(
        {"Content-Type": "application/json", **_headers_forwarded_to(_client_headers(), "bedrock")}
    )

    assert without_oauth["Authorization"].startswith(SIGV4_PREFIX)
    assert _signed_headers_component(with_oauth["Authorization"], "SignedHeaders") == (
        _signed_headers_component(without_oauth["Authorization"], "SignedHeaders")
    )


def test_bedrock_get_request_headers_keeps_the_sigv4_signature():
    forwarded = _headers_forwarded_to(_client_headers(), "bedrock")

    with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": ""}):
        prepped = BaseAWSLLM().get_request_headers(
            credentials=Credentials(
                SIGV4_OPTIONAL_PARAMS["aws_access_key_id"],
                SIGV4_OPTIONAL_PARAMS["aws_secret_access_key"],
            ),
            aws_region_name=BEDROCK_REGION,
            extra_headers=forwarded,
            endpoint_url=BEDROCK_ENDPOINT,
            data=json.dumps(BEDROCK_REQUEST_DATA),
            headers={"Content-Type": "application/json", **forwarded},
        )

    authorizations = _authorization_values(prepped.headers)
    assert len(authorizations) == 1
    assert authorizations[0].startswith(SIGV4_PREFIX)


def test_bedrock_api_key_deployment_keeps_its_own_bearer_token():
    forwarded = _headers_forwarded_to(_client_headers(), "bedrock")

    signed = _signed_headers_for_bedrock(
        {"Content-Type": "application/json", **forwarded}, api_key=BEDROCK_API_KEY
    )

    assert _authorization_values(signed) == [f"Bearer {BEDROCK_API_KEY}"]


def test_deliberately_configured_authorization_still_overrides_sigv4():
    signed = _signed_headers_for_bedrock(
        {"Content-Type": "application/json", "Authorization": CROSS_ACCOUNT_AUTHORIZATION}
    )

    assert _authorization_values(signed) == [CROSS_ACCOUNT_AUTHORIZATION]


def test_vertex_sends_exactly_one_authorization_header():
    forwarded = _headers_forwarded_to(_client_headers(), "vertex_ai")

    vertex_request_headers = {
        "content-type": "application/json",
        "Authorization": GOOGLE_ACCESS_TOKEN,
    }
    vertex_request_headers.update(forwarded)

    assert _authorization_values(vertex_request_headers) == [GOOGLE_ACCESS_TOKEN]
