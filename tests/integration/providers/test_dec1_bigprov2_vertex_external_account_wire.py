"""Vertex AI with `external_account` credentials (Google workload identity federation) on the wire.

A deployment's `vertex_credentials` can be an `external_account` JSON instead of a service account key.
`VertexBase._credentials_from_json` (`litellm/llms/vertex_ai/vertex_llm_base.py:237`) picks the google-auth
class from its shape: a `credential_source` whose `environment_id` contains `aws` is an AWS source (and with
LiteLLM's explicit `aws_*` keys it goes through `VertexAIAwsWifAuth.credentials_from_explicit_aws`, which
signs a GetCallerIdentity request locally instead of reading the EC2 metadata service), an `executable`
source is pluggable, and anything else (`file`, `url`) is an identity pool. Every form ends in the same STS
token exchange at the JSON's `token_url`, optionally followed by service account impersonation, and the
resulting access token is the `Authorization: Bearer` on the Vertex call.

These are the forms customers on EKS, GKE fleets, GitHub Actions and AWS use to reach Vertex without a
key file, and none of them was pinned before (`index/test_dec1_vertex_.md`, "Not covered"). A Rust port of
Vertex auth has to reproduce the STS form exactly, keep the token cached across calls, and refuse cleanly
when the exchange fails. One owned peer (`wif_peer`) is the subject URL, the STS endpoint, IAM credentials
and the Vertex host at once. The deployment always names `vertex_project`, because an `external_account`
JSON carries no project id. UNVERIFIED: the exact google-auth form fields are taken from its STS contract
(google-auth is not installed here), and these tests assume the installed google-auth does not restrict
`token_url` to `googleapis.com` hosts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply
from integration.providers.dec1_bigprov2_support import (
    ACCESS_TOKEN_TYPE,
    AUDIENCE,
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_TOKEN_TYPE,
    CLOUD_PLATFORM,
    IMPERSONATED_ACCOUNT,
    JWT_TOKEN_TYPE,
    SUBJECT_TOKEN,
    TOKEN_EXCHANGE_GRANT,
    FederationPlan,
    aws_external_account,
    aws_subject,
    aws_subject_headers,
    bearer_of,
    body_of,
    chat,
    error_message,
    form_of,
    identity_pool_file,
    identity_pool_url,
    impersonation,
    models_path,
    path_of,
    register_vertex,
    wif_peer,
)


def _subject_file(tmp_path: Path, token: str = SUBJECT_TOKEN) -> str:
    path: Final = tmp_path / "dec1-oidc-token"
    path.write_text(token)
    return str(path)


def test_a_file_sourced_identity_pool_exchanges_the_file_token_at_the_sts_url(gateway: Gateway, tmp_path: Path) -> None:
    """An `external_account` whose `credential_source.file` holds `dec1-oidc-subject-token` serves one chat.

    The peer's STS endpoint must receive one form POST with `grant_type` token-exchange, the JSON's
    `audience`, `subject_token` equal to the file content, `subject_token_type` jwt, `requested_token_type`
    access_token and `scope` cloud-platform. A rewrite that reads the file but posts JSON to STS, or forgets
    the scope, is refused by Google.
    """
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, identity_pool_file(peer.url, _subject_file(tmp_path)))
        assert chat(gateway, model).status_code == 200
        (exchange,) = peer.sts_calls()
        assert exchange.method == "POST"
        assert exchange.headers.get("content-type", "").startswith("application/x-www-form-urlencoded")
        form: Final = form_of(exchange)
        assert form["grant_type"] == TOKEN_EXCHANGE_GRANT
        assert form["audience"] == AUDIENCE
        assert form["subject_token"] == SUBJECT_TOKEN
        assert form["subject_token_type"] == JWT_TOKEN_TYPE
        assert form["requested_token_type"] == ACCESS_TOKEN_TYPE
        assert form["scope"] == CLOUD_PLATFORM


def test_the_sts_access_token_is_the_bearer_on_the_vertex_call(gateway: Gateway, tmp_path: Path) -> None:
    """With no impersonation URL, the `access_token` the STS endpoint issues (`sts-token-1`) is sent as
    `Authorization: Bearer sts-token-1` on the one `:generateContent` call to the deployment's project and
    location."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, identity_pool_file(peer.url, _subject_file(tmp_path)))
        assert chat(gateway, model).status_code == 200
        (call,) = peer.model_calls()
        assert path_of(call) == models_path() + ":generateContent"
        assert bearer_of(call) == "sts-token-1"


def test_a_url_sourced_identity_pool_reads_the_subject_token_with_its_headers(gateway: Gateway) -> None:
    """`credential_source.url` points at `<peer>/subject` with header `Metadata-Flavor: dec1`. The proxy (through
    google-auth) must GET that URL with the header and use the plain text body as `subject_token` in the STS
    exchange. This is how Azure managed identity and GKE fleet sources are configured."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, identity_pool_url(peer.url))
        assert chat(gateway, model).status_code == 200
        subject_calls: Final = peer.subject_calls()
        assert len(subject_calls) >= 1
        assert subject_calls[0].method == "GET"
        assert subject_calls[0].headers.get("metadata-flavor") == "dec1"
        assert form_of(peer.sts_calls()[0])["subject_token"] == SUBJECT_TOKEN


def test_a_json_formatted_url_source_takes_the_named_field_as_the_subject_token(gateway: Gateway) -> None:
    """With `format: {"type": "json", "subject_token_field_name": "id_token"}` the subject URL answers
    `{"id_token": "dec1-json-subject"}` and the STS `subject_token` must be exactly `dec1-json-subject`,
    not the whole JSON document."""
    plan: Final = FederationPlan(subject="dec1-json-subject", subject_json_field="id_token")
    with wif_peer(plan) as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, identity_pool_url(peer.url, json_field="id_token"))
        assert chat(gateway, model).status_code == 200
        assert form_of(peer.sts_calls()[0])["subject_token"] == "dec1-json-subject"


def test_impersonation_trades_the_sts_token_for_a_service_account_token(gateway: Gateway, tmp_path: Path) -> None:
    """With `service_account_impersonation_url` set, the STS token (`sts-token-1`) is sent as the bearer of one
    POST to `.../serviceAccounts/<account>:generateAccessToken` whose JSON body asks for scope
    cloud-platform, and the Vertex call carries the impersonated `accessToken` (`impersonated-token-1`), never
    the STS token."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        credentials: Final = {**identity_pool_file(peer.url, _subject_file(tmp_path)), **impersonation(peer.url)}
        model: Final = register_vertex(scenario, peer, credentials)
        assert chat(gateway, model).status_code == 200
        (minted,) = peer.impersonation_calls()
        assert IMPERSONATED_ACCOUNT in path_of(minted)
        assert bearer_of(minted) == "sts-token-1"
        assert CLOUD_PLATFORM in str(body_of(minted)["scope"])
        (call,) = peer.model_calls()
        assert bearer_of(call) == "impersonated-token-1"


def test_impersonation_asks_sts_for_the_iam_scope_rather_than_the_caller_scope(
    gateway: Gateway, tmp_path: Path
) -> None:
    """When impersonating, google-auth asks STS for the IAM scope (`https://www.googleapis.com/auth/cloud-platform`
    in today's google-auth) and the requested scopes go into the `generateAccessToken` body instead.
    UNVERIFIED: read from google-auth's `external_account.Credentials.refresh`, which swaps the scope when an
    impersonation URL is set; the STS form must still carry exactly one `scope`."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(
            scenario, peer, {**identity_pool_file(peer.url, _subject_file(tmp_path)), **impersonation(peer.url)}
        )
        assert chat(gateway, model).status_code == 200
        assert form_of(peer.sts_calls()[0])["scope"] == CLOUD_PLATFORM


def test_the_federated_token_is_reused_across_calls(gateway: Gateway, tmp_path: Path) -> None:
    """Two chat calls on one external account deployment make exactly one STS exchange (the token is valid
    for 3600 seconds) and both Vertex calls carry `sts-token-1`. A port that exchanges per request would
    multiply STS traffic and hit Google's federation quota."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, identity_pool_file(peer.url, _subject_file(tmp_path)))
        assert chat(gateway, model).status_code == 200
        assert chat(gateway, model).status_code == 200
        assert len(peer.sts_calls()) == 1
        assert [bearer_of(call) for call in peer.model_calls()] == ["sts-token-1", "sts-token-1"]


def test_a_rejected_federation_exchange_fails_the_call_without_reaching_vertex(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The STS endpoint answers 400 `{"error": "invalid_grant", "error_description": "dec1 subject expired"}`.
    The caller gets an error status and Vertex receives no request. UNVERIFIED: the status is expected to be
    500 like the service account `invalid_grant` case (`test_dec1_vertex_credentials_tokens.py`), since the
    google-auth error carries no HTTP status of its own."""
    rejected: Final = Reply(status=400, body=b'{"error": "invalid_grant", "error_description": "dec1 subject expired"}')
    with wif_peer(FederationPlan(sts_error=rejected)) as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, identity_pool_file(peer.url, _subject_file(tmp_path)))
        response: Final = chat(gateway, model, num_retries=0)
        assert response.status_code == 500, response.text
        assert peer.model_calls() == ()
        assert len(peer.sts_calls()) >= 1


def test_a_missing_subject_token_file_fails_the_call_without_any_exchange(gateway: Gateway, tmp_path: Path) -> None:
    """`credential_source.file` names a path that does not exist. The call fails with an error status, no STS
    exchange is attempted and Vertex receives nothing; a port must not fall back to another credential
    source (for example the proxy's own ADC) when the configured one is broken."""
    missing: Final = str(tmp_path / "not-there")
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, identity_pool_file(peer.url, missing))
        response: Final = chat(gateway, model, num_retries=0)
        assert response.status_code >= 400, response.text
        assert peer.sts_calls() == ()
        assert peer.model_calls() == ()


def test_explicit_aws_keys_sign_a_get_caller_identity_subject_for_the_exchange(gateway: Gateway) -> None:
    """An AWS `external_account` with LiteLLM's `aws_access_key_id`, `aws_secret_access_key` and
    `aws_region_name: us-east-1` serves one chat. The STS form must carry `subject_token_type`
    `urn:ietf:params:aws:token-type:aws4_request` and a subject token that, URL decoded, is a JSON request
    description: `method` POST, `url` `https://sts.us-east-1.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15`.
    UNVERIFIED: the subject token encoding is google-auth's (`aws.Credentials.retrieve_subject_token`)."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, aws_external_account(peer.url))
        assert chat(gateway, model).status_code == 200
        (exchange,) = peer.sts_calls()
        form: Final = form_of(exchange)
        assert form["subject_token_type"] == AWS_TOKEN_TYPE
        assert form["audience"] == AUDIENCE
        subject: Final = aws_subject(exchange)
        assert subject["method"] == "POST"
        assert subject["url"] == f"https://sts.{AWS_REGION}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"


def test_the_aws_subject_is_sigv4_signed_with_the_explicit_key_and_names_the_audience(gateway: Gateway) -> None:
    """The headers inside the AWS subject token must include an `authorization` starting
    `AWS4-HMAC-SHA256 Credential=AKIADEC1WIFSYNTHETIC/` and scoped to `us-east-1/sts/aws4_request`, and
    `x-goog-cloud-target-resource` equal to the audience. Google verifies the signature by replaying the
    request to AWS, so a port that signs for the wrong region or drops the target header is refused."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, aws_external_account(peer.url))
        assert chat(gateway, model).status_code == 200
        headers: Final = aws_subject_headers(peer.sts_calls()[0])
        assert headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={AWS_ACCESS_KEY}/")
        assert f"/{AWS_REGION}/sts/aws4_request" in headers["authorization"]
        assert headers["x-goog-cloud-target-resource"] == AUDIENCE


def test_an_aws_session_token_travels_inside_the_signed_subject(gateway: Gateway) -> None:
    """With `aws_session_token: dec1-session` added, the subject token headers must include
    `x-amz-security-token: dec1-session`, so temporary credentials from AssumeRole keep working."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, aws_external_account(peer.url, aws_session_token="dec1-session"))
        assert chat(gateway, model).status_code == 200
        assert aws_subject_headers(peer.sts_calls()[0])["x-amz-security-token"] == "dec1-session"


def test_explicit_aws_keys_without_a_region_are_refused_before_any_exchange(gateway: Gateway) -> None:
    """The AWS `external_account` has `aws_access_key_id` and `aws_secret_access_key` but no
    `aws_region_name`. The call fails with a message containing `aws_region_name is required in the WIF
    credential JSON` (`vertex_ai_aws_wif.py:80`), and neither STS nor Vertex is called."""
    with wif_peer() as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(scenario, peer, aws_external_account(peer.url, aws_region_name=None))
        response: Final = chat(gateway, model, num_retries=0)
        assert response.status_code >= 400, response.text
        assert "aws_region_name is required in the WIF credential JSON" in error_message(response)
        assert peer.sts_calls() == ()
        assert peer.model_calls() == ()


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_the_federated_bearer_reaches_vertex_claude_raw_predict(gateway: Gateway, tmp_path: Path, stream: bool) -> None:
    """A `vertex_ai/claude-sonnet-4-5@20250929` deployment in `us-east5` with the same file sourced external
    account calls `.../publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict` (or
    `:streamRawPredict`) with `Authorization: Bearer sts-token-1`. Claude on Vertex uses a separate request
    path from Gemini, and both must take the federated token."""
    from integration.providers.dec1_vertex_support import CLAUDE, claude_echo, claude_path

    with wif_peer(respond=claude_echo) as peer, gateway.scenario() as scenario:
        model: Final = register_vertex(
            scenario,
            peer,
            identity_pool_file(peer.url, _subject_file(tmp_path)),
            backend=f"vertex_ai/{CLAUDE}",
            vertex_location="us-east5",
        )
        response: Final = chat(gateway, model, stream=stream)
        assert response.status_code == 200, response.text
        (call,) = peer.model_calls()
        method: Final = "streamRawPredict" if stream else "rawPredict"
        assert path_of(call) == f"{claude_path(project='dec1-wif-project')}:{method}"
        assert bearer_of(call) == "sts-token-1"
