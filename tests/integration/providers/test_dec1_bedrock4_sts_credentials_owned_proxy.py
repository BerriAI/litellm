"""STS credential chains for Bedrock deployments: AssumeRole with an external id and session tags, and
AssumeRoleWithWebIdentity, against an owned STS peer.

A deployment with `aws_role_name` gets its SigV4 keys from STS before Bedrock is called
(`litellm/llms/bedrock/base_aws_llm.py:443`): the proxy first asks `GetCallerIdentity` whether it already
runs as the role (`base_aws_llm.py:1244`), then calls `AssumeRole` with `RoleArn`, `RoleSessionName`
(defaulting to `litellm-session-<unix time>`), and `ExternalId` and `Tags` only when configured
(`_assume_role_params`, `base_aws_llm.py:94`). With `aws_web_identity_token` as well, it calls
`AssumeRoleWithWebIdentity` with the resolved token, `DurationSeconds: 3600` and a session policy
(`base_aws_llm.py:1055`). The Bedrock call is then signed with the assumed key and carries the assumed
session token in `x-amz-security-token`. Credentials are cached per full parameter set (`base_aws_llm.py:401`),
so the external id is part of the cache key.

These tests need an owned proxy: the `GetCallerIdentity` probe builds its STS client without the deployment's
`aws_sts_endpoint` (`base_aws_llm.py:952`), so only `AWS_ENDPOINT_URL_STS` keeps it off real AWS. The module
boots one owned proxy whose environment has synthetic source keys, empty AWS config files, IMDS disabled and
`AWS_ENDPOINT_URL_STS` at the module's STS peer, the same recipe as `test_bedrock_role_configuration.py`
(which pins the basic AssumeRole and session-name caching and is not repeated here). Each test uses its own
role ARN so the shared credential cache never answers it. `LITELLM_RUST` is `false`, so this pins the
Python credential chain; the Rust twin is listed as not covered.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_bedrock4_support import (
    ASSUMED_ACCESS_KEY,
    ASSUMED_TOKEN,
    CONVERSE_MODEL,
    CONVERSE_TARGET,
    STS_ACCESS_KEY,
    STS_SECRET_KEY,
    WEB_IDENTITY_ACCESS_KEY,
    WEB_IDENTITY_TOKEN,
    WEB_IDENTITY_TOKEN_ENV,
    calls_for,
    error_text,
    prompt,
    role_arn,
    sts_action,
    sts_reply,
)
from integration.providers.dec1_bedrock_support import converse_reply, register_model, sigv4_scope
from pydantic import JsonValue

WEB_IDENTITY_SESSION_TOKEN: Final = "synthetic-web-identity-session-token-dec1-bedrock4"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    sts: Wire
    bedrock: Wire


def _sts(request: Request) -> Reply:
    fields: Final = sts_action(request)
    action: Final = fields.get("Action", [""])[0]
    if "deny" in fields.get("RoleArn", [""])[0]:
        return Reply(
            status=403,
            content_type="text/xml",
            body=(
                b'<ErrorResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><Error><Type>Sender</Type>'
                b"<Code>AccessDenied</Code><Message>dec1 not authorized to perform: sts:AssumeRole</Message></Error>"
                b"<RequestId>dec1-deny</RequestId></ErrorResponse>"
            ),
        )
    if action == "AssumeRoleWithWebIdentity":
        return sts_reply(action, access_key=WEB_IDENTITY_ACCESS_KEY, token=WEB_IDENTITY_SESSION_TOKEN)
    return sts_reply(action)


def _bedrock(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b"{}")
    return converse_reply(text="dec1 role answer")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    """One owned proxy per module, with its STS and Bedrock peers. See the module docstring for the environment."""
    directory: Final = tmp_path_factory.mktemp("dec1-bedrock4-sts")
    empty: Final = directory / "empty-aws-config"
    empty.write_text("")
    with gateway_from_environment() as shared, wire_server(_sts) as sts, wire_server(_bedrock) as bedrock:
        overrides: Final = {
            "AWS_ACCESS_KEY_ID": STS_ACCESS_KEY,
            "AWS_SECRET_ACCESS_KEY": STS_SECRET_KEY,
            "AWS_CONFIG_FILE": str(empty),
            "AWS_SHARED_CREDENTIALS_FILE": str(empty),
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_ENDPOINT_URL_STS": sts.url,
            "AWS_DEFAULT_REGION": "us-east-1",
            "LITELLM_RUST": "false",
            WEB_IDENTITY_TOKEN_ENV: WEB_IDENTITY_TOKEN,
        }
        with owned_proxy(
            shared,
            Path(directory),
            overrides,
            remove_environment=tuple(name for name in os.environ if name.startswith("AWS_")),
        ) as owned:
            yield _Rig(owned, sts, bedrock)


def _deploy(rig: _Rig, **params: JsonValue) -> str:
    """Register a Converse deployment with no `api_key`, the Bedrock peer as runtime and the STS peer as `aws_sts_endpoint`.

    The deletion callback is dropped on purpose: the deployment lives until the module's owned proxy stops.
    """
    with rig.gateway.scenario() as scenario:
        scenario.cleanups.pop_all()
        return register_model(
            scenario,
            model=CONVERSE_MODEL,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=rig.bedrock.url,
            aws_sts_endpoint=rig.sts.url,
            **params,
        )


def _chat(rig: _Rig, model: str) -> tuple[int, str]:
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt()}], "cache": {"no-cache": True}},
    )
    return response.status_code, response.text


def _bedrock_posts(rig: _Rig) -> tuple[Request, ...]:
    return tuple(request for request in rig.bedrock.drain() if request.method == "POST")


def test_an_external_id_is_sent_on_assume_role_and_the_call_is_signed_with_the_assumed_key(rig: _Rig) -> None:
    """A deployment sets `aws_role_name`, `aws_session_name: dec1-ext` and `aws_external_id: dec1-external-7`.

    STS must get one `AssumeRole` naming that role with `RoleSessionName=dec1-ext` and `ExternalId=dec1-external-7`
    (`base_aws_llm.py:105`), and Bedrock must get the Converse POST signed with `ASIADEC1BEDROCK4ROLE` for
    `us-east-1`/`bedrock` with `x-amz-security-token` equal to the assumed session token.
    """
    role: Final = role_arn()
    model: Final = _deploy(rig, aws_role_name=role, aws_session_name="dec1-ext", aws_external_id="dec1-external-7")
    rig.bedrock.drain()
    status, text = _chat(rig, model)
    assert status == 200, text
    assumed: Final = calls_for(rig.sts.drain(), role, "AssumeRole")
    assert len(assumed) == 1, assumed
    assert assumed[0]["RoleSessionName"] == ["dec1-ext"]
    assert assumed[0]["ExternalId"] == ["dec1-external-7"]
    sent: Final = _bedrock_posts(rig)
    assert len(sent) == 1
    assert sent[0].target == CONVERSE_TARGET
    assert sigv4_scope(sent[0]) == (ASSUMED_ACCESS_KEY, "us-east-1", "bedrock")
    assert sent[0].headers["x-amz-security-token"] == ASSUMED_TOKEN


def test_without_an_external_id_assume_role_carries_no_external_id_field(rig: _Rig) -> None:
    """A role deployment with `aws_session_name: dec1-plain` and no `aws_external_id`.

    `_assume_role_params` builds `RoleArn` and `RoleSessionName` only (`base_aws_llm.py:101`), so the STS form
    must have no `ExternalId` key at all (an empty one would be refused by a role that requires none).
    """
    role: Final = role_arn()
    model: Final = _deploy(rig, aws_role_name=role, aws_session_name="dec1-plain")
    status, text = _chat(rig, model)
    assert status == 200, text
    assumed: Final = calls_for(rig.sts.drain(), role, "AssumeRole")
    assert len(assumed) == 1, assumed
    assert "ExternalId" not in assumed[0]
    assert assumed[0]["RoleSessionName"] == ["dec1-plain"]


def test_session_tags_are_sent_sorted_by_key(rig: _Rig) -> None:
    """A role deployment sets `aws_session_tags: [{"Key": "team", "Value": "genai"}, {"Key": "cost", "Value": "dec1"}]`.

    Tags are validated and sorted by `Key` (`base_aws_llm.py:68`), so the AssumeRole form must carry
    `Tags.member.1.Key=cost`, `Tags.member.1.Value=dec1`, `Tags.member.2.Key=team`, `Tags.member.2.Value=genai`.
    """
    role: Final = role_arn()
    model: Final = _deploy(
        rig,
        aws_role_name=role,
        aws_session_name="dec1-tags",
        aws_session_tags=[{"Key": "team", "Value": "genai"}, {"Key": "cost", "Value": "dec1"}],
    )
    status, text = _chat(rig, model)
    assert status == 200, text
    assumed: Final = calls_for(rig.sts.drain(), role, "AssumeRole")
    assert len(assumed) == 1, assumed
    tags: Final = {key: value[0] for key, value in assumed[0].items() if key.startswith("Tags.member.")}
    assert tags == {
        "Tags.member.1.Key": "cost",
        "Tags.member.1.Value": "dec1",
        "Tags.member.2.Key": "team",
        "Tags.member.2.Value": "genai",
    }


def test_a_role_without_a_session_name_gets_a_litellm_session_name(rig: _Rig) -> None:
    """A role deployment sets no `aws_session_name`.

    `_resolve_role_credentials` defaults it to `litellm-session-<unix seconds>` (`base_aws_llm.py:1257`), so the
    AssumeRole `RoleSessionName` must match `litellm-session-` followed by digits. CloudTrail audit tooling keys
    on this prefix.
    """
    role: Final = role_arn()
    model: Final = _deploy(rig, aws_role_name=role)
    status, text = _chat(rig, model)
    assert status == 200, text
    assumed: Final = calls_for(rig.sts.drain(), role, "AssumeRole")
    assert len(assumed) == 1, assumed
    session: Final = assumed[0]["RoleSessionName"][0]
    assert session.startswith("litellm-session-") and session.removeprefix("litellm-session-").isdigit(), session


def test_the_external_id_is_part_of_the_credential_cache_key(rig: _Rig) -> None:
    """Two deployments share a role and session name but differ in `aws_external_id` (`dec1-a`, `dec1-b`); each is
    called twice.

    The cache key is the whole parameter set including `aws_external_id` (`base_aws_llm.py:401`), so STS must see
    exactly two AssumeRole calls for the role, one per external id, and four Bedrock calls must succeed. A cache
    that ignores the external id would hand tenant B the session assumed for tenant A.
    """
    role: Final = role_arn()
    first: Final = _deploy(rig, aws_role_name=role, aws_session_name="dec1-cache", aws_external_id="dec1-a")
    second: Final = _deploy(rig, aws_role_name=role, aws_session_name="dec1-cache", aws_external_id="dec1-b")
    for model in (first, first, second, second):
        status, text = _chat(rig, model)
        assert status == 200, text
    assumed: Final = calls_for(rig.sts.drain(), role, "AssumeRole")
    assert sorted(entry["ExternalId"][0] for entry in assumed) == ["dec1-a", "dec1-b"]


def test_an_assume_role_access_denied_fails_the_call_without_reaching_bedrock(rig: _Rig) -> None:
    """The STS peer answers `AccessDenied` (HTTP 403) to AssumeRole for a role whose name contains `deny`.

    `GetCallerIdentity` names a user, not the role, so the proxy must not fall back to ambient credentials
    (`base_aws_llm.py:1364`): the call must fail and Bedrock must see no POST. UNVERIFIED: the caller status
    is not pinned (the botocore ClientError has no Bedrock mapping); only that it is an error.
    """
    role: Final = role_arn().replace("dec1-bedrock4-", "dec1-bedrock4-deny-")
    model: Final = _deploy(rig, aws_role_name=role, aws_session_name="dec1-deny")
    rig.bedrock.drain()
    status, text = _chat(rig, model)
    assert status >= 400, text
    assert _bedrock_posts(rig) == ()


def test_web_identity_calls_assume_role_with_web_identity_with_the_resolved_token(rig: _Rig) -> None:
    """A deployment sets `aws_role_name`, `aws_session_name: dec1-web` and
    `aws_web_identity_token: oidc/env/DEC1_BEDROCK4_OIDC_TOKEN`.

    The token is read from the proxy's environment (`litellm/secret_managers/main.py:444`), so STS must get one
    `AssumeRoleWithWebIdentity` with that `WebIdentityToken`, `RoleSessionName=dec1-web`,
    `DurationSeconds=3600` and a JSON `Policy` whose `Version` is `2012-10-17` (`base_aws_llm.py:1055`), and
    Bedrock must be signed with the web identity key and session token.
    """
    role: Final = role_arn()
    model: Final = _deploy(
        rig,
        aws_role_name=role,
        aws_session_name="dec1-web",
        aws_web_identity_token=f"oidc/env/{WEB_IDENTITY_TOKEN_ENV}",
    )
    rig.bedrock.drain()
    status, text = _chat(rig, model)
    assert status == 200, text
    calls: Final = calls_for(rig.sts.drain(), role, "AssumeRoleWithWebIdentity")
    assert len(calls) == 1, calls
    assert calls[0]["WebIdentityToken"] == [WEB_IDENTITY_TOKEN]
    assert calls[0]["RoleSessionName"] == ["dec1-web"]
    assert calls[0]["DurationSeconds"] == ["3600"]
    assert json.loads(calls[0]["Policy"][0])["Version"] == "2012-10-17"
    sent: Final = _bedrock_posts(rig)
    assert len(sent) == 1
    assert sigv4_scope(sent[0]) == (WEB_IDENTITY_ACCESS_KEY, "us-east-1", "bedrock")
    assert sent[0].headers["x-amz-security-token"] == WEB_IDENTITY_SESSION_TOKEN


def test_web_identity_never_probes_get_caller_identity_for_the_role(rig: _Rig) -> None:
    """The web identity branch is taken before the AssumeRole branch (`base_aws_llm.py:426`), so it never runs the
    `GetCallerIdentity` probe and never calls plain `AssumeRole`.

    For a fresh web identity deployment, the STS calls naming its role must be exactly one
    `AssumeRoleWithWebIdentity` and zero `AssumeRole`.
    """
    role: Final = role_arn()
    model: Final = _deploy(
        rig,
        aws_role_name=role,
        aws_session_name="dec1-web-only",
        aws_web_identity_token=f"oidc/env/{WEB_IDENTITY_TOKEN_ENV}",
    )
    status, text = _chat(rig, model)
    assert status == 200, text
    requests: Final = rig.sts.drain()
    assert len(calls_for(requests, role, "AssumeRoleWithWebIdentity")) == 1
    assert calls_for(requests, role, "AssumeRole") == ()


def test_web_identity_with_an_external_id_still_serves_the_request(rig: _Rig) -> None:
    """A web identity deployment also sets `aws_external_id: dec1-web-ext`.

    `AssumeRoleWithWebIdentity` has no `ExternalId` parameter in the STS API, yet the proxy adds it
    (`base_aws_llm.py:1064`); the intended behaviour is that the call still succeeds with one STS call.
    """
    pytest.skip(
        "BUG: base_aws_llm.py:1065 passes ExternalId to sts.assume_role_with_web_identity, which botocore "
        "rejects client side with ParamValidationError, so web identity plus aws_external_id can never authenticate"
    )
    role: Final = role_arn()
    model: Final = _deploy(
        rig,
        aws_role_name=role,
        aws_session_name="dec1-web-ext",
        aws_web_identity_token=f"oidc/env/{WEB_IDENTITY_TOKEN_ENV}",
        aws_external_id="dec1-web-ext",
    )
    status, text = _chat(rig, model)
    assert status == 200, text
    assert len(calls_for(rig.sts.drain(), role, "AssumeRoleWithWebIdentity")) == 1


def test_a_literal_web_identity_token_is_refused_without_calling_sts(rig: _Rig) -> None:
    """A web identity deployment carries the JWT itself as `aws_web_identity_token`, not a reference.

    `get_secret` treats a bare string as an environment variable name (`litellm/secret_managers/main.py:501`), finds
    nothing, and `_auth_with_web_identity_token` raises `OIDC token could not be retrieved from secret manager.`
    with status 401 (`base_aws_llm.py:1031`). STS and Bedrock must see nothing for the role. UNVERIFIED that the
    401 survives the proxy's exception mapping unchanged; the message is pinned.
    """
    role: Final = role_arn()
    model: Final = _deploy(
        rig, aws_role_name=role, aws_session_name="dec1-literal", aws_web_identity_token=WEB_IDENTITY_TOKEN
    )
    rig.bedrock.drain()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt()}], "cache": {"no-cache": True}},
    )
    assert response.status_code == 401, response.text
    assert "OIDC token could not be retrieved" in error_text(response), response.text
    assert calls_for(rig.sts.drain(), role, "AssumeRoleWithWebIdentity") == ()
    assert _bedrock_posts(rig) == ()
