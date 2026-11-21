"""Bedrock regions, cross-region inference profiles, credentials and endpoints, as the upstream sees them.

Every Bedrock call has to pick three things before it is sent: the model id that goes into the URL
(a cross-region inference profile such as `us.`, `eu.`, `apac.`, `ca.` (#52152), `jp.`, `au.` or `global.`
is sent verbatim, while a `bedrock/<region>/` path segment or an ARN is not), the AWS region (deployment
`aws_region_name`, else a region segment in the model path, else the region inside an ARN, else
`AWS_REGION_NAME`, `AWS_REGION`, boto3, `us-west-2`; `base_aws_llm.py:716` and `converse_handler.py:316`),
and the credential (a Bedrock API key sent as `Authorization: Bearer`, or SigV4 over static keys, a session
token or an assumed role, signed for service `bedrock` in that region). The endpoint is `api_base`, else
`aws_bedrock_runtime_endpoint`, else `AWS_BEDROCK_RUNTIME_ENDPOINT`, else
`https://bedrock-runtime.<region>.<partition suffix>`, and on Converse a configured
`aws_bedrock_runtime_endpoint` is where the request is actually sent even when `api_base` is also set
(`get_runtime_endpoint`, `base_aws_llm.py:1483`).

The cross-region prefix also feeds pricing: the cost map prices `us.`, `eu.` and `apac.` profiles at a
regional premium and `global.` at the base rate. The environment-variable branches need a proxy with its
own environment and live in `test_dec1_bedrock_credentials_env_owned_proxy.py`. Role assumption against a
real STS endpoint is already pinned in `test_bedrock_role_configuration.py`.
"""

from __future__ import annotations

import urllib.parse
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_bedrock_support import (
    ACCESS_KEY,
    CLAUDE_HAIKU_55,
    SESSION_TOKEN,
    TOKEN,
    assert_sigv4_signed,
    bedrock,
    body_of,
    converse_reply,
    path_of,
    signed_headers,
    sigv4_params,
    sigv4_scope,
)


def _ok(_: Request) -> Reply:
    return converse_reply()


@pytest.mark.parametrize("prefix", ["us", "eu", "apac", "ca", "jp", "au", "global"])
def test_a_cross_region_profile_id_is_sent_verbatim_to_converse(gateway: Gateway, prefix: str) -> None:
    """`bedrock/<prefix>.anthropic.claude-haiku-5-5` names a cross-region inference profile.

    Bedrock routes on the profile id, so the request must go to `/model/<prefix>.anthropic.claude-haiku-5-5/converse`
    with the prefix kept, and the route must be Converse for every prefix in
    `get_bedrock_cross_region_inference_regions` (`common_utils.py:1195`), including `ca` added by #52152.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{prefix}.{CLAUDE_HAIKU_55}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == f"/model/{prefix}.{CLAUDE_HAIKU_55}/converse"


def test_reasoning_effort_on_the_canada_profile_reaches_converse_like_on_the_us_profile(gateway: Gateway) -> None:
    """Before #52152 the `ca.` prefix was not stripped when looking up model capabilities, so effort was lost.

    `reasoning_effort: "max"` on `bedrock/ca.anthropic.claude-haiku-5-5` must reach Bedrock as
    `additionalModelRequestFields.output_config == {"effort": "max"}`, exactly what the `us.` profile gets
    (unit twin: `test_reasoning_effort_reaches_the_request_on_the_canada_inference_profile`).
    """
    fields: dict[str, object] = {}
    for prefix in ("us", "ca"):
        with bedrock(gateway, _ok, model=f"bedrock/{prefix}.{CLAUDE_HAIKU_55}") as rig:
            response = rig.chat(reasoning_effort="max", max_tokens=4000, drop_params=True)
            assert response.status_code == 200, response.text
            additional = body_of(rig.one_post())["additionalModelRequestFields"]
            assert isinstance(additional, dict), additional
            fields[prefix] = additional
    assert fields["ca"]["output_config"] == {"effort": "max"}  # pyright: ignore[reportIndexIssue]
    assert fields["ca"] == fields["us"]


@pytest.mark.parametrize(
    ("prefix", "input_rate", "output_rate"),
    [
        ("us", 1.1e-07, 5.5e-07),
        ("eu", 1.1e-07, 5.5e-07),
        ("apac", 1.1e-07, 5.5e-07),
        ("global", 1e-07, 5e-07),
    ],
)
def test_cross_region_profiles_are_priced_from_their_own_cost_map_rows(
    gateway: Gateway, prefix: str, input_rate: float, output_rate: float
) -> None:
    """With no deployment prices, each profile's own cost map row prices the call.

    `us.`, `eu.` and `apac.anthropic.claude-haiku-5-5` carry a 10 percent regional premium (1.1e-07 in,
    5.5e-07 out) and `global.` the base rate (1e-07, 5e-07), so 11 input and 7 output tokens must cost
    `11 * input + 7 * output` in `x-litellm-response-cost`.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{prefix}.{CLAUDE_HAIKU_55}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(11 * input_rate + 7 * output_rate)


def test_the_canada_profile_without_a_cost_map_row_is_priced_at_the_base_model_rate(gateway: Gateway) -> None:
    """`ca.anthropic.claude-haiku-5-5` has no cost map row of its own.

    The cost lookup strips the profile prefix (`_is_bedrock_cross_region_profile_of`, `cost_calculator.py:1168`)
    and must fall back to `anthropic.claude-haiku-5-5` (1e-07 in, 5e-07 out): 11 and 7 tokens cost 4.6e-06,
    never 0 and never an error. UNVERIFIED: the exact fallback chain; the regional premium is not applied.
    """
    with bedrock(gateway, _ok, model=f"bedrock/ca.{CLAUDE_HAIKU_55}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(11 * 1e-07 + 7 * 5e-07)


def test_a_bearer_key_deployment_sends_no_sigv4_headers(gateway: Gateway) -> None:
    """A deployment with `api_key` set (a Bedrock API key) authenticates with `Authorization: Bearer <key>` only.

    Bedrock must see no `x-amz-date`, no `x-amz-security-token` and no `AWS4-HMAC-SHA256` signature
    (`get_request_headers`, `base_aws_llm.py:1579`), and the proxy master key must never appear.
    """
    with bedrock(gateway, _ok, model=f"bedrock/us.{CLAUDE_HAIKU_55}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.headers["authorization"] == f"Bearer {TOKEN}"
        assert "x-amz-date" not in sent.headers
        assert "x-amz-security-token" not in sent.headers
        assert gateway.key not in str(dict(sent.headers))


def test_static_keys_sign_the_converse_request_with_sigv4_for_the_deployment_region(gateway: Gateway) -> None:
    """A deployment with `aws_access_key_id`, `aws_secret_access_key`, `aws_region_name: eu-west-1` and a blank `api_key`.

    Bedrock must see `Authorization: AWS4-HMAC-SHA256` whose credential scope is
    `<access key>/<date>/eu-west-1/bedrock/aws4_request`, whose `SignedHeaders` include `host` and
    `x-amz-date`, and whose signature matches an independent SigV4 computation over the exact bytes received.
    """
    with bedrock(
        gateway, _ok, model=f"bedrock/eu.{CLAUDE_HAIKU_55}", **sigv4_params(aws_region_name="eu-west-1")
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sigv4_scope(sent) == (ACCESS_KEY, "eu-west-1", "bedrock")
        assert {"host", "x-amz-date"} <= set(signed_headers(sent))
        assert_sigv4_signed(sent, region="eu-west-1")


def test_a_session_token_is_sent_and_covered_by_the_signature(gateway: Gateway) -> None:
    """Temporary credentials: `aws_session_token` next to the keys.

    Bedrock must see `x-amz-security-token: <token>`, that header must be listed in `SignedHeaders`, and the
    signature must still verify.
    """
    with bedrock(
        gateway,
        _ok,
        model=f"bedrock/us.{CLAUDE_HAIKU_55}",
        **sigv4_params(aws_region_name="us-east-2", aws_session_token=SESSION_TOKEN),
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.headers["x-amz-security-token"] == SESSION_TOKEN
        assert "x-amz-security-token" in signed_headers(sent)
        assert_sigv4_signed(sent, region="us-east-2")


def test_a_region_segment_in_the_model_path_sets_the_signing_region_and_leaves_the_url(gateway: Gateway) -> None:
    """`bedrock/us-west-2/anthropic.claude-haiku-5-5` with keys and no `aws_region_name`.

    `converse_handler.py:316` strips the region segment from the model id and uses it as the region, so
    Bedrock must see `/model/anthropic.claude-haiku-5-5/converse` (no `us-west-2/` in the path) and a SigV4
    scope region of `us-west-2`.
    """
    with bedrock(
        gateway, _ok, model=f"bedrock/us-west-2/{CLAUDE_HAIKU_55}", **sigv4_params(aws_region_name=None)
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{CLAUDE_HAIKU_55}/converse"
        assert sigv4_scope(sent)[1] == "us-west-2"


def test_the_deployment_region_wins_over_a_region_segment_in_the_model_path(gateway: Gateway) -> None:
    """`bedrock/us-west-2/anthropic.claude-haiku-5-5` with `aws_region_name: ap-south-1`.

    The path region is only injected when `aws_region_name` is absent (`converse_handler.py:331`), so the
    SigV4 scope must name `ap-south-1` while the URL still drops the `us-west-2/` segment.
    """
    with bedrock(
        gateway, _ok, model=f"bedrock/us-west-2/{CLAUDE_HAIKU_55}", **sigv4_params(aws_region_name="ap-south-1")
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{CLAUDE_HAIKU_55}/converse"
        assert sigv4_scope(sent)[1] == "ap-south-1"


def test_the_region_inside_an_inference_profile_arn_sets_the_signing_region(gateway: Gateway) -> None:
    """A deployment names `model_id: arn:aws:bedrock:eu-central-1:123456789012:application-inference-profile/dec1abc` and no region.

    `_get_aws_region_from_model_arn` (`base_aws_llm.py:516`) reads `eu-central-1` from the ARN, so the SigV4
    scope must name it, and the URL must carry the whole ARN percent-encoded as one path segment.
    """
    arn: Final = "arn:aws:bedrock:eu-central-1:123456789012:application-inference-profile/dec1abc"
    with bedrock(
        gateway,
        _ok,
        model=f"bedrock/converse/{CLAUDE_HAIKU_55}",
        model_id=arn,
        **sigv4_params(aws_region_name=None),
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == f"/model/{urllib.parse.quote(arn, safe='')}/converse", sent.target
        assert sigv4_scope(sent)[1] == "eu-central-1"


def test_aws_bedrock_runtime_endpoint_is_used_when_no_api_base_is_set(gateway: Gateway) -> None:
    """A Converse deployment with `aws_bedrock_runtime_endpoint: <peer>` and no `api_base`.

    The request must reach that endpoint at `/model/<id>/converse`, instead of the public
    `bedrock-runtime.<region>.amazonaws.com` host.
    """
    with (
        wire_server(_ok) as endpoint,
        bedrock(
            gateway, _ok, model=f"bedrock/{CLAUDE_HAIKU_55}", api_base=None, aws_bedrock_runtime_endpoint=endpoint.url
        ) as rig,
    ):
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        posts: Final = [request for request in endpoint.drain() if request.method == "POST"]
        assert [path_of(request) for request in posts] == [f"/model/{CLAUDE_HAIKU_55}/converse"]


def test_aws_bedrock_runtime_endpoint_wins_over_api_base_on_converse(gateway: Gateway) -> None:
    """A Converse deployment with both `api_base: <peer A>` and `aws_bedrock_runtime_endpoint: <peer B>`.

    `get_runtime_endpoint` returns the runtime endpoint as `proxy_endpoint_url`, and the Converse handler
    sends to that URL (`converse_handler.py:360`), so peer B must receive the one POST and peer A none.
    Pinned current behaviour; a rewrite that prefers `api_base` would silently move traffic.
    """
    with (
        wire_server(_ok) as endpoint,
        bedrock(gateway, _ok, model=f"bedrock/{CLAUDE_HAIKU_55}", aws_bedrock_runtime_endpoint=endpoint.url) as rig,
    ):
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert len([request for request in endpoint.drain() if request.method == "POST"]) == 1
        assert rig.posts() == ()


def test_a_sigv4_request_signs_the_body_bytes_it_sends(gateway: Gateway) -> None:
    """With SigV4 the signature covers the payload hash, so any re-serialization after signing breaks auth.

    For a request with tools and unicode text the independent SigV4 check over the exact received body
    must still pass, and `x-amz-content-sha256` (when present) must equal the SHA-256 of those bytes.
    UNVERIFIED: whether botocore adds `x-amz-content-sha256` for service `bedrock`; only checked when present.
    """
    import hashlib

    with bedrock(
        gateway, _ok, model=f"bedrock/us.{CLAUDE_HAIKU_55}", **sigv4_params(aws_region_name="us-east-1")
    ) as rig:
        response: Final = rig.chat("café über 中文", max_tokens=20)
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert_sigv4_signed(sent, region="us-east-1")
        if "x-amz-content-sha256" in sent.headers:
            assert sent.headers["x-amz-content-sha256"] == hashlib.sha256(sent.body).hexdigest()


def test_an_invalid_region_name_is_refused_before_any_request(gateway: Gateway) -> None:
    """`aws_region_name: "US_EAST_1"` fails `_VALID_AWS_REGION_PATTERN` (lowercase letters, digits and hyphens).

    `_validate_aws_region_name` raises before signing (`base_aws_llm.py:786`), so no request may reach the
    peer and the caller gets an error naming `Invalid AWS region format`. UNVERIFIED: the HTTP status the
    `ValueError` maps to (400 expected, 500 possible), so only "not 200" is asserted.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{CLAUDE_HAIKU_55}", **sigv4_params(aws_region_name="US_EAST_1")) as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code != 200, response.text
        assert "Invalid AWS region format" in response.text
        assert rig.posts() == ()
