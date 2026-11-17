"""`/v1/messages` on Bedrock InvokeModel: model id encoding, cross-region profiles, regions, guardrails and cache spend.

Since #51330 (Nov 12) a Claude deployment on Bedrock InvokeModel is served by the Rust Messages crate only
(`litellm/rust_bridge/catalog.py:209-214`, `Target.BEDROCK_INVOKE`), and #52042 (Nov 16) closed the high
severity parity bugs on that path. The URL is built by `invoke_url`
(`litellm/rust/crates/llms/src/bedrock/messages/invoke_transformations/anthropic_claude_transformation.rs:149`),
a port of Python's `AmazonInvokeConfig.get_complete_url`; the model id by `get_bedrock_model_id` (`:99`) and
`encode_model_id` (`:87`); the signing region by `resolve_bedrock_region` (`litellm/rust/crates/auth-aws/src/
aws.rs:1067-1097`, `aws_region_name` first, then the ARN's region, then env, then `us-west-2`).

What a rewrite could silently change, and these tests pin: a plain profile id keeps its `:` as is in the path
while an ARN or a `model_id` override is percent-encoded byte by byte (`:` to `%3A`, `/` to `%2F`); every
cross-region prefix (`global.`, `eu.`, `jp.`, `au.`, `apac.`) reaches Bedrock verbatim and is priced from its own
cost map row; a region segment inside the model name stays in the path and never picks the signing region;
`aws_bedrock_runtime_endpoint` is used when no `api_base` is set; the deployment `guardrailConfig` becomes the
`X-Amzn-Bedrock-*` headers or a local 400; and cache read and cache creation tokens on a unary InvokeModel answer
are relayed and billed.

The rounds before this one (`test_dec1_msgprov_bedrock_invoke.py`, `providers/test_dec1_bedrock_messages_gaps_
wire.py`, `providers/test_dec1_bedrock_credentials_env_owned_proxy.py`) already cover the body, betas, bearer vs
SigV4 auth including the empty `api_key` env bearer case (AUTH-084), stream reframing, errors and the `ca.`
profile; nothing here repeats them.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request, wire_server
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig, marker
from integration.messages_endpoint.dec1_msgprov3_support import raw_path
from integration.messages_endpoint.dec1_msgprov_support import (
    ACCESS_KEY,
    BEDROCK_MODEL_ID,
    BEDROCK_TOKEN,
    PRICES,
    SECRET_KEY,
    invoke_echo,
    invoke_message_reply,
    message_id,
    only_spend_row,
    provider_deployment,
    sigv4_scope,
    spend_of,
)
from pydantic import JsonValue

_SONNET_NAME: Final = "claude-sonnet-4-5-20250929"
"""How Bedrock InvokeModel names Claude in its answer, so pricing falls back to the deployment's profile row."""
_ARN: Final = "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
_ENCODED_ARN: Final = "arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Ainference-profile%2Feu.anthropic.claude-sonnet-4-5-20250929-v1%3A0"
_PLAIN_ID: Final = "us.anthropic.claude-sonnet-5"
_GUARDRAIL_IDENTIFIER: Final = "gr-msgprov3"


def _bearer(
    gateway: Gateway,
    model_id: str = BEDROCK_MODEL_ID,
    respond: Callable[[Request], Reply] = invoke_echo,
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=f"bedrock/invoke/{model_id}",
        api_key=BEDROCK_TOKEN,
        aws_region_name="us-east-1",
        **params,
    )


def _signed(gateway: Gateway, model: str, **params: JsonValue) -> AbstractContextManager[Rig]:
    """A deployment with static keys and no bearer, so the call is SigV4 signed and the scope can be read back."""
    return provider_deployment(
        gateway,
        invoke_echo,
        model=model,
        api_key=None,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        **params,
    )


def _endpoint_answer(request: Request) -> Reply:
    return invoke_echo(request) if request.method == "POST" else Reply(body=b"{}")


# ---------------------------------------------------------------- model id encoding


def test_a_profile_id_keeps_its_colon_unescaped_in_the_invoke_path(gateway: Gateway) -> None:
    """For `bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0` the raw request path Bedrock receives is
    `/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke`, with a literal `:` and not `%3A`: a plain id is
    not run through `encode_model_id` (`anthropic_claude_transformation.rs:99-115`, case `regular_id` at `:609`).
    A port that encodes every id would change the path SigV4 signs."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert raw_path(rig.one_call()) == f"/model/{BEDROCK_MODEL_ID}/invoke"


def test_a_streamed_profile_id_keeps_its_colon_unescaped(gateway: Gateway) -> None:
    """The same deployment with `stream: true` posts to the raw path
    `/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke-with-response-stream` (`INVOKE_STREAM_PATH`,
    `anthropic_claude_transformation.rs:167-171`)."""
    with _bearer(gateway) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        assert raw_path(rig.one_call()) == f"/model/{BEDROCK_MODEL_ID}/invoke-with-response-stream"


def test_an_inference_profile_arn_is_percent_encoded_in_the_invoke_path(gateway: Gateway) -> None:
    """For `bedrock/invoke/<inference profile ARN>` the ARN is encoded byte by byte, so the raw path is
    `/model/arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Ainference-profile%2Feu.anthropic...v1%3A0/invoke`
    (`get_bedrock_model_id` ARN branch, `anthropic_claude_transformation.rs:104-106`, case `compound_prefixed_arn`
    at `:597`). Bedrock routes on the escaped form; a raw `/` would split the ARN into two path segments."""
    with _bearer(gateway, _ARN) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert raw_path(rig.one_call()) == f"/model/{_ENCODED_ARN}/invoke"


def test_an_arn_model_streams_to_the_encoded_arn_stream_path(gateway: Gateway) -> None:
    """Streaming on the ARN deployment posts to `/model/<encoded ARN>/invoke-with-response-stream`
    (`an_arn_url_is_percent_encoded_in_the_arn_region`, `anthropic_claude_transformation.rs:723`)."""
    with _bearer(gateway, _ARN) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        assert raw_path(rig.one_call()) == f"/model/{_ENCODED_ARN}/invoke-with-response-stream"


def test_a_model_id_arn_is_percent_encoded_in_place_of_the_model(gateway: Gateway) -> None:
    """A deployment `bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0` with `model_id: <ARN>` posts to
    `/model/<encoded ARN>/invoke`: the `model_id` parameter replaces the model in the URL and is always encoded
    (`get_bedrock_model_id`, `anthropic_claude_transformation.rs:100-102`, case `model_id_override` at `:610`)."""
    with _bearer(gateway, "anthropic.claude-sonnet-4-5-20250929-v1:0", model_id=_ARN) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert raw_path(rig.one_call()) == f"/model/{_ENCODED_ARN}/invoke"


def test_a_model_id_with_a_slash_is_percent_encoded(gateway: Gateway) -> None:
    """`model_id: "dec1/profile"` is sent as `/model/dec1%2Fprofile/invoke`: `encode_model_id` escapes every byte
    outside `A-Za-z0-9_.-~` (`anthropic_claude_transformation.rs:87-97`, `the_model_id_param_overrides_the_url_model`
    at `:741`)."""
    with _bearer(gateway, "anthropic.claude-sonnet-4-5-20250929-v1:0", model_id="dec1/profile") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert raw_path(rig.one_call()) == "/model/dec1%2Fprofile/invoke"


def test_the_body_never_carries_the_model_id_or_the_arn(gateway: Gateway) -> None:
    """On the ARN deployment the InvokeModel body has no `model` (and no `stream`) member; Bedrock takes the model
    only from the path (`wire_body`, `anthropic_claude_transformation.rs:309-318`), so neither the ARN nor the
    public model name may appear in the body."""
    with _bearer(gateway, _ARN) as rig:
        assert rig.post(rig.body()).status_code == 200
        body: Final = rig.one_body()
        assert "model" not in body and "stream" not in body, body
        assert "arn:aws" not in str(body), body


# ---------------------------------------------------------------- cross-region inference profiles


@pytest.mark.parametrize(
    "profile",
    (
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "au.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "apac.anthropic.claude-sonnet-4-20250514-v1:0",
    ),
)
def test_every_cross_region_prefix_is_kept_in_the_invoke_path(gateway: Gateway, profile: str) -> None:
    """A deployment `bedrock/invoke/<profile>` posts to `/model/<profile>/invoke` with the geography prefix intact:
    the prefix selects Bedrock's cross-region routing and only `invoke/` and a `nova/` spec are stripped from the id
    (`get_bedrock_model_id`, `anthropic_claude_transformation.rs:107-114`)."""
    with _bearer(gateway, profile) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert raw_path(rig.one_call()) == f"/model/{profile}/invoke"


@pytest.mark.parametrize(
    ("profile", "input_cost", "output_cost"),
    (
        ("global.anthropic.claude-sonnet-4-5-20250929-v1:0", 3e-06, 1.5e-05),
        ("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.3e-06, 1.65e-05),
        ("jp.anthropic.claude-sonnet-4-5-20250929-v1:0", 3.3e-06, 1.65e-05),
    ),
)
def test_a_cross_region_profile_is_priced_from_its_own_cost_map_row(
    gateway: Gateway, profile: str, input_cost: float, output_cost: float
) -> None:
    """With no deployment prices and a reply naming `claude-sonnet-4-5-20250929`, 1000 input and 100 output tokens
    are priced on the profile's own cost map row (`litellm/model_prices_and_context_window_backup.json`): the
    `global.` row has the base price (3e-06 / 1.5e-05, spend 0.0045) while `eu.` and `jp.` carry the 10 percent
    regional premium (3.3e-06 / 1.65e-05, spend 0.00495). Same fallback as the `us.` case pinned in
    `test_dec1_msgprov_bedrock_invoke.py:660` (`_served_or_priced_requested_model`, `litellm/cost_calculator.py`).
    UNVERIFIED: that the Rust cost path resolves the `global.` and `jp.` rows the same way as `us.`."""
    reply: Final = invoke_message_reply(
        message_id(), model=_SONNET_NAME, usage={"input_tokens": 1000, "output_tokens": 100}
    )
    with _bearer(gateway, profile, lambda _request: reply) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert spend_of(only_spend_row(rig)) == pytest.approx(1000 * input_cost + 100 * output_cost)


# ---------------------------------------------------------------- regions and endpoints


def test_an_arn_model_without_a_region_is_signed_in_the_arn_region(gateway: Gateway) -> None:
    """A SigV4 deployment `bedrock/invoke/<ARN in eu-west-1>` with static keys and no `aws_region_name` is signed
    with the scope region `eu-west-1` for service `bedrock`, taken from the ARN's fourth field
    (`get_aws_region_from_model_arn`, `anthropic_claude_transformation.rs:140-147`; Python
    `_get_aws_region_name`, `litellm/llms/bedrock/base_aws_llm.py:736-741`). UNVERIFIED: that no
    `AWS_REGION_NAME`/`AWS_REGION` in the shared proxy's environment outranks it (both rank below the ARN)."""
    with _signed(gateway, f"bedrock/invoke/{_ARN}") as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert sigv4_scope(call) == (ACCESS_KEY, "eu-west-1", "bedrock")
        assert raw_path(call) == f"/model/{_ENCODED_ARN}/invoke"


def test_a_configured_region_beats_the_arn_region_for_signing(gateway: Gateway) -> None:
    """The same ARN deployment with `aws_region_name: us-east-1` is signed for `us-east-1`: the deployment region
    is read before the ARN (`resolve_aws_region`, `litellm/rust/crates/auth-aws/src/aws.rs:1067-1076`; Python
    `base_aws_llm.py:733-741` only reads the ARN when `aws_region_name` is None)."""
    with _signed(gateway, f"bedrock/invoke/{_ARN}", aws_region_name="us-east-1") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert sigv4_scope(rig.one_call())[1] == "us-east-1"


def test_a_region_segment_in_the_model_stays_in_the_path_and_does_not_pick_the_signing_region(gateway: Gateway) -> None:
    """`bedrock/invoke/us-west-2/us.anthropic.claude-sonnet-5` with `aws_region_name: eu-central-1` posts to
    `/model/us-west-2/us.anthropic.claude-sonnet-5/invoke` and is signed for `eu-central-1`: a region-looking
    segment is part of the id, not a region selector (cases `params_region_hosts_a_region_segment` and
    `region_segment_does_not_pick_the_region`, `anthropic_claude_transformation.rs:648-660`)."""
    with _signed(gateway, f"bedrock/invoke/us-west-2/{_PLAIN_ID}", aws_region_name="eu-central-1") as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert raw_path(call) == f"/model/us-west-2/{_PLAIN_ID}/invoke"
        assert sigv4_scope(call)[1] == "eu-central-1"


def test_aws_bedrock_runtime_endpoint_is_used_when_no_api_base_is_set(gateway: Gateway) -> None:
    """A deployment with `api_base: null` and `aws_bedrock_runtime_endpoint: <peer>` posts InvokeModel to that
    endpoint at `/model/<id>/invoke`: the endpoint precedence is `api_base`, then `aws_bedrock_runtime_endpoint`,
    then `AWS_BEDROCK_RUNTIME_ENDPOINT`, then the regional host (`invoke_url`, `anthropic_claude_transformation.rs:
    178-186`, case `params_endpoint_outranks_the_environment` at `:429`)."""

    with (
        wire_server(_endpoint_answer) as endpoint,
        _bearer(gateway, api_base=None, aws_bedrock_runtime_endpoint=endpoint.url) as rig,
    ):
        assert rig.post(rig.body()).status_code == 200
        posts: Final = tuple(request for request in endpoint.drain() if request.method == "POST")
        assert [raw_path(request) for request in posts] == [f"/model/{BEDROCK_MODEL_ID}/invoke"]
        assert rig.calls() == ()


def test_a_blank_api_base_does_not_hide_the_runtime_endpoint(gateway: Gateway) -> None:
    """`api_base: "  "` counts as unset (`configured` trims and drops empty values, `anthropic_claude_transformation.rs:
    172-177`, case `a_blank_api_base_does_not_hide_the_params_endpoint` at `:422`), so the call goes to
    `aws_bedrock_runtime_endpoint`. UNVERIFIED: that the proxy stores a whitespace `api_base` unchanged instead of
    rejecting or trimming it at `/model/new`."""
    with (
        wire_server(_endpoint_answer) as endpoint,
        _bearer(gateway, api_base="  ", aws_bedrock_runtime_endpoint=endpoint.url) as rig,
    ):
        assert rig.post(rig.body()).status_code == 200
        assert len([request for request in endpoint.drain() if request.method == "POST"]) == 1
        assert rig.calls() == ()


# ---------------------------------------------------------------- guardrail headers


def test_guardrail_trace_disabled_is_uppercased_and_a_missing_version_sends_no_version_header(gateway: Gateway) -> None:
    """`guardrailConfig: {"guardrailIdentifier": "gr-msgprov3", "trace": "disabled"}` reaches InvokeModel as
    `X-Amzn-Bedrock-GuardrailIdentifier: gr-msgprov3` and `X-Amzn-Bedrock-Trace: DISABLED`, and no
    `X-Amzn-Bedrock-GuardrailVersion` header at all, because absent values are filtered out
    (`bedrock_invoke_guardrail_headers`, `litellm/llms/bedrock/chat/invoke_transformations/base_invoke_transformation.py:
    95-114`; Rust `base_invoke_transformation.rs:83-121`)."""
    guardrail: Final[dict[str, JsonValue]] = {"guardrailIdentifier": _GUARDRAIL_IDENTIFIER, "trace": "disabled"}
    with _bearer(gateway, guardrailConfig=guardrail) as rig:
        assert rig.post(rig.body()).status_code == 200
        headers: Final = rig.one_call().headers
        assert headers["x-amzn-bedrock-guardrailidentifier"] == _GUARDRAIL_IDENTIFIER
        assert headers["x-amzn-bedrock-trace"] == "DISABLED"
        assert "x-amzn-bedrock-guardrailversion" not in headers


def test_an_unknown_guardrail_trace_value_is_refused_with_400_before_bedrock(gateway: Gateway) -> None:
    """`guardrailConfig.trace: "verbose"` fails the config validation, so the caller gets HTTP 400 whose message
    starts with `Invalid guardrailConfig=` and names the allowed values `'enabled', 'disabled' or 'enabled_full'`,
    and Bedrock is never called (`base_invoke_transformation.py:96-101`; Rust `GUARDRAIL_TRACE_VALUES` check,
    `base_invoke_transformation.rs:60-75`)."""
    guardrail: Final[dict[str, JsonValue]] = {"guardrailIdentifier": _GUARDRAIL_IDENTIFIER, "trace": "verbose"}
    with _bearer(gateway, guardrailConfig=guardrail) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 400, response.text
        assert "Invalid guardrailConfig=" in response.text
        assert "'enabled', 'disabled' or 'enabled_full'" in response.text
        assert rig.calls() == ()


def test_a_non_object_guardrail_config_is_refused_with_400_before_bedrock(gateway: Gateway) -> None:
    """`guardrailConfig: "gr-msgprov3"` (a bare string, a common YAML mistake) is HTTP 400 with `Invalid
    guardrailConfig=` and `valid dictionary` in the message, and Bedrock is never called (Rust
    `bedrock_invoke_guardrail_headers` non-object branch, `base_invoke_transformation.rs:86-92`)."""
    with _bearer(gateway, guardrailConfig=_GUARDRAIL_IDENTIFIER) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 400, response.text
        assert "Invalid guardrailConfig=" in response.text
        assert "valid dictionary" in response.text
        assert rig.calls() == ()


# ---------------------------------------------------------------- usage and cache tokens


_CACHE_USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": 100,
    "output_tokens": 10,
    "cache_read_input_tokens": 1000,
    "cache_creation_input_tokens": 200,
}


def test_unary_invoke_cache_usage_is_relayed_unchanged(gateway: Gateway) -> None:
    """A unary InvokeModel answer with `usage: {input_tokens: 100, output_tokens: 10, cache_read_input_tokens: 1000,
    cache_creation_input_tokens: 200}` reaches the caller with those four numbers unchanged: InvokeModel speaks
    Anthropic's wire format, so nothing is translated (the Rust handler relays the provider body)."""
    reply: Final = invoke_message_reply(message_id(), usage=_CACHE_USAGE)
    with _bearer(gateway, respond=lambda _request: reply, **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        usage: Final = object_value(JSON_OBJECT.validate_json(response.content)["usage"])
        assert {name: usage.get(name) for name in _CACHE_USAGE} == _CACHE_USAGE


def test_unary_invoke_cache_tokens_are_billed_at_both_cache_rates(gateway: Gateway) -> None:
    """With deployment prices (input 0.001, output 0.002, cache read 0.0001, cache creation 0.003) the same answer
    costs `100 * 0.001 + 10 * 0.002 + 1000 * 0.0001 + 200 * 0.003 = 0.82`, on the SpendLogs row and in
    `x-litellm-response-cost`, and the row's `prompt_tokens` is 1300 (input plus both cache counts, as the Anthropic
    usage is folded into the OpenAI shaped Usage) with `completion_tokens` 10. The stream variant is pinned in `test_dec1_msgprov_bedrock_invoke.py`; a port that
    only reads cache tokens from `invocationMetrics` would bill the unary call at 0.12."""
    reply: Final = invoke_message_reply(message_id(), usage=_CACHE_USAGE)
    with _bearer(gateway, respond=lambda _request: reply, **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.82)
        row: Final = only_spend_row(rig)
        assert spend_of(row) == pytest.approx(0.82)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (1300, 10)


# ---------------------------------------------------------------- request metadata header


def test_an_allowed_key_alias_is_sent_in_the_bedrock_request_metadata_header(gateway: Gateway) -> None:
    """A deployment with `bedrock_request_metadata_fields: ["user_api_key_alias"]`, called by a key whose alias is
    `dec1-msgprov3-alias-<hex>`, sends `X-Amzn-Bedrock-Request-Metadata: {"user_api_key_alias":"<that alias>"}`
    (compact JSON, `bedrock_request_metadata_headers`, `litellm/llms/bedrock/request_metadata.py:227`; Rust
    `litellm/rust/crates/llms/src/bedrock/request_metadata.rs:185`), so AWS invocation logs can attribute the call.
    UNVERIFIED: that the proxy's `user_api_key_alias` reaches the Rust `litellm_metadata` extra on this route."""
    with _bearer(gateway, bedrock_request_metadata_fields=["user_api_key_alias"]) as rig:
        alias: Final = marker("dec1-msgprov3-alias")
        key: Final = rig.scenario.key(models=[rig.model], key_alias=alias)
        assert rig.post(rig.body(), key=key).status_code == 200
        header: Final = rig.one_call().headers["x-amzn-bedrock-request-metadata"]
        assert header == '{"user_api_key_alias":"' + alias + '"}'


def test_without_allowed_fields_no_request_metadata_header_is_sent(gateway: Gateway) -> None:
    """Without `bedrock_request_metadata_fields` the InvokeModel call carries no `X-Amzn-Bedrock-Request-Metadata`
    header, even for a key with an alias: an empty allowlist returns before any header is built
    (`request_metadata.rs:188-191`)."""
    with _bearer(gateway) as rig:
        key: Final = rig.scenario.key(models=[rig.model], key_alias=marker("dec1-msgprov3-alias"))
        assert rig.post(rig.body(), key=key).status_code == 200
        assert "x-amzn-bedrock-request-metadata" not in rig.one_call().headers
