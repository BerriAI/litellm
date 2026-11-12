"""`POST /v1/messages` served by a Bedrock Claude deployment through InvokeModel.

A deployment named `bedrock/invoke/<id>` or `bedrock/<claude id>` has a native Messages config
(`AmazonAnthropicClaudeMessagesConfig`, `litellm/llms/bedrock/messages/invoke_transformations/`),
so the caller's Anthropic body goes to `POST {api_base}/model/{id}/invoke` (or
`/invoke-with-response-stream` for `stream: true`) with these changes: `model` and `stream` leave
the body, `anthropic_version: bedrock-2023-05-31` is added, a string `system` becomes text blocks,
billing-header system blocks are dropped, the caller's `anthropic-beta` values move into an
`anthropic_beta` body list filtered through the `bedrock` row of `anthropic_beta_headers_config.json`,
and every top-level field outside `BedrockInvokeAnthropicMessagesRequest` is removed. Auth is a
Bedrock API key as `Authorization: Bearer` when the deployment has `api_key`, otherwise AWS SigV4
for service `bedrock` in the deployment's region. The streamed reply is an AWS event stream of
`chunk` frames, each carrying one base64 Anthropic event, which the proxy re-frames as Anthropic SSE.

Bedrock and Vertex are the next providers the Rust Messages route will serve
(`litellm-rust/crates/inference-messages/src/common_utils.rs` already lists both configs). Today
`litellm/rust_bridge/catalog.py` keeps every non-Anthropic provider on Python, so these tests pin the
Python behaviour the Rust port has to reproduce byte for byte at the provider and caller edges.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import (
    JSON_OBJECT,
    NO_CACHE,
    Rig,
    anthropic_error,
    sse_events,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    ACCESS_KEY,
    BEDROCK_BASE_MODEL_ID,
    BEDROCK_MODEL_ID,
    BEDROCK_PLAIN_MODEL_ID,
    BEDROCK_TOKEN,
    BEDROCK_VERSION,
    CACHE_CREATION_COST,
    CACHE_READ_COST,
    INPUT_COST,
    OUTPUT_COST,
    PNG_BASE64,
    PRICES,
    SECRET_KEY,
    SESSION_TOKEN,
    TOOL,
    assert_sigv4_signed,
    at,
    aws_error,
    body_of,
    invoke_chunk,
    invoke_echo,
    invoke_exception,
    invoke_message_reply,
    invoke_stream_reply,
    message_id,
    only_spend_row,
    path_of,
    provider_deployment,
    signed_headers,
    sigv4_scope,
    spend_of,
    unique_text,
)
from pydantic import JsonValue

_ANTHROPIC_SONNET_NAME: Final = "claude-sonnet-4-5-20250929"
_INVOKE: Final = f"/model/{BEDROCK_MODEL_ID}/invoke"
_INVOKE_STREAM: Final = f"/model/{BEDROCK_MODEL_ID}/invoke-with-response-stream"
_BILLING_BLOCK: Final[dict[str, JsonValue]] = {
    "type": "text",
    "text": "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli",
}


def _bearer(
    gateway: Gateway,
    respond: Callable[[Request], Reply] = invoke_echo,
    *,
    model: str = f"bedrock/invoke/{BEDROCK_MODEL_ID}",
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway, respond, model=model, api_key=BEDROCK_TOKEN, aws_region_name="us-east-1", **params
    )


def _sigv4(
    gateway: Gateway,
    respond: Callable[[Request], Reply] = invoke_echo,
    *,
    model: str = f"bedrock/invoke/{BEDROCK_PLAIN_MODEL_ID}",
    region: str = "us-west-2",
    **params: JsonValue,
) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=model,
        api_key=None,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_region_name=region,
        **params,
    )


# ---------------------------------------------------------------- URL, method and auth


def test_a_bedrock_invoke_deployment_posts_to_the_model_invoke_path_with_the_bearer_token(gateway: Gateway) -> None:
    """A `bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0` deployment with `api_key` set
    receives exactly one `POST /model/<id>/invoke`, with `Authorization: Bearer <api_key>` and
    `Content-Type: application/json`, and no SigV4 `x-amz-date` header. A Bedrock API key is a bearer
    token, not an `x-api-key`, so a port that copies the Anthropic auth scheme would be refused."""
    with _bearer(gateway) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert (call.method, path_of(call)) == ("POST", _INVOKE)
        assert call.headers["authorization"] == f"Bearer {BEDROCK_TOKEN}"
        assert call.headers["content-type"].startswith("application/json")
        assert "x-amz-date" not in call.headers and "x-api-key" not in call.headers


def test_a_streaming_request_posts_to_invoke_with_response_stream(gateway: Gateway) -> None:
    """With `stream: true` the provider call goes to `POST /model/<id>/invoke-with-response-stream`
    instead of `/invoke`, and the caller gets HTTP 200 with `content-type: text/event-stream`."""
    with _bearer(gateway) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert path_of(rig.one_call()) == _INVOKE_STREAM


def test_a_bare_bedrock_claude_model_reaches_invoke_on_the_messages_route(gateway: Gateway) -> None:
    """A deployment named `bedrock/<claude id>` with no `invoke/` or `converse/` route prefix is served
    on `/v1/messages` through InvokeModel (`BedrockModelInfo.get_bedrock_provider_config_for_messages_api`
    returns the invoke config whenever `claude` is in the id), so the peer sees `/model/<id>/invoke`,
    never `/converse`. Chat completions on the same deployment would use Converse, so the route
    choice is per endpoint and easy to lose in a rewrite."""
    with _bearer(gateway, model=f"bedrock/{BEDROCK_MODEL_ID}") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == _INVOKE


def test_the_model_id_parameter_replaces_the_id_in_the_invoke_path(gateway: Gateway) -> None:
    """A deployment can name an application inference profile ARN in `litellm_params.model_id`.
    The ARN must replace the model id in the path, URL-encoded so its `:` and `/` stay one path
    segment: the raw target contains `arn%3Aaws%3Abedrock` and the decoded path is
    `/model/<arn>/invoke` (`BaseAWSLLM.get_bedrock_model_id`)."""
    arn: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/msgprov0001"
    with _bearer(gateway, model_id=arn) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert path_of(call) == f"/model/{arn}/invoke"
        assert "arn%3Aaws%3Abedrock" in call.target, call.target


def test_a_sigv4_deployment_signs_the_invoke_request_for_service_bedrock(gateway: Gateway) -> None:
    """A deployment with `aws_access_key_id`, `aws_secret_access_key` and `aws_region_name: us-west-2`
    and no `api_key` sends an `Authorization: AWS4-HMAC-SHA256` header whose credential scope is
    `<date>/us-west-2/bedrock/aws4_request`, whose signed headers include `host` and `x-amz-date`,
    and whose signature matches an independent SigV4 computation over the exact bytes received."""
    with _sigv4(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert path_of(call) == f"/model/{BEDROCK_PLAIN_MODEL_ID}/invoke"
        assert {"host", "x-amz-date"}.issubset(signed_headers(call))
        assert_sigv4_signed(call, region="us-west-2")


@pytest.mark.parametrize("region", ["us-east-1", "eu-west-1", "ap-northeast-1"])
def test_aws_region_name_sets_the_sigv4_scope_region(gateway: Gateway, region: str) -> None:
    """The SigV4 credential scope names the deployment's `aws_region_name`, whatever the `us.` prefix
    of the inference profile says. Bedrock refuses a signature scoped to the wrong region, so the
    region source must survive the port."""
    with _sigv4(gateway, region=region) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert sigv4_scope(rig.one_call()) == (ACCESS_KEY, region, "bedrock")


def test_an_aws_session_token_is_sent_and_signed(gateway: Gateway) -> None:
    """With temporary credentials (`aws_session_token` set) the request carries
    `x-amz-security-token: <token>` and that header is part of `SignedHeaders`, as AWS requires."""
    with _sigv4(gateway, aws_session_token=SESSION_TOKEN) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert call.headers["x-amz-security-token"] == SESSION_TOKEN
        assert "x-amz-security-token" in signed_headers(call)


def test_a_blank_api_key_next_to_aws_keys_is_signed_not_sent_as_an_empty_bearer(gateway: Gateway) -> None:
    """`api_key: ""` next to AWS keys means no Bedrock API key (`bedrock_bearer_token` treats an empty
    token as absent), so the request is SigV4 signed and never carries `Authorization: Bearer `."""
    with provider_deployment(
        gateway,
        invoke_echo,
        model=f"bedrock/invoke/{BEDROCK_PLAIN_MODEL_ID}",
        api_key="",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_region_name="us-west-2",
    ) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert call.headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        assert_sigv4_signed(call, region="us-west-2")


def test_the_callers_proxy_key_never_reaches_bedrock(gateway: Gateway) -> None:
    """The caller authenticates to the proxy with the master key. No header the peer receives may
    contain that key, and there is no `x-api-key` header at all on a Bedrock call."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body()).status_code == 200
        call: Final = rig.one_call()
        assert all(gateway.key not in value for value in call.headers.values()), dict(call.headers)
        assert "x-api-key" not in call.headers


# ---------------------------------------------------------------- body translation


@pytest.mark.parametrize("stream", [False, True])
def test_the_invoke_body_carries_bedrock_anthropic_version_and_no_model_or_stream(
    gateway: Gateway, stream: bool
) -> None:
    """For a plain request the body sent to Bedrock is exactly
    `{"anthropic_version": "bedrock-2023-05-31", "max_tokens": 64, "messages": [...]}`: `model` and
    `stream` are removed because InvokeModel rejects them (the route carries both), and the Bedrock
    version string is added."""
    with _bearer(gateway) as rig:
        body: Final = rig.body(stream=stream)
        if stream:
            rig.stream(body)
        else:
            assert rig.post(body).status_code == 200
        assert rig.one_body() == {"anthropic_version": BEDROCK_VERSION, "max_tokens": 64, "messages": body["messages"]}


def test_a_string_system_prompt_becomes_one_text_block(gateway: Gateway) -> None:
    """`system: "<text>"` arrives as `[{"type": "text", "text": "<text>"}]`. Bedrock, Vertex and Foundry
    all go through `_normalize_system_role_messages`, which rebuilds `system` as blocks, unlike the
    first-party Anthropic route that forwards the string untouched."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(system="You are terse.")).status_code == 200
        assert rig.one_body()["system"] == [{"type": "text", "text": "You are terse."}]


def test_a_leading_system_role_message_is_hoisted_into_system(gateway: Gateway) -> None:
    """A caller that puts `{"role": "system", ...}` first in `messages` (OpenAI habit) gets it moved:
    Bedrock receives `system == [{"type": "text", "text": "<that text>"}]` and `messages` starting at
    the user turn, because Claude rejects a leading system-role message."""
    prompt: Final = unique_text()
    with _bearer(gateway) as rig:
        body: Final = rig.body(prompt)
        body["messages"] = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": prompt}]
        assert rig.post(body).status_code == 200
        sent: Final = rig.one_body()
        assert sent["system"] == [{"type": "text", "text": "Be brief."}]
        assert sent["messages"] == [{"role": "user", "content": prompt}]


def test_the_claude_code_billing_header_block_is_stripped_from_system(gateway: Gateway) -> None:
    """Claude Code sends a first system block starting `x-anthropic-billing-header:`. Bedrock bills by
    AWS account, so that block is removed (`should_strip_billing_metadata` is true for Bedrock) and
    only the real instruction block reaches the provider."""
    with _bearer(gateway) as rig:
        instruction: Final[dict[str, JsonValue]] = {"type": "text", "text": "You are Claude Code."}
        assert rig.post(rig.body(system=[_BILLING_BLOCK, instruction])).status_code == 200
        assert rig.one_body()["system"] == [instruction]


def test_tools_tool_choice_stop_sequences_metadata_and_sampling_are_forwarded(gateway: Gateway) -> None:
    """`tools`, a named `tool_choice`, `stop_sequences`, `metadata.user_id`, `temperature` and `top_k`
    are all in the InvokeModel allowlist and must arrive with the values the caller sent."""
    fields: Final[dict[str, JsonValue]] = {
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "stop_sequences": ["###", "END"],
        "metadata": {"user_id": "msgprov-user-1"},
        "temperature": 0.2,
        "top_k": 40,
    }
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(**fields)).status_code == 200
        sent: Final = rig.one_body()
        assert {name: sent.get(name) for name in fields} == fields


def test_fields_outside_the_invoke_allowlist_are_removed(gateway: Gateway) -> None:
    """InvokeModel answers 400 "Extra inputs are not permitted" for unknown top-level keys, so
    `_strip_unsupported_bedrock_invoke_fields` keeps only `BedrockInvokeAnthropicMessagesRequest`
    keys. A request carrying `service_tier` and `inference_geo` (valid on api.anthropic.com) must
    reach Bedrock without either key."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(service_tier="auto", inference_geo="us")).status_code == 200
        sent: Final = rig.one_body()
        assert "service_tier" not in sent and "inference_geo" not in sent, sent


def test_a_tool_custom_object_is_hoisted_and_dropped(gateway: Gateway) -> None:
    """A tool carrying a non-spec `custom: {"defer_loading": true}` object (a Claude Code shape) reaches
    Bedrock with `defer_loading: true` on the tool itself and no `custom` key
    (`normalize_custom_field_on_tools`). This is also a Python-only feature today (TOOL_CUSTOM_FIELD)."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(tools=[{**TOOL, "custom": {"defer_loading": True}}])).status_code == 200
        tool: Final = object_value(at(rig.one_body(), "tools", 0))
        assert tool.get("defer_loading") is True and "custom" not in tool, tool


def test_an_enabled_thinking_block_is_forwarded_unchanged(gateway: Gateway) -> None:
    """`thinking: {"type": "enabled", "budget_tokens": 2048}` on Claude Sonnet 4.5 (not an adaptive
    thinking model) must reach Bedrock as the same object next to `max_tokens: 4096`."""
    thinking: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 2048}
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(max_tokens=4096, thinking=thinking)).status_code == 200
        sent: Final = rig.one_body()
        assert (sent["thinking"], sent["max_tokens"]) == (thinking, 4096)


def test_an_image_block_is_forwarded_with_its_base64_source(gateway: Gateway) -> None:
    """A user turn with an `image` block whose source is `{"type": "base64", "media_type": "image/png",
    "data": ...}` reaches InvokeModel unchanged; InvokeModel speaks the Anthropic content format."""
    prompt: Final = unique_text()
    content: Final[list[JsonValue]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}},
        {"type": "text", "text": prompt},
    ]
    with _bearer(gateway) as rig:
        body: Final = rig.body()
        body["messages"] = [{"role": "user", "content": content}]
        assert rig.post(body).status_code == 200
        assert rig.one_body()["messages"] == [{"role": "user", "content": content}]


def test_cache_control_keeps_a_ttl_on_claude_4_5_and_drops_scope(gateway: Gateway) -> None:
    """Bedrock accepts only `type` and `ttl` in `cache_control`, and `ttl` only on Claude 4.5 and newer.
    A block marked `{"type": "ephemeral", "ttl": "1h", "scope": "global"}` must arrive as
    `{"type": "ephemeral", "ttl": "1h"}` (`_remove_ttl_from_cache_control`). UNVERIFIED: that
    `supports_bedrock_prompt_cache_ttl` is true for the Sonnet 4.5 profile; look there if this fails."""
    prompt: Final = unique_text()
    with _bearer(gateway) as rig:
        body: Final = rig.body()
        body["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"},
                    }
                ],
            }
        ]
        assert rig.post(body).status_code == 200
        block: Final = object_value(at(rig.one_body(), "messages", 0, "content", 0))
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_a_supported_caller_beta_moves_into_the_body_anthropic_beta_list(gateway: Gateway) -> None:
    """InvokeModel reads betas from the body. A caller header `anthropic-beta: context-1m-2025-08-07`
    (kept by the `bedrock` row of the beta config) must produce `anthropic_beta: ["context-1m-2025-08-07"]`
    in the body sent to Bedrock."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-beta": "context-1m-2025-08-07"}).status_code == 200
        assert rig.one_body()["anthropic_beta"] == ["context-1m-2025-08-07"]


def test_a_beta_bedrock_does_not_support_is_dropped_from_the_body(gateway: Gateway) -> None:
    """`files-api-2025-04-14` maps to null in the `bedrock` row, so a caller sending only that beta gets
    a body with no `anthropic_beta` key at all; forwarding it would earn a Bedrock 400."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-beta": "files-api-2025-04-14"}).status_code == 200
        assert "anthropic_beta" not in rig.one_body()


def test_a_beta_with_a_bedrock_name_is_renamed(gateway: Gateway) -> None:
    """The `bedrock` row renames `advanced-tool-use-2025-11-20` to `tool-search-tool-2025-10-19`, so the
    body's `anthropic_beta` is `["tool-search-tool-2025-10-19"]`. UNVERIFIED: no other beta is added
    for a request without tools; see `_get_bedrock_invoke_anthropic_beta_headers` if this differs."""
    with _bearer(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-beta": "advanced-tool-use-2025-11-20"}).status_code == 200
        assert rig.one_body()["anthropic_beta"] == ["tool-search-tool-2025-10-19"]


def test_the_server_side_web_search_tool_is_refused_before_bedrock(gateway: Gateway) -> None:
    """Bedrock cannot run Anthropic's `web_search_20250305` tool. Without web search interception the
    proxy answers HTTP 400 in the Anthropic envelope (`invalid_request_error`) with a message starting
    "Bedrock does not support Anthropic's server-side web search tool", and never calls Bedrock."""
    tool: Final[dict[str, JsonValue]] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
    with _bearer(gateway) as rig:
        response: Final = rig.post(rig.body(tools=[tool]))
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "Bedrock does not support Anthropic's server-side web search tool" in str(error["message"])
        assert rig.calls() == ()


# ---------------------------------------------------------------- response translation


def test_the_invoke_message_is_returned_with_the_public_model_name(gateway: Gateway) -> None:
    """Bedrock answers an Anthropic message whose `model` is the Bedrock base id. The caller gets the
    same `id`, `content`, `stop_reason` and `usage`, with `model` set to the proxy model name it asked
    for, never the Bedrock id."""
    identity: Final = message_id()
    with _bearer(gateway, lambda _r: invoke_message_reply(identity)) as rig:
        response: Final = rig.post(rig.body())
        payload: Final = JSON_OBJECT.validate_json(response.content)
        assert (payload["id"], payload["model"], payload["stop_reason"]) == (identity, rig.model, "end_turn")
        assert payload["content"] == [{"type": "text", "text": "pong"}]
        assert payload["usage"] == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.parametrize(
    ("stop_reason", "stop_sequence"),
    [("stop_sequence", "###"), ("max_tokens", None), ("tool_use", None), ("refusal", None)],
)
def test_the_bedrock_stop_reason_and_stop_sequence_pass_through(
    gateway: Gateway, stop_reason: str, stop_sequence: str | None
) -> None:
    """InvokeModel already speaks Anthropic, so `stop_reason` and `stop_sequence` reach the caller
    verbatim, including `stop_sequence` naming the matched string. Converse loses this (see the
    Converse file), so a port that routes InvokeModel through the chat bridge would break it."""
    identity: Final = message_id()
    reply: Final = invoke_message_reply(identity, stop_reason=stop_reason, extra={"stop_sequence": stop_sequence})
    with _bearer(gateway, lambda _r: reply) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body()).content)
        assert (payload["stop_reason"], payload["stop_sequence"]) == (stop_reason, stop_sequence)


def test_a_tool_use_block_is_returned_unchanged(gateway: Gateway) -> None:
    """A `tool_use` content block (`id`, `name`, `input`) from Bedrock reaches the caller unchanged next
    to `stop_reason: "tool_use"`."""
    block: Final[dict[str, JsonValue]] = {
        "type": "tool_use",
        "id": "toolu_msgprov01",
        "name": "get_weather",
        "input": {"city": "Lisbon"},
    }
    reply: Final = invoke_message_reply(message_id(), content=[block], stop_reason="tool_use")
    with _bearer(gateway, lambda _r: reply) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body(tools=[TOOL])).content)
        assert payload["content"] == [block]


# ---------------------------------------------------------------- streaming


def test_the_aws_event_stream_is_reframed_as_anthropic_sse_in_order(gateway: Gateway) -> None:
    """Bedrock streams six `chunk` frames, each a base64 Anthropic event. The caller must get exactly
    the six named SSE events in order (`message_start`, `content_block_start`, `content_block_delta`,
    `content_block_stop`, `message_delta`, `message_stop`), the delta text `hello stream`, and
    `message_start.message.model` equal to the proxy model name."""
    identity: Final = message_id()
    with _bearer(gateway, lambda _r: invoke_stream_reply(identity)) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert [name for name, _data in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ], text
        start: Final = object_value(events[0][1]["message"])
        assert (start["id"], start["model"]) == (identity, rig.model)
        assert object_value(events[2][1]["delta"])["text"] == "hello stream"
        assert "[DONE]" not in text


def test_invocation_metrics_never_leak_and_fill_the_final_usage(gateway: Gateway) -> None:
    """Real Bedrock appends `amazon-bedrock-invocationMetrics` to the last chunk. The decoder pops it,
    so no SSE frame the caller sees contains that key, and its counts surface as Anthropic usage:
    `message_delta.usage` reports `output_tokens: 7` and `input_tokens: 11`
    (`_promote_message_stop_usage`). UNVERIFIED: whether `input_tokens` is promoted onto
    `message_delta` when the delta already had usage; see `bedrock_sse_wrapper`."""
    metrics: Final[dict[str, JsonValue]] = {"inputTokenCount": 11, "outputTokenCount": 7, "invocationLatency": 5}
    with _bearer(gateway, lambda _r: invoke_stream_reply(message_id(), metrics=metrics)) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        assert "amazon-bedrock-invocationMetrics" not in text
        delta: Final = next(data for name, data in sse_events(text) if name == "message_delta")
        usage: Final = object_value(delta["usage"])
        assert (usage["output_tokens"], usage.get("input_tokens")) == (7, 11), delta


def test_stream_cache_tokens_from_invocation_metrics_are_billed(gateway: Gateway) -> None:
    """When the cache breakdown only arrives in `amazon-bedrock-invocationMetrics`
    (`cacheReadInputTokenCount: 5`, `cacheWriteInputTokenCount: 9`, `inputTokenCount: 10`,
    `outputTokenCount: 4`), the SpendLogs row must still price it:
    10 * 0.001 + 4 * 0.002 + 5 * 0.0001 + 9 * 0.003 = 0.0455. Dropping the metrics loses cache
    tokens and bills them at zero (the decoder docstring calls this out)."""
    metrics: Final[dict[str, JsonValue]] = {
        "inputTokenCount": 10,
        "outputTokenCount": 4,
        "cacheReadInputTokenCount": 5,
        "cacheWriteInputTokenCount": 9,
    }
    reply: Final = invoke_stream_reply(message_id(), input_tokens=10, output_tokens=4, metrics=metrics)
    with _bearer(gateway, lambda _r: reply, **PRICES) as rig:
        rig.stream(rig.body(stream=True))
        assert spend_of(only_spend_row(rig)) == pytest.approx(
            10 * INPUT_COST + 4 * OUTPUT_COST + 5 * CACHE_READ_COST + 9 * CACHE_CREATION_COST
        )


def test_a_mid_stream_throttling_exception_ends_the_stream_with_an_error_event(gateway: Gateway) -> None:
    """Bedrock can send an event-stream `exception` frame (`throttlingException`) after content
    started. The HTTP status is already 200, so the caller must see the earlier events and then an
    Anthropic `error` event, not a silently truncated stream. UNVERIFIED: the error event's
    `error.type`; `rate_limit_error` is expected from the 429 mapping."""
    identity: Final = message_id()

    def respond(_request: Request) -> Reply:
        frames: Final = invoke_stream_reply(identity).chunks or ()
        return Reply(
            content_type="application/vnd.amazon.eventstream",
            chunks=(*frames[:3], invoke_exception("throttlingException", "Too many tokens, please wait.")),
        )

    with _bearer(gateway, respond) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        names: Final = [name for name, _data in sse_events(text)]
        assert names[:3] == ["message_start", "content_block_start", "content_block_delta"], text
        assert names[-1] == "error", text
        error: Final = object_value(sse_events(text)[-1][1]["error"])
        assert error["type"] == "rate_limit_error", text


def test_an_error_on_stream_open_is_an_http_error_not_a_200_stream(gateway: Gateway) -> None:
    """When `invoke-with-response-stream` itself answers 429 ThrottlingException, nothing was streamed
    yet, so the caller must get HTTP 429 with a JSON Anthropic envelope of type `rate_limit_error`."""
    reply: Final = aws_error(429, "ThrottlingException", "Too many requests, please wait before trying again.")
    with _bearer(gateway, lambda _r: reply) as rig:
        response, text = rig.stream(rig.body(stream=True))
        assert response.status_code == 429, text
        assert object_value(JSON_OBJECT.validate_json(text)["error"])["type"] == "rate_limit_error"


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "error_class", "message", "answered", "anthropic_type"),
    [
        (400, "ValidationException", "messages.0.content: Field required", 400, "invalid_request_error"),
        (
            403,
            "AccessDeniedException",
            "You don't have access to the model with the specified model ID.",
            403,
            "permission_error",
        ),
        (404, "ResourceNotFoundException", "Could not resolve the foundation model.", 404, "not_found_error"),
        (429, "ThrottlingException", "Too many requests, please wait before trying again.", 429, "rate_limit_error"),
        (429, "ModelNotReadyException", "Model is not ready to serve inference requests.", 429, "rate_limit_error"),
        (503, "ServiceUnavailableException", "Bedrock is unable to process your request.", 503, "api_error"),
    ],
)
def test_a_bedrock_error_is_answered_with_its_status_and_anthropic_type(
    gateway: Gateway, status: int, error_class: str, message: str, answered: int, anthropic_type: str
) -> None:
    """Bedrock's `{"message": ...}` errors are mapped by `_map_bedrock_exception` and answered in the
    Anthropic envelope: validation 400, access denied 403, not found 404, throttling and model not
    ready 429, service unavailable 503. The provider text survives in `error.message`, and the call is
    made once (no retry budget on the shared proxy)."""
    with _bearer(gateway, lambda _r: aws_error(status, error_class, message)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == answered, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == anthropic_type
        assert message in str(error["message"])
        assert len(rig.calls()) == 1


def test_a_bedrock_500_is_answered_as_503(gateway: Gateway) -> None:
    """Pinned current behaviour: `_map_bedrock_exception` raises `ServiceUnavailableError` for a Bedrock
    500 InternalServerException, so the caller gets HTTP 503 with `error.type == "api_error"`, unlike
    the first-party Anthropic route, which answers a provider 500 with 500."""
    with _bearer(gateway, lambda _r: aws_error(500, "InternalServerException", "Internal failure.")) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 503, response.text
        assert anthropic_error(response)["type"] == "api_error"


def test_an_invalid_security_token_is_answered_with_bedrocks_403_as_a_permission_error(gateway: Gateway) -> None:
    """Pinned current behaviour: Bedrock answers bad SigV4 credentials with 403
    UnrecognizedClientException "The security token included in the request is invalid.". The mapper
    matches that text and raises `AuthenticationError` (`litellm/litellm_core_utils/exception_mapping_utils.py:1057`),
    whose own `status_code = 401` is overwritten by the OpenAI base class with the status of the upstream
    response it is given (`litellm/exceptions.py:194` then `super().__init__(..., response=...)`), unlike
    `ServiceUnavailableError`, which rebuilds the response with its own status. So the caller gets HTTP 403 and
    `permission_error`, while `error.message` still names `AuthenticationError` and keeps Bedrock's text.
    Run 43314424795 saw exactly that; a rewrite that answers 401 here must change this cell on purpose."""
    reply: Final = aws_error(
        403, "UnrecognizedClientException", "The security token included in the request is invalid."
    )
    with _sigv4(gateway, lambda _r: reply) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 403, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "permission_error"
        assert "AuthenticationError" in str(error["message"])
        assert "The security token included in the request is invalid." in str(error["message"])


def test_a_too_long_input_is_a_400_invalid_request_error(gateway: Gateway) -> None:
    """Bedrock rejects an oversized prompt with 400 "Input is too long for requested model.". The
    mapper turns it into a context window error, which the caller sees as HTTP 400
    `invalid_request_error` with the Bedrock text in the message."""
    reply: Final = aws_error(400, "ValidationException", "Input is too long for requested model.")
    with _bearer(gateway, lambda _r: reply) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "Input is too long" in str(error["message"])


def test_the_error_message_names_bedrock(gateway: Gateway) -> None:
    """Error messages from this route are prefixed with the provider, e.g.
    `BedrockException - ...`, so a caller can tell a Bedrock refusal from a proxy refusal.
    UNVERIFIED: only the substring `Bedrock` is asserted; the exact prefix may carry `litellm.`."""
    with _bearer(gateway, lambda _r: aws_error(400, "ValidationException", "bad field")) as rig:
        assert "Bedrock" in str(anthropic_error(rig.post(rig.body()))["message"])


# ---------------------------------------------------------------- spend


def test_the_spend_row_names_bedrock_and_prices_the_tokens(gateway: Gateway) -> None:
    """One successful call writes one SpendLogs row with `call_type = "anthropic_messages"`,
    `custom_llm_provider = "bedrock"`, `prompt_tokens = 11`, `completion_tokens = 7`, the Bedrock id in
    `model`, and `spend = 11 * 0.001 + 7 * 0.002 = 0.025` from the deployment prices."""
    with _bearer(gateway, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"], row["status"]) == (
            "anthropic_messages",
            "bedrock",
            "success",
        )
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert BEDROCK_MODEL_ID in str(row["model"])
        assert spend_of(row) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_the_cost_map_price_of_the_us_inference_profile_applies_without_deployment_prices(gateway: Gateway) -> None:
    """With no prices on the deployment, the cost map row of `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
    (3.3e-06 input, 1.65e-05 output, the regional profile premium over the base id) prices
    1000 input and 100 output tokens at 0.00495.
    The reply names the model the way Bedrock InvokeModel does for Claude, by its Anthropic name
    `claude-sonnet-4-5-20250929`. The cost is priced from the answered model first
    (`_served_or_priced_requested_model`, `litellm/cost_calculator.py:1137`) and falls back to the deployment's
    profile row only when the answered name has no Bedrock row of its own; a reply naming the base Bedrock id
    `anthropic.claude-sonnet-4-5-20250929-v1:0`, as run 43314424795 sent, is priced on the base row (0.0045) on
    `/v1/messages` and `/v1/chat/completions` alike, so that run's miss was the reply, not the route."""
    reply: Final = invoke_message_reply(
        message_id(), model=_ANTHROPIC_SONNET_NAME, usage={"input_tokens": 1000, "output_tokens": 100}
    )
    with _bearer(gateway, lambda _r: reply) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert spend_of(only_spend_row(rig)) == pytest.approx(1000 * 3.3e-06 + 100 * 1.65e-05)


def test_the_response_cost_header_matches_the_spend_row(gateway: Gateway) -> None:
    """The `x-litellm-response-cost` header on a Bedrock answer must equal the row's `spend`, 0.025."""
    with _bearer(gateway, **PRICES) as rig:
        response: Final = rig.post(rig.body())
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.025)
        assert spend_of(only_spend_row(rig)) == pytest.approx(0.025)


def test_a_failed_bedrock_call_writes_a_failure_row(gateway: Gateway) -> None:
    """A Bedrock 429 still writes exactly one row for the model group with `status = "failure"` and
    `custom_llm_provider = "bedrock"`, so throttled traffic is visible in spend reports."""
    with _bearer(gateway, lambda _r: aws_error(429, "ThrottlingException", "slow down")) as rig:
        assert rig.post(rig.body()).status_code == 429
        row: Final = only_spend_row(rig)
        assert (row["status"], row["custom_llm_provider"]) == ("failure", "bedrock")


# ---------------------------------------------------------------- count tokens


def _count_reply(request: Request) -> Reply:
    if path_of(request).endswith("/count-tokens"):
        return Reply(body=json.dumps({"inputTokens": 42}).encode())
    return invoke_echo(request)


def test_count_tokens_with_string_content_uses_the_converse_count_input(gateway: Gateway) -> None:
    """`POST /v1/messages/count_tokens` on this deployment calls Bedrock CountTokens at
    `/model/anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens` (the base model, the cost map's
    `token_count_model`) with `{"input": {"converse": {"messages": [{"role": "user", "content":
    [{"text": <prompt>}]}]}}}` and answers `input_tokens: 42`. UNVERIFIED: the count source fields
    next to `input_tokens`."""
    prompt: Final = unique_text()
    with _bearer(gateway, _count_reply) as rig:
        response: Final = rig.post(
            {"model": rig.model, "messages": [{"role": "user", "content": prompt}]}, path="/v1/messages/count_tokens"
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["input_tokens"] == 42
        call: Final = rig.one_call()
        assert path_of(call) == f"/model/{BEDROCK_BASE_MODEL_ID}/count-tokens"
        assert body_of(call) == {"input": {"converse": {"messages": [{"role": "user", "content": [{"text": prompt}]}]}}}


def test_count_tokens_with_content_blocks_uses_the_invoke_model_body(gateway: Gateway) -> None:
    """Anthropic content blocks (`{"type": "text", ...}`) cannot go through the Converse count shape,
    so the count request carries `input.invokeModel.body`, a base64 JSON body that has
    `anthropic_version: bedrock-2023-05-31`, the caller's `messages`, a `max_tokens` default and no
    `model` key."""
    content: Final[list[JsonValue]] = [{"type": "text", "text": unique_text()}]
    with _bearer(gateway, _count_reply) as rig:
        response: Final = rig.post(
            {"model": rig.model, "messages": [{"role": "user", "content": content}]}, path="/v1/messages/count_tokens"
        )
        assert response.status_code == 200, response.text
        encoded: Final = object_value(object_value(body_of(rig.one_call())["input"])["invokeModel"])["body"]
        decoded: Final = JSON_OBJECT.validate_json(base64.b64decode(str(encoded)))
        assert decoded["anthropic_version"] == BEDROCK_VERSION and "model" not in decoded
        assert decoded["messages"] == [{"role": "user", "content": content}]
        assert isinstance(decoded["max_tokens"], int)


# ---------------------------------------------------------------- Python path parity


def test_the_invoke_body_is_the_same_with_and_without_the_response_cache(gateway: Gateway) -> None:
    """The response cache forces the Python path (RESPONSE_CACHE) and `cache: {"no-cache": true}` lifts
    that. Two requests that differ only in that control must send Bedrock the same body apart from
    the prompt text, so whichever implementation serves a call, Bedrock sees one contract."""
    with _bearer(gateway) as rig:
        shape: Final[dict[str, JsonValue]] = {"system": "same", "stop_sequences": ["###"], "temperature": 0.5}
        assert rig.post(rig.body("first " + unique_text(), **shape)).status_code == 200
        assert rig.post(rig.body("second " + unique_text(), **shape, **NO_CACHE)).status_code == 200
        bodies: Final = [body_of(call) for call in rig.calls()]
        assert len(bodies) == 2
        assert [{k: v for k, v in body.items() if k != "messages"} for body in bodies] == [
            {k: v for k, v in bodies[0].items() if k != "messages"}
        ] * 2


def test_the_invoke_stream_frames_are_not_split_or_merged(gateway: Gateway) -> None:
    """A chunk frame carrying a text delta of 500 characters must reach the caller as one
    `content_block_delta` with all 500 characters; the AWS frame boundary is the event boundary."""
    identity: Final = message_id()
    long_text: Final = "x" * 500

    def respond(_request: Request) -> Reply:
        return invoke_stream_reply(identity, text=long_text)

    with _bearer(gateway, respond) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        deltas: Final = [data for name, data in sse_events(text) if name == "content_block_delta"]
        assert [object_value(delta["delta"])["text"] for delta in deltas] == [long_text]


def test_an_invoke_chunk_frame_helper_round_trips(gateway: Gateway) -> None:
    """Guard for this file's own fixture: a `chunk` frame built by `invoke_chunk` for a `ping` event is
    accepted by the proxy's decoder, which relays a single valid stream when the frame is first.
    The caller must get HTTP 200 and a `message_stop` at the end."""
    identity: Final = message_id()

    def respond(_request: Request) -> Reply:
        frames: Final = invoke_stream_reply(identity).chunks or ()
        return Reply(
            content_type="application/vnd.amazon.eventstream", chunks=(invoke_chunk({"type": "ping"}), *frames)
        )

    with _bearer(gateway, respond) as rig:
        response, text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        assert sse_events(text)[-1][0] == "message_stop", text
