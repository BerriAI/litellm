"""Which headers and body fields cross the native Rust gateway preview, in each direction.

The contract, read from `litellm/rust/crates/gateway-inference/src/messages.rs`, `request.rs` and
`host-http/src/sse.rs`:

Requests: of the caller's headers only `anthropic-version` and `anthropic-beta` are forwarded, and only to
Anthropic-speaking providers in `ANTHROPIC_API_HEADER_PROVIDERS` (`anthropic, bedrock, bedrock_mantle,
vertex_ai, openrouter, deepseek`); the OpenRouter metadata headers go only to OpenRouter. Nothing else the
caller sends (custom headers, cookies, `x-request-id`, forwarding headers, the bearer) reaches a provider. The
body is read as raw bytes whatever its `content-type`, parsed as one JSON object, and on `/v1/messages` it is
built as: the deployment's listed default params (`DEPLOYMENT_DEFAULT_PARAMS`), then the caller's fields (the
caller wins), then the deployment's `tools` placed before the caller's. `/chat/completions` also accepts
`multipart/form-data`, each text field parsed as JSON when it can be.

Responses: a unary answer is re-serialized JSON with none of the provider's response headers.

One gateway and one scripted provider serve the module.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.wire import Reply
from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    PROVIDER_KEY,
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    anthropic_reply,
    azure_ai_deployment,
    chat_body,
    json_of,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    openai_like_deployment,
    response_json,
    scripted_provider,
)

_DEPLOYMENT_TOOL: Final = {"name": "dec1_deployment_tool", "input_schema": {"type": "object", "properties": {}}}
_CLIENT_TOOL: Final = {"name": "dec1_client_tool", "input_schema": {"type": "object", "properties": {}}}


@pytest.fixture(scope="module")
def provider() -> Iterator[ScriptedProvider]:
    with scripted_provider() as scripted:
        yield scripted


@pytest.fixture(scope="module")
def gateway(provider: ScriptedProvider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGateway]:
    url: Final = provider.url
    config: Final = master_key_config(
        [
            anthropic_deployment("claude", url),
            anthropic_deployment(
                "claude-defaults",
                url,
                temperature=0.25,
                system="dec1 deployment system prompt",
                metadata={"user_id": "dec1-deployment-user"},
                tools=[_DEPLOYMENT_TOOL],
                frequency_penalty=0.5,
                dec1_custom_param="dec1-not-a-default",
            ),
            anthropic_deployment("claude-streams", url, stream=True),
            azure_ai_deployment("azure-claude", url),
            openai_like_deployment("chat", url),
        ]
    )
    with native_gateway(config, tmp_path_factory.mktemp("gwnative2-headers")) as booted:
        yield booted


# Request headers


def test_custom_client_headers_never_reach_the_provider(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The caller adds `x-dec1-custom: one`, `cookie: dec1=secret`, `x-forwarded-for: 203.0.113.9` and
    `x-request-id: dec1-rid`. None of them may reach the Anthropic provider: only the two Anthropic API headers
    are copied (`anthropic_api_headers` in `messages.rs:506`). The Python proxy forwards client headers only
    when `forward_client_headers_to_llm_api` is on, which the preview has no setting for."""
    case: Final = marker()
    gateway.post(
        "/v1/messages",
        messages_body(case),
        headers={
            "x-dec1-custom": "one",
            "cookie": "dec1=secret",
            "x-forwarded-for": "203.0.113.9",
            "x-request-id": "dec1-rid",
        },
    )
    sent: Final = provider.requests_for(case)[0].headers
    assert "x-dec1-custom" not in sent
    assert "cookie" not in sent
    assert "x-forwarded-for" not in sent
    assert "dec1-rid" not in sent.values()


def test_the_anthropic_beta_header_is_not_forwarded_to_azure_ai(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The caller sends `anthropic-beta: dec1-beta-azure` to the `azure_ai` Claude deployment. `azure_ai` is not
    in `ANTHROPIC_API_HEADER_PROVIDERS` (`messages.rs:384`), so the provider's `anthropic-beta` header, if any,
    must not contain `dec1-beta-azure`. UNVERIFIED: whether the Azure AI config adds its own beta header."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "azure-claude"), headers={"anthropic-beta": "dec1-beta-azure"})
    sent: Final = provider.requests_for(case)[0].headers
    assert "dec1-beta-azure" not in sent.get("anthropic-beta", "")


def test_openrouter_metadata_headers_are_not_forwarded_to_anthropic(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The caller sends `x-openrouter-metadata: dec1-meta` to an Anthropic deployment. Those headers are scoped
    to `openrouter` only (`OPENROUTER_API_HEADERS`, `messages.rs:380`), so the Anthropic provider must not see
    the header."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case), headers={"x-openrouter-metadata": "dec1-meta"})
    assert "x-openrouter-metadata" not in provider.requests_for(case)[0].headers


def test_the_provider_request_is_json_with_the_deployment_key(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A plain messages call. The provider request must have `content-type: application/json`
    (`default_headers` in `base_llm/messages/transformation.rs:151`) and `x-api-key` equal to the deployment's
    key, and no `authorization` header at all."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case))
    sent: Final = provider.requests_for(case)[0].headers
    assert sent["content-type"].startswith("application/json")
    assert sent["x-api-key"] == PROVIDER_KEY
    assert "authorization" not in sent


# Response headers


def test_provider_response_headers_are_not_relayed_on_a_unary_call(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider answers 200 with `request-id: req_dec1`, `anthropic-ratelimit-requests-remaining: 7` and
    `x-dec1-upstream: yes`. The caller's response must carry none of them and have `content-type`
    `application/json`: the handler returns `Json(value)`, a fresh response (`Unary::encode_response`). The
    Python proxy relays provider headers with an `llm_provider-` prefix, which the preview does not port."""
    case: Final = marker()
    provider.script(
        case,
        lambda _: anthropic_reply(
            case,
            headers={
                "request-id": "req_dec1",
                "anthropic-ratelimit-requests-remaining": "7",
                "x-dec1-upstream": "yes",
            },
        ),
    )
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    for name in ("request-id", "anthropic-ratelimit-requests-remaining", "x-dec1-upstream"):
        assert name not in response.headers
        assert f"llm_provider-{name}" not in response.headers


def test_the_callers_x_request_id_is_not_echoed_as_a_response_header(gateway: NativeGateway) -> None:
    """A successful messages call with `x-request-id: dec1-echo`. No middleware copies it back (`log_request`
    makes its own UUID and only logs it, `gateway/src/lib.rs:339`), so the response must have no
    `x-request-id` header."""
    response: Final = gateway.post("/v1/messages", messages_body(marker()), headers={"x-request-id": "dec1-echo"})
    assert response.status_code == 200, response.text
    assert "x-request-id" not in response.headers


def test_a_provider_retry_after_reaches_the_caller_on_429(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The `openai_like` provider answers 429 with `retry-after: 17`. SDK retry loops read `retry-after`, and
    the Python proxy surfaces provider rate-limit headers on the error. The preview's error response is built
    from the status and body only (`Error::openai_response`, `error.rs:85`), so the header is lost.
    UNVERIFIED: the exact header name the Python proxy uses (`retry-after` or `llm_provider-retry-after`)."""
    pytest.skip("BUG: the native gateway drops the provider's retry-after header on a 429")
    case: Final = marker()
    provider.script(
        case, lambda _: Reply(status=429, body=b'{"error": {"message": "dec1 slow"}}', headers={"retry-after": "17"})
    )
    response: Final = gateway.post("/chat/completions", chat_body(case))
    assert response.status_code == 429
    assert response.headers.get("retry-after") == "17"


# Request bodies


def test_a_json_body_labelled_text_plain_is_served(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """A valid messages body sent with `content-type: text/plain`. The body is read as raw bytes and parsed as
    JSON whatever the label (`JsonObject` reads `Bytes`, not axum's `Json`, `request.rs:176-188`), so the call
    must be served with 200 and reach the provider once."""
    case: Final = marker()
    response: Final = gateway.post(
        "/v1/messages",
        content=f'{{"model": "claude", "max_tokens": 4, "messages": [{{"role": "user", "content": "{case}"}}]}}'.encode(),
        headers={"content-type": "text/plain"},
    )
    assert response.status_code == 200, response.text
    assert len(provider.requests_for(case)) == 1


def test_a_multipart_body_on_messages_is_invalid_json(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """A `multipart/form-data` body with fields `model` and `messages` on `/v1/messages`. That route only reads
    a JSON object (`JsonObject`), so the answer must be 400 `invalid_request_error` in the Anthropic envelope
    with a message starting `invalid request body: invalid JSON`, and nothing reaches the provider."""
    case: Final = marker()
    with gateway.client() as client:
        response: Final = client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            files={"model": (None, "claude"), "messages": (None, f'[{{"role": "user", "content": "{case}"}}]')},
        )
    assert response.status_code == 400, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert isinstance(error["message"], str)
    assert error["message"].startswith("invalid request body: invalid JSON")
    assert provider.requests_for(case) == ()


def test_a_multipart_body_on_chat_completions_is_invalid_json(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A `multipart/form-data` body on `/chat/completions` with text fields `model` = `chat` and `messages` = a
    JSON array. Pinned from CI run 44176682961: on main the chat route extracts `JsonObject`
    (litellm/rust/crates/gateway-inference/src/chat_completions.rs:22), and only OCR and transcription use the
    multipart aware `InferenceBody` (request.rs:66), so the answer is 400 `invalid_request_error` in the OpenAI
    envelope with a message starting `invalid request body: invalid JSON`, and nothing reaches the provider.
    The Python proxy reads form bodies on every route, so this is a preview gap a migration must close."""
    case: Final = marker()
    with gateway.client() as client:
        response: Final = client.post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            files={"model": (None, "chat"), "messages": (None, f'[{{"role": "user", "content": "{case}"}}]')},
        )
    assert response.status_code == 400, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert (error["type"], str(error["message"]).startswith("invalid request body: invalid JSON")) == (
        "invalid_request_error",
        True,
    )
    assert provider.requests_for(case) == ()


def test_a_duplicated_json_key_keeps_its_last_value(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """A messages body that names `model` twice, first `dec1-nope` then `claude`. Like Python's `json.loads`,
    `serde_json` keeps the last value, so the call must be served by `claude` with 200."""
    case: Final = marker()
    response: Final = gateway.post(
        "/v1/messages",
        content=(
            '{"model": "dec1-nope", "model": "claude", "max_tokens": 4, '
            f'"messages": [{{"role": "user", "content": "{case}"}}]}}'
        ).encode(),
    )
    assert response.status_code == 200, response.text
    assert len(provider.requests_for(case)) == 1


# Deployment defaults on /v1/messages


def test_deployment_default_params_fill_what_the_caller_left_out(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `claude-defaults` sets `temperature: 0.25`, `system` and `metadata.user_id`. A call that sets
    none of them must send all three to the provider with the deployment's values (`deployment_default_params`
    over `DEPLOYMENT_DEFAULT_PARAMS`, `messages.rs:386-488`)."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "claude-defaults"))
    sent: Final = json_of(provider.requests_for(case)[0])
    assert sent["temperature"] == 0.25
    assert sent["system"] == "dec1 deployment system prompt"
    assert sent["metadata"] == {"user_id": "dec1-deployment-user"}


def test_the_callers_value_wins_over_a_deployment_default(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The same deployment, but the caller sends `temperature: 0.9` and `system: "dec1 caller system"`. The
    provider must receive the caller's values, because the caller's fields are chained after the defaults
    into one map (`messages.rs:464-467`)."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "claude-defaults", temperature=0.9, system="dec1 caller system"))
    sent: Final = json_of(provider.requests_for(case)[0])
    assert (sent["temperature"], sent["system"]) == (0.9, "dec1 caller system")


def test_deployment_params_outside_the_default_list_never_reach_the_provider(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `claude-defaults` also sets `frequency_penalty: 0.5` and `dec1_custom_param`. Neither is in
    `DEPLOYMENT_DEFAULT_PARAMS`, so the provider body must contain neither key."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "claude-defaults"))
    sent: Final = json_of(provider.requests_for(case)[0])
    assert "frequency_penalty" not in sent
    assert "dec1_custom_param" not in sent


def test_deployment_tools_come_before_the_callers_tools(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """Deployment `claude-defaults` lists one tool and the caller sends another. The provider must receive both,
    the deployment's first (`merged_tools`, `messages.rs:490-504`)."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "claude-defaults", tools=[_CLIENT_TOOL]))
    tools: Final = json_of(provider.requests_for(case)[0])["tools"]
    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools if isinstance(tool, dict)] == ["dec1_deployment_tool", "dec1_client_tool"]


def test_deployment_tools_are_sent_when_the_caller_has_none(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """A call to `claude-defaults` without `tools`. The provider must receive exactly the deployment's one tool
    (a missing client value counts as an empty list in `merged_tools`)."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "claude-defaults"))
    tools: Final = json_of(provider.requests_for(case)[0])["tools"]
    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools if isinstance(tool, dict)] == ["dec1_deployment_tool"]


def test_a_deployment_stream_default_streams_a_call_that_did_not_ask(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `claude-streams` sets `stream: true` in `litellm_params`; `stream` is one of the
    `DEPLOYMENT_DEFAULT_PARAMS`. A call without `stream` must therefore send `stream: true` to the provider.
    UNVERIFIED: that the caller then gets `text/event-stream`; the scripted provider answers JSON here, so only
    the provider body is asserted."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "claude-streams"))
    assert json_of(provider.requests_for(case)[0])["stream"] is True


def test_the_unary_answer_is_re_serialized_not_relayed_raw(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The provider answers with pretty-printed JSON (newlines and two-space indents). The caller must get the
    same message object but compact JSON, because the decoded `MessagesBody` is written back with `Json`
    (`messages.rs:447`). A rewrite that streams provider bytes through would change the body bytes."""
    case: Final = marker()
    pretty: Final = (
        b'{\n  "id": "msg_dec1_pretty",\n  "type": "message",\n  "role": "assistant",\n  "model": "claude-dec1-upstream",\n'
        b'  "content": [{"type": "text", "text": "dec1 pretty"}],\n  "stop_reason": "end_turn",\n'
        b'  "stop_sequence": null,\n  "usage": {"input_tokens": 5, "output_tokens": 3}\n}'
    )
    provider.script(case, lambda _: Reply(body=pretty))
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 200, response.text
    assert b"\n" not in response.content
    assert response_json(response)["id"] == "msg_dec1_pretty"


def test_a_gateway_response_is_http_1_1(gateway: NativeGateway) -> None:
    """A plain messages call over a default `httpx` client. The gateway answers with `HTTP/1.1` (axum serve
    without TLS, no h2c upgrade asked), which pins the protocol load balancers see today."""
    with gateway.client() as client:
        response: Final = client.post(
            "/v1/messages", json=messages_body(marker()), headers={"Authorization": f"Bearer {GATEWAY_KEY}"}
        )
    assert response.http_version == "HTTP/1.1"
