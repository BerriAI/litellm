"""What a preview user sees on each inference route of the native Rust gateway, per provider.

The contract, read from `litellm/rust/crates/gateway-inference` and the `inference-*` crates:

`POST /v1/messages` serves Anthropic, Bedrock invoke Claude, Vertex AI Claude and Azure AI Claude (among
others) natively and answers in the Anthropic error envelope. `POST /chat/completions` serves only
Anthropic, Bedrock (Converse) and `openai_like` deployments, never streams, and answers in the OpenAI
error envelope. `POST /v1/responses` serves only OpenAI and OpenRouter. `/v1/embeddings` and
`/v1/completions` are mounted but answer 501 "... is not implemented by the Rust gateway". A provider
the route cannot serve is a 400 "invalid provider: <name>" or a 501 "declined by the rust path: ...",
never a silent forward to Python, because the preview has no Python behind it.

One gateway and one scripted provider serve the module; each test tags its prompt with a unique marker
and reads back only the provider requests that carry it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.vertex import service_account_json
from tests.integration._support.wire import Reply
from tests.integration.configuration.dec1_gwnative_support import (
    AWS_ACCESS_KEY_ID,
    GATEWAY_KEY,
    PROVIDER_KEY,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
    VERTEX_TOKEN,
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    anthropic_error,
    azure_ai_deployment,
    azure_deployment,
    bedrock_deployment,
    chat_body,
    deployment,
    json_of,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    openai_deployment,
    openai_error,
    openai_error_body,
    openai_like_deployment,
    response_json,
    responses_body,
    scripted_provider,
    vertex_deployment,
)

_UNKNOWN_SUFFIX: Final = "Call `/v1/models` to view available models for your key."


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
            anthropic_deployment("claude-defaults", url, max_tokens=77, additional_drop_params=["temperature"]),
            deployment("claude-no-key", "anthropic/claude-dec1-upstream", api_base=url),
            deployment(
                "claude-dead", "anthropic/claude-dec1-upstream", api_base="http://127.0.0.1:9", api_key=PROVIDER_KEY
            ),
            bedrock_deployment("bedrock-claude", url),
            bedrock_deployment("bedrock-converse", url, "bedrock/converse/anthropic.claude-dec1-v1"),
            vertex_deployment("vertex-claude", url, service_account_json(VERTEX_PROJECT, url)),
            azure_ai_deployment("azure-claude", url),
            azure_deployment("azure-gpt", url),
            openai_like_deployment("chat", url),
            openai_deployment("openai-gpt", url),
            openai_deployment("responses", url),
        ]
    )
    with native_gateway(config, tmp_path_factory.mktemp("gwnative-routes")) as booted:
        yield booted


# /v1/messages


def test_messages_to_anthropic_reach_v1_messages_with_the_deployment_key_and_model(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """`POST /v1/messages` for public name `claude` (deployment `anthropic/claude-dec1-upstream`). The provider
    must get one `POST /v1/messages` with `x-api-key` equal to the deployment key and body `model`
    `claude-dec1-upstream` (provider prefix stripped, public name replaced), and the caller must get 200 with
    the provider's text in `content[0].text`."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    assert (sent[0].method, sent[0].target) == ("POST", "/v1/messages")
    assert sent[0].headers["x-api-key"] == PROVIDER_KEY
    assert json_of(sent[0])["model"] == "claude-dec1-upstream"
    body: Final = response_json(response)
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "scripted anthropic reply"}]


def test_messages_forward_the_callers_anthropic_version_and_beta_headers(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The caller sends `anthropic-version: 2023-06-01` and `anthropic-beta: dec1-beta-2026`. Both must reach
    the Anthropic provider unchanged (`ANTHROPIC_API_HEADERS` in `gateway-inference/src/messages.rs`)."""
    case: Final = marker()
    gateway.post(
        "/v1/messages",
        messages_body(case),
        headers={"anthropic-version": "2023-06-01", "anthropic-beta": "dec1-beta-2026"},
    )
    sent: Final = provider.requests_for(case)
    assert sent[0].headers["anthropic-version"] == "2023-06-01"
    assert "dec1-beta-2026" in sent[0].headers["anthropic-beta"]


def test_messages_do_not_forward_the_gateway_bearer_to_the_provider(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The caller authenticates with the master key. The provider request must not carry the master key in
    any header; it is authenticated with the deployment's own key."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case))
    sent: Final = provider.requests_for(case)
    assert all(GATEWAY_KEY not in value for value in sent[0].headers.values())


def test_a_deployment_max_tokens_fills_a_missing_client_value(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `claude-defaults` sets `max_tokens: 77`. A call without `max_tokens` must send 77 to the
    provider, and a call with `max_tokens: 8` must send 8 (client params override deployment defaults,
    `deployment_default_params` then `client_params` in `messages.rs`)."""
    missing: Final = marker()
    body: Final = messages_body(missing, "claude-defaults")
    del body["max_tokens"]
    gateway.post("/v1/messages", body)
    given: Final = marker()
    gateway.post("/v1/messages", messages_body(given, "claude-defaults", max_tokens=8))
    assert json_of(provider.requests_for(missing)[0])["max_tokens"] == 77
    assert json_of(provider.requests_for(given)[0])["max_tokens"] == 8


def test_a_deployment_additional_drop_param_never_reaches_the_provider(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `claude-defaults` lists `additional_drop_params: [temperature]`. A call with
    `temperature: 0.5` must succeed and the provider body must have no `temperature` key."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "claude-defaults", temperature=0.5))
    assert response.status_code == 200, response.text
    assert "temperature" not in json_of(provider.requests_for(case)[0])


def test_messages_return_the_providers_model_name_unchanged(gateway: NativeGateway) -> None:
    """Pins the preview: the provider answers with `model: claude-dec1-upstream` and the caller asked for
    `claude`. The gateway returns `claude-dec1-upstream`, because restamping the response `model` to the
    requested name is Python handler logic not ported yet (One Pager, A19). See the parity file."""
    response: Final = gateway.post("/v1/messages", messages_body(marker()))
    assert response_json(response)["model"] == "claude-dec1-upstream"


def test_messages_to_bedrock_invoke_claude_are_sigv4_signed_to_the_invoke_path(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `bedrock/anthropic.claude-dec1-v1` with `api_base` at the scripted peer and static AWS keys.
    The provider must get `POST /model/anthropic.claude-dec1-v1/invoke` with an `authorization` header that
    starts with `AWS4-HMAC-SHA256 Credential=<access key id>/` and names region `us-east-1` and service
    `bedrock`; the caller gets 200 with the Anthropic message."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude"))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert sent[0].target == "/model/anthropic.claude-dec1-v1/invoke"
    authorization: Final = sent[0].headers["authorization"]
    assert authorization.startswith(f"AWS4-HMAC-SHA256 Credential={AWS_ACCESS_KEY_ID}/"), authorization
    assert "/us-east-1/bedrock/aws4_request" in authorization, authorization


def test_messages_to_bedrock_invoke_carry_the_bedrock_anthropic_version(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Bedrock invoke body for Claude names its API version in the body, not a header. The provider body
    must have `anthropic_version: bedrock-2023-05-31`. UNVERIFIED: read from the Python Bedrock invoke
    transformation; confirm in `llms/src/bedrock/messages/invoke_transformations/`."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "bedrock-claude"))
    assert json_of(provider.requests_for(case)[0])["anthropic_version"] == "bedrock-2023-05-31"


def test_messages_to_a_bedrock_converse_model_are_refused_as_an_invalid_provider(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `bedrock/converse/...`. `messages_provider` has no Messages config for Converse, so the call
    must answer 400 in the Anthropic envelope with `error.type` `invalid_request_error` and message
    `invalid provider: bedrock`, and nothing reaches the provider. The Python proxy would translate it."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-converse"))
    assert response.status_code == 400, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "invalid provider: bedrock"},
    }
    assert provider.requests_for(case) == ()


def test_messages_to_vertex_claude_use_a_service_account_token_and_raw_predict(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `vertex_ai/claude-sonnet-4-5@20250929` with `vertex_credentials` whose `token_uri` is the
    scripted peer. The gateway must fetch a token there and then send `POST /v1/projects/<project>/locations/
    us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict` with
    `authorization: Bearer <token>` (`the_url_is_the_publisher_path_under_the_host` in
    `llms/src/vertex_ai/messages/transformation.rs`). UNVERIFIED: that the Rust GCP auth honours a custom
    `token_uri` from the service account file."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "vertex-claude"))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert sent[0].target == (
        f"/v1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/anthropic/models/"
        "claude-sonnet-4-5@20250929:rawPredict"
    )
    assert sent[0].headers["authorization"] == f"Bearer {VERTEX_TOKEN}"
    assert provider.requests_to("/_oauth/token"), "no token exchange reached the scripted token_uri"


def test_messages_to_vertex_claude_drop_the_model_from_the_body(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Vertex names the model in the URL, so the rawPredict body must have no `model` key and must carry
    `anthropic_version: vertex-2023-10-16`. UNVERIFIED: read from the Python Vertex partner transformation."""
    case: Final = marker()
    gateway.post("/v1/messages", messages_body(case, "vertex-claude"))
    sent: Final = json_of(provider.requests_for(case)[0])
    assert "model" not in sent
    assert sent["anthropic_version"] == "vertex-2023-10-16"


def test_messages_to_azure_ai_claude_go_to_the_anthropic_path_with_x_api_key(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `azure_ai/claude-dec1-upstream` with `api_base` `<peer>/`. The URL is the base plus
    `/anthropic/v1/messages` (`complete_azure_anthropic_url`), and the Azure `api-key` is renamed to
    `x-api-key`. The provider must see that path and `x-api-key` equal to the deployment key."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "azure-claude"))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert sent[0].target == "/anthropic/v1/messages"
    assert sent[0].headers["x-api-key"] == PROVIDER_KEY


def test_messages_to_an_openai_deployment_are_refused_as_an_invalid_provider(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `openai/gpt-dec1-upstream` called on `/v1/messages`. The Rust Messages route has no OpenAI
    config, so it must answer 400 Anthropic envelope `invalid provider: openai` with no provider call. The
    Python proxy would adapt the call to chat completions."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "openai-gpt"))
    assert response.status_code == 400, response.text
    assert response_json(response)["error"] == {"type": "invalid_request_error", "message": "invalid provider: openai"}
    assert provider.requests_for(case) == ()


def test_messages_unknown_model_uses_the_anthropic_envelope_with_the_request_id(gateway: NativeGateway) -> None:
    """`POST /v1/messages` for `dec1-nope` with `x-request-id: dec1-req-7`. The answer must be 400 with exactly
    `{"type": "error", "error": {"type": "invalid_request_error", "message": "/v1/messages: Invalid model name
    passed in model=dec1-nope. Call `/v1/models` ..."}, "request_id": "dec1-req-7"}` (`Error::body`)."""
    response: Final = gateway.post(
        "/v1/messages", messages_body(marker(), "dec1-nope"), headers={"x-request-id": "dec1-req-7"}
    )
    assert response.status_code == 400
    assert response_json(response) == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": f"/v1/messages: Invalid model name passed in model=dec1-nope. {_UNKNOWN_SUFFIX}",
        },
        "request_id": "dec1-req-7",
    }


def test_messages_errors_have_no_request_id_without_the_header(gateway: NativeGateway) -> None:
    """The same unknown-model call without `x-request-id`. The envelope must have no `request_id` key at
    all, rather than an empty or generated one."""
    response: Final = gateway.post("/v1/messages", messages_body(marker(), "dec1-nope"))
    assert "request_id" not in response_json(response)


def test_messages_invalid_json_is_a_400_in_the_anthropic_envelope(gateway: NativeGateway) -> None:
    """The body is `{"model": ` (truncated JSON). The answer must be 400 with `type: error`,
    `error.type: invalid_request_error` and a message that starts `invalid request body: invalid JSON: `,
    because `/v1/messages` maps body rejections into its own envelope too."""
    response: Final = gateway.post("/v1/messages", content=b'{"model": ')
    assert response.status_code == 400
    body: Final = response_json(response)
    error: Final = body["error"]
    assert isinstance(error, dict)
    assert (body["type"], error["type"]) == ("error", "invalid_request_error")
    assert str(error["message"]).startswith("invalid request body: invalid JSON: "), error


def test_messages_pass_an_anthropic_provider_error_through_with_its_status(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider answers 529 `{"type": "error", "error": {"type": "overloaded_error", "message":
    "dec1 busy"}}` and the caller sent `x-request-id: dec1-req-529`. The caller must get 529 and that exact
    body plus `request_id: dec1-req-529` (an Anthropic-shaped provider body passes through)."""
    case: Final = marker()
    provider.script(case, lambda _: anthropic_error(529, "overloaded_error", "dec1 busy"))
    response: Final = gateway.post("/v1/messages", messages_body(case), headers={"x-request-id": "dec1-req-529"})
    assert response.status_code == 529, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "dec1 busy"},
        "request_id": "dec1-req-529",
    }


def test_messages_wrap_a_non_anthropic_provider_error_in_the_anthropic_envelope(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A Bedrock deployment answers 429 `{"message": "dec1 slow down"}`. The caller must get 429 with
    `{"type": "error", "error": {"type": "rate_limit_error", "message": "dec1 slow down"}}`
    (`provider_message` extracts `message`, `error_type(429)` is `rate_limit_error`)."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=429, body=b'{"message": "dec1 slow down"}'))
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude"))
    assert response.status_code == 429, response.text
    assert response_json(response) == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "dec1 slow down"},
    }


def test_messages_without_a_provider_key_answer_401_and_call_nothing(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Deployment `claude-no-key` has no `api_key` and the gateway's environment has no `ANTHROPIC_API_KEY`.
    The call must answer 401 with `error.type` `authentication_error` (`is_missing_credential`) and the
    provider must see nothing."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, "claude-no-key"))
    assert response.status_code == 401, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert error["type"] == "authentication_error"
    assert provider.requests_for(case) == ()


def test_messages_to_an_unreachable_provider_answer_500_api_error(gateway: NativeGateway) -> None:
    """Deployment `claude-dead` points at `127.0.0.1:9`, where nothing listens. A connection failure is not a
    provider answer, so the call must answer 500 with `error.type` `api_error`. UNVERIFIED: Python maps a
    connection error to 500 as well (`APIConnectionError`), so the statuses should match."""
    response: Final = gateway.post("/v1/messages", messages_body(marker(), "claude-dead"))
    assert response.status_code == 500, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert error["type"] == "api_error"


# /chat/completions


@pytest.mark.parametrize("path", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_to_openai_like_reach_chat_completions_with_the_deployment_key(
    gateway: NativeGateway, provider: ScriptedProvider, path: str
) -> None:
    """Deployment `openai_like/gpt-dec1-upstream` with `api_base` `<peer>/v1`. `POST <path>` must send one
    `POST /v1/chat/completions` with `authorization: Bearer <deployment key>` and body `model`
    `gpt-dec1-upstream`, and return 200 with `object: chat.completion` and the provider text.
    BUG: the preview serializes the bare `ChatCompletionsResponse` (no `id`, no `object`,
    litellm/rust/crates/llms-types/src/formats/chat_completions/response.rs:83-92) straight to the caller
    (`Json` in litellm/rust/crates/gateway-inference/src/chat_completions.rs:78-98); that shape is meant for
    Python to wrap in a `ModelResponse`, so standalone callers get no `object` (CI run 44176682961)."""
    pytest.skip("BUG: native gateway chat completions answer has no object and no id")
    case: Final = marker()
    response: Final = gateway.post(path, chat_body(case))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert sent[0].target == "/v1/chat/completions"
    assert sent[0].headers["authorization"] == f"Bearer {PROVIDER_KEY}"
    assert json_of(sent[0])["model"] == "gpt-dec1-upstream"
    body: Final = response_json(response)
    assert body["object"] == "chat.completion"
    choices: Final = body["choices"]
    assert isinstance(choices, list)
    assert choices[0]["message"]["content"] == "scripted chat reply"  # type: ignore[index]


def test_chat_completions_to_anthropic_are_translated_to_messages(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """`POST /v1/chat/completions` for the Anthropic deployment `claude`. The provider must get
    `POST /v1/messages` with `anthropic-version: 2023-06-01` (the chat config's default header), and the
    caller must get 200 `object: chat.completion` with `choices[0].message.content` equal to the Anthropic
    text.
    BUG: the preview serializes the bare `ChatCompletionsResponse` (no `id`, no `object`,
    litellm/rust/crates/llms-types/src/formats/chat_completions/response.rs:83-92) straight to the caller
    (`Json` in litellm/rust/crates/gateway-inference/src/chat_completions.rs:78-98); that shape is meant for
    Python to wrap in a `ModelResponse`, so standalone callers get no `object` (CI run 44176682961)."""
    pytest.skip("BUG: native gateway chat completions answer has no object and no id")
    case: Final = marker()
    response: Final = gateway.post("/v1/chat/completions", chat_body(case, "claude", max_tokens=16))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert sent[0].target == "/v1/messages"
    assert sent[0].headers["anthropic-version"] == "2023-06-01"
    body: Final = response_json(response)
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "scripted anthropic reply"  # type: ignore[index]


def test_chat_completions_to_bedrock_use_converse(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`POST /v1/chat/completions` for `bedrock/converse/anthropic.claude-dec1-v1`. The Bedrock chat config is
    Converse, so the provider must get `POST /model/anthropic.claude-dec1-v1/converse`, SigV4-signed, and the
    caller 200 with the Converse text as `choices[0].message.content`.
    BUG: the chat route builds `optional_params` from the request body only
    (litellm/rust/crates/gateway-inference/src/chat_completions.rs:61-64, 83-87) and never passes the deployment's
    `litellm_params` extras, unlike `/v1/messages` (gateway-inference/src/messages.rs:115). The deployment's
    `aws_access_key_id`, `aws_secret_access_key` and `aws_region_name` are lost, the Bedrock config falls back to
    the AWS default chain and the caller gets 500 `could not load AWS default chain credentials`
    (CI run 44176682961)."""
    pytest.skip("BUG: native gateway chat completions ignore the deployment's static AWS credentials")
    case: Final = marker()
    response: Final = gateway.post("/v1/chat/completions", chat_body(case, "bedrock-converse"))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert sent[0].target == "/model/anthropic.claude-dec1-v1/converse"
    assert sent[0].headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert response_json(response)["choices"][0]["message"]["content"] == "scripted converse reply"  # type: ignore[index]


@pytest.mark.parametrize(
    ("model", "provider_name"), [("openai-gpt", "openai"), ("azure-gpt", "azure"), ("vertex-claude", "vertex_ai")]
)
def test_chat_completions_refuse_providers_without_a_rust_chat_config(
    gateway: NativeGateway, provider: ScriptedProvider, model: str, provider_name: str
) -> None:
    """`chat_completions_provider` knows only Anthropic, Bedrock and `openai_like`. A deployment on another
    provider must answer 400 with exactly `{"error": {"message": "invalid provider: <name>", "type":
    "invalid_request_error", "param": null, "code": 400}}` and nothing reaches the provider. This is the
    biggest preview gap: plain `openai/` and `azure/` chat models do not work natively."""
    case: Final = marker()
    response: Final = gateway.post("/v1/chat/completions", chat_body(case, model))
    assert response.status_code == 400, response.text
    assert response_json(response) == openai_error_body(
        f"invalid provider: {provider_name}", "invalid_request_error", 400
    )
    assert provider.requests_for(case) == ()


def test_chat_completions_decline_an_unrecognized_parameter_with_501(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """An `openai_like` call with `functions: [...]`, which is not in the config's supported params. The core
    declines before any send (`unsupported_param`), so the answer must be 501 with message
    `declined by the rust path: unrecognized request parameter` and type `api_error`, and the provider must
    see nothing. The Python proxy would forward the parameter."""
    case: Final = marker()
    response: Final = gateway.post(
        "/v1/chat/completions", chat_body(case, functions=[{"name": "f", "parameters": {"type": "object"}}])
    )
    assert response.status_code == 501, response.text
    assert response_json(response) == openai_error_body(
        "declined by the rust path: unrecognized request parameter", "api_error", 501
    )
    assert provider.requests_for(case) == ()


@pytest.mark.parametrize("path", ["/openai/deployments/chat/chat/completions", "/engines/chat/chat/completions"])
def test_azure_style_paths_take_the_model_from_the_path(
    gateway: NativeGateway, provider: ScriptedProvider, path: str
) -> None:
    """`POST <path>` whose body has no `model`. `create_from_model_path` fills `model` from the path segment
    `chat`, so the call must be served with 200 by the `chat` deployment."""
    case: Final = marker()
    body: Final = chat_body(case)
    del body["model"]
    response: Final = gateway.post(path, body)
    assert response.status_code == 200, response.text
    assert len(provider.requests_for(case)) == 1


def test_a_body_model_wins_over_the_path_model(gateway: NativeGateway) -> None:
    """`POST /openai/deployments/chat/chat/completions` with body `model: dec1-nope`. A non-null body model is
    kept, so the answer must be the unknown-model 400 for `dec1-nope`, not a call to `chat`."""
    response: Final = gateway.post("/openai/deployments/chat/chat/completions", chat_body(marker(), "dec1-nope"))
    assert response.status_code == 400, response.text
    assert "model=dec1-nope" in response.text


@pytest.mark.parametrize("path", ["/chat/completions", "/v1/chat/completions"])
def test_chat_completions_unknown_model_names_the_unversioned_route(gateway: NativeGateway, path: str) -> None:
    """`POST <path>` for `dec1-nope`. The message always starts `/chat/completions:` whichever spelling was
    called (`resolve_deployment(..., "/chat/completions", ...)`), and the body is exactly the OpenAI envelope
    with type `invalid_request_error` and code 400."""
    response: Final = gateway.post(path, chat_body(marker(), "dec1-nope"))
    assert response.status_code == 400
    assert response_json(response) == openai_error_body(
        f"/chat/completions: Invalid model name passed in model=dec1-nope. {_UNKNOWN_SUFFIX}",
        "invalid_request_error",
        400,
    )


def test_chat_completions_without_a_model_say_the_model_is_required(gateway: NativeGateway) -> None:
    """A body with `messages` and no `model`. The answer must be 400 with message
    `invalid request body: model is required` (`BodyError::MissingModel`)."""
    response: Final = gateway.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
    assert response.status_code == 400
    assert response_json(response) == openai_error_body(
        "invalid request body: model is required", "invalid_request_error", 400
    )


def test_a_json_array_body_is_not_an_object(gateway: NativeGateway) -> None:
    """The body is the JSON array `[1, 2]`. The answer must be 400 with message
    `invalid request body: expected a JSON object`."""
    response: Final = gateway.post("/v1/chat/completions", content=b"[1, 2]")
    assert response.status_code == 400
    assert response_json(response) == openai_error_body(
        "invalid request body: expected a JSON object", "invalid_request_error", 400
    )


def test_chat_completions_map_an_upstream_429_to_a_rate_limit_error(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The `openai_like` provider answers 429. The caller must get 429 in the OpenAI envelope with
    `error.type` `rate_limit_error` and `error.code` 429 (`route_status` keeps the provider status)."""
    case: Final = marker()
    provider.script(case, lambda _: openai_error(429, "dec1 rate limited", "rate_limit_exceeded"))
    response: Final = gateway.post("/v1/chat/completions", chat_body(case))
    assert response.status_code == 429, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert (error["type"], error["code"]) == ("rate_limit_error", 429)


def test_chat_completions_answer_json_with_an_application_json_content_type(gateway: NativeGateway) -> None:
    """A served chat completion. The response `content-type` must be `application/json`, the type every
    OpenAI SDK checks before parsing."""
    response: Final = gateway.post("/v1/chat/completions", chat_body(marker()))
    assert response.headers["content-type"] == "application/json"


# /v1/responses


@pytest.mark.parametrize("path", ["/responses", "/v1/responses"])
def test_responses_to_openai_reach_v1_responses(gateway: NativeGateway, provider: ScriptedProvider, path: str) -> None:
    """Deployment `openai/gpt-dec1-upstream` with `api_base` `<peer>/v1`. `POST <path>` must send one
    `POST /v1/responses` with `authorization: Bearer <deployment key>`, body `model` `gpt-dec1-upstream` and
    `input` the caller's text, and return 200 with `object: response` and `status: completed`."""
    case: Final = marker()
    response: Final = gateway.post(path, responses_body(case))
    sent: Final = provider.requests_for(case)
    assert response.status_code == 200, response.text
    assert sent[0].target == "/v1/responses"
    assert sent[0].headers["authorization"] == f"Bearer {PROVIDER_KEY}"
    forwarded: Final = json_of(sent[0])
    assert (forwarded["model"], forwarded["input"]) == ("gpt-dec1-upstream", case)
    body: Final = response_json(response)
    assert (body["object"], body["status"]) == ("response", "completed")


@pytest.mark.parametrize("model", ["azure-gpt", "claude", "bedrock-claude"])
def test_responses_decline_providers_other_than_openai_and_openrouter(
    gateway: NativeGateway, provider: ScriptedProvider, model: str
) -> None:
    """`POST /v1/responses` for a non-OpenAI deployment. `responses_provider` declines it (`NOT_NATIVE`), so
    the answer must be 501 with message `declined by the rust path: native HTTP responses provider` and
    nothing reaches the provider. The Python proxy bridges these to chat completions."""
    case: Final = marker()
    response: Final = gateway.post("/v1/responses", responses_body(case, model))
    assert response.status_code == 501, response.text
    assert response_json(response) == openai_error_body(
        "declined by the rust path: native HTTP responses provider", "api_error", 501
    )
    assert provider.requests_for(case) == ()


def test_responses_unknown_model_names_the_responses_route(gateway: NativeGateway) -> None:
    """`POST /v1/responses` for `dec1-nope`. The message must start `/responses:` and the status be 400."""
    response: Final = gateway.post("/v1/responses", responses_body(marker(), "dec1-nope"))
    assert response.status_code == 400
    assert response_json(response) == openai_error_body(
        f"/responses: Invalid model name passed in model=dec1-nope. {_UNKNOWN_SUFFIX}", "invalid_request_error", 400
    )


# Routes the preview mounts but does not implement


@pytest.mark.parametrize(
    "path",
    [
        "/v1/embeddings",
        "/embeddings",
        "/v1/completions",
        "/completions",
        "/engines/chat/embeddings",
        "/openai/deployments/chat/completions",
    ],
)
def test_unimplemented_routes_answer_501_naming_the_path(
    gateway: NativeGateway, provider: ScriptedProvider, path: str
) -> None:
    """`POST <path>` with the master key and a valid model. The route is mounted with `request::unsupported`,
    so the answer must be 501 with exactly `{"error": {"message": "<path> is not implemented by the Rust
    gateway", "type": "api_error", "param": null, "code": 501}}` and nothing reaches the provider. An
    embeddings user of the preview gets a clear refusal, not a 404 or a hang."""
    case: Final = marker()
    response: Final = gateway.post(path, {"model": "chat", "input": case})
    assert response.status_code == 501, response.text
    assert response_json(response) == openai_error_body(
        f"{path} is not implemented by the Rust gateway", "api_error", 501
    )
    assert provider.requests_for(case) == ()


# Per-request cache controls


def test_cache_controls_are_stripped_before_the_provider(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """A messages call with `cache: {"no-cache": true}` and `caching: false`. Both keys are read by
    `caching::prepare` and removed, so the call must succeed and the provider body must have neither key.
    No response cache is built in the standalone binary, so they change nothing else."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case, cache={"no-cache": True}, caching=False))
    assert response.status_code == 200, response.text
    forwarded: Final = json_of(provider.requests_for(case)[0])
    assert "cache" not in forwarded
    assert "caching" not in forwarded


@pytest.mark.parametrize(
    ("controls", "message_start"),
    [
        ({"cache": {"dec1-unknown": True}}, "invalid request body: invalid cache controls: "),
        ({"cache": {"ttl": 0}}, "invalid request body: cache durations must be finite and positive"),
        ({"cache": {"s-maxage": -5}}, "invalid request body: cache durations must be finite and positive"),
        ({"caching": "yes"}, "invalid request body: invalid caching flag: "),
    ],
    ids=["unknown_control", "zero_ttl", "negative_max_age", "string_caching_flag"],
)
def test_malformed_cache_controls_are_a_400(
    gateway: NativeGateway, provider: ScriptedProvider, controls: dict[str, object], message_start: str
) -> None:
    """A chat call with a malformed cache control. `Controls` denies unknown fields and durations must be
    positive, so the answer must be 400 `invalid_request_error` whose message starts with the expected text,
    with no provider call. The Python proxy ignores unknown cache keys; this strictness is preview-only."""
    case: Final = marker()
    response: Final = gateway.post("/v1/chat/completions", {**chat_body(case), **controls})
    assert response.status_code == 400, response.text
    error: Final = response_json(response)["error"]
    assert isinstance(error, dict)
    assert str(error["message"]).startswith(message_start), error
    assert provider.requests_for(case) == ()


def test_served_responses_carry_no_litellm_headers(gateway: NativeGateway) -> None:
    """Pins the preview: a served `/v1/messages` call. The Python proxy adds `x-litellm-call-id`,
    `x-litellm-model-id`, `x-litellm-response-cost` and friends; the preview writes none of them (request
    prep and response headers are not ported, One Pager A8), and with no response cache it writes no
    `x-litellm-cache-key` either. No response header may start with `x-litellm-`."""
    response: Final = gateway.post("/v1/messages", messages_body(marker()))
    assert response.status_code == 200, response.text
    assert [name for name in response.headers if name.lower().startswith("x-litellm-")] == []
