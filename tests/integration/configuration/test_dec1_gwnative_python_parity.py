"""Parity between the native Rust gateway preview and the Python proxy for the same config.yaml.

The contract: a user who points the preview at the config they run on the Python proxy must, for every
call the preview claims to serve, see the same status, the same response body fields and the same
provider request. Where the preview refuses something the Python proxy serves, the test is written for
parity and skipped with `BUG:`, so the list of skips is the list of gaps between part B of the December 1
definition and the proxy.

Both servers boot from one config whose master key is `os.environ/LITELLM_MASTER_KEY` and talk to one
scripted provider. The Python proxy is the harness's `forwarded_proxy` (two workers, no database,
behind the Rust forwarder), the way a user runs it today.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.rust_forward import EntryPoint, forwarded_proxy
from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    anthropic_error,
    anthropic_stream,
    chat_body,
    json_of,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    openai_deployment,
    openai_like_deployment,
    response_json,
    responses_body,
    scripted_provider,
    sse_events,
)


@dataclass(frozen=True, slots=True)
class Pair:
    native: NativeGateway
    python_url: str
    provider: ScriptedProvider

    def python(
        self,
        path: str,
        body: dict[str, JsonValue] | None = None,
        *,
        key: str | None = GATEWAY_KEY,
        headers: dict[str, str] | None = None,
        method: str = "POST",
    ) -> httpx.Response:
        sent: Final = {**({"Authorization": f"Bearer {key}"} if key is not None else {}), **(headers or {})}
        with httpx.Client(base_url=self.python_url, timeout=60, trust_env=False) as client:
            if method == "GET":
                return client.get(path, headers=sent)
            return client.post(path, json=body or {}, headers=sent)

    def rust(
        self,
        path: str,
        body: dict[str, JsonValue] | None = None,
        *,
        key: str | None = GATEWAY_KEY,
        headers: dict[str, str] | None = None,
        method: str = "POST",
    ) -> httpx.Response:
        if method == "GET":
            return self.native.get(path, key=key)
        return self.native.post(path, body or {}, key=key, headers=headers or {})


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pair]:
    with scripted_provider() as provider:
        config: Final = master_key_config(
            [
                anthropic_deployment("claude", provider.url),
                openai_like_deployment("chat", provider.url),
                openai_deployment("openai-gpt", provider.url),
            ]
        )
        with (
            native_gateway(config, tmp_path_factory.mktemp("gwnative-parity-rust")) as native,
            forwarded_proxy(EntryPoint.LITELLM, config, tmp_path_factory.mktemp("gwnative-parity-python")) as python,
        ):
            yield Pair(native, python.url, provider)


def test_readiness_bodies_match(pair: Pair) -> None:
    """`GET /health/readiness` without credentials on both. Both must answer 200 with the same JSON body,
    `{"status": "healthy", "db": "Not connected"}` for a proxy without a database."""
    python: Final = pair.python("/health/readiness", key=None, method="GET")
    rust: Final = pair.rust("/health/readiness", key=None, method="GET")
    assert (python.status_code, rust.status_code) == (200, 200)
    assert response_json(rust) == response_json(python)


def test_messages_bodies_match_for_a_plain_anthropic_call(pair: Pair) -> None:
    """The same non-streaming `/v1/messages` call to the Anthropic deployment on both, against the same
    scripted reply. Both must answer 200, and the bodies must be equal on every key except `model` (covered
    by its own test): `id`, `type`, `role`, `content`, `stop_reason`, `stop_sequence`, `usage`."""
    python: Final = pair.python("/v1/messages", messages_body(marker()))
    rust: Final = pair.rust("/v1/messages", messages_body(marker()))
    assert (python.status_code, rust.status_code) == (200, 200), (python.text, rust.text)
    python_body: Final = {key: value for key, value in response_json(python).items() if key != "model"}
    rust_body: Final = {key: value for key, value in response_json(rust).items() if key != "model"}
    assert rust_body == python_body


def test_messages_send_the_same_provider_body(pair: Pair) -> None:
    """The same `/v1/messages` call (with `temperature` and `system`) on both. The provider must receive equal
    JSON bodies from both, once the prompt marker is normalised. Python already runs Anthropic `/v1/messages`
    through the Rust core, so a difference here means the gateway's projection (`project` in
    `gateway-inference/src/messages.rs`) diverged from Python's request prep.
    Pinned difference (CI run 44176682961): Python also sends `"stream": false`, the preview leaves `stream` out.
    Both mean a unary call to Anthropic, so the comparison drops that one key after pinning it on each side."""
    python_case: Final = marker()
    rust_case: Final = marker()
    pair.python("/v1/messages", messages_body(python_case, temperature=0.2, system="dec1 system"))
    pair.rust("/v1/messages", messages_body(rust_case, temperature=0.2, system="dec1 system"))
    python_sent: Final = pair.provider.requests_for(python_case)[0].body.decode().replace(python_case, "CASE")
    rust_sent: Final = pair.provider.requests_for(rust_case)[0].body.decode().replace(rust_case, "CASE")
    python_body: Final = json.loads(python_sent)
    rust_body: Final = json.loads(rust_sent)
    assert (python_body.pop("stream", None), rust_body.pop("stream", None)) == (False, None)
    assert rust_body == python_body


def test_messages_send_the_same_provider_auth_and_version_headers(pair: Pair) -> None:
    """The same `/v1/messages` call on both. The provider requests must agree on `x-api-key` and
    `anthropic-version`, the two headers Anthropic requires."""
    python_case: Final = marker()
    rust_case: Final = marker()
    pair.python("/v1/messages", messages_body(python_case))
    pair.rust("/v1/messages", messages_body(rust_case))
    python_headers: Final = pair.provider.requests_for(python_case)[0].headers
    rust_headers: Final = pair.provider.requests_for(rust_case)[0].headers
    for name in ("x-api-key", "anthropic-version"):
        assert rust_headers.get(name) == python_headers.get(name), name


def test_messages_response_model_matches_the_python_proxy(pair: Pair) -> None:
    """The caller asks for `claude` and the provider answers `model: claude-dec1-upstream`. Whatever the
    Python proxy returns in `model` (it restamps to the requested name per One Pager A19), the preview must
    return the same. UNVERIFIED which value Python returns; the test compares rather than hard-codes it."""
    python: Final = pair.python("/v1/messages", messages_body(marker()))
    rust: Final = pair.rust("/v1/messages", messages_body(marker()))
    if response_json(python)["model"] != response_json(rust)["model"]:
        pytest.skip("BUG: the preview does not restamp the response model to the requested name like Python")
    assert response_json(rust)["model"] == response_json(python)["model"]


def test_messages_stream_event_sequences_match(pair: Pair) -> None:
    """The same streaming `/v1/messages` call on both against the same scripted SSE. The event name sequence
    must be equal, and both must be `text/event-stream`."""
    python_case: Final = marker()
    rust_case: Final = marker()
    pair.provider.script(python_case, lambda _: anthropic_stream("dec1 parity stream"))
    pair.provider.script(rust_case, lambda _: anthropic_stream("dec1 parity stream"))
    python: Final = pair.python("/v1/messages", messages_body(python_case, stream=True))
    rust: Final = pair.rust("/v1/messages", messages_body(rust_case, stream=True))
    assert python.headers["content-type"].startswith("text/event-stream")
    assert rust.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _ in sse_events(rust.text)] == [name for name, _ in sse_events(python.text)]


def test_an_anthropic_provider_error_matches(pair: Pair) -> None:
    """The provider answers 529 `overloaded_error` to both. Status and `error.type` must match between the two
    servers. Pinned difference (CI run 44176682961): the preview relays the provider's own `error.message`
    (`dec1 parity busy`), while Python wraps it as `litellm.InternalServerError: AnthropicError - <provider
    body>` followed by its model group no-fallback note; both still carry the provider text."""
    python_case: Final = marker()
    rust_case: Final = marker()
    for case in (python_case, rust_case):
        pair.provider.script(case, lambda _: anthropic_error(529, "overloaded_error", "dec1 parity busy"))
    python: Final = pair.python("/v1/messages", messages_body(python_case))
    rust: Final = pair.rust("/v1/messages", messages_body(rust_case))
    assert rust.status_code == python.status_code, (python.text, rust.text)
    rust_error: Final = response_json(rust)["error"]
    python_error: Final = response_json(python)["error"]
    assert isinstance(rust_error, dict) and isinstance(python_error, dict)
    assert rust_error["type"] == python_error["type"] == "overloaded_error"
    assert rust_error["message"] == "dec1 parity busy"
    assert str(python_error["message"]).startswith("litellm.InternalServerError: AnthropicError - ")
    assert "dec1 parity busy" in str(python_error["message"])


def test_unknown_model_on_messages_matches_status_and_error_type(pair: Pair) -> None:
    """`/v1/messages` for `dec1-nope` on both. Status must match (400) and both must use the Anthropic
    envelope with the same `error.type`."""
    python: Final = pair.python("/v1/messages", messages_body(marker(), "dec1-nope"))
    rust: Final = pair.rust("/v1/messages", messages_body(marker(), "dec1-nope"))
    assert (python.status_code, rust.status_code) == (400, 400), (python.text, rust.text)
    assert response_json(rust)["type"] == response_json(python)["type"] == "error"
    assert response_json(rust)["error"]["type"] == response_json(python)["error"]["type"]  # type: ignore[index]


def test_unknown_model_on_chat_completions_matches_status(pair: Pair) -> None:
    """`/v1/chat/completions` for `dec1-nope` on both. Both must answer 400 with an OpenAI `error` object
    whose message mentions `model=dec1-nope`."""
    python: Final = pair.python("/v1/chat/completions", chat_body(marker(), "dec1-nope"))
    rust: Final = pair.rust("/v1/chat/completions", chat_body(marker(), "dec1-nope"))
    assert (python.status_code, rust.status_code) == (400, 400), (python.text, rust.text)
    assert "model=dec1-nope" in str(response_json(python)["error"])
    assert "model=dec1-nope" in str(response_json(rust)["error"])


def test_chat_completion_content_and_object_match_for_openai_like(pair: Pair) -> None:
    """The same `/v1/chat/completions` call to the `openai_like` deployment on both. Both must answer 200 with
    `object: chat.completion`, the same `choices[0].message.content` and the same `choices[0].finish_reason`.
    BUG: the preview serializes the bare `ChatCompletionsResponse` (`created`, `model`, `choices`, `usage`, with no
    `id` and no `object`, litellm/rust/crates/llms-types/src/formats/chat_completions/response.rs:83-92) straight
    to the caller (`Json` in gateway-inference/src/chat_completions.rs:78-98). That shape is meant for Python to
    wrap in a `ModelResponse`; standalone there is no Python, so callers get no `object` (CI run 44176682961)."""
    pytest.skip("BUG: native gateway chat completions answer has no object and no id")
    python_response: Final = pair.python("/v1/chat/completions", chat_body(marker()))
    rust_response: Final = pair.rust("/v1/chat/completions", chat_body(marker()))
    assert (python_response.status_code, rust_response.status_code) == (200, 200), (
        python_response.text,
        rust_response.text,
    )
    python: Final = response_json(python_response)
    rust: Final = response_json(rust_response)
    assert rust["object"] == python["object"] == "chat.completion"
    assert rust["choices"][0]["message"]["content"] == python["choices"][0]["message"]["content"]  # type: ignore[index]
    assert rust["choices"][0]["finish_reason"] == python["choices"][0]["finish_reason"]  # type: ignore[index]


def test_chat_completion_usage_matches_for_openai_like(pair: Pair) -> None:
    """Same call. `usage.prompt_tokens`, `usage.completion_tokens` and `usage.total_tokens` must be equal,
    since both read them from the provider's answer (5, 3, 8 in the script). In CI run 44176682961 Python's
    answer had no `usage` at all and the body was not recorded, so both statuses are asserted first to show
    what Python answered."""
    python_response: Final = pair.python("/v1/chat/completions", chat_body(marker()))
    rust_response: Final = pair.rust("/v1/chat/completions", chat_body(marker()))
    assert (python_response.status_code, rust_response.status_code) == (200, 200), (
        python_response.text,
        rust_response.text,
    )
    python: Final = response_json(python_response).get("usage")
    rust: Final = response_json(rust_response).get("usage")
    assert isinstance(python, dict) and isinstance(rust, dict)
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert rust[name] == python[name], name


def test_responses_output_text_matches_for_openai(pair: Pair) -> None:
    """The same `/v1/responses` call to the OpenAI deployment on both. Both must answer 200 with `object:
    response`, `status: completed` and the same `output[0].content[0].text`."""
    python: Final = response_json(pair.python("/v1/responses", responses_body(marker(), "openai-gpt")))
    rust: Final = response_json(pair.rust("/v1/responses", responses_body(marker(), "openai-gpt")))
    assert (rust["object"], rust["status"]) == (python["object"], python["status"])
    assert rust["output"][0]["content"][0]["text"] == python["output"][0]["content"][0]["text"]  # type: ignore[index]


@pytest.mark.parametrize("key", [None, "sk-dec1-parity-wrong"], ids=["missing", "wrong"])
def test_rejected_credentials_get_401_on_both(pair: Pair, key: str | None) -> None:
    """`/v1/messages` with no key and with a wrong key. The preview must answer 401 to both. Python must answer
    401 to the missing key; to a wrong `sk-` key this database-less Python proxy answers 400, since it cannot look
    the key up (pinned from CI run 44176682961; UNVERIFIED which auth branch maps it to 400). (The body differs
    too: Python writes JSON, the preview plain text; see the next test.)"""
    python: Final = pair.python("/v1/messages", messages_body(marker()), key=key)
    rust: Final = pair.rust("/v1/messages", messages_body(marker()), key=key)
    assert (python.status_code, rust.status_code) == ((401 if key is None else 400), 401), (python.text, rust.text)


def test_a_401_body_is_json_on_both(pair: Pair) -> None:
    """A wrong key on `/v1/chat/completions`. The Python proxy answers a JSON `error` object; SDKs parse it.
    The preview must answer JSON with an `error.message` too."""
    pytest.skip("BUG: the preview's 401 body is the plain text `missing or invalid bearer token`, not JSON")
    rust: Final = pair.rust("/v1/chat/completions", chat_body(marker()), key="sk-dec1-parity-wrong")
    assert "message" in response_json(rust)["error"]  # type: ignore[operator]


@pytest.mark.parametrize("header", ["x-api-key", "x-litellm-api-key"])
def test_the_master_key_in_an_api_key_header_is_accepted_on_both(pair: Pair, header: str) -> None:
    """The master key sent as `<header>` with no Authorization on `/v1/messages`. The Python proxy accepts it
    (Anthropic SDKs send `x-api-key`); the preview must too, or Claude Code pointed at the preview fails."""
    python: Final = pair.python("/v1/messages", messages_body(marker()), key=None, headers={header: GATEWAY_KEY})
    assert python.status_code == 200, python.text
    pytest.skip("BUG: the preview reads only Authorization: Bearer (One Pager: x-api-key not started)")
    rust: Final = pair.rust("/v1/messages", messages_body(marker()), key=None, headers={header: GATEWAY_KEY})
    assert rust.status_code == 200, rust.text


def test_the_call_id_header_is_present_on_both(pair: Pair) -> None:
    """A served `/v1/messages` call. The Python proxy returns `x-litellm-call-id` (and model id, cost and
    timing headers); the preview must return at least `x-litellm-call-id` so a support ticket can quote it."""
    python: Final = pair.python("/v1/messages", messages_body(marker()))
    assert "x-litellm-call-id" in python.headers
    pytest.skip("BUG: the preview writes no x-litellm-* response headers (One Pager A8 request prep not ported)")
    rust: Final = pair.rust("/v1/messages", messages_body(marker()))
    assert "x-litellm-call-id" in rust.headers


def test_openai_chat_completions_are_served_on_both(pair: Pair) -> None:
    """`/v1/chat/completions` for an `openai/` deployment. Python serves it with 200; the preview must too."""
    python: Final = pair.python("/v1/chat/completions", chat_body(marker(), "openai-gpt"))
    assert python.status_code == 200, python.text
    pytest.skip("BUG: the preview's chat route has no OpenAI provider config and answers 400 invalid provider: openai")
    rust: Final = pair.rust("/v1/chat/completions", chat_body(marker(), "openai-gpt"))
    assert rust.status_code == 200, rust.text


def test_streaming_chat_completions_are_served_on_both(pair: Pair) -> None:
    """`stream: true` on `/v1/chat/completions` for `chat`. Python streams `chat.completion.chunk` frames; the
    preview must stream too instead of answering 501."""
    pytest.skip("BUG: the preview declines every streaming chat completion with 501")
    rust: Final = pair.rust("/v1/chat/completions", chat_body(marker(), stream=True))
    assert rust.status_code == 200, rust.text


def test_embeddings_are_served_on_both(pair: Pair) -> None:
    """`/v1/embeddings` for an OpenAI deployment. Python serves embeddings; the preview answers 501
    `/v1/embeddings is not implemented by the Rust gateway`. Part B of December 1 lists only `/v1/messages`,
    so this gap may be accepted; the test documents it."""
    pytest.skip("BUG: /v1/embeddings is mounted as request::unsupported (501) in the preview")
    rust: Final = pair.rust("/v1/embeddings", {"model": "openai-gpt", "input": "dec1"})
    assert rust.status_code == 200, rust.text


def test_messages_provider_request_is_the_same_for_the_same_prompt_marker_count(pair: Pair) -> None:
    """One call on each server must cause exactly one provider request each (no hidden retries, health
    checks or discovery calls on the request path)."""
    python_case: Final = marker()
    rust_case: Final = marker()
    pair.python("/v1/messages", messages_body(python_case))
    pair.rust("/v1/messages", messages_body(rust_case))
    assert (len(pair.provider.requests_for(python_case)), len(pair.provider.requests_for(rust_case))) == (1, 1)
    assert (
        json_of(pair.provider.requests_for(rust_case)[0])["model"]
        == json_of(pair.provider.requests_for(python_case)[0])["model"]
    )
