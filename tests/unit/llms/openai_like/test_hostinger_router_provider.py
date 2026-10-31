import copy
import json
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_hostinger_router_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hostinger-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="hostinger_router/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5"
    assert provider == "hostinger_router"
    assert api_key == "hostinger-test-key"
    assert api_base == "https://router.hostinger.com/v1"


def test_hostinger_router_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HOSTINGER_ROUTER_API_KEY", "hostinger-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="hostinger_router/claude-sonnet-5",
        custom_llm_provider=None,
        api_base="https://router.internal.example/v1",
        api_key="hostinger-explicit-key",
    )

    assert provider == "hostinger_router"
    assert api_key == "hostinger-explicit-key"
    assert api_base == "https://router.internal.example/v1"


@pytest.mark.parametrize("model", ["gpt-5-nano", "claude-sonnet-5"])
def test_hostinger_router_api_base_keeps_unprefixed_model_provider(model: str):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    _, provider_without_api_base, _, _ = get_llm_provider(model=model)
    _, provider_with_hostinger_api_base, _, _ = get_llm_provider(
        model=model,
        api_base="https://router.hostinger.com/v1",
        api_key="hostinger-test-key",
    )

    assert provider_with_hostinger_api_base == provider_without_api_base


HOSTINGER_ROUTER_MODELS = tuple(sorted(name for name in litellm.model_cost if name.startswith("hostinger_router/")))
HOSTINGER_ROUTER_CHAT_MODELS = tuple(
    name for name in HOSTINGER_ROUTER_MODELS if litellm.get_model_info(name)["mode"] == "chat"
)


@pytest.mark.parametrize("model", HOSTINGER_ROUTER_CHAT_MODELS)
def test_hostinger_router_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="hostinger_router",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert model_info["output_cost_per_token"] > 0
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "hostinger_router"
    assert model_info["mode"] == "chat"
    if model_info.get("supports_prompt_caching"):
        assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    capability_checks: Final = (
        ("supports_function_calling", litellm.supports_function_calling),
        ("supports_reasoning", litellm.supports_reasoning),
        ("supports_response_schema", litellm.supports_response_schema),
        ("supports_vision", litellm.supports_vision),
    )
    assert {name: check(model) for name, check in capability_checks} == {
        name: model_info.get(name) is True for name, _ in capability_checks
    }


def test_hostinger_router_chat_completions_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_hostinger",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "claude-sonnet-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Hostinger Router"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="hostinger_router/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="hostinger-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert response.choices[0].message.content == "Hello from Hostinger Router"


def test_hostinger_router_chat_completions_streaming_request():
    chunks = [
        'data: {"id":"chatcmpl_hostinger","object":"chat.completion.chunk","created":1789550000,"model":"claude-sonnet-5","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl_hostinger","object":"chat.completion.chunk","created":1789550000,"model":"claude-sonnet-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n',
        "data: [DONE]\n\n",
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/chat/completions").respond(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(chunks).encode(),
        )
        response: Final = litellm.completion(
            model="hostinger_router/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            api_key="hostinger-test-key",
        )
        collected = [chunk for chunk in response]

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert body["stream"] is True
    assert any(chunk.choices and chunk.choices[0].delta.content == "Hi" for chunk in collected)


def test_hostinger_router_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/responses").respond(
            200,
            json={
                "id": "resp_hostinger",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "gpt-5.2",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_hostinger",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Hostinger Router", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="hostinger_router/gpt-5.2",
            input="Say hello",
            api_key="hostinger-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://router.hostinger.com/v1/responses"
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert body["model"] == "gpt-5.2"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Hostinger Router"


EMBEDDING_RESPONSE: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}


def test_hostinger_router_embeddings_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        response: Final = litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert body["model"] == "text-embedding-3-small"
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_hostinger_router_embeddings_send_extra_headers_as_http_headers():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={"X-Trace-Id": "trace-123"},
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["x-trace-id"] == "trace-123"
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert request.headers["content-type"] == "application/json"
    assert set(body) == {"model", "input"}


def test_hostinger_router_embeddings_extra_headers_replace_key_headers_of_any_case():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={"authorization": "Bearer caller-key", "x-trace-id": "trace-1"},
        )

    request: Final = route.calls.last.request
    assert request.headers.get_list("authorization") == ["Bearer caller-key"]
    assert request.headers.get_list("content-type") == ["application/json"]
    assert request.headers.get_list("x-trace-id") == ["trace-1"]


@pytest.mark.parametrize(("configured", "sent"), [(12345, "12345"), (1.5, "1.5")])
def test_hostinger_router_embeddings_send_numeric_extra_headers_as_text(configured: int | float, sent: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={"X-Org-Id": configured},
        )

    assert route.calls.last.request.headers.get_list("x-org-id") == [sent]


@pytest.mark.parametrize("configured", [True, None, ["a", "b"]])
def test_hostinger_router_embeddings_refuse_non_text_extra_headers(configured: object):
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        with pytest.raises(litellm.UnsupportedParamsError, match="X-Debug") as refused:
            litellm.embedding(
                model="hostinger_router/text-embedding-3-small",
                input=["Say hello"],
                api_key="hostinger-test-key",
                extra_headers={"X-Debug": configured},
            )

    assert refused.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.parametrize("configured", [True, None, ["a", "b"]])
def test_hostinger_router_embeddings_drop_non_text_extra_headers_under_drop_params(configured: object):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={"X-Debug": configured, "X-Trace-Id": "trace-1"},
            drop_params=True,
        )

    request: Final = route.calls.last.request
    assert "x-debug" not in request.headers
    assert request.headers.get_list("x-trace-id") == ["trace-1"]
    assert set(json.loads(request.content)) == {"model", "input"}


TRANSPORT_EXTRA_HEADERS: Final = [
    ("Host", "other.example"),
    ("content-length", "1"),
    ("Transfer-Encoding", "chunked"),
    ("Connection", "close"),
    ("Keep-Alive", "timeout=5"),
    ("Proxy-Connection", "close"),
    ("TE", "gzip"),
    ("Trailer", "X-Checksum"),
    ("Upgrade", "websocket"),
    ("Expect", "100-continue"),
]


@pytest.mark.parametrize(("name", "configured"), TRANSPORT_EXTRA_HEADERS)
def test_hostinger_router_embeddings_ignore_transport_extra_headers(name: str, configured: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={name: configured, "X-Trace-Id": "trace-1"},
        )

    request: Final = route.calls.last.request
    assert request.headers.get_list("host") == ["router.hostinger.com"]
    assert request.headers.get_list("content-length") == [str(len(request.content))]
    assert "transfer-encoding" not in request.headers
    assert configured not in request.headers.get_list(name)
    assert request.headers.get_list("x-trace-id") == ["trace-1"]


MALFORMED_HEADER_NAMES: Final = ["Host:evil.example", "X Note", "X-Caf\u00e9"]


@pytest.mark.parametrize("name", MALFORMED_HEADER_NAMES)
def test_hostinger_router_embeddings_refuse_malformed_extra_header_names(name: str):
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        with pytest.raises(litellm.UnsupportedParamsError) as refused:
            litellm.embedding(
                model="hostinger_router/text-embedding-3-small",
                input=["Say hello"],
                api_key="hostinger-test-key",
                extra_headers={name: "x"},
            )

    assert refused.value.status_code == 400
    assert repr(name) in refused.value.message
    assert route.call_count == 0


@pytest.mark.parametrize("name", MALFORMED_HEADER_NAMES)
def test_hostinger_router_embeddings_drop_malformed_extra_header_names_under_drop_params(name: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={name: "x", "X-Trace-Id": "trace-1"},
            drop_params=True,
        )

    request: Final = route.calls.last.request
    assert request.headers.get_list("host") == ["router.hostinger.com"]
    assert request.headers.get_list("x-trace-id") == ["trace-1"]
    assert name.lower() not in request.headers.keys()


@pytest.mark.parametrize("configured", ["a\r\nHost: evil.example", "line\nbreak", "caf\u00e9", "a\x7fb"])
def test_hostinger_router_embeddings_drop_extra_header_values_http_cannot_carry(configured: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={"x-litellm-user_api_key_alias": configured, "X-Trace-Id": "trace-1"},
        )

    request: Final = route.calls.last.request
    assert request.headers.get_list("host") == ["router.hostinger.com"]
    assert request.headers.get_list("x-trace-id") == ["trace-1"]
    assert "x-litellm-user_api_key_alias" not in request.headers


@pytest.mark.parametrize(
    ("configured", "sent"),
    [(" x ", "x"), ("\t", ""), (" a\tb ", "a\tb"), ("env-secret\n", "env-secret"), ("\r\n x \r\n", "x")],
)
def test_hostinger_router_embeddings_trim_whitespace_around_extra_header_values(configured: str, sent: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            extra_headers={"X-Rig": configured},
        )

    assert route.calls.last.request.headers.get_list("x-rig") == [sent]


def test_hostinger_router_embeddings_extra_headers_stay_with_their_request():
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "shared",
                "litellm_params": {
                    "model": "hostinger_router/text-embedding-3-small",
                    "api_key": "hostinger-test-key",
                    "headers": {"X-Org-Id": "org-1"},
                },
            }
        ]
    )

    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        router.embedding(
            model="shared", input=["first"], extra_headers={"Authorization": "Bearer caller-a", "X-Leak": "a"}
        )
        router.embedding(model="shared", input=["second"])

    first: Final = route.calls[0].request
    second: Final = route.calls[1].request
    assert first.headers.get_list("x-leak") == ["a"]
    assert second.headers.get_list("authorization") == ["Bearer hostinger-test-key"]
    assert "x-leak" not in second.headers
    assert second.headers.get_list("x-org-id") == ["org-1"]


@pytest.mark.parametrize("param", ["headers", "extra_headers"])
def test_hostinger_router_embeddings_send_headers_given_as_name_value_pairs(param: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            **{param: [["X-Trace-Id", "trace-1"]]},
        )

    request: Final = route.calls.last.request
    assert request.headers.get_list("x-trace-id") == ["trace-1"]
    assert request.headers.get_list("authorization") == ["Bearer hostinger-test-key"]


@pytest.mark.parametrize("param", ["headers", "extra_headers"])
def test_hostinger_router_embeddings_refuse_headers_that_are_not_an_object(param: str):
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        with pytest.raises(litellm.UnsupportedParamsError, match=rf"\b{param} must be a JSON object") as refused:
            litellm.embedding(
                model="hostinger_router/text-embedding-3-small",
                input=["Say hello"],
                api_key="hostinger-test-key",
                **{param: "X-Trace-Id: trace-1"},
            )

    assert refused.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.parametrize("param", ["headers", "extra_headers"])
def test_hostinger_router_embeddings_drop_headers_that_are_not_an_object_under_drop_params(param: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        litellm.embedding(
            model="hostinger_router/text-embedding-3-small",
            input=["Say hello"],
            api_key="hostinger-test-key",
            drop_params=True,
            **{param: "X-Trace-Id: trace-1"},
        )

    request: Final = route.calls.last.request
    assert "x-trace-id" not in request.headers
    assert request.headers.get_list("authorization") == ["Bearer hostinger-test-key"]
    assert set(json.loads(request.content)) == {"model", "input"}


def test_hostinger_router_embeddings_never_borrow_another_providers_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HOSTINGER_ROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_LIKE_API_KEY", "other-provider-key")
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-key")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/embeddings").respond(200, json=EMBEDDING_RESPONSE)
        with pytest.raises(litellm.BadRequestError, match="Missing API Key"):
            litellm.embedding(model="hostinger_router/text-embedding-3-small", input=["Say hello"])

    assert route.call_count == 0


def test_json_provider_without_embeddings_endpoint_refuses_embeddings():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    slug: Final = next(
        slug
        for slug in JSONProviderRegistry.list_providers()
        if not JSONProviderRegistry.supports_embeddings(slug) and slug not in litellm.provider_list
    )

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.route().respond(200, json=EMBEDDING_RESPONSE)
        with pytest.raises(litellm.BadRequestError):
            litellm.embedding(model=f"{slug}/text-embedding-3-small", input=["Say hello"], api_key="test-key")

    assert route.call_count == 0


@pytest.mark.asyncio
async def test_hostinger_router_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://router.hostinger.com/v1/messages").respond(
            200,
            json={
                "id": "msg_hostinger",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "Hello from Hostinger Router"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="hostinger_router/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="hostinger-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://router.hostinger.com/v1/messages"
    assert request.headers["authorization"] == "Bearer hostinger-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Hostinger Router"


CHAT_COMPLETION_RESPONSE: Final = {
    "id": "chatcmpl_hostinger",
    "object": "chat.completion",
    "created": 1_789_550_000,
    "model": "gpt-5.2",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Hostinger Router"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
UNSUPPORTED_ENDPOINT: Final = {"error": {"message": "Model does not support this endpoint."}}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.2", "gpt-5-nano"])
async def test_hostinger_router_messages_bridge_models_without_native_messages(
    model: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock(assert_all_called=False) as upstream:
        native: Final = upstream.post("https://router.hostinger.com/v1/messages").respond(
            403, json=UNSUPPORTED_ENDPOINT
        )
        chat: Final = upstream.post("https://router.hostinger.com/v1/chat/completions").respond(
            200, json=CHAT_COMPLETION_RESPONSE
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"hostinger_router/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="hostinger-test-key",
        )

    assert native.call_count == 0
    assert chat.call_count == 1
    assert json.loads(chat.calls.last.request.content)["model"] == model
    assert response["content"][0]["text"] == "Hello from Hostinger Router"


def test_hostinger_router_responses_bridge_models_without_native_responses():
    with respx.mock(assert_all_called=False) as upstream:
        native: Final = upstream.post("https://router.hostinger.com/v1/responses").respond(
            403, json=UNSUPPORTED_ENDPOINT
        )
        chat: Final = upstream.post("https://router.hostinger.com/v1/chat/completions").respond(
            200, json=CHAT_COMPLETION_RESPONSE
        )
        response: Final = litellm.responses(
            model="hostinger_router/claude-sonnet-5",
            input="Say hello",
            api_key="hostinger-test-key",
        )

    assert native.call_count == 0
    assert chat.call_count == 1
    assert json.loads(chat.calls.last.request.content)["model"] == "claude-sonnet-5"
    assert response.output[0].content[0].text == "Hello from Hostinger Router"


ANTHROPIC_MESSAGE_RESPONSE: Final = {
    "id": "msg_hostinger",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "Hello from Hostinger Router"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 4, "output_tokens": 3},
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sibling_model_info",
    [
        {"supported_endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/responses"]},
        {"supported_endpoints": "/v1/messages"},
    ],
)
async def test_hostinger_router_endpoint_opt_in_stays_with_its_own_deployment(
    sibling_model_info: dict[str, object], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    deployment_params: Final = {"model": "hostinger_router/gpt-5-nano", "api_key": "hostinger-test-key"}
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "opted-in", "litellm_params": deployment_params, "model_info": sibling_model_info},
            {"model_name": "plain", "litellm_params": deployment_params},
        ]
    )

    with respx.mock(assert_all_called=False) as upstream:
        native_messages: Final = upstream.post("https://router.hostinger.com/v1/messages").respond(
            403, json=UNSUPPORTED_ENDPOINT
        )
        native_responses: Final = upstream.post("https://router.hostinger.com/v1/responses").respond(
            403, json=UNSUPPORTED_ENDPOINT
        )
        chat: Final = upstream.post("https://router.hostinger.com/v1/chat/completions").respond(
            200, json=CHAT_COMPLETION_RESPONSE
        )
        messages_response: Final = await router.aanthropic_messages(
            model="plain", messages=[{"role": "user", "content": "Say hello"}], max_tokens=32
        )
        responses_response: Final = await router.aresponses(model="plain", input="Say hello")

    assert native_messages.call_count == 0
    assert native_responses.call_count == 0
    assert chat.call_count == 2
    assert messages_response["content"][0]["text"] == "Hello from Hostinger Router"
    assert responses_response.output[0].content[0].text == "Hello from Hostinger Router"


@pytest.mark.asyncio
async def test_hostinger_router_deployment_cache_ttl_opt_in_keeps_one_hour_ttl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "opted-in",
                "litellm_params": {
                    "model": "hostinger_router/claude-haiku-4-5",
                    "api_base": "https://router.hostinger.com/v1",
                    "api_key": "hostinger-test-key",
                },
                "model_info": {"supported_endpoints": ["/v1/messages"], "cache_control_ttl": True},
            }
        ]
    )
    one_hour_cache: Final = {"type": "ephemeral", "ttl": "1h"}

    with respx.mock() as upstream:
        native: Final = upstream.post("https://router.hostinger.com/v1/messages").respond(
            200, json=ANTHROPIC_MESSAGE_RESPONSE
        )
        await router.aanthropic_messages(
            model="opted-in",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Say hello", "cache_control": one_hour_cache}]}
            ],
            max_tokens=32,
        )

    body: Final = json.loads(native.calls.last.request.content)
    assert body["messages"][0]["content"][0]["cache_control"] == one_hour_cache
