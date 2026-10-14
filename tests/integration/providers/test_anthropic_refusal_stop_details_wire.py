import asyncio
import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal, TypeAlias
from urllib.parse import unquote, urlsplit

import anthropic
import openai
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.anthropic_refusal import (
    CITED_FIELDS,
    DOCS_REFUSAL,
    DOCS_STOP_DETAILS,
    FIXTURES,
    MALFORMED_STOP_DETAILS,
    REFUSAL_FIELDS,
    Ledger,
    Location,
    frame_types,
    frames,
    hidden_params_probe,
    openai_id,
    refusal_events,
    refusal_prompt,
    relayed,
    respond,
    rewrite_cached_message_fields,
    sink_config,
)
from integration._support.anthropic_thinking import (
    BEDROCK_MODEL,
    JSON_LIST,
    JSON_OBJECT,
    MODEL,
    NO_CACHE,
    Event,
    aws_chunks,
    chunks_of,
    sse_chunks,
)
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, held
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Wire, wire_server
from pydantic import BaseModel, JsonValue

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("anthropic_refusal_stop_details_wire"),
]

_ANTHROPIC_BASE: Final = "http://api.anthropic.com"
_ANTHROPIC_KEY: Final = "scripted-anthropic-key"
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_VERTEX_HOST: Final = f"{_VERTEX_LOCATION}-aiplatform.googleapis.com"
_VERTEX_MODEL_PATH: Final = (
    f"/v1/projects/{_VERTEX_PROJECT}/locations/{_VERTEX_LOCATION}/publishers/anthropic/models/{MODEL}"
)
_CHAT_FIELDS: Final[Location] = ("choices", 0, "message", "provider_specific_fields")
_DELTA_FIELDS: Final[Location] = ("choices", 0, "delta", "provider_specific_fields")
_RESPONSE_FIELDS: Final[Location] = ("provider_specific_fields",)
_STREAMED_RESPONSE_FIELDS: Final[Location] = ("response", "provider_specific_fields")

Client: TypeAlias = Literal["sdk", "async_sdk", "httpx"]
Provider: TypeAlias = Literal[
    "anthropic",
    "bedrock_invoke",
    "claude_platform",
    "vertex_ai",
    "azure_ai",
    "bedrock_mantle",
    "snowflake",
    "openai_compatible",
]
Surface: TypeAlias = Literal["chat", "chat_stream", "messages", "messages_stream", "responses", "responses_stream"]

CLIENTS: Final[tuple[Client, ...]] = ("sdk", "async_sdk", "httpx")
FAMILY: Final[tuple[Provider, ...]] = (
    "bedrock_invoke",
    "claude_platform",
    "vertex_ai",
    "azure_ai",
    "bedrock_mantle",
    "snowflake",
)
_STREAM_TARGETS: Final[Mapping[Provider, tuple[str, str]]] = {
    "anthropic": ("/v1/messages", "/v1/messages"),
    "bedrock_invoke": (f"/model/{BEDROCK_MODEL}/invoke", f"/model/{BEDROCK_MODEL}/invoke-with-response-stream"),
    "claude_platform": ("/v1/messages", "/v1/messages"),
    "vertex_ai": (f"{_VERTEX_MODEL_PATH}:rawPredict", f"{_VERTEX_MODEL_PATH}:streamRawPredict"),
    "azure_ai": ("/anthropic/v1/messages", "/anthropic/v1/messages"),
    "bedrock_mantle": ("/anthropic/v1/messages", "/anthropic/v1/messages"),
    "snowflake": ("/api/v2/cortex/v1/messages", "/api/v2/cortex/v1/messages"),
}


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    wire: Wire
    vertex: Wire
    sink: Ledger
    forbidden: Ledger
    absent: Ledger
    hidden: Ledger
    credentials: str


def _service_account_json(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _VERTEX_PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_VERTEX_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def _port(wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    assert port is not None, wire.url
    return port


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("anthropic-refusal-stop-details")
    certificate: Final = write_self_signed_cert(directory, (_VERTEX_HOST,))
    with (
        gateway_from_environment() as parent,
        wire_server(respond) as wire,
        wire_server(respond, tls=server_context(*certificate)) as vertex,
        tunnelling_forward_proxy(_port(vertex)) as tunnel,
        wire_server(lambda _request: Reply()) as sink,
        wire_server(lambda _request: Reply(status=403)) as forbidden,
        wire_server(lambda _request: Reply(status=404)) as absent,
        wire_server(lambda _request: Reply()) as hidden,
    ):
        credentials: Final = _service_account_json(parent.upstream_url.rstrip("/"))
        credentials_path: Final = directory / "vertex-service-account.json"
        credentials_path.write_text(credentials)
        config: Final = sink_config(
            directory,
            {"refusal_sink": sink.url, "refusal_sink_403": forbidden.url, "refusal_sink_404": absent.url},
            extra_callbacks=(hidden_params_probe(directory, hidden.url),),
            store_prompts=True,
        )
        overrides: Final = {
            "ANTHROPIC_API_BASE": _ANTHROPIC_BASE,
            "ANTHROPIC_API_KEY": _ANTHROPIC_KEY,
            "AIOHTTP_TRUST_ENV": "True",
            "HTTP_PROXY": wire.url,
            "http_proxy": wire.url,
            "HTTPS_PROXY": tunnel.url,
            "https_proxy": tunnel.url,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "SSL_CERT_FILE": str(certificate[0]),
            "DEFAULT_VERTEXAI_PROJECT": _VERTEX_PROJECT,
            "DEFAULT_VERTEXAI_LOCATION": _VERTEX_LOCATION,
            "DEFAULT_GOOGLE_APPLICATION_CREDENTIALS": str(credentials_path),
        }
        with owned_proxy(parent, directory, overrides, config=config, workers=2) as owned:
            yield Rig(owned, wire, vertex, Ledger(sink), Ledger(forbidden), Ledger(absent), Ledger(hidden), credentials)


@pytest.fixture(autouse=True)
def _drained_upstreams(rig: Rig) -> None:
    rig.wire.drain()
    rig.vertex.drain()


def _marker() -> str:
    return uuid.uuid4().hex


def _deployment(scenario: Scenario, provider: Provider, rig: Rig) -> str:
    wire_url: Final = rig.wire.url
    match provider:
        case "anthropic":
            return scenario.model(model=f"anthropic/{MODEL}", api_base=wire_url, api_key=_ANTHROPIC_KEY)
        case "bedrock_invoke":
            return scenario.model(
                model=f"bedrock/invoke/{BEDROCK_MODEL}",
                api_base=wire_url,
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )
        case "claude_platform":
            return scenario.model(
                model=f"bedrock/claude_platform/{MODEL}",
                api_base=wire_url,
                api_key="scripted-platform-key",
                aws_region_name="us-east-1",
                workspace_id="scripted-workspace",
            )
        case "vertex_ai":
            return scenario.model(
                model=f"vertex_ai/{MODEL}",
                api_base=f"{wire_url}{_VERTEX_MODEL_PATH}",
                api_key=None,
                vertex_project=_VERTEX_PROJECT,
                vertex_location=_VERTEX_LOCATION,
                vertex_credentials=rig.credentials,
            )
        case "azure_ai":
            return scenario.model(model=f"azure_ai/{MODEL}", api_base=wire_url, api_key="scripted-azure-key")
        case "bedrock_mantle":
            return scenario.model(
                model=f"bedrock_mantle/{BEDROCK_MODEL}",
                api_base=wire_url,
                api_key="scripted-mantle-key",
                aws_region_name="us-east-1",
            )
        case "snowflake":
            return scenario.model(model=f"snowflake/{MODEL}", api_base=wire_url, api_key="scripted-snowflake-key")
        case "openai_compatible":
            return scenario.model(model="openai/gpt-4o-mini", api_base=f"{wire_url}/v1", api_key="scripted-openai-key")


def _base_url(rig: Rig) -> str:
    return str(rig.gateway.client.base_url).rstrip("/")


def _dump(model: BaseModel) -> Event:
    return JSON_OBJECT.validate_python(model.model_dump())


def _post(rig: Rig, path: str, body: Mapping[str, JsonValue], key: str) -> bytes:
    response: Final = rig.gateway.request("POST", path, body, key=key)
    assert response.status_code == 200, response.text
    return response.content


def _chat_request(model: str, text: str, *, stream: bool, cached: bool) -> Event:
    request: Final[Event] = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": stream,
        "max_tokens": 64,
    }
    return request if cached else {**request, **NO_CACHE}


def _chat(rig: Rig, key: str, model: str, text: str, client: Client = "httpx", *, cached: bool = False) -> Event:
    match client:
        case "httpx":
            return JSON_OBJECT.validate_json(
                _post(rig, "/v1/chat/completions", _chat_request(model, text, stream=False, cached=cached), key)
            )
        case "sdk":
            with openai.OpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                return _dump(
                    sdk.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": text}], max_tokens=64, extra_body=NO_CACHE
                    )
                )
        case "async_sdk":

            async def call() -> Event:
                async with openai.AsyncOpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                    return _dump(
                        await sdk.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": text}],
                            max_tokens=64,
                            extra_body=NO_CACHE,
                        )
                    )

            return asyncio.run(call())


def _chat_stream(
    rig: Rig, key: str, model: str, text: str, client: Client = "httpx", *, cached: bool = False
) -> tuple[Event, ...]:
    match client:
        case "httpx":
            return chunks_of(
                _post(rig, "/v1/chat/completions", _chat_request(model, text, stream=True, cached=cached), key).decode()
            )
        case "sdk":
            with openai.OpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                return tuple(
                    _dump(chunk)
                    for chunk in sdk.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": text}],
                        max_tokens=64,
                        stream=True,
                        extra_body=NO_CACHE,
                    )
                )
        case "async_sdk":

            async def call() -> tuple[Event, ...]:
                async with openai.AsyncOpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                    stream: Final = await sdk.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": text}],
                        max_tokens=64,
                        stream=True,
                        extra_body=NO_CACHE,
                    )
                    return tuple([_dump(chunk) async for chunk in stream])

            return asyncio.run(call())


def _messages_request(model: str, text: str, *, stream: bool, cached: bool) -> Event:
    request: Final[Event] = {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "messages": [{"role": "user", "content": text}],
    }
    return request if cached else {**request, **NO_CACHE}


def _messages(rig: Rig, key: str, model: str, text: str, client: Client = "httpx", *, cached: bool = False) -> Event:
    match client:
        case "httpx":
            return JSON_OBJECT.validate_json(
                _post(rig, "/v1/messages", _messages_request(model, text, stream=False, cached=cached), key)
            )
        case "sdk":
            with anthropic.Anthropic(base_url=_base_url(rig), api_key=key, max_retries=0) as sdk:
                return _dump(
                    sdk.messages.create(
                        model=model, max_tokens=64, messages=[{"role": "user", "content": text}], extra_body=NO_CACHE
                    )
                )
        case "async_sdk":

            async def call() -> Event:
                async with anthropic.AsyncAnthropic(base_url=_base_url(rig), api_key=key, max_retries=0) as sdk:
                    return _dump(
                        await sdk.messages.create(
                            model=model,
                            max_tokens=64,
                            messages=[{"role": "user", "content": text}],
                            extra_body=NO_CACHE,
                        )
                    )

            return asyncio.run(call())


def _messages_stream(
    rig: Rig, key: str, model: str, text: str, client: Client = "httpx", *, cached: bool = False
) -> tuple[Event, ...]:
    match client:
        case "httpx":
            return frames(
                _post(rig, "/v1/messages", _messages_request(model, text, stream=True, cached=cached), key).decode()
            )
        case "sdk":
            with anthropic.Anthropic(base_url=_base_url(rig), api_key=key, max_retries=0) as sdk:
                return tuple(
                    _dump(event)
                    for event in sdk.messages.create(
                        model=model,
                        max_tokens=64,
                        messages=[{"role": "user", "content": text}],
                        stream=True,
                        extra_body=NO_CACHE,
                    )
                )
        case "async_sdk":

            async def call() -> tuple[Event, ...]:
                async with anthropic.AsyncAnthropic(base_url=_base_url(rig), api_key=key, max_retries=0) as sdk:
                    stream: Final = await sdk.messages.create(
                        model=model,
                        max_tokens=64,
                        messages=[{"role": "user", "content": text}],
                        stream=True,
                        extra_body=NO_CACHE,
                    )
                    return tuple([_dump(event) async for event in stream])

            return asyncio.run(call())


def _responses_request(model: str, text: str, *, stream: bool, cached: bool) -> Event:
    request: Final[Event] = {"model": model, "input": text, "max_output_tokens": 64, "stream": stream}
    return request if cached else {**request, **NO_CACHE}


def _responses(rig: Rig, key: str, model: str, text: str, client: Client = "httpx", *, cached: bool = False) -> Event:
    match client:
        case "httpx":
            return JSON_OBJECT.validate_json(
                _post(rig, "/v1/responses", _responses_request(model, text, stream=False, cached=cached), key)
            )
        case "sdk":
            with openai.OpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                return _dump(sdk.responses.create(model=model, input=text, max_output_tokens=64, extra_body=NO_CACHE))
        case "async_sdk":

            async def call() -> Event:
                async with openai.AsyncOpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                    return _dump(
                        await sdk.responses.create(model=model, input=text, max_output_tokens=64, extra_body=NO_CACHE)
                    )

            return asyncio.run(call())


def _responses_stream(rig: Rig, key: str, model: str, text: str, client: Client = "httpx") -> tuple[Event, ...]:
    match client:
        case "httpx":
            return frames(
                _post(rig, "/v1/responses", _responses_request(model, text, stream=True, cached=False), key).decode()
            )
        case "sdk":
            with openai.OpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                return tuple(
                    _dump(event)
                    for event in sdk.responses.create(
                        model=model, input=text, max_output_tokens=64, stream=True, extra_body=NO_CACHE
                    )
                )
        case "async_sdk":

            async def call() -> tuple[Event, ...]:
                async with openai.AsyncOpenAI(base_url=f"{_base_url(rig)}/v1", api_key=key, max_retries=0) as sdk:
                    stream: Final = await sdk.responses.create(
                        model=model, input=text, max_output_tokens=64, stream=True, extra_body=NO_CACHE
                    )
                    return tuple([_dump(event) async for event in stream])

            return asyncio.run(call())


def _relayed_in(events: Sequence[JsonValue]) -> tuple[tuple[Location, Event], ...]:
    return tuple(chain.from_iterable(relayed(event) for event in events))


def _choice(chunk: Mapping[str, JsonValue]) -> Event:
    choices: Final = JSON_LIST.validate_python(chunk.get("choices") or [])
    return JSON_OBJECT.validate_python(choices[0]) if choices else {}


def _finish_reasons(chunks: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    return tuple(_choice(chunk)["finish_reason"] for chunk in chunks if _choice(chunk).get("finish_reason"))


def _streamed_text(chunks: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(
        str(JSON_OBJECT.validate_python(_choice(chunk).get("delta") or {}).get("content") or "") for chunk in chunks
    )


def _message_text(body: Mapping[str, JsonValue]) -> JsonValue:
    return JSON_OBJECT.validate_python(_choice(body)["message"]).get("content")


def _native_stops(events: Sequence[Mapping[str, JsonValue]]) -> tuple[Event, ...]:
    deltas: Final = tuple(
        JSON_OBJECT.validate_python(event["delta"]) for event in events if event.get("type") == "message_delta"
    )
    return tuple({name: delta.get(name) for name in ("stop_reason", "stop_details")} for delta in deltas)


def _native_deltas(events: Sequence[Mapping[str, JsonValue]]) -> tuple[Event, ...]:
    deltas: Final = tuple(
        JSON_OBJECT.validate_python(event["delta"]) for event in events if event.get("type") == "message_delta"
    )
    return tuple({name: delta[name] for name in ("stop_reason", "stop_details") if name in delta} for delta in deltas)


def _targets(wire: Wire) -> tuple[str, ...]:
    return tuple(unquote(request.target) for request in wire.drain())


def _paths(wire: Wire) -> tuple[str, ...]:
    return tuple(urlsplit(target).path for target in _targets(wire))


def _logged(rig: Rig, key: str, count: int = 1) -> tuple[Event, ...]:
    events: Final = rig.sink.await_key(key, count)
    assert len(events) == count, events
    return events


def _logged_response(rig: Rig, key: str) -> JsonValue:
    return _logged(rig, key)[0]["response"]


@pytest.mark.parametrize("client", CLIENTS)
def test_chat_refusal_relays_the_docs_stop_details_to_the_caller_and_the_log(rig: Rig, client: Client) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _chat(rig, key, model, refusal_prompt("refusal", _marker()), client)
        assert relayed(body) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), body
        assert _choice(body)["finish_reason"] == "content_filter", body
        assert _paths(rig.wire) == ("/v1/messages",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("client", CLIENTS)
def test_chat_stream_refusal_relays_the_docs_stop_details_in_one_chunk_and_the_log(rig: Rig, client: Client) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        chunks: Final = _chat_stream(rig, key, model, refusal_prompt("refusal", _marker()), client)
        assert _relayed_in(chunks) == ((_DELTA_FIELDS, REFUSAL_FIELDS),), chunks
        assert _finish_reasons(chunks) == ("content_filter",), chunks
        assert _streamed_text(chunks) == "", chunks
        assert _paths(rig.wire) == ("/v1/messages",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("client", CLIENTS)
def test_messages_refusal_returns_the_docs_body_and_logs_the_stop_details(rig: Rig, client: Client) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _messages(rig, key, model, refusal_prompt("refusal", _marker()), client)
        assert (body["stop_reason"], body["stop_details"], body["content"]) == ("refusal", DOCS_STOP_DETAILS, []), body
        assert relayed(body) == (), body
        assert _paths(rig.wire) == ("/v1/messages",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("client", CLIENTS)
def test_messages_stream_refusal_keeps_the_client_frames_and_logs_the_stop_details(rig: Rig, client: Client) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        events: Final = _messages_stream(rig, key, model, refusal_prompt("refusal", _marker()), client)
        assert frame_types(events) == frame_types(FIXTURES["refusal"].events), events
        assert _native_stops(events) == (REFUSAL_FIELDS,), events
        assert _relayed_in(events) == (), events
        assert _paths(rig.wire) == ("/v1/messages",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("client", CLIENTS)
def test_responses_refusal_relays_the_docs_stop_details_at_the_top_level(rig: Rig, client: Client) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _responses(rig, key, model, refusal_prompt("refusal", _marker()), client)
        assert relayed(body) == ((_RESPONSE_FIELDS, REFUSAL_FIELDS),), body
        assert (body["status"], body["incomplete_details"]) == ("incomplete", {"reason": "content_filter"}), body
        assert _paths(rig.wire) == ("/v1/messages",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("client", CLIENTS)
def test_responses_stream_refusal_relays_the_docs_stop_details_on_the_incomplete_event(
    rig: Rig, client: Client
) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        events: Final = _responses_stream(rig, key, model, refusal_prompt("refusal", _marker()), client)
        assert _relayed_in(events) == ((_STREAMED_RESPONSE_FIELDS, REFUSAL_FIELDS),), events
        assert frame_types(events)[-1] == "response.incomplete", events
        assert _paths(rig.wire) == ("/v1/messages",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_anthropic_passthrough_refusal_relays_upstream_bytes_and_logs_the_stop_details(rig: Rig, stream: bool) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        body: Final = {
            "model": MODEL,
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": refusal_prompt("refusal", _marker())}],
        }
        content: Final = _post(rig, "/anthropic/v1/messages", body, key)
        if stream:
            assert content == b"".join(sse_chunks(refusal_events(DOCS_REFUSAL))), content
        else:
            assert JSON_OBJECT.validate_json(content) == DOCS_REFUSAL, content
        assert _targets(rig.wire) == (f"{_ANTHROPIC_BASE}/v1/messages",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("provider", FAMILY)
def test_family_chat_refusal_relays_the_docs_stop_details(rig: Rig, provider: Provider) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, rig)
        key: Final = scenario.key()
        body: Final = _chat(rig, key, model, refusal_prompt("refusal", _marker()))
        expected: Final = () if provider == "snowflake" else ((_CHAT_FIELDS, REFUSAL_FIELDS),)
        assert relayed(body) == expected, body
        assert _choice(body)["finish_reason"] == ("stop" if provider == "snowflake" else "content_filter"), body
        assert _paths(rig.wire) == (_STREAM_TARGETS[provider][0],)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == expected, logged


@pytest.mark.parametrize("provider", FAMILY)
def test_family_chat_stream_refusal_relays_the_docs_stop_details(rig: Rig, provider: Provider) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, rig)
        key: Final = scenario.key()
        chunks: Final = _chat_stream(rig, key, model, refusal_prompt("refusal", _marker()))
        assert _relayed_in(chunks) == ((_DELTA_FIELDS, REFUSAL_FIELDS),), chunks
        assert _finish_reasons(chunks) == ("content_filter",), chunks
        assert _paths(rig.wire) == (_STREAM_TARGETS[provider][1],)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("provider", FAMILY)
def test_family_messages_stream_refusal_keeps_frames_and_logs_the_stop_details(rig: Rig, provider: Provider) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, rig)
        key: Final = scenario.key()
        events: Final = _messages_stream(rig, key, model, refusal_prompt("refusal", _marker()))
        native: Final = {"stop_reason": "refusal"} if provider == "snowflake" else REFUSAL_FIELDS
        assert _native_deltas(events) == (native,), events
        assert _relayed_in(events) == (), events
        assert _paths(rig.wire) == (_STREAM_TARGETS[provider][1],)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


def _bedrock_passthrough_deployment(scenario: Scenario, rig: Rig) -> str:
    return scenario.model(
        model=f"bedrock/{BEDROCK_MODEL}",
        api_base=rig.wire.url,
        aws_access_key_id="AKIASCRIPTEDPROVIDER",
        aws_secret_access_key="scripted-secret",
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=rig.wire.url,
    )


@pytest.mark.parametrize("stream", [False, True], ids=["invoke", "invoke-with-response-stream"])
def test_bedrock_passthrough_refusal_relays_upstream_bytes_and_logs_the_stop_details(rig: Rig, stream: bool) -> None:
    with rig.gateway.scenario() as scenario:
        deployment: Final = _bedrock_passthrough_deployment(scenario, rig)
        key: Final = scenario.key()
        action: Final = "invoke-with-response-stream" if stream else "invoke"
        body: Final = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": refusal_prompt("refusal", _marker())}],
        }
        content: Final = _post(rig, f"/bedrock/model/{deployment}/{action}", body, key)
        if stream:
            assert content == b"".join(aws_chunks(refusal_events(DOCS_REFUSAL))), content
        else:
            assert JSON_OBJECT.validate_json(content) == DOCS_REFUSAL, content
        assert _paths(rig.wire) == (f"/model/{BEDROCK_MODEL}/{action}",)
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


@pytest.mark.parametrize("stream", [False, True], ids=["rawPredict", "streamRawPredict"])
def test_vertex_passthrough_refusal_relays_upstream_bytes_and_logs_the_stop_details(rig: Rig, stream: bool) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        action: Final = "streamRawPredict" if stream else "rawPredict"
        body: Final = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 64,
            "stream": stream,
            "messages": [{"role": "user", "content": refusal_prompt("refusal", _marker())}],
        }
        content: Final = _post(rig, f"/vertex_ai{_VERTEX_MODEL_PATH}:{action}", body, key)
        if stream:
            assert content == b"".join(sse_chunks(refusal_events(DOCS_REFUSAL))), content
        else:
            assert JSON_OBJECT.validate_json(content) == DOCS_REFUSAL, content
        assert _paths(rig.vertex) == (f"{_VERTEX_MODEL_PATH}:{action}",)
        assert _targets(rig.wire) == ()
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


def _surface_call(rig: Rig, surface: Surface, key: str, model: str, text: str) -> JsonValue:
    match surface:
        case "chat":
            return _chat(rig, key, model, text, cached=True)
        case "chat_stream":
            return list(_chat_stream(rig, key, model, text, cached=True))
        case "messages":
            return _messages(rig, key, model, text, cached=True)
        case "messages_stream":
            return list(_messages_stream(rig, key, model, text, cached=True))
        case "responses":
            return _responses(rig, key, model, text, cached=True)
        case "responses_stream":
            return list(_responses_stream(rig, key, model, text))


@dataclass(frozen=True, slots=True)
class CacheTwin:
    fields: tuple[tuple[Location, Event], ...]
    served_from_cache: tuple[bool, bool]


_CACHE_TWINS: Final[Mapping[Surface, CacheTwin]] = {
    "chat": CacheTwin(((_CHAT_FIELDS, REFUSAL_FIELDS),), (False, True)),
    "chat_stream": CacheTwin(((_DELTA_FIELDS, REFUSAL_FIELDS),), (False, True)),
    "messages": CacheTwin((), (False, False)),
    "messages_stream": CacheTwin((), (False, False)),
}


def _relayed_anywhere(value: JsonValue) -> tuple[tuple[Location, Event], ...]:
    return _relayed_in(JSON_LIST.validate_python(value)) if isinstance(value, list) else relayed(value)


@pytest.mark.parametrize("surface", tuple(_CACHE_TWINS))
def test_cache_hit_twin_carries_the_same_stop_details_as_the_miss(rig: Rig, surface: Surface) -> None:
    twin: Final = _CACHE_TWINS[surface]
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        text: Final = refusal_prompt("refusal", _marker())
        first: Final = _surface_call(rig, surface, key, model, text)
        _logged(rig, key)
        second: Final = _surface_call(rig, surface, key, model, text)
        logged: Final = _logged(rig, key, 2)
        upstream: Final = _paths(rig.wire)
        served_from_cache: Final = tuple(event.get("cache_hit") is True for event in logged)
        evidence: Final = {"cache_hit": served_from_cache, "upstream": upstream, "first": first, "second": second}
        assert (served_from_cache, len(upstream)) == (twin.served_from_cache, twin.served_from_cache.count(False)), (
            evidence
        )
        assert (_relayed_anywhere(first), _relayed_anywhere(second)) == (twin.fields, twin.fields), evidence
        assert tuple(relayed(event["response"]) for event in logged) == (((_CHAT_FIELDS, REFUSAL_FIELDS),),) * 2, (
            served_from_cache,
            logged,
        )


def test_responses_cache_hit_logs_the_same_stop_details_as_the_miss(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        text: Final = refusal_prompt("refusal", _marker())
        first: Final = _surface_call(rig, "responses", key, model, text)
        _logged(rig, key)
        _surface_call(rig, "responses", key, model, text)
        logged: Final = _logged(rig, key, 2)
        served_from_cache: Final = tuple(event.get("cache_hit") is True for event in logged)
        assert (len(_paths(rig.wire)), served_from_cache.count(True)) == (1, 1), (served_from_cache, logged)
        assert relayed(first) == ((_RESPONSE_FIELDS, REFUSAL_FIELDS),), first
        assert tuple(relayed(event["response"]) for event in logged) == (((_CHAT_FIELDS, REFUSAL_FIELDS),),) * 2, (
            served_from_cache,
            logged,
        )


def test_passthrough_is_not_served_from_the_response_cache(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        body: Final = {
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": refusal_prompt("refusal", _marker())}],
        }
        first: Final = _post(rig, "/anthropic/v1/messages", body, key)
        second: Final = _post(rig, "/anthropic/v1/messages", body, key)
        assert JSON_OBJECT.validate_json(first) == JSON_OBJECT.validate_json(second) == DOCS_REFUSAL
        assert _targets(rig.wire) == (f"{_ANTHROPIC_BASE}/v1/messages",) * 2
        logged: Final = _logged(rig, key, 2)
        assert tuple(relayed(event["response"]) for event in logged) == (((_CHAT_FIELDS, REFUSAL_FIELDS),),) * 2


@pytest.mark.parametrize("variant", tuple(MALFORMED_STOP_DETAILS))
def test_malformed_stop_details_are_relayed_as_sent_without_breaking_the_turn(rig: Rig, variant: str) -> None:
    value: Final = MALFORMED_STOP_DETAILS[variant]
    expected: Final[Event] = {"stop_reason": "end_turn", "stop_details": value}
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _chat(rig, key, model, refusal_prompt(variant, _marker()))
        assert relayed(body) == ((_CHAT_FIELDS, expected),), body
        assert (_choice(body)["finish_reason"], _message_text(body)) == ("stop", "ready"), body
        chunks: Final = _chat_stream(rig, key, model, refusal_prompt(variant, _marker()))
        assert _relayed_in(chunks) == ((_DELTA_FIELDS, expected),), chunks
        assert (_finish_reasons(chunks), _streamed_text(chunks)) == (("stop",), "ready"), chunks
        assert len(_paths(rig.wire)) == 2
        logged: Final = _logged(rig, key, 2)
        assert tuple(relayed(event["response"]) for event in logged) == (((_CHAT_FIELDS, expected),),) * 2, logged
        assert rig.gateway.client.get("/health/liveliness").status_code == 200


def test_duplicated_stop_details_key_relays_the_docs_object_once(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _chat(rig, key, model, refusal_prompt("duplicate", _marker()))
        assert relayed(body) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), body
        logged: Final = _logged_response(rig, key)
        assert relayed(logged) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged


def test_non_object_upstream_body_fails_the_call_and_leaves_other_keys_serving(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        broken_key: Final = scenario.key()
        healthy_key: Final = scenario.key()
        broken: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_request(model, refusal_prompt("nondict", _marker()), stream=False, cached=False),
            key=broken_key,
        )
        assert broken.status_code == 500, broken.text
        assert "error" in JSON_OBJECT.validate_json(broken.content), broken.text
        healthy: Final = _chat(rig, healthy_key, model, refusal_prompt("refusal", _marker()))
        assert relayed(healthy) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), healthy
        assert len(_paths(rig.wire)) == 2
        failure: Final = _logged(rig, broken_key)[0]
        assert (failure["status"], relayed(failure["response"])) == ("failure", ()), failure


@pytest.mark.parametrize(
    ("fixture", "status", "message"),
    [("unauthorized", 401, "invalid x-api-key"), ("invalid", 400, "max_tokens: Field required")],
)
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_provider_errors_reach_the_caller_and_log_a_failure_without_stop_fields(
    rig: Rig, fixture: str, status: int, message: str, stream: bool
) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_request(model, refusal_prompt(fixture, _marker()), stream=stream, cached=False),
            key=key,
        )
        assert response.status_code == status, response.text
        assert message in response.text, response.text
        assert _paths(rig.wire) == ("/v1/messages",)
        failure: Final = _logged(rig, key)[0]
        assert (failure["status"], relayed(failure["response"])) == ("failure", ()), failure


def test_failing_sinks_get_one_attempt_each_and_the_proxy_stays_healthy(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _chat(rig, key, model, refusal_prompt("refusal", _marker()))
        assert relayed(body) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), body
        for ledger in (rig.sink, rig.forbidden, rig.absent):
            events = ledger.await_key(key)
            assert tuple(relayed(event["response"]) for event in events) == (((_CHAT_FIELDS, REFUSAL_FIELDS),),)
        held(
            lambda: tuple(len(ledger.for_key(key)) for ledger in (rig.sink, rig.forbidden, rig.absent)),
            lambda counts: counts == (1, 1, 1),
            holding=3,
        )
        readiness: Final = rig.gateway.client.get("/health/readiness")
        assert (readiness.status_code, readiness.json()["status"]) == (200, "healthy"), readiness.text


def test_unauthenticated_refusal_request_never_reaches_the_upstream_or_the_sink(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        stranger: Final = "sk-not-a-virtual-key-" + _marker()
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_request(model, refusal_prompt("refusal", _marker()), stream=False, cached=False),
            key=stranger,
        )
        assert response.status_code == 401, response.text
        assert _targets(rig.wire) == ()
        assert rig.sink.for_key(stranger) == ()


@pytest.mark.parametrize("fixture", ["null", "missing"])
def test_normal_turn_adds_no_stop_fields_on_any_surface(rig: Rig, fixture: str) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _chat(rig, key, model, refusal_prompt(fixture, _marker()))
        assert relayed(body) == (), body
        assert (_choice(body)["finish_reason"], _message_text(body)) == ("stop", "ready"), body
        chunks: Final = _chat_stream(rig, key, model, refusal_prompt(fixture, _marker()))
        assert _relayed_in(chunks) == (), chunks
        assert (_finish_reasons(chunks), _streamed_text(chunks)) == (("stop",), "ready"), chunks
        events: Final = _messages_stream(rig, key, model, refusal_prompt(fixture, _marker()))
        expected_native: Final = {"stop_reason": "end_turn", "stop_details": None}
        assert _native_stops(events) == (expected_native,), events
        assert len(_paths(rig.wire)) == 3
        logged: Final = _logged(rig, key, 3)
        assert tuple(relayed(event["response"]) for event in logged) == ((), (), ()), logged


def test_refusal_after_partial_text_keeps_the_text_and_relays_the_stop_details(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        chat_key: Final = scenario.key()
        messages_key: Final = scenario.key()
        chunks: Final = _chat_stream(rig, chat_key, model, refusal_prompt("partial", _marker()))
        assert _streamed_text(chunks) == "Hello..", chunks
        assert _relayed_in(chunks) == ((_DELTA_FIELDS, REFUSAL_FIELDS),), chunks
        assert _finish_reasons(chunks) == ("content_filter",), chunks
        logged_chat: Final = _logged_response(rig, chat_key)
        assert relayed(logged_chat) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged_chat
        assert _message_text(JSON_OBJECT.validate_python(logged_chat)) == "Hello..", logged_chat
        events: Final = _messages_stream(rig, messages_key, model, refusal_prompt("partial", _marker()))
        assert _native_stops(events) == (REFUSAL_FIELDS,), events
        assert frame_types(events) == frame_types(FIXTURES["partial"].events), events
        logged_messages: Final = _logged_response(rig, messages_key)
        assert relayed(logged_messages) == ((_CHAT_FIELDS, REFUSAL_FIELDS),), logged_messages
        assert len(_paths(rig.wire)) == 2


def test_repeated_identical_refusals_log_once_per_request(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        text: Final = refusal_prompt("refusal", _marker())
        bodies: Final = (_chat(rig, key, model, text), _chat(rig, key, model, text))
        assert tuple(relayed(body) for body in bodies) == (((_CHAT_FIELDS, REFUSAL_FIELDS),),) * 2, bodies
        assert len(_paths(rig.wire)) == 2
        logged: Final = _logged(rig, key, 2)
        assert len({event["litellm_call_id"] for event in logged}) == 2, logged
        assert tuple(relayed(event["response"]) for event in logged) == (((_CHAT_FIELDS, REFUSAL_FIELDS),),) * 2


@pytest.mark.parametrize("fixture", ["refusal", "int", "null"])
def test_chat_stream_logged_response_keeps_the_received_model_id(rig: Rig, fixture: str) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        chunks: Final = _chat_stream(rig, key, model, refusal_prompt(fixture, _marker()))
        assert len({chunk["id"] for chunk in chunks}) == 1, chunks
        probed: Final = rig.hidden.await_key(key)
        assert len(probed) == 1, probed
        assert (probed[0]["response_id"], probed[0]["received_model_id"]) == (chunks[0]["id"], chunks[0]["id"]), probed


def _delta_fields(chunks: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    deltas: Final = tuple(JSON_OBJECT.validate_python(_choice(chunk).get("delta") or {}) for chunk in chunks)
    return tuple(
        delta["provider_specific_fields"] for delta in deltas if delta.get("provider_specific_fields") is not None
    )


def _message_fields(response: JsonValue) -> tuple[JsonValue, ...]:
    message: Final = JSON_OBJECT.validate_python(_choice(JSON_OBJECT.validate_python(response))["message"])
    fields: Final = message.get("provider_specific_fields")
    return () if fields is None else (fields,)


def test_openai_compatible_stream_logged_response_keeps_the_received_model_id(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai_compatible", rig)
        key: Final = scenario.key()
        marker: Final = _marker()
        chunks: Final = _chat_stream(rig, key, model, refusal_prompt("plain", marker))
        assert (_streamed_text(chunks), _finish_reasons(chunks), _delta_fields(chunks)) == ("ready now", ("stop",), ())
        assert {chunk["id"] for chunk in chunks} == {openai_id(marker)}, chunks
        assert _paths(rig.wire) == ("/v1/chat/completions",)
        probed: Final = rig.hidden.await_key(key)
        assert len(probed) == 1, probed
        assert (probed[0]["response_id"], probed[0]["received_model_id"]) == (openai_id(marker),) * 2, probed


@pytest.mark.parametrize(("fixture", "fields"), [("plain", ()), ("cited", (CITED_FIELDS,))], ids=["plain", "cited"])
def test_openai_compatible_stream_cache_hit_replays_the_provider_fields_of_the_miss(
    rig: Rig, fixture: str, fields: tuple[JsonValue, ...]
) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai_compatible", rig)
        key: Final = scenario.key()
        text: Final = refusal_prompt(fixture, _marker())
        first: Final = _chat_stream(rig, key, model, text, cached=True)
        _logged(rig, key)
        second: Final = _chat_stream(rig, key, model, text, cached=True)
        logged: Final = _logged(rig, key, 2)
        served_from_cache: Final = tuple(event.get("cache_hit") is True for event in logged)
        evidence: Final = {"cache_hit": served_from_cache, "first": first, "second": second, "logged": logged}
        assert (served_from_cache, _paths(rig.wire)) == ((False, True), ("/v1/chat/completions",)), evidence
        streamed: Final = tuple(
            (_streamed_text(chunks), _finish_reasons(chunks), _delta_fields(chunks)) for chunks in (first, second)
        )
        assert streamed == (("ready now", ("stop",), fields),) * 2, evidence
        assert tuple(_message_fields(event["response"]) for event in logged) == (fields, fields), evidence


def _spend_rows(rig: Rig, key: str) -> tuple[Event, ...]:
    response: Final = rig.gateway.request("GET", "/spend/logs", params={"api_key": key})
    assert response.status_code == 200, response.text
    return tuple(JSON_OBJECT.validate_python(row) for row in JSON_LIST.validate_json(response.content))


def test_spend_logs_endpoint_reports_the_stop_details_of_a_refusal(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        body: Final = _chat(rig, key, model, refusal_prompt("refusal", _marker()))
        rows: Final = eventually(lambda: _spend_rows(rig, key), lambda found: len(found) >= 1, seconds=70)
        assert len(rows) == 1, rows
        reported: Final = (rows[0]["request_id"], relayed(rows[0]["response"]))
        expected: Final = (body["id"], ((_CHAT_FIELDS, REFUSAL_FIELDS),))
        assert ((body["id"], relayed(body)), reported) == (expected, expected), rows


def _usage_counts(usage: JsonValue, total: str, details: str) -> tuple[JsonValue, JsonValue]:
    counted: Final = JSON_OBJECT.validate_python(usage)
    return (counted[total], JSON_OBJECT.validate_python(counted[details])["reasoning_tokens"])


def _chat_usage_counts(chunks: Sequence[Mapping[str, JsonValue]]) -> tuple[tuple[JsonValue, JsonValue], ...]:
    return tuple(
        _usage_counts(chunk["usage"], "completion_tokens", "completion_tokens_details")
        for chunk in chunks
        if chunk.get("usage")
    )


def _cached_stream_usage_counts(
    rig: Rig, surface: Literal["chat_stream", "responses_stream"], key: str, model: str, text: str
) -> tuple[tuple[JsonValue, JsonValue], ...]:
    match surface:
        case "chat_stream":
            request: Final = {
                **_chat_request(model, text, stream=True, cached=True),
                "stream_options": {"include_usage": True},
            }
            return _chat_usage_counts(chunks_of(_post(rig, "/v1/chat/completions", request, key).decode()))
        case "responses_stream":
            events: Final = frames(
                _post(rig, "/v1/responses", _responses_request(model, text, stream=True, cached=True), key).decode()
            )
            responses: Final = tuple(
                JSON_OBJECT.validate_python(event["response"])
                for event in events
                if isinstance(event.get("response"), dict)
            )
            return tuple(
                _usage_counts(response["usage"], "output_tokens", "output_tokens_details")
                for response in responses
                if response.get("usage")
            )


def _logged_usage_counts(logged: Sequence[Mapping[str, JsonValue]]) -> tuple[tuple[JsonValue, JsonValue], ...]:
    return tuple(
        _usage_counts(
            JSON_OBJECT.validate_python(event["response"])["usage"], "completion_tokens", "completion_tokens_details"
        )
        for event in logged
    )


@pytest.mark.parametrize("fixture", ["plainusage", "citedusage"], ids=["plain", "cited"])
def test_openai_compatible_stream_cache_hit_replays_the_usage_details_of_the_miss(rig: Rig, fixture: str) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai_compatible", rig)
        key: Final = scenario.key()
        text: Final = refusal_prompt(fixture, _marker())
        first: Final = _cached_stream_usage_counts(rig, "chat_stream", key, model, text)
        _logged(rig, key)
        second: Final = _cached_stream_usage_counts(rig, "chat_stream", key, model, text)
        logged: Final = _logged(rig, key, 2)
        served_from_cache: Final = tuple(event.get("cache_hit") is True for event in logged)
        evidence: Final = {"cache_hit": served_from_cache, "first": first, "second": second, "logged": logged}
        assert (served_from_cache, _paths(rig.wire)) == ((False, True), ("/v1/chat/completions",)), evidence
        assert (first, second, _logged_usage_counts(logged)) == (((7, 4),), ((7, 4),), ((7, 4), (7, 4))), evidence


def test_text_completion_stream_cache_hit_replays_the_usage_details_of_the_miss(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai_compatible", rig)
        key: Final = scenario.key()
        request: Final[Event] = {
            "model": model,
            "prompt": refusal_prompt("plainusage", _marker()),
            "max_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first: Final = _chat_usage_counts(chunks_of(_post(rig, "/v1/completions", request, key).decode()))
        _logged(rig, key)
        second: Final = _chat_usage_counts(chunks_of(_post(rig, "/v1/completions", request, key).decode()))
        logged: Final = _logged(rig, key, 2)
        served_from_cache: Final = tuple(event.get("cache_hit") is True for event in logged)
        evidence: Final = {"cache_hit": served_from_cache, "first": first, "second": second, "logged": logged}
        assert (served_from_cache, _paths(rig.wire)) == ((False, True), ("/v1/chat/completions",)), evidence
        assert (first, second, _logged_usage_counts(logged)) == (((7, 4),), ((7, 4),), ((7, 4), (7, 4))), evidence


@pytest.mark.parametrize("surface", ["chat_stream", "responses_stream"])
def test_anthropic_stream_cache_hit_keeps_the_usage_details_of_the_miss(
    rig: Rig, surface: Literal["chat_stream", "responses_stream"]
) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", rig)
        key: Final = scenario.key()
        text: Final = refusal_prompt("reasoned", _marker())
        first: Final = _cached_stream_usage_counts(rig, surface, key, model, text)
        _logged(rig, key)
        second: Final = _cached_stream_usage_counts(rig, surface, key, model, text)
        logged: Final = _logged(rig, key, 2)
        served_from_cache: Final = tuple(event.get("cache_hit") is True for event in logged)
        evidence: Final = {"cache_hit": served_from_cache, "first": first, "second": second, "logged": logged}
        assert (served_from_cache, _paths(rig.wire)) == ((False, True), ("/v1/messages",)), evidence
        assert (first, second, _logged_usage_counts(logged)) == (((10, 4),), ((10, 4),), ((10, 4), (10, 4))), evidence


def _response_id(event: Mapping[str, JsonValue]) -> JsonValue:
    return JSON_OBJECT.validate_python(event["response"]).get("id")


@pytest.mark.parametrize("variant", tuple(MALFORMED_STOP_DETAILS))
def test_stream_cache_hit_skips_cached_provider_fields_that_are_not_an_object(rig: Rig, variant: str) -> None:
    with rig.gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "openai_compatible", rig)
        key: Final = scenario.key()
        marker: Final = _marker()
        text: Final = refusal_prompt("plain", marker)
        filled: Final = chunks_of(
            _post(
                rig,
                "/v1/chat/completions",
                {**_chat_request(model, text, stream=True, cached=True), "cache": {"ttl": 2}},
                key,
            ).decode()
        )
        assert _streamed_text(filled) == "ready now", filled
        _logged(rig, key)
        stale_id: Final = rewrite_cached_message_fields(openai_id(marker), MALFORMED_STOP_DETAILS[variant])
        replayed: Final = eventually(
            lambda: _chat_stream(rig, key, model, text, cached=True),
            lambda chunks: {chunk["id"] for chunk in chunks} == {stale_id},
            seconds=30,
        )
        streamed: Final = (_streamed_text(replayed), _finish_reasons(replayed), _delta_fields(replayed))
        assert streamed == ("ready now", ("stop",), ()), replayed
        assert _paths(rig.wire) == ("/v1/chat/completions",)
        stale_logs: Final = eventually(
            lambda: tuple(event for event in rig.sink.for_key(key) if _response_id(event) == stale_id),
            lambda found: len(found) >= 1,
        )
        assert tuple(_message_fields(event["response"]) for event in stale_logs) == ((),) * len(stale_logs), stale_logs
