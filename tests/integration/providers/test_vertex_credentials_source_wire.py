from __future__ import annotations

import base64
import itertools
import json
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, Scenario, list_value, object_value, string_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

CHAT_BACKEND: Final = "gemini-3.7-flash"
EMBED_BACKEND: Final = "gemini-embedding-2-preview"
PROJECT: Final = "scripted-project"
LOCATION: Final = "us-central1"
MODELS_PATH: Final = f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models"
VALUES: Final = [0.5, 0.25]
FILE_MESSAGE: Final = "Unable to load the vertex credentials file. LiteLLM's log names the path and the reason."
VALUE_ADVICE: Final = "Set it to a readable file path, or to the credentials JSON itself."
PROXY_ROOT: Final = Path(__file__).resolve().parents[3]
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
Endpoint = Literal["chat", "chat_stream", "embeddings", "messages", "responses"]
ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "chat_stream", "embeddings", "messages", "responses")
BURST_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "chat_stream", "embeddings")


def prompt_text(body: dict[str, JsonValue]) -> str:
    content: Final = body["content"] if "content" in body else list_value(body["contents"])[-1]
    return string_value(object_value(list_value(object_value(content)["parts"])[0])["text"])


def gemini_reply(text: str) -> dict[str, JsonValue]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        "modelVersion": CHAT_BACKEND,
    }


def vertex_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.headers["authorization"] == "Bearer scripted-token", request.headers
    echo: Final = f"echo {prompt_text(JSON_OBJECT.validate_json(request.body))}"
    if request.target == f"{MODELS_PATH}/{EMBED_BACKEND}:embedContent":
        return Reply(body=json.dumps({"embedding": {"values": VALUES}}).encode())
    if request.target == f"{MODELS_PATH}/{CHAT_BACKEND}:streamGenerateContent?alt=sse":
        return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(gemini_reply(echo))}\n\n".encode(),))
    assert request.target == f"{MODELS_PATH}/{CHAT_BACKEND}:generateContent", request.target
    return Reply(body=json.dumps(gemini_reply(echo)).encode())


@dataclass(frozen=True, slots=True)
class Deployments:
    chat: str
    embeddings: str


def register(scenario: Scenario, wire: Wire, credentials: JsonValue) -> Deployments:
    return Deployments(
        chat=deployment(scenario, wire, credentials, CHAT_BACKEND),
        embeddings=deployment(scenario, wire, credentials, EMBED_BACKEND),
    )


def deployment(scenario: Scenario, wire: Wire, credentials: JsonValue, backend: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_key=None,
        api_base=wire.url,
        vertex_project=PROJECT,
        vertex_location=LOCATION,
        vertex_credentials=credentials,
    )


def body_of(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def stream_text(response: httpx.Response) -> str:
    lines: Final = tuple(line for line in response.text.splitlines() if line.startswith("data: "))
    assert lines[-1] == "data: [DONE]", response.text
    chunks: Final = tuple(JSON_OBJECT.validate_json(line.removeprefix("data: ")) for line in lines[:-1])
    deltas: Final = (
        object_value(object_value(list_value(chunk["choices"])[0])["delta"]) for chunk in chunks if chunk["choices"]
    )
    return "".join(string_value(delta["content"]) for delta in deltas if delta.get("content"))


def responses_text(body: dict[str, JsonValue]) -> str:
    messages: Final = tuple(object_value(item) for item in list_value(body["output"]))
    contents: Final = tuple(list_value(message["content"]) for message in messages if message["type"] == "message")
    parts: Final = tuple(object_value(part) for part in itertools.chain.from_iterable(contents))
    return "".join(string_value(part["text"]) for part in parts if part["type"] == "output_text")


def call(gateway: Gateway, endpoint: Endpoint, deployments: Deployments, text: str) -> httpx.Response:
    match endpoint:
        case "chat":
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": deployments.chat, "messages": [{"role": "user", "content": text}]},
            )
        case "chat_stream":
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": deployments.chat, "messages": [{"role": "user", "content": text}], "stream": True},
            )
        case "embeddings":
            return gateway.request("POST", "/v1/embeddings", {"model": deployments.embeddings, "input": [text]})
        case "messages":
            return gateway.request(
                "POST",
                "/v1/messages",
                {"model": deployments.chat, "max_tokens": 64, "messages": [{"role": "user", "content": text}]},
            )
        case "responses":
            return gateway.request("POST", "/v1/responses", {"model": deployments.chat, "input": text})
        case _:
            assert_never(endpoint)


def answered(endpoint: Endpoint, response: httpx.Response, text: str) -> None:
    assert response.status_code == 200, response.text
    match endpoint:
        case "chat":
            assert (
                object_value(object_value(list_value(body_of(response)["choices"])[0])["message"])["content"]
                == f"echo {text}"
            )
        case "chat_stream":
            assert stream_text(response) == f"echo {text}"
        case "embeddings":
            assert object_value(list_value(body_of(response)["data"])[0])["embedding"] == VALUES
        case "messages":
            assert object_value(list_value(body_of(response)["content"])[0])["text"] == f"echo {text}"
        case "responses":
            assert responses_text(body_of(response)) == f"echo {text}"
        case _:
            assert_never(endpoint)


def provider_saw(wire: Wire, text: str) -> None:
    received: Final = wire.drain()
    assert [prompt_text(JSON_OBJECT.validate_json(item.body)) for item in received] == [text], received


@contextmanager
def proxy_cwd_file(contents: str) -> Generator[str]:
    name: Final = f"{{vertex-{uuid.uuid4().hex}}}.json"
    path: Final = PROXY_ROOT / name
    path.write_text(contents)
    try:
        yield name
    finally:
        path.unlink()


def written(directory: Path, name: str, contents: str | bytes) -> str:
    path: Final = directory / name
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    else:
        path.write_text(contents)
    return str(path)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("form", ("inline", "file", "brace_named_file"))
def test_loadable_credentials_reach_vertex_on_every_endpoint(
    gateway: Gateway, tmp_path: Path, form: str, endpoint: Endpoint
) -> None:
    key: Final = service_account_json(PROJECT, gateway.upstream_url.rstrip("/"))
    text: Final = f"loadable {form} {endpoint} {uuid.uuid4().hex}"
    with wire_server(vertex_peer) as wire, gateway.scenario() as scenario, proxy_cwd_file(key) as brace_name:
        credentials: Final = {
            "inline": key,
            "file": written(tmp_path, "key.json", key),
            "brace_named_file": brace_name,
        }[form]
        deployments: Final = register(scenario, wire, credentials)
        answered(endpoint, call(gateway, endpoint, deployments, text), text)
        provider_saw(wire, text)


@dataclass(frozen=True, slots=True)
class Unloadable:
    name: str
    value: Callable[[Path, str], JsonValue]
    message: str


def _missing(directory: Path, _: str) -> JsonValue:
    return str(directory / "missing.json")


def _directory(directory: Path, _: str) -> JsonValue:
    return str(directory)


def _invalid_json_file(directory: Path, _: str) -> JsonValue:
    return written(directory, "truncated.json", '{"type": "service_account",')


def _array_file(directory: Path, _: str) -> JsonValue:
    return written(directory, "array.json", "[]")


def _latin1_file(directory: Path, _: str) -> JsonValue:
    return written(directory, "latin1.json", b'{"type": "\xe9"}')


def _oversized_file(directory: Path, _: str) -> JsonValue:
    return written(directory, "huge.json", json.dumps({"type": "service_account", "pad": "a" * (1 << 20)}))


def _incomplete_key_file(directory: Path, _: str) -> JsonValue:
    return written(directory, "incomplete.json", json.dumps({"type": "service_account", "project_id": PROJECT}))


def _quoted_path(directory: Path, token_url: str) -> JsonValue:
    return f'"{written(directory, "key.json", service_account_json(PROJECT, token_url))}"'


def _long_value(_: Path, __: str) -> JsonValue:
    return "x" * 5000


def _truncated_inline(_: Path, __: str) -> JsonValue:
    return '{"type": "service_account", "private_key": "-----BEGIN'


def _base64_inline(_: Path, token_url: str) -> JsonValue:
    return base64.b64encode(service_account_json(PROJECT, token_url).encode()).decode()


def _bom_inline(_: Path, token_url: str) -> JsonValue:
    return "﻿" + service_account_json(PROJECT, token_url)


def _blank(_: Path, __: str) -> JsonValue:
    return "   "


UNLOADABLE: Final = (
    Unloadable("missing_file", _missing, FILE_MESSAGE),
    Unloadable("directory", _directory, FILE_MESSAGE),
    Unloadable("invalid_json_file", _invalid_json_file, FILE_MESSAGE),
    Unloadable("array_file", _array_file, FILE_MESSAGE),
    Unloadable("latin1_file", _latin1_file, FILE_MESSAGE),
    Unloadable("oversized_file", _oversized_file, FILE_MESSAGE),
    Unloadable("incomplete_key_file", _incomplete_key_file, FILE_MESSAGE),
    Unloadable("quoted_path", _quoted_path, FILE_MESSAGE),
    Unloadable("long_value", _long_value, FILE_MESSAGE),
    Unloadable(
        "truncated_inline",
        _truncated_inline,
        "The inline `vertex_credentials` value is not valid JSON: ",
    ),
    Unloadable("base64_inline", _base64_inline, f"value is base64-encoded JSON, not the JSON itself. {VALUE_ADVICE}"),
    Unloadable("bom_inline", _bom_inline, "Remove the quotes or byte-order mark before the opening `{`."),
    Unloadable("blank", _blank, f"The `vertex_credentials` value is empty. {VALUE_ADVICE}"),
)


def refused(gateway: Gateway, wire: Wire, response: httpx.Response, message: str, hidden: str) -> None:
    assert response.status_code == 500, response.text
    assert message in response.text, response.text
    assert hidden not in response.text, response.text
    assert wire.drain() == (), "a request with unloadable credentials reached the provider"
    assert gateway.client.get("/health/liveliness").status_code == 200


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("case", UNLOADABLE, ids=tuple(case.name for case in UNLOADABLE))
def test_unloadable_credentials_answer_the_loader_message_without_the_path(
    gateway: Gateway, tmp_path: Path, case: Unloadable, endpoint: Endpoint
) -> None:
    with wire_server(vertex_peer) as wire, gateway.scenario() as scenario:
        deployments: Final = register(scenario, wire, case.value(tmp_path, gateway.upstream_url.rstrip("/")))
        response: Final = call(gateway, endpoint, deployments, f"unloadable {case.name} {uuid.uuid4().hex}")
        refused(gateway, wire, response, case.message, str(tmp_path))


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_unreadable_brace_named_file_answers_the_file_message(gateway: Gateway, endpoint: Endpoint) -> None:
    key: Final = service_account_json(PROJECT, gateway.upstream_url.rstrip("/"))
    with wire_server(vertex_peer) as wire, gateway.scenario() as scenario, proxy_cwd_file(key) as name:
        (PROXY_ROOT / name).chmod(0)
        deployments: Final = register(scenario, wire, name)
        response: Final = call(gateway, endpoint, deployments, f"unreadable {uuid.uuid4().hex}")
        refused(gateway, wire, response, FILE_MESSAGE, name)


def test_a_key_file_created_after_a_failure_is_used_on_the_next_request(gateway: Gateway, tmp_path: Path) -> None:
    path: Final = tmp_path / "late.json"
    with wire_server(vertex_peer) as wire, gateway.scenario() as scenario:
        deployments: Final = register(scenario, wire, str(path))
        refused(gateway, wire, call(gateway, "chat", deployments, "before the file"), FILE_MESSAGE, str(path))
        path.write_text(service_account_json(PROJECT, gateway.upstream_url.rstrip("/")))
        answered("chat", call(gateway, "chat", deployments, "after the file"), "after the file")
        provider_saw(wire, "after the file")


def test_repeated_failures_answer_identically_and_leave_other_deployments_working(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(vertex_peer) as wire, gateway.scenario() as scenario:
        broken: Final = register(scenario, wire, str(tmp_path / "missing.json"))
        working: Final = register(scenario, wire, service_account_json(PROJECT, gateway.upstream_url.rstrip("/")))
        first: Final = call(gateway, "chat", broken, "same request")
        second: Final = call(gateway, "chat", broken, "same request")
        refused(gateway, wire, first, FILE_MESSAGE, str(tmp_path))
        refused(gateway, wire, second, FILE_MESSAGE, str(tmp_path))
        assert body_of(first)["error"] == body_of(second)["error"]
        answered("chat", call(gateway, "chat", working, "still working"), "still working")
        provider_saw(wire, "still working")


@pytest.mark.parametrize("value", (5, ["a"]), ids=("int", "list"))
def test_non_string_credentials_are_refused_at_registration_without_a_provider_call(
    gateway: Gateway, value: JsonValue
) -> None:
    with wire_server(vertex_peer) as wire:
        response: Final = gateway.request(
            "POST",
            "/model/new",
            {
                "model_name": f"integration-{uuid.uuid4().hex}",
                "litellm_params": {
                    "model": f"vertex_ai/{CHAT_BACKEND}",
                    "api_base": wire.url,
                    "vertex_project": PROJECT,
                    "vertex_location": LOCATION,
                    "vertex_credentials": value,
                },
            },
        )
        assert response.status_code == 422, response.text
        assert "vertex_credentials" in response.text, response.text
        assert wire.drain() == ()
        assert gateway.client.get("/health/liveliness").status_code == 200


@dataclass(frozen=True, slots=True)
class Shot:
    healthy: bool
    endpoint: Endpoint
    text: str


def test_concurrent_mixed_traffic_keeps_each_answer_on_its_own_deployment(gateway: Gateway, tmp_path: Path) -> None:
    key_path: Final = written(tmp_path, "key.json", service_account_json(PROJECT, gateway.upstream_url.rstrip("/")))
    with wire_server(vertex_peer) as wire, gateway.scenario() as scenario:
        good: Final = register(scenario, wire, key_path)
        bad: Final = register(scenario, wire, str(tmp_path / "missing.json"))
        plan: Final = tuple(
            Shot(index % 2 == 0, BURST_ENDPOINTS[index % 3], f"burst {index} {uuid.uuid4().hex}") for index in range(24)
        )

        def fire(shot: Shot) -> httpx.Response:
            return call(gateway, shot.endpoint, good if shot.healthy else bad, shot.text)

        with ThreadPoolExecutor(max_workers=12) as pool:
            responses: Final = tuple(pool.map(fire, plan))
        for shot, response in zip(plan, responses, strict=True):
            if shot.healthy:
                answered(shot.endpoint, response, shot.text)
            else:
                assert response.status_code == 500 and FILE_MESSAGE in response.text, response.text
        received: Final = sorted(prompt_text(JSON_OBJECT.validate_json(item.body)) for item in wire.drain())
        assert received == sorted(shot.text for shot in plan if shot.healthy)
        assert gateway.client.get("/health/liveliness").status_code == 200


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_rust_messages_route_answers_the_file_message_and_loads_a_key_file(gateway: Gateway, tmp_path: Path) -> None:
    key_path: Final = written(tmp_path, "key.json", service_account_json(PROJECT, gateway.upstream_url.rstrip("/")))
    (tmp_path / "rust").mkdir()
    with (
        wire_server(vertex_peer) as wire,
        owned_proxy(gateway, tmp_path / "rust", {"LITELLM_RUST": "true"}, workers=2) as candidate,
        candidate.scenario() as scenario,
    ):
        good: Final = register(scenario, wire, key_path)
        bad: Final = register(scenario, wire, str(tmp_path / "missing.json"))
        answered("messages", call(candidate, "messages", good, "rust loads the file"), "rust loads the file")
        provider_saw(wire, "rust loads the file")
        refused(candidate, wire, call(candidate, "messages", bad, "rust refuses"), FILE_MESSAGE, str(tmp_path))
