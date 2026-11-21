"""Shared rig for the December 1 Google Vertex AI contracts, round 3 (`test_dec1_vertex3_*`).

Round 1 (`dec1_vertex_support.py`) gives one owned TCP peer that is both the Vertex model host (through the
deployment's `api_base`) and the OAuth token endpoint (through the generated service account's `token_uri`).
Round 3 reuses that rig unchanged and adds what it lacks:

- OpenAI shaped replies, because Vertex Model-as-a-Service partner models (Llama, DeepSeek, Qwen, gpt-oss,
  MiniMax, Kimi, GLM), Mistral and AI21 rawPredict, and Model Garden endpoints all answer in the OpenAI chat
  completion wire format.
- An "open" peer that records every request, GET included. The round 1 peer only keeps POSTs, which hides the
  `cachedContents` list call and its `pageToken` paging.
- Small JSON readers shared by the six files.

Nothing here ever builds a URL on a Google host: every deployment and vector store carries an `api_base` on the
peer, and every token mint is diverted by `token_uri`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, list_value, object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_vertex_support import (
    JSON_OBJECT,
    PROJECT,
    TOKEN_PATH,
    TokenPlan,
    body_of,
    path_of,
    service_account_text,
)
from pydantic import JsonValue

LLAMA: Final = "meta/llama-3.3-70b-instruct-maas"
DEEPSEEK: Final = "deepseek-ai/deepseek-r1-0528-maas"
DEEPSEEK_GLOBAL_ONLY: Final = "deepseek-ai/deepseek-v3.2-maas"
"""`supported_regions: ["global"]` in the cost map, so any other location is rerouted to `global`."""
QWEN_GLOBAL_ONLY: Final = "qwen/qwen3-coder-480b-a35b-instruct-maas"
GPT_OSS: Final = "openai/gpt-oss-120b-maas"
MINIMAX: Final = "minimaxai/minimax-m2-maas"
KIMI: Final = "moonshotai/kimi-k2-thinking-maas"
GLM_GLOBAL_ONLY: Final = "zai-org/glm-4.7-maas"
MISTRAL: Final = "mistral-large@2407"
MISTRAL_SMALL: Final = "mistral-small-2503@001"
CODESTRAL: Final = "codestral@2405"
JAMBA: Final = "jamba-1.5-large@001"
GARDEN_ENDPOINT_ID: Final = "5464397967697903616"
GARDEN_PUBLISHER: Final = "xai/grok-4.1-fast-non-reasoning"
PSC_ENDPOINT_ID: Final = "4965075652664360960"
GEMINI_EMBEDDING_2: Final = "gemini-embedding-2"
MULTIMODAL_EMBEDDING: Final = "multimodalembedding@001"


def location_path(location: str, *, project: str = PROJECT, version: str = "v1") -> str:
    return f"/{version}/projects/{project}/locations/{location}"


def openapi_chat_path(location: str = "us-central1", *, project: str = PROJECT) -> str:
    """The shared MaaS OpenAI compatible route every Llama-family partner model posts to."""
    return f"{location_path(location, project=project)}/endpoints/openapi/chat/completions"


def raw_predict_path(publisher: str, model: str, *, location: str = "us-central1", version: str = "v1") -> str:
    return f"{location_path(location, version=version)}/publishers/{publisher}/models/{model}"


# ---------------------------------------------------------------- OpenAI shaped replies


def openai_chat_json(
    model: str = "partner-model",
    *,
    text: str | None = "pong",
    prompt: int = 11,
    completion: int = 7,
    finish: str = "stop",
    tool_calls: Sequence[JsonValue] | None = None,
    **message: JsonValue,
) -> dict[str, JsonValue]:
    assistant: dict[str, JsonValue] = {"role": "assistant", "content": text, **message}
    if tool_calls is not None:
        assistant["tool_calls"] = list(tool_calls)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion",
        "created": 1_763_000_000,
        "model": model,
        "choices": [{"index": 0, "message": assistant, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


def openai_reply(**fields: object) -> Reply:
    return Reply(body=json.dumps(openai_chat_json(**fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def openai_sse(
    model: str = "partner-model", pieces: Sequence[str] = ("hello ", "maas"), *, usage: bool = True
) -> Reply:
    base: Final[dict[str, JsonValue]] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:20]}",
        "object": "chat.completion.chunk",
        "created": 1_763_000_000,
        "model": model,
    }
    frames: Final[list[dict[str, JsonValue]]] = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}]}
        for piece in pieces
    ]
    frames.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage:
        frames.append(
            {**base, "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
        )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames), b"data: [DONE]\n\n"),
    )


def openai_echo(request: Request) -> Reply:
    """Answer an OpenAI shaped chat call: SSE when the body asks for `stream: true`, JSON otherwise."""
    body: Final = body_of(request)
    if body.get("stream") is True or ":streamRawPredict" in path_of(request):
        return openai_sse()
    return openai_reply()


# ---------------------------------------------------------------- a peer that records GETs too


@dataclass(frozen=True, slots=True)
class OpenPeer:
    """One owned TCP peer that mints tokens on `TOKEN_PATH` and records every other request, any method."""

    wire: Wire
    plan: TokenPlan
    seen: list[Request]

    @property
    def url(self) -> str:
        return self.wire.url.rstrip("/")

    def calls(self) -> tuple[Request, ...]:
        """Every non-token call, except a GET outside `cachedContents`: the proxy may probe a new deployment
        with a GET, and only the `cachedContents` list is a GET these tests care about."""
        self.seen.extend(self.wire.drain())
        return tuple(
            request
            for request in self.seen
            if path_of(request) != TOKEN_PATH and (request.method != "GET" or "cachedContents" in path_of(request))
        )

    def token_calls(self) -> tuple[Request, ...]:
        self.seen.extend(self.wire.drain())
        return tuple(request for request in self.seen if path_of(request) == TOKEN_PATH)

    def shape(self) -> list[tuple[str, str]]:
        """`(method, path)` of every non-token call in arrival order."""
        return [(request.method, path_of(request)) for request in self.calls()]


@contextmanager
def open_peer(respond: Callable[[Request], Reply], plan: TokenPlan | None = None) -> Iterator[OpenPeer]:
    tokens: Final = plan if plan is not None else TokenPlan()

    def answer(request: Request) -> Reply:
        if request.method == "POST" and path_of(request) == TOKEN_PATH:
            return tokens.answer()
        return respond(request)

    with wire_server(answer) as wire:
        yield OpenPeer(wire, tokens, [])


def register_open_vertex(
    scenario: Scenario, peer: OpenPeer, backend: str, *, base_suffix: str = "", **params: JsonValue
) -> str:
    """Register `vertex_ai/<backend>` against an open peer, with service account credentials minted there."""
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_key=None,
        api_base=peer.url + base_suffix,
        vertex_project=PROJECT,
        vertex_location="us-central1",
        vertex_credentials=service_account_text(peer.url),
        **params,
    )


# ---------------------------------------------------------------- readers


def json_of(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def output_items(response: httpx.Response) -> list[dict[str, JsonValue]]:
    """The `output` list of a Responses API answer."""
    return [object_value(item) for item in list_value(json_of(response)["output"])]


def output_of_type(response: httpx.Response, kind: str) -> list[dict[str, JsonValue]]:
    return [item for item in output_items(response) if item.get("type") == kind]


def event_types(text: str) -> list[str]:
    """`type` of every JSON `data:` frame in a Responses API stream."""
    frames: Final = (line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: "))
    return [str(JSON_OBJECT.validate_json(frame).get("type")) for frame in frames if frame.strip() != "[DONE]"]


def bearer(request: Request) -> str:
    return request.headers["authorization"].removeprefix("Bearer ")


def is_token_from(plan: TokenPlan, request: Request) -> bool:
    return bearer(request) in plan.issued


def mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    return object_value(value)


def register_vector_store(
    gateway: Gateway,
    scenario: Scenario,
    store_id: str,
    api_type: str,
    api_base: str,
    token_base: str,
    **extra: JsonValue,
) -> None:
    """`POST /vector_store/new` for a Vertex RAG Engine (`rag_api`) or Vertex AI Search (`search_api`) store.

    The shape follows `test_vector_store_search_wire.py`: `custom_llm_provider` names the API flavour and
    `litellm_params` carry the Vertex project, location, credentials and the peer as `api_base`.
    """
    registered: Final = gateway.request(
        "POST",
        "/vector_store/new",
        {
            "vector_store_id": store_id,
            "custom_llm_provider": f"vertex_ai/{api_type}",
            "litellm_params": {
                "api_base": api_base,
                "vertex_project": PROJECT,
                "vertex_location": "us-central1",
                "vertex_credentials": service_account_text(token_base),
                **extra,
            },
        },
    )
    assert registered.status_code == 200, registered.text
    scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store_id})
