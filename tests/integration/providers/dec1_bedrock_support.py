"""Shared rig for the December 1 AWS Bedrock contracts on the OpenAI-shaped surfaces of the proxy.

The `test_dec1_bedrock_*` files pin what the proxy does when a deployment behind `/v1/chat/completions`,
`/v1/embeddings`, `/v1/responses` or `/v1/messages` is AWS Bedrock (Converse, InvokeModel and the embedding
`/invoke` route). Each test registers one deployment on the shared proxy whose `api_base` points at an
owned TCP peer (`wire_server`), so the exact request Bedrock would receive (method, path, headers, SigV4
`Authorization`, JSON body) can be read back, and the peer answers in Bedrock's own wire format: a
Converse document, an InvokeModel document, or a binary `application/vnd.amazon.eventstream` stream.

Harness facts these helpers bake in, learned from the first full CI runs of the December 1 suite:
the shared proxy runs with `litellm_settings.cache: true`, so every prompt carries a fresh marker;
the proxy may probe a new deployment with a `GET`, so only `POST` requests count as provider calls;
SpendLogs rows land asynchronously and are polled with `eventually`, never slept for; and the shared
proxy sets `router_settings.disable_cooldowns: true` with no `num_retries`, so tests that count
upstream attempts pass `num_retries` in the request body explicitly.

Most generic Bedrock wire helpers (Converse JSON, event stream frames, AWS error replies, SigV4
verification) already live in `messages_endpoint/dec1_msgprov_support.py` and are re-exported here so
the two suites agree on one encoding of the AWS wire.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.upstream import _aws_event_frame, aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.messages_endpoint.dec1_msgprov_support import (
    ACCESS_KEY,
    EVENT_STREAM,
    SECRET_KEY,
    SESSION_TOKEN,
    assert_sigv4_signed,
    aws_error,
    body_of,
    converse_json,
    converse_reply,
    converse_stream_reply,
    invoke_chunk,
    invoke_exception,
    path_of,
    signed_headers,
    sigv4_scope,
)
from pydantic import JsonValue

__all__ = (
    "ACCESS_KEY",
    "EVENT_STREAM",
    "SECRET_KEY",
    "SESSION_TOKEN",
    "assert_sigv4_signed",
    "aws_error",
    "body_of",
    "converse_json",
    "converse_reply",
    "converse_stream_reply",
    "invoke_chunk",
    "invoke_exception",
    "path_of",
    "signed_headers",
    "sigv4_scope",
)

TOKEN: Final = "synthetic-bedrock-bearer"
"""A Bedrock API key (bearer token). Synthetic; Bedrock never sees it."""
REGION: Final = "us-east-1"

CLAUDE_SONNET_45: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
"""In `BEDROCK_CONVERSE_MODELS` (`litellm/constants.py:1511`), so a bare `bedrock/<id>` chat call goes to Converse."""
CLAUDE_HAIKU_55: Final = "anthropic.claude-haiku-5-5"
"""A Claude id with no `:` version suffix, so the request path and the SigV4 canonical path agree byte for byte."""
NOVA_PRO: Final = "amazon.nova-pro-v1:0"
MISTRAL_7B: Final = "mistral.mistral-7b-instruct-v0:2"
"""Cost map row `litellm_provider: bedrock` and not in the Converse list, so a bare id goes to InvokeModel."""
LLAMA2_13B: Final = "meta.llama2-13b-chat-v1"
TITAN_TEXT: Final = "amazon.titan-text-express-v1"
COHERE_COMMAND_R: Final = "cohere.command-r-v1:0"
COHERE_COMMAND_TEXT: Final = "cohere.command-text-v14"
TITAN_EMBED_V2: Final = "amazon.titan-embed-text-v2:0"
TITAN_EMBED_V1: Final = "amazon.titan-embed-text-v1"
COHERE_EMBED_V3: Final = "cohere.embed-english-v3"
COHERE_EMBED_V4: Final = "cohere.embed-v4:0"

INPUT_COST: Final = 0.001
OUTPUT_COST: Final = 0.002
CACHE_READ_COST: Final = 0.0001
CACHE_CREATION_COST: Final = 0.003
PRICES: Final[Mapping[str, JsonValue]] = {
    "input_cost_per_token": INPUT_COST,
    "output_cost_per_token": OUTPUT_COST,
    "cache_read_input_token_cost": CACHE_READ_COST,
    "cache_creation_input_token_cost": CACHE_CREATION_COST,
}

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

SPEND_COLUMNS: Final = (
    "SELECT request_id, call_type, spend, total_tokens, prompt_tokens, completion_tokens, model, model_id, "
    "model_group, custom_llm_provider, api_base, status, metadata "
    'FROM "LiteLLM_SpendLogs" '
)


def marker(prefix: str = "dec1-bedrock") -> str:
    """A unique token for prompts, so the shared proxy's response cache can never answer a test."""
    return f"{prefix}-{uuid.uuid4().hex}"


def _answering_probes(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        return respond(request)

    return answer


@dataclass(frozen=True, slots=True)
class Rig:
    """One owned Bedrock peer plus the model group name that routes to it."""

    gateway: Gateway
    scenario: Scenario
    wire: Wire
    model: str

    def posts(self) -> tuple[Request, ...]:
        """Every POST the peer received since the last call. GET probes are dropped."""
        return tuple(request for request in self.wire.drain() if request.method == "POST")

    def one_post(self) -> Request:
        posts: Final = self.posts()
        assert len(posts) == 1, [(request.method, request.target) for request in posts]
        return posts[0]

    def chat(self, prompt: str | None = None, **fields: JsonValue) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt if prompt is not None else marker()}],
            **fields,
        }
        return self.gateway.request("POST", "/v1/chat/completions", body)

    def chat_body(self, body: Mapping[str, JsonValue]) -> httpx.Response:
        return self.gateway.request("POST", "/v1/chat/completions", {"model": self.model, **body})

    def embed(self, input_value: JsonValue, **fields: JsonValue) -> httpx.Response:
        return self.gateway.request("POST", "/v1/embeddings", {"model": self.model, "input": input_value, **fields})

    def responses(self, **fields: JsonValue) -> httpx.Response:
        return self.gateway.request("POST", "/v1/responses", {"model": self.model, **fields})

    def messages(self, **fields: JsonValue) -> httpx.Response:
        return self.gateway.request(
            "POST",
            "/v1/messages",
            {"model": self.model, "max_tokens": 64, **fields},
            headers={"anthropic-version": "2023-06-01"},
        )


@contextmanager
def bedrock(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    model: str,
    model_info: Mapping[str, JsonValue] | None = None,
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Register `model` (a full `bedrock/...` id) with `api_base` at a new owned peer.

    Defaults: bearer `api_key` (Bedrock API key auth) and `aws_region_name: us-east-1`. A SigV4
    deployment passes `api_key=""` (see `sigv4_params`), which `bedrock_bearer_token` treats as no
    bearer (`litellm/llms/bedrock/base_aws_llm.py:134`). A value of `None` drops the parameter
    entirely (the deployment is registered through `register_model`, which adds no harness defaults),
    so `api_base=None` exercises endpoint resolution from `aws_bedrock_runtime_endpoint` and
    `aws_region_name=None` exercises region resolution.
    """
    with wire_server(_answering_probes(respond)) as wire, gateway.scenario() as scenario:
        parameters: Final[dict[str, JsonValue]] = {
            "api_base": wire.url,
            "api_key": TOKEN,
            "aws_region_name": REGION,
            **litellm_params,
        }
        cleaned: Final = {key: value for key, value in parameters.items() if value is not None}
        name: Final = register_model(scenario, model=model, model_info=model_info, **cleaned)
        yield Rig(gateway, scenario, wire, name)


def register_model(
    scenario: Scenario, *, model: str, model_info: Mapping[str, JsonValue] | None = None, **litellm_params: JsonValue
) -> str:
    """`POST /model/new` with exactly these `litellm_params`.

    Unlike `Scenario.model`, this injects no harness `api_key` or `api_base`, so a deployment can truly
    lack either one (needed to exercise `aws_bedrock_runtime_endpoint` and env based credentials).
    The deployment is deleted when the scenario closes. Assumes a single worker shared proxy, as
    `Scenario.model` does when `gateway.workers == 1`.
    """
    name: Final = f"dec1-bedrock-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, **litellm_params},
            "model_info": dict(model_info) if model_info is not None else {},
        },
    )
    identity: Final = str(created["model_info"]["id"])  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
    scenario.cleanups.callback(scenario.delete_model, identity)
    return name


def sigv4_params(**extra: JsonValue) -> dict[str, JsonValue]:
    """SigV4 static keys and no `api_key` at all (`None` is dropped by `bedrock`), so no bearer token is used."""
    return {"api_key": None, "aws_access_key_id": ACCESS_KEY, "aws_secret_access_key": SECRET_KEY, **extra}


def json_reply(
    payload: Mapping[str, JsonValue], *, status: int = 200, headers: Mapping[str, str] | None = None
) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode(), headers=dict(headers or {}))


def converse_frames(events: Sequence[tuple[str, Mapping[str, JsonValue]]]) -> Reply:
    """A Converse stream reply from `(event type, payload)` pairs, each one AWS event stream frame."""
    return Reply(
        content_type=EVENT_STREAM,
        chunks=tuple(_aws_event_frame(kind, payload, "sc", "u") for kind, payload in events),
    )


def invoke_frames(payloads: Sequence[Mapping[str, JsonValue]]) -> Reply:
    """An InvokeModel `invoke-with-response-stream` reply: each payload base64 wrapped in one `chunk` frame."""
    return Reply(content_type=EVENT_STREAM, chunks=tuple(invoke_chunk(payload) for payload in payloads))


def exception_frame(kind: str, message: str) -> bytes:
    return aws_event_stream_frame(
        {":exception-type": kind, ":content-type": "application/json", ":message-type": "exception"},
        json.dumps({"message": message}).encode(),
    )


def sse_events(response: httpx.Response) -> list[dict[str, JsonValue]]:
    """The JSON `data:` payloads of an OpenAI-style SSE answer, without the `[DONE]` sentinel."""
    events: Final[list[dict[str, JsonValue]]] = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: ") :]))
    return events


def streamed_text(events: Sequence[Mapping[str, JsonValue]]) -> str:
    pieces: Final[list[str]] = []
    for event in events:
        for choice in event.get("choices", []) or []:  # pyright: ignore[reportUnknownVariableType]
            content = choice.get("delta", {}).get("content")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(content, str):
                pieces.append(content)
    return "".join(pieces)


def error_message(response: httpx.Response) -> str:
    payload: Final = response.json()
    error: Final = payload.get("error", payload)
    return str(error.get("message", "")) if isinstance(error, dict) else str(error)


def spend_row(request_id: str, *, seconds: float = 90) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(SPEND_COLUMNS + "WHERE request_id=%s", (request_id,)),
        lambda values: len(values) == 1,
        seconds=seconds,
    )
    return rows[0]


def b64(payload: Mapping[str, JsonValue]) -> str:
    return base64.b64encode(json.dumps(payload).encode()).decode()
