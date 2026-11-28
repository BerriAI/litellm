"""Shared rig for the December 1 Azure contracts (`test_dec1_azure_*`).

The `test_dec1_azure_*` files pin what the proxy does when the deployment behind `/v1/chat/completions`,
`/v1/responses`, `/v1/embeddings`, `/v1/messages` or the Azure compatible `/openai/deployments/...` routes
is Azure OpenAI (`azure/...`) or Azure AI Foundry (`azure_ai/...`). Each test registers one deployment on
the shared proxy whose `api_base` points at an owned TCP peer (`wire_server`), so the exact request Azure
would receive (method, path, `api-version` query, `api-key` or `Authorization` header, JSON body) can be read
back, and the peer answers in Azure's own wire format.

Harness facts these helpers bake in, learned from the first full CI runs of the December 1 suite: the
shared proxy runs with `litellm_settings.cache: true`, so every prompt carries a fresh marker; the proxy
may probe a new deployment with a `GET`, so only `POST` requests count as provider calls; SpendLogs rows
land asynchronously and are polled with `eventually`, never slept for; and the shared proxy sets
`router_settings.disable_cooldowns: true` with no `num_retries`, so tests that count upstream attempts pass
`num_retries` in the request body. Azure OpenAI chat and embeddings go through the OpenAI SDK's
`AzureOpenAI` client, which retries 429 and 5xx on its own `max_retries` (litellm default 2,
`litellm/constants.py:145`), so tests that count attempts also set `max_retries: 0` on the deployment
unless they pin the SDK retry itself.

Deployments are registered through `POST /model/new` with exactly the given `litellm_params` (no harness
`api_key` default), so a test can truly omit the key to exercise Entra ID bearer tokens.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

API_KEY: Final = "synthetic-azure-key"
"""Deployment level Azure OpenAI key. Synthetic; never valid anywhere."""
FOUNDRY_KEY: Final = "synthetic-foundry-key"
ENTRA_TOKEN: Final = "synthetic-entra-access-token"
"""A static `azure_ad_token` value; Azure would see it as `Authorization: Bearer <token>`."""

DEPLOYMENT: Final = "gpt-4o-dec1"
"""An Azure deployment name. Deliberately not a model name, so URL shape tests can tell them apart."""
GPT5_DEPLOYMENT: Final = "gpt-5.4-nano"
EMBEDDING_DEPLOYMENT: Final = "text-embedding-3-small"
DATED_API_VERSION: Final = "2024-10-21"
"""A dated GA api-version that supports tools, tool_choice and response_format."""
PREVIEW_API_VERSION: Final = "2025-04-01-preview"

DEFAULT_AZURE_API_VERSION: Final = "2025-02-01-preview"
"""`litellm.AZURE_DEFAULT_API_VERSION` (`litellm/__init__.py:338`), used when no api_version is configured
and `AZURE_API_VERSION` is unset in the proxy environment."""

INPUT_COST: Final = 0.001
OUTPUT_COST: Final = 0.002
CACHE_READ_COST: Final = 0.0001
PRICES: Final[Mapping[str, JsonValue]] = {
    "input_cost_per_token": INPUT_COST,
    "output_cost_per_token": OUTPUT_COST,
    "cache_read_input_token_cost": CACHE_READ_COST,
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

USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def marker(prefix: str = "dec1-azure") -> str:
    """A unique token for prompts, so the shared proxy's response cache can never answer a test."""
    return f"{prefix}-{uuid.uuid4().hex}"


def _answering_probes(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        return respond(request)

    return answer


def path_of(request: Request) -> str:
    return urlsplit(request.target).path


def query_of(request: Request) -> dict[str, str]:
    """The query string as a flat dict (last value wins). An absent query is `{}`."""
    return {name: values[-1] for name, values in parse_qs(urlsplit(request.target).query).items()}


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


@dataclass(frozen=True, slots=True)
class Rig:
    """One owned Azure peer plus the model group name that routes to it."""

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
        body: Final[dict[str, JsonValue]] = {"model": self.model, "input": marker(), **fields}
        return self.gateway.request("POST", "/v1/responses", body)

    def messages(self, headers: Mapping[str, str] | None = None, **fields: JsonValue) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": marker()}],
            **fields,
        }
        return self.gateway.request(
            "POST", "/v1/messages", body, headers={"anthropic-version": "2023-06-01", **(headers or {})}
        )


@contextmanager
def azure(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    model: str = f"azure/{DEPLOYMENT}",
    model_info: Mapping[str, JsonValue] | None = None,
    base_path: str = "",
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Register `model` (a full `azure/...` or `azure_ai/...` id) with `api_base` at a new owned peer.

    Defaults: `api_key: API_KEY` and `api_version: DATED_API_VERSION`. `base_path` is appended to the peer
    URL to form `api_base`. A value of `None` drops the parameter entirely, so `api_key=None` exercises
    Entra ID auth and `api_version=None` exercises the api-version defaults.
    """
    with wire_server(_answering_probes(respond)) as wire, gateway.scenario() as scenario:
        parameters: Final[dict[str, JsonValue]] = {
            "api_base": f"{wire.url}{base_path}",
            "api_key": API_KEY,
            "api_version": DATED_API_VERSION,
            **litellm_params,
        }
        cleaned: Final = {key: value for key, value in parameters.items() if value is not None}
        name: Final = register_model(scenario, model=model, model_info=model_info, **cleaned)
        yield Rig(gateway, scenario, wire, name)


@contextmanager
def foundry(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    model: str,
    model_info: Mapping[str, JsonValue] | None = None,
    base_path: str = "",
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Register an `azure_ai/...` deployment at a new owned peer with `api_key: FOUNDRY_KEY` and no api_version."""
    with azure(
        gateway,
        respond,
        model=model,
        model_info=model_info,
        base_path=base_path,
        **{"api_key": FOUNDRY_KEY, "api_version": None, **litellm_params},
    ) as rig:
        yield rig


def register_model(
    scenario: Scenario, *, model: str, model_info: Mapping[str, JsonValue] | None = None, **litellm_params: JsonValue
) -> str:
    """`POST /model/new` with exactly these `litellm_params`; deleted when the scenario closes.

    Unlike `Scenario.model`, this injects no harness `api_key` or `api_base`, so a deployment can truly lack
    a key. Assumes a single worker shared proxy, as `Scenario.model` does when `gateway.workers == 1`.
    """
    name: Final = f"dec1-azure-{uuid.uuid4().hex}"
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


def json_reply(
    payload: Mapping[str, JsonValue], *, status: int = 200, headers: Mapping[str, str] | None = None
) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode(), headers=dict(headers or {}))


def chat_json(
    content: str | None = "pong",
    *,
    model: str = DEPLOYMENT,
    finish_reason: str = "stop",
    usage: Mapping[str, JsonValue] | None = None,
    message_extra: Mapping[str, JsonValue] | None = None,
    choice_extra: Mapping[str, JsonValue] | None = None,
    top_extra: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """An Azure OpenAI chat completion document."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **(message_extra or {})},
                "finish_reason": finish_reason,
                **(choice_extra or {}),
            }
        ],
        "usage": dict(usage) if usage is not None else dict(USAGE),
        **(top_extra or {}),
    }


def chat_reply(content: str | None = "pong", *, headers: Mapping[str, str] | None = None, **fields: JsonValue) -> Reply:
    return json_reply(chat_json(content, **fields), headers=headers)  # pyright: ignore[reportArgumentType]


def sse(frames: Sequence[Mapping[str, JsonValue] | str], *, abort_after: int | None = None) -> Reply:
    """A `text/event-stream` reply: each mapping is one `data:` frame; a string is sent as a raw data payload."""
    chunks: Final = tuple(
        f"data: {frame if isinstance(frame, str) else json.dumps(frame)}\n\n".encode() for frame in frames
    )
    return Reply(content_type="text/event-stream", chunks=chunks, abort_after=abort_after)


def chat_chunk(
    identity: str, delta: Mapping[str, JsonValue], *, finish_reason: str | None = None
) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": DEPLOYMENT,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }


def chat_stream(
    texts: Sequence[str], *, usage: Mapping[str, JsonValue] | None = None, finish_reason: str = "stop"
) -> Reply:
    """An Azure chat stream: a role chunk, one chunk per text, a finish chunk, an optional usage chunk, `[DONE]`."""
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    frames: Final[list[Mapping[str, JsonValue] | str]] = [chat_chunk(identity, {"role": "assistant", "content": ""})]
    frames.extend(chat_chunk(identity, {"content": text}) for text in texts)
    frames.append(chat_chunk(identity, {}, finish_reason=finish_reason))
    if usage is not None:
        frames.append({**chat_chunk(identity, {}), "choices": [], "usage": dict(usage)})
    frames.append("[DONE]")
    return sse(frames)


def azure_error(
    status: int,
    code: str | None,
    message: str,
    *,
    inner_error: Mapping[str, JsonValue] | None = None,
    error_type: str | None = None,
    param: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Reply:
    """An Azure OpenAI error envelope: `{"error": {"code", "message", "type", "param", "innererror"?}}`."""
    error: Final[dict[str, JsonValue]] = {"code": code, "message": message, "type": error_type, "param": param}
    if inner_error is not None:
        error["innererror"] = dict(inner_error)
    return json_reply({"error": error}, status=status, headers=headers)


CONTENT_FILTER_RESULT: Final[dict[str, JsonValue]] = {
    "hate": {"filtered": True, "severity": "high"},
    "self_harm": {"filtered": False, "severity": "safe"},
    "sexual": {"filtered": False, "severity": "safe"},
    "violence": {"filtered": False, "severity": "safe"},
}
CONTENT_FILTER_MESSAGE: Final = (
    "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. "
    "Please modify your prompt and retry."
)


def content_filter_error() -> Reply:
    """The 400 Azure answers when the prompt trips its content filter."""
    return azure_error(
        400,
        "content_filter",
        CONTENT_FILTER_MESSAGE,
        param="prompt",
        inner_error={"code": "ResponsibleAIPolicyViolation", "content_filter_result": CONTENT_FILTER_RESULT},
    )


def embedding_json(vectors: Sequence[Sequence[float]], *, prompt_tokens: int = 6) -> dict[str, JsonValue]:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": list(vector)} for index, vector in enumerate(vectors)
        ],
        "model": EMBEDDING_DEPLOYMENT,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


def responses_json(text: str = "pong", *, usage: Mapping[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": GPT5_DEPLOYMENT,
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": dict(usage) if usage is not None else {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    }


def anthropic_message_json(text: str = "pong", *, model: str = "claude-sonnet-4-5") -> dict[str, JsonValue]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


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


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    payload: Final = response.json()
    error: Final = payload.get("error", payload)
    assert isinstance(error, dict), response.text
    return error


def error_message(response: httpx.Response) -> str:
    return str(error_of(response).get("message", ""))


def always(reply: Reply) -> Callable[[Request], Reply]:
    def respond(_: Request) -> Reply:
        return reply

    return respond


def failing_then(failures: Sequence[Reply], success: Reply) -> Callable[[Request], Reply]:
    """Answer each POST with the next scripted failure, then `success` forever."""
    remaining: Final = list(failures)

    def respond(_: Request) -> Reply:
        return remaining.pop(0) if remaining else success

    return respond


def spend_row(request_id: str, *, seconds: float = 90) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(SPEND_COLUMNS + "WHERE request_id=%s", (request_id,)),
        lambda values: len(values) == 1,
        seconds=seconds,
    )
    return rows[0]


def spend_rows_for_group(model_group: str, count: int = 1, *, seconds: float = 90) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_COLUMNS + "WHERE model_group=%s", (model_group,)),
        lambda values: len(values) >= count,
        seconds=seconds,
    )
