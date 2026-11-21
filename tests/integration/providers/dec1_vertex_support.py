"""Shared rig for the December 1 Google Vertex AI contracts (`test_dec1_vertex_*`).

Every test registers one Vertex deployment on the shared proxy whose `api_base` points at an owned
TCP peer (`wire_server`). The same peer also serves the OAuth token endpoint: the deployment's
service account JSON names `<peer>/_oauth/token` as its `token_uri`, so google-auth's JWT bearer
exchange (and every refresh after it) is diverted to the peer and never reaches
`oauth2.googleapis.com`. Model calls and token calls are told apart by path, so a test can count
token mints, script their expiry, and read back the exact Vertex request (method, path, query,
headers, body).

Harness facts that shape these helpers: the shared proxy has `litellm_settings.cache: true`, so every
prompt carries a fresh marker; the proxy may probe a new deployment with a `GET`, so only `POST`
requests count as provider calls; and no helper here ever builds a URL on a Google host.
"""

from __future__ import annotations

import base64
import itertools
import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, Scenario, list_value, object_value, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

PROJECT: Final = "dec1-vertex-project"
"""The project every deployment names in `vertex_project` unless a test resolves it another way."""
SA_PROJECT: Final = "dec1-sa-project"
"""The `project_id` inside the generated service account JSON, distinct from `PROJECT` on purpose."""
QUOTA_PROJECT: Final = "dec1-quota-project"
LOCATION: Final = "us-central1"
CLAUDE_LOCATION: Final = "us-east5"
GEMINI: Final = "gemini-2.5-flash"
"""A Gemini 2.x model: thinking maps to `thinkingBudget`, json schema maps to `response_json_schema`."""
CLAUDE: Final = "claude-sonnet-4-5@20250929"
EMBEDDING: Final = "text-embedding-005"
IMAGEN: Final = "imagegeneration@006"
GEMINI_IMAGE: Final = "gemini-2.5-flash-image"
SCOPE: Final = "https://www.googleapis.com/auth/cloud-platform"
JWT_BEARER: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
TOKEN_PATH: Final = "/_oauth/token"
PNG_B64: Final = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082"
    )
).decode()


def marker() -> str:
    """A fresh prompt suffix, so the shared proxy's response cache never answers a test."""
    return uuid.uuid4().hex


def models_path(model: str = GEMINI, *, project: str = PROJECT, location: str = LOCATION, version: str = "v1") -> str:
    return f"/{version}/projects/{project}/locations/{location}/publishers/google/models/{model}"


def claude_path(*, project: str = PROJECT, location: str = CLAUDE_LOCATION, model: str = CLAUDE) -> str:
    return f"/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}"


def path_of(request: Request) -> str:
    return unquote(urlsplit(request.target).path)


def query_of(request: Request) -> str:
    return urlsplit(request.target).query


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def json_of(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def at(value: JsonValue, *path: str | int) -> JsonValue:
    """The value at `path` inside nested JSON, asserting each step exists."""
    if not path:
        return value
    head, *rest = path
    if isinstance(head, int):
        assert isinstance(value, list) and len(value) > head, (value, head)
        return at(value[head], *rest)
    assert isinstance(value, dict) and head in value, (value, head)
    return at(value[head], *rest)


# ---------------------------------------------------------------- credentials


def _private_key_pem() -> str:
    return (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )


def service_account(token_base: str, *, project: str = SA_PROJECT, **extra: JsonValue) -> dict[str, JsonValue]:
    """A fresh service account (new RSA key, so a fresh google-auth cache entry) whose token URI is the peer."""
    return {
        "type": "service_account",
        "project_id": project,
        "private_key_id": f"dec1-{marker()[:12]}",
        "private_key": _private_key_pem(),
        "client_email": f"dec1-vertex@{project}.iam.gserviceaccount.com",
        "client_id": "0",
        "auth_uri": f"{token_base}/_oauth/authorize",
        "token_uri": f"{token_base}{TOKEN_PATH}",
        **extra,
    }


def service_account_text(token_base: str, **fields: JsonValue) -> str:
    return json.dumps(service_account(token_base, **fields))


def authorized_user(token_base: str, *, quota_project: str | None = QUOTA_PROJECT) -> dict[str, JsonValue]:
    """`gcloud auth application-default login` output, with its token URI diverted to the peer."""
    return {
        "type": "authorized_user",
        "client_id": f"dec1-{marker()[:8]}.apps.googleusercontent.com",
        "client_secret": "synthetic-client-secret",
        "refresh_token": f"synthetic-refresh-{marker()}",
        "token_uri": f"{token_base}{TOKEN_PATH}",
        **({"quota_project_id": quota_project} if quota_project is not None else {}),
    }


def jwt_claims(assertion: str) -> dict[str, JsonValue]:
    """The unverified claim set of a compact JWT, enough to check what google-auth signed."""
    payload: Final = assertion.split(".")[1]
    return JSON_OBJECT.validate_json(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def form_of(request: Request) -> dict[str, str]:
    return {name: values[0] for name, values in parse_qs(request.body.decode()).items()}


# ---------------------------------------------------------------- the peer


@dataclass
class TokenPlan:
    """How the peer's token endpoint answers: `token-1`, `token-2`, ... each valid for `expires_in` seconds."""

    expires_in: int = 3600
    error: Reply | None = None
    issued: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def answer(self) -> Reply:
        if self.error is not None:
            return self.error
        with self._lock:
            token: Final = f"token-{len(self.issued) + 1}"
            self.issued.append(token)
        return Reply(
            body=json.dumps({"access_token": token, "token_type": "Bearer", "expires_in": self.expires_in}).encode()
        )


@dataclass(frozen=True, slots=True)
class Peer:
    """One owned TCP peer that is both the OAuth token endpoint and the Vertex model host."""

    wire: Wire
    plan: TokenPlan
    seen: list[Request]

    @property
    def url(self) -> str:
        return self.wire.url.rstrip("/")

    def _drain(self) -> None:
        self.seen.extend(request for request in self.wire.drain() if request.method == "POST")

    def token_calls(self) -> tuple[Request, ...]:
        self._drain()
        return tuple(request for request in self.seen if path_of(request) == TOKEN_PATH)

    def model_calls(self) -> tuple[Request, ...]:
        self._drain()
        return tuple(request for request in self.seen if path_of(request) != TOKEN_PATH)

    def one_call(self) -> Request:
        calls: Final = self.model_calls()
        assert len(calls) == 1, [call.target for call in calls]
        return calls[0]

    def one_body(self) -> dict[str, JsonValue]:
        return body_of(self.one_call())


@contextmanager
def peer(respond: Callable[[Request], Reply], plan: TokenPlan | None = None) -> Iterator[Peer]:
    tokens: Final = plan if plan is not None else TokenPlan()

    def answer(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object":"list","data":[]}')
        if path_of(request) == TOKEN_PATH:
            return tokens.answer()
        return respond(request)

    with wire_server(answer) as wire:
        yield Peer(wire, tokens, [])


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    scenario: Scenario
    peer: Peer
    model: str

    def chat(self, text: str | None = None, **fields: JsonValue) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "messages": [{"role": "user", "content": text if text is not None else f"ping {marker()}"}],
            **fields,
        }
        return self.gateway.request("POST", "/v1/chat/completions", body)

    def post(
        self, path: str, body: Mapping[str, JsonValue], headers: Mapping[str, str] | None = None
    ) -> httpx.Response:
        return self.gateway.request("POST", path, {"model": self.model, **body}, headers=headers)

    def stream(self, path: str, body: Mapping[str, JsonValue]) -> tuple[httpx.Response, str]:
        with self.gateway.client.stream(
            "POST",
            path,
            json={"model": self.model, **body, "stream": True},
            headers={"Authorization": f"Bearer {self.gateway.key}"},
        ) as response:
            text: Final = "".join(response.iter_text())
            return response, text


@contextmanager
def vertex(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    backend: str = GEMINI,
    plan: TokenPlan | None = None,
    credentials: Callable[[str], JsonValue] | None = None,
    base_suffix: str = "",
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> Iterator[Rig]:
    """Register `vertex_ai/<backend>` with `api_base` at a new peer whose token endpoint mints the tokens.

    `credentials` builds the `vertex_credentials` value from the peer's base URL (a service account by
    default). `params` override `vertex_project` / `vertex_location` (pass `None` to leave one unset) or add
    other litellm params.
    """
    with peer(respond, plan) as owned, gateway.scenario() as scenario:
        build: Final = credentials if credentials is not None else service_account_text
        settings: Final[dict[str, JsonValue]] = {
            "api_key": None,
            "api_base": owned.url + base_suffix,
            "vertex_project": PROJECT,
            "vertex_location": CLAUDE_LOCATION if backend.startswith("claude") else LOCATION,
            "vertex_credentials": build(owned.url),
            **params,
        }
        name: Final = scenario.model(
            model=f"vertex_ai/{backend}",
            model_info=model_info,
            **{key: value for key, value in settings.items() if value is not None or key == "api_key"},
        )
        yield Rig(gateway, scenario, owned, name)


# ---------------------------------------------------------------- Gemini wire


def usage_metadata(prompt: int = 11, candidates: int = 7, **extra: JsonValue) -> dict[str, JsonValue]:
    total: Final = extra.pop("totalTokenCount", prompt + candidates)
    return {"promptTokenCount": prompt, "candidatesTokenCount": candidates, "totalTokenCount": total, **extra}


def gemini_json(
    parts: Sequence[JsonValue] | None = None,
    *,
    finish: str = "STOP",
    usage: Mapping[str, JsonValue] | None = None,
    text: str = "pong",
) -> dict[str, JsonValue]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": list(parts) if parts is not None else [{"text": text}]},
                "finishReason": finish,
                "index": 0,
            }
        ],
        "usageMetadata": dict(usage) if usage is not None else usage_metadata(),
        "modelVersion": GEMINI,
        "responseId": f"resp-{marker()[:16]}",
    }


def gemini_reply(**fields: object) -> Reply:
    return Reply(body=json.dumps(gemini_json(**fields)).encode())  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def gemini_sse(
    pieces: Sequence[str] = ("hello ", "stream"), *, usage: Mapping[str, JsonValue] | None = None, finish: str = "STOP"
) -> Reply:
    frames: Final = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": piece}]}, "index": 0}], "modelVersion": GEMINI}
        for piece in pieces
    ]
    frames[-1] = {
        **frames[-1],
        "candidates": [{**object_value(list_value(frames[-1]["candidates"])[0]), "finishReason": finish}],
        "usageMetadata": dict(usage) if usage is not None else usage_metadata(),
    }
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\r\n\r\n".encode() for frame in frames),
    )


def gemini_echo(request: Request) -> Reply:
    if path_of(request).endswith(":streamGenerateContent"):
        return gemini_sse()
    return gemini_reply()


def google_error(status: int, code_name: str, message: str) -> Reply:
    """Google's JSON error envelope, `{"error": {"code", "message", "status"}}`."""
    return Reply(
        status=status,
        body=json.dumps({"error": {"code": status, "message": message, "status": code_name}}).encode(),
    )


def choice(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(list_value(json_of(response)["choices"])[0])


def message_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(choice(response)["message"])


def sse_payloads(text: str) -> tuple[dict[str, JsonValue], ...]:
    """The JSON payload of every `data:` line except `[DONE]`."""
    lines: Final = (line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: "))
    return tuple(JSON_OBJECT.validate_json(line) for line in lines if line.strip() != "[DONE]")


def streamed_text(text: str) -> str:
    chunks: Final = sse_payloads(text)
    deltas: Final = (
        object_value(object_value(list_value(chunk["choices"])[0]).get("delta") or {})
        for chunk in chunks
        if chunk.get("choices")
    )
    return "".join(string_value(delta["content"]) for delta in deltas if delta.get("content"))


# ---------------------------------------------------------------- Claude on Vertex (Anthropic wire)


def claude_json(
    *,
    text: str = "pong",
    usage: Mapping[str, JsonValue] | None = None,
    content: Sequence[JsonValue] | None = None,
    stop_reason: str = "end_turn",
) -> dict[str, JsonValue]:
    return {
        "id": f"msg_vrtx_{marker()[:20]}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": list(content) if content is not None else [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": dict(usage) if usage is not None else {"input_tokens": 11, "output_tokens": 7},
    }


def claude_sse(text: str = "hello stream") -> Reply:
    identity: Final = f"msg_vrtx_{marker()[:20]}"
    events: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events),
    )


def claude_echo(request: Request) -> Reply:
    if ":streamRawPredict" in path_of(request):
        return claude_sse()
    return Reply(body=json.dumps(claude_json()).encode())


def anthropic_error(status: int, error_type: str, message: str) -> Reply:
    return Reply(
        status=status, body=json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()
    )


def counter() -> Callable[[], int]:
    """A thread safe 1, 2, 3, ... sequence for peers that answer differently per call."""
    numbers: Final = itertools.count(1)
    lock: Final = threading.Lock()

    def take() -> int:
        with lock:
            return next(numbers)

    return take


# ---------------------------------------------------------------- embeddings and images


def embedding_peer(request: Request) -> Reply:
    """Answer a text embedding `:predict` with one 2-value vector per instance, `token_count` 3, 4, 5, ...
    and `metadata.billableCharacterCount` equal to the summed instance lengths."""
    instances: Final = list_value(body_of(request)["instances"])
    texts: Final = [string_value(object_value(item)["content"]) for item in instances]
    return Reply(
        body=json.dumps(
            {
                "predictions": [
                    {
                        "embeddings": {
                            "values": [0.5, float(index)],
                            "statistics": {"token_count": index + 3, "truncated": False},
                        }
                    }
                    for index, _ in enumerate(texts)
                ],
                "metadata": {"billableCharacterCount": sum(len(text) for text in texts)},
            }
        ).encode()
    )


def gemini_image_reply(*, prompt_tokens: int = 10, image_tokens: int = 1290) -> Reply:
    """A Gemini image model's `generateContent` answer holding one PNG as `inlineData`."""
    return Reply(
        body=json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"inlineData": {"mimeType": "image/png", "data": PNG_B64}}],
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": prompt_tokens,
                    "candidatesTokenCount": image_tokens,
                    "totalTokenCount": prompt_tokens + image_tokens,
                    "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": image_tokens}],
                },
            }
        ).encode()
    )


def imagen_reply(count: int) -> Reply:
    return Reply(
        body=json.dumps(
            {"predictions": [{"bytesBase64Encoded": PNG_B64, "mimeType": "image/png"} for _ in range(count)]}
        ).encode()
    )
