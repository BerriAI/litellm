"""Shared rig for the `test_dec1_reliab_*` files: one peer that speaks every big provider, and owned proxies.

The router core (`litellm/rust/crates/router-core`) decides retries, cooldowns and fallbacks from the
exception litellm's provider mapping raises (`litellm/litellm_core_utils/exception_mapping_utils.py`), not
from the raw HTTP status. Each provider maps its own error bodies differently: a Bedrock 500 becomes a 503
`ServiceUnavailableError`, Google's 503 overload text becomes a 500 `InternalServerError`, an Anthropic 529
becomes a 500 `InternalServerError` with `provider_status_code: 529`, a Vertex 502 becomes an
`APIConnectionError`, and so on. These files pin how those real error shapes drive the router.

One in-process peer answers every deployment. Each deployment's `api_base` carries a label as its first path
segment (`<peer>/<label>/...`), so the peer knows which deployment a request was meant for whatever the
provider's own URL shape is. `Peer.plan(label, *replies)` queues the answers for a label; once the queue is
empty the label answers `Peer.always(label, reply)` if one was set, else a provider shaped 200 whose text
is `served <label>`. Every request is recorded with its label and arrival time, so a test counts attempts
per deployment and measures the wait between them.

Provider URL shapes the peer recognises: OpenAI and Azure `.../chat/completions`, Anthropic `.../v1/messages`,
Bedrock `.../model/<id>/converse` and `.../converse-stream`, Gemini and Vertex `...:generateContent` and
`...:streamGenerateContent`, and the Vertex service account token endpoint at `/_oauth/token`.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from itertools import pairwise
from pathlib import Path
from typing import Final
from urllib.parse import unquote

import httpx
from pydantic import JsonValue

from tests.integration._support.anthropic_sse import message_stream, stream_reply
from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.openai_wire import chat_reply
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.messages_endpoint.dec1_msgprov_support import (
    aws_error,
    converse_reply,
    converse_stream_reply,
)
from tests.integration.providers.dec1_vertex_support import TOKEN_PATH, service_account_text
from tests.integration.routing.dec1_routing_support import routing_proxy

PROVIDER_KEY: Final = "dec1-reliab-provider-key"
BEDROCK_BEARER: Final = "dec1-reliab-bedrock-bearer"
BEDROCK_MODEL: Final = "amazon.nova-pro-v1:0"
ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
GEMINI_MODEL: Final = "gemini-2.5-flash"
OPENAI_MODEL: Final = "gpt-4o-mini"
AZURE_API_VERSION: Final = "2024-10-21"
VERTEX_PROJECT: Final = "dec1-reliab-project"
VERTEX_LOCATION: Final = "us-central1"
SLOW_SECONDS: Final = 4
NO_DEPLOYMENTS: Final = "No deployments available for selected model"

# Real provider error texts, synthetic but shaped like the providers' own.
BEDROCK_THROTTLE_TEXT: Final = "Too many requests, please wait before trying again."
BEDROCK_NOT_READY_TEXT: Final = "Model is not ready to serve inference requests."
BEDROCK_UNAVAILABLE_TEXT: Final = "Service is temporarily unavailable."
BEDROCK_INTERNAL_TEXT: Final = "Internal failure"
BEDROCK_VALIDATION_TEXT: Final = "Malformed input request: #: extraneous key [dec1] is not permitted"
BEDROCK_TOO_LONG_TEXT: Final = "Input is too long for requested model."
BEDROCK_ACCESS_TEXT: Final = "You don't have access to the model with the specified model ID. AccessDeniedException"
BEDROCK_TOKEN_TEXT: Final = "The security token included in the request is invalid."
GOOGLE_EXHAUSTED_TEXT: Final = "Resource has been exhausted (e.g. check quota)."
GOOGLE_QUOTA_TEXT: Final = "Quota exceeded for aiplatform.googleapis.com/generate_content_requests_per_minute"
GOOGLE_OVERLOADED_TEXT: Final = "The model is overloaded. Please try again later."
GOOGLE_UNAVAILABLE_TEXT: Final = "The service is currently unavailable."
GOOGLE_INVALID_TEXT: Final = "Invalid value at 'generation_config.temperature' (TYPE_FLOAT), \"hot\""
GOOGLE_TOO_LONG_TEXT: Final = "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."
GOOGLE_PERMISSION_TEXT: Final = "Permission 'aiplatform.endpoints.predict' denied on resource (or it may not exist)."
GOOGLE_BLOCKED_TEXT: Final = "The response was blocked."
GOOGLE_BAD_GATEWAY_TEXT: Final = "Bad Gateway"
ANTHROPIC_OVERLOADED_TEXT: Final = "Overloaded"
ANTHROPIC_RATE_TEXT: Final = "Number of request tokens has exceeded your per-minute rate limit"
ANTHROPIC_TOO_LONG_TEXT: Final = "prompt is too long: 210000 tokens > 200000 maximum"
ANTHROPIC_FILTER_TEXT: Final = "Output blocked by content filtering policy"
OPENAI_RATE_TEXT: Final = "Rate limit reached for gpt-4o-mini in organization org-dec1 on requests per min (RPM)"
OPENAI_QUOTA_TEXT: Final = "You exceeded your current quota, please check your plan and billing details."
OPENAI_SERVER_TEXT: Final = "The server had an error while processing your request. Sorry about that!"
OPENAI_UNAVAILABLE_TEXT: Final = "The engine is currently overloaded, please try again later."
OPENAI_TOO_LONG_TEXT: Final = (
    "This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens."
)
OPENAI_POLICY_TEXT: Final = "Your request was rejected as a result of our safety system."
AZURE_RATE_TEXT: Final = (
    "Requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2024-10-21 have exceeded "
    "token rate limit of your current OpenAI S0 pricing tier."
)
AZURE_FILTER_TEXT: Final = (
    "The response was filtered due to the prompt triggering Azure OpenAI's content management policy."
)
AZURE_GENERIC_500_TEXT: Final = "The service encountered a transient error."
AZURE_INTERNAL_TEXT: Final = "Internal server error"
AZURE_TIMEOUT_TEXT: Final = "The operation was timeout."


@dataclass(frozen=True, slots=True)
class Hit:
    label: str
    path: str
    body: Mapping[str, JsonValue]
    at: float
    headers: Mapping[str, str]


def _label_of(path: str) -> str:
    """The first path segment, without a Vertex `:method` suffix."""
    return unquote(path.lstrip("/").split("/", 1)[0]).split(":", 1)[0]


def _body(request: Request) -> Mapping[str, JsonValue]:
    try:
        return JSON_OBJECT.validate_json(request.body) if request.body else {}
    except ValueError:
        return {}


def _gemini_json(text: str) -> dict[str, JsonValue]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18},
        "modelVersion": GEMINI_MODEL,
    }


def _anthropic_json(text: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_dec1reliab{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": ANTHROPIC_MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def success_for(path: str, body: Mapping[str, JsonValue], text: str) -> Reply:
    """A provider shaped 200 for whatever endpoint `path` is."""
    stream: Final = body.get("stream") is True
    if path.endswith("/chat/completions"):
        return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", OPENAI_MODEL, text, stream=stream)
    if path.endswith("/v1/messages"):
        if stream:
            return stream_reply(message_stream(f"msg_dec1reliab{uuid.uuid4().hex}", ANTHROPIC_MODEL, text))
        return Reply(body=json.dumps(_anthropic_json(text)).encode())
    if path.endswith("/converse-stream"):
        return converse_stream_reply((text,))
    if path.endswith("/converse"):
        return converse_reply(content=[{"text": text}])
    if ":streamGenerateContent" in path:
        frame: Final = f"data: {json.dumps(_gemini_json(text))}\r\n\r\n".encode()
        return Reply(content_type="text/event-stream", chunks=(frame,))
    if ":generateContent" in path:
        return Reply(body=json.dumps(_gemini_json(text)).encode())
    return Reply(status=404, body=b'{"error": {"message": "dec1 reliab peer: unknown path"}}')


# ---------------------------------------------------------------- provider error shapes


def openai_error(
    status: int,
    message: str,
    *,
    kind: str = "invalid_request_error",
    code: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Reply:
    """OpenAI's and Azure's envelope `{"error": {"message", "type", "param", "code"}}`."""
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": kind, "param": None, "code": code}}).encode(),
        headers=dict(headers or {}),
    )


def openai_rate_limit(code: str, headers: Mapping[str, str] | None = None) -> Reply:
    """OpenAI's 429, either `rate_limit_exceeded` (a per-minute limit) or `insufficient_quota` (billing)."""
    message: Final = OPENAI_QUOTA_TEXT if code == "insufficient_quota" else OPENAI_RATE_TEXT
    kind: Final = "insufficient_quota" if code == "insufficient_quota" else "requests"
    return openai_error(429, message, kind=kind, code=code, headers=headers)


def azure_rate_limit(headers: Mapping[str, str] | None = None) -> Reply:
    """Azure OpenAI's 429 `{"error": {"code": "429", "message": ...}}`, usually with `retry-after` and `retry-after-ms`."""
    return Reply(
        status=429,
        body=json.dumps({"error": {"code": "429", "message": AZURE_RATE_TEXT}}).encode(),
        headers=dict(headers or {}),
    )


def azure_content_filter() -> Reply:
    """Azure's prompt filter refusal: 400 `code: content_filter` with `innererror.code: ResponsibleAIPolicyViolation`."""
    return Reply(
        status=400,
        body=json.dumps(
            {
                "error": {
                    "message": AZURE_FILTER_TEXT,
                    "type": None,
                    "param": "prompt",
                    "code": "content_filter",
                    "status": 400,
                    "innererror": {
                        "code": "ResponsibleAIPolicyViolation",
                        "content_filter_result": {"hate": {"filtered": True, "severity": "high"}},
                    },
                }
            }
        ).encode(),
    )


def anthropic_error(status: int, kind: str, message: str, headers: Mapping[str, str] | None = None) -> Reply:
    """Anthropic's envelope `{"type": "error", "error": {"type", "message"}}`."""
    return Reply(
        status=status,
        body=json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode(),
        headers=dict(headers or {}),
    )


def anthropic_overloaded() -> Reply:
    return anthropic_error(529, "overloaded_error", ANTHROPIC_OVERLOADED_TEXT)


def google_error(status: int, code_name: str, message: str, headers: Mapping[str, str] | None = None) -> Reply:
    """Google's envelope `{"error": {"code", "message", "status"}}`, shared by Gemini and Vertex."""
    return Reply(
        status=status,
        body=json.dumps({"error": {"code": status, "message": message, "status": code_name}}).encode(),
        headers=dict(headers or {}),
    )


def bedrock_error(status: int, error_type: str, message: str) -> Reply:
    """AWS's `{"message": ...}` with the error class in `x-amzn-ErrorType`."""
    return aws_error(status, error_type, message)


def http_date_in(seconds: int) -> str:
    """An RFC 7231 date `seconds` from now, the other form `retry-after` may take."""
    return format_datetime(datetime.now(UTC) + timedelta(seconds=seconds), usegmt=True)


# ---------------------------------------------------------------- the peer


@dataclass(frozen=True, slots=True)
class Peer:
    wire: Wire
    log: list[Hit] = field(default_factory=list)  # mutable-ok: appended by the peer's handler threads
    plans: dict[str, deque[Reply]] = field(default_factory=dict)  # mutable-ok: queued by tests
    fixed: dict[str, Reply] = field(default_factory=dict)  # mutable-ok: set by tests
    slow: set[str] = field(default_factory=set)  # mutable-ok: labels that answer after SLOW_SECONDS

    @property
    def url(self) -> str:
        return self.wire.url

    def plan(self, label: str, *replies: Reply) -> None:
        self.plans[label] = deque(replies)

    def always(self, label: str, reply: Reply) -> None:
        self.fixed[label] = reply

    def hits(self, label: str) -> tuple[Hit, ...]:
        return tuple(hit for hit in tuple(self.log) if hit.label == label)

    def count(self, label: str) -> int:
        return len(self.hits(label))

    def gaps(self, label: str) -> tuple[float, ...]:
        times: Final = tuple(hit.at for hit in self.hits(label))
        return tuple(later - earlier for earlier, later in pairwise(times))

    def order(self, *labels: str) -> tuple[str, ...]:
        return tuple(hit.label for hit in tuple(self.log) if hit.label in labels)


def _token_reply() -> Reply:
    return Reply(
        body=json.dumps({"access_token": "dec1-reliab-token", "expires_in": 3600, "token_type": "Bearer"}).encode()
    )


def _answer(peer: Peer, request: Request) -> Reply:
    path: Final = request.target.split("?", 1)[0]
    if path == TOKEN_PATH:
        return _token_reply()
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    label: Final = _label_of(path)
    body: Final = _body(request)
    peer.log.append(Hit(label, path, body, time.monotonic(), request.headers))
    queued: Final = peer.plans.get(label)
    if queued:
        return queued.popleft()
    if label in peer.fixed:
        return peer.fixed[label]
    if label in peer.slow:
        time.sleep(SLOW_SECONDS)
    return success_for(path, body, f"served {label}")


@contextmanager
def provider_peer() -> Iterator[Peer]:
    holder: Final[list[Peer]] = []  # mutable-ok: the handler needs the Peer built from its own wire

    def respond(request: Request) -> Reply:
        return _answer(holder[0], request)

    with wire_server(respond) as wire:
        holder.append(Peer(wire))
        yield holder[0]


# ---------------------------------------------------------------- deployments


def _entry(
    group: str, label: str, model: str, params: Mapping[str, JsonValue], model_info: Mapping[str, JsonValue] | None
) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {"model": model, **params},
        "model_info": {"id": f"dec1-reliab-{label}", **(model_info or {})},
    }


def openai_deployment(
    group: str, label: str, base: str, *, model_info: Mapping[str, JsonValue] | None = None, **params: JsonValue
) -> dict[str, JsonValue]:
    """`openai/gpt-4o-mini` at `<peer>/<label>/v1`, with the OpenAI SDK's own retries off."""
    return _entry(
        group,
        label,
        f"openai/{OPENAI_MODEL}",
        {"api_base": f"{base}/{label}/v1", "api_key": PROVIDER_KEY, "max_retries": 0, **params},
        model_info,
    )


def azure_deployment(
    group: str, label: str, base: str, *, model_info: Mapping[str, JsonValue] | None = None, **params: JsonValue
) -> dict[str, JsonValue]:
    """`azure/<label>` at `<peer>/<label>`, with the AzureOpenAI SDK's own retries off."""
    return _entry(
        group,
        label,
        f"azure/{label}",
        {
            "api_base": f"{base}/{label}",
            "api_key": PROVIDER_KEY,
            "api_version": AZURE_API_VERSION,
            "max_retries": 0,
            **params,
        },
        model_info,
    )


def anthropic_deployment(
    group: str, label: str, base: str, *, model_info: Mapping[str, JsonValue] | None = None, **params: JsonValue
) -> dict[str, JsonValue]:
    """`anthropic/<model>` at `<peer>/<label>` (the handler appends `/v1/messages`)."""
    return _entry(
        group,
        label,
        f"anthropic/{ANTHROPIC_MODEL}",
        {"api_base": f"{base}/{label}", "api_key": PROVIDER_KEY, **params},
        model_info,
    )


def bedrock_deployment(
    group: str, label: str, base: str, *, model_info: Mapping[str, JsonValue] | None = None, **params: JsonValue
) -> dict[str, JsonValue]:
    """`bedrock/converse/<nova>` at `<peer>/<label>` with a bearer API key, so no SigV4 is needed."""
    return _entry(
        group,
        label,
        f"bedrock/converse/{BEDROCK_MODEL}",
        {"api_base": f"{base}/{label}", "api_key": BEDROCK_BEARER, "aws_region_name": "us-east-1", **params},
        model_info,
    )


def gemini_deployment(
    group: str, label: str, base: str, *, model_info: Mapping[str, JsonValue] | None = None, **params: JsonValue
) -> dict[str, JsonValue]:
    """`gemini/gemini-2.5-flash` (AI Studio) at `<peer>/<label>/v1beta`."""
    return _entry(
        group,
        label,
        f"gemini/{GEMINI_MODEL}",
        {"api_base": f"{base}/{label}/v1beta", "api_key": PROVIDER_KEY, **params},
        model_info,
    )


def vertex_deployment(
    group: str, label: str, base: str, *, model_info: Mapping[str, JsonValue] | None = None, **params: JsonValue
) -> dict[str, JsonValue]:
    """`vertex_ai/gemini-2.5-flash` at `<peer>/<label>` with a service account whose token URI is the peer.

    An `api_base` with a path is used as `{api_base}:generateContent` (`_check_custom_proxy` in
    `litellm/llms/vertex_ai/vertex_llm_base.py`), which still starts with the label.
    """
    return _entry(
        group,
        label,
        f"vertex_ai/{GEMINI_MODEL}",
        {
            "api_base": f"{base}/{label}",
            "vertex_project": VERTEX_PROJECT,
            "vertex_location": VERTEX_LOCATION,
            "vertex_credentials": service_account_text(base),
            **params,
        },
        model_info,
    )


Builder = Callable[..., dict[str, JsonValue]]
BUILDERS: Final[Mapping[str, Builder]] = {
    "openai": openai_deployment,
    "azure": azure_deployment,
    "anthropic": anthropic_deployment,
    "bedrock": bedrock_deployment,
    "gemini": gemini_deployment,
    "vertex": vertex_deployment,
}


def failing_then_healthy(
    provider: str, group: str, failing: str, healthy: str, base: str, **failing_params: JsonValue
) -> tuple[dict[str, JsonValue], ...]:
    """A failing deployment with `weight: 1` and a healthy OpenAI sibling without one.

    Simple shuffle normalises weights, so the weighted deployment takes every request while it is healthy
    and the sibling is reached only once the weighted one is in cooldown (the pattern of
    `test_dec1_routing_cooldown_defaults.py`)."""
    return (
        BUILDERS[provider](group, failing, base, weight=1, **failing_params),
        openai_deployment(group, healthy, base),
    )


def pair_of(provider: str, group: str, label: str, base: str, **params: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    """Two deployments of one provider in one group, labelled `<label>-a` and `<label>-b`.

    A group with more than one deployment never sleeps between retries (`time_to_sleep` in
    `router-core/src/call.rs:1205`), so retry counts are observed without waiting for backoff."""
    return tuple(BUILDERS[provider](group, f"{label}-{suffix}", base, **params) for suffix in ("a", "b"))


# ---------------------------------------------------------------- owned proxy


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    peer: Peer

    def both(self, label: str) -> int:
        """Attempts that reached either deployment of a `pair_of` group."""
        return self.peer.count(f"{label}-a") + self.peer.count(f"{label}-b")


@contextmanager
def reliab_rig(
    directory: Path,
    models: Callable[[str], Sequence[Mapping[str, JsonValue]]],
    router_settings: Mapping[str, JsonValue],
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Rig]:
    """The provider peer plus one owned proxy whose `model_list` is `models(peer.url)`."""
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        peer: Final = stack.enter_context(provider_peer())
        gateway: Final = stack.enter_context(
            routing_proxy(
                shared,
                directory,
                model_list=models(peer.url),
                router_settings=router_settings,
                general_settings=general_settings,
            )
        )
        yield Rig(gateway, peer)


# ---------------------------------------------------------------- calls and reads


def marker() -> str:
    return f"dec1-reliab-{uuid.uuid4().hex}"


def chat(
    gateway: Gateway, model: str, *, headers: Mapping[str, str] | None = None, **extra: JsonValue
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": marker()}], **extra}
    return gateway.request("POST", "/v1/chat/completions", body, headers=headers)


def stream_chat(gateway: Gateway, model: str, **extra: JsonValue) -> tuple[int, str]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": marker()}],
        **extra,
    }
    with gateway.client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        return response.status_code, response.read().decode()


def burst(gateway: Gateway, model: str, requests: int) -> tuple[int, ...]:
    return tuple(chat(gateway, model).status_code for _ in range(requests))


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def content_of(response: httpx.Response) -> str:
    choices: Final = JSON_OBJECT.validate_json(response.content)["choices"]
    assert isinstance(choices, list) and choices, response.text
    return string_value(object_value(object_value(choices[0])["message"])["content"])
