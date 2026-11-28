"""Shared building blocks for the ``test_dec1_budlim2_*`` contracts.

These files pin budgets, rate limits and model capacity on the request surfaces other than
``/v1/chat/completions``: ``/v1/messages`` (and its ``count_tokens``), ``/v1/responses``, ``/v1/embeddings``,
``/v1/completions``, ``/v1/audio/transcriptions``, ``/v1/images/edits``, ``/v1/batches``, the ``/v1/realtime``
websocket and MCP tool calls.

Where the pieces live in the product:

    budget hold at auth     litellm/proxy/spend_tracking/budget_reservation.py ``reserve_budget_for_request``
                            (estimate in Python, hold taken by the Rust engine
                            ``litellm/rust/crates/callbacks-budgets/src/engine.rs`` since #51535)
    hold release            litellm/proxy/middleware/budget_reservation_release_middleware.py
    rate limiter            litellm/proxy/hooks/parallel_request_limiter_v3.py
    Anthropic error shape   litellm/proxy/common_utils/anthropic_error_payload.py

Price used by every hold test here (``surface_model``): input tokens are free and every output token costs
``OUTPUT_RATE`` (0.004). With free input, the admission estimate of a request is exactly
``min(requested output cap or 16384, model ceiling) * 0.004 * max(n, best_of, 1)``
(``_max_cost_for_cost_info``), so every hold below is a literal. The scripted provider is the one from
``dec1_spend_support.respond``, whose fixed usage gives these real costs at that price:

    /v1/messages        3 output tokens   -> 0.012
    /v1/responses       4 output tokens   -> 0.016
    /v1/completions     2 output tokens   -> 0.008
    /v1/chat/completions 5 output tokens  -> 0.02

A prompt that contains ``HOLD_MARKER`` is kept waiting by the provider until ``release()``, so a test can read
the Redis spend counter while the request is in flight.
"""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_spend_support import respond as spend_respond

OUTPUT_RATE: Final = 0.004
NO_CAP_OUTPUT_TOKENS: Final = 16384
NO_CAP_HOLD: Final = NO_CAP_OUTPUT_TOKENS * OUTPUT_RATE
MESSAGES_REAL_COST: Final = 3 * OUTPUT_RATE
RESPONSES_REAL_COST: Final = 4 * OUTPUT_RATE
TEXT_REAL_COST: Final = 2 * OUTPUT_RATE
CHAT_REAL_COST: Final = 5 * OUTPUT_RATE
COUNTED_INPUT_TOKENS: Final = 12
HOLD_MARKER: Final = "dec1-budlim2-hold"
PROMPT_TOKENS: Final = len("dec1-budlim2 " + uuid.uuid4().hex) // 4
HELD_PROMPT_TOKENS: Final = len(f"{HOLD_MARKER} " + uuid.uuid4().hex) // 4
WINDOW_SECONDS: Final = 60
ARRIVAL_SECONDS: Final = 30
ANTHROPIC_VERSION: Final = {"anthropic-version": "2023-06-01"}
MESSAGES_PROVIDER_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
WAV_BYTES: Final = (
    b"RIFF\x30\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x08\x00\x00\x00"
    b"\x00\x01\x02\x03\x04\x05\x06\x07"
)
WAV_CONTENT_TYPE: Final = mimetypes.guess_type("a.wav")[0] or "application/octet-stream"
PNG_BYTES: Final = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
TRANSCRIPT: Final = "dec1 budlim2 transcript"


def _transcription_reply() -> Reply:
    return Reply(body=json.dumps({"text": TRANSCRIPT}).encode())


def _image_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {"created": 1, "data": [{"b64_json": "aW1hZ2U=", "revised_prompt": None}], "usage": None}
        ).encode()
    )


def _count_reply() -> Reply:
    return Reply(body=json.dumps({"input_tokens": COUNTED_INPUT_TOKENS}).encode())


@dataclass
class SurfaceProvider:
    """A scripted provider for every surface in this family.

    Generation routes answer with the fixed usage of ``dec1_spend_support.respond``; ``count_tokens`` answers
    ``{"input_tokens": 12}``; transcriptions answer a fixed text; image edits answer one base64 image. A request
    whose body contains ``HOLD_MARKER`` waits until ``release()``. ``failing_status`` makes every generation
    call fail with that status and an OpenAI shaped body.
    """

    wire: Wire | None = None
    release_gate: threading.Event = field(default_factory=threading.Event)
    arrived: threading.Condition = field(default_factory=threading.Condition)
    arrivals: int = 0
    failing_status: int | None = None

    def answer(self, request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        path: Final = request.target.split("?", 1)[0]
        if path.endswith("/count_tokens"):
            return _count_reply()
        if HOLD_MARKER.encode() in request.body:
            with self.arrived:
                self.arrivals += 1
                self.arrived.notify_all()
            assert self.release_gate.wait(timeout=60), "the held call was never released"
        if self.failing_status is not None:
            return Reply(
                status=self.failing_status,
                body=json.dumps(
                    {"error": {"message": "dec1 budlim2 scripted failure", "type": "server_error", "code": None}}
                ).encode(),
            )
        if path.endswith("/audio/transcriptions"):
            return _transcription_reply()
        if path.endswith("/images/edits"):
            return _image_reply()
        return spend_respond(request)

    def wait_for(self, count: int, seconds: float = ARRIVAL_SECONDS) -> None:
        with self.arrived:
            assert self.arrived.wait_for(lambda: self.arrivals >= count, timeout=seconds), (
                f"only {self.arrivals} of {count} held calls reached the provider"
            )

    def release(self) -> None:
        self.release_gate.set()

    def posts(self) -> tuple[Request, ...]:
        """Every non-GET request the provider received since the last call."""
        assert self.wire is not None
        return tuple(request for request in self.wire.drain() if request.method != "GET")

    @property
    def url(self) -> str:
        assert self.wire is not None
        return self.wire.url


@contextmanager
def surface_provider(failing_status: int | None = None) -> Iterator[SurfaceProvider]:
    provider: Final = SurfaceProvider(failing_status=failing_status)
    with wire_server(provider.answer) as wire:
        provider.wire = wire
        try:
            yield provider
        finally:
            provider.release()


def surface_model(scenario: Scenario, provider: SurfaceProvider, **parameters: JsonValue) -> str:
    """An ``openai/gpt-4o-mini`` deployment (override with ``model=``) with free input and 0.004 per output token."""
    return scenario.model(
        **{
            "model": "openai/gpt-4o-mini",
            "api_base": f"{provider.url}/v1",
            "api_key": "dec1-budlim2-provider-key",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": OUTPUT_RATE,
            **parameters,
        }
    )


def messages_model(scenario: Scenario, provider: SurfaceProvider, **parameters: JsonValue) -> str:
    """An ``anthropic/claude-sonnet-4-5`` deployment on the scripted provider, priced like ``surface_model``."""
    return surface_model(scenario, provider, model=MESSAGES_PROVIDER_MODEL, api_base=provider.url, **parameters)


def text_model(scenario: Scenario, provider: SurfaceProvider) -> str:
    return surface_model(scenario, provider, model="text-completion-openai/gpt-3.5-turbo-instruct")


def embedding_model(scenario: Scenario, provider: SurfaceProvider, **parameters: JsonValue) -> str:
    return surface_model(scenario, provider, model="openai/text-embedding-3-small", **parameters)


def transcription_model(scenario: Scenario, provider: SurfaceProvider) -> str:
    """An ``openai/whisper-1`` deployment; whisper is priced per second in the cost map, not per token."""
    return scenario.model(model="openai/whisper-1", api_base=f"{provider.url}/v1", api_key="dec1-budlim2-key")


def image_edit_model(scenario: Scenario, provider: SurfaceProvider, **model_info: JsonValue) -> str:
    return scenario.model(
        model="openai/gpt-image-1",
        api_base=f"{provider.url}/v1",
        api_key="dec1-budlim2-key",
        model_info=dict(model_info) if model_info else None,
    )


def prompt(hold: bool = False) -> str:
    """A unique prompt, so the response cache never answers; with ``hold`` the provider keeps the call waiting."""
    return f"{HOLD_MARKER if hold else 'dec1-budlim2'} {uuid.uuid4().hex}"


def messages(
    gateway: Gateway,
    model: str,
    key: str,
    *,
    hold: bool = False,
    max_tokens: int | None = 16,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """One ``POST /v1/messages``; ``max_tokens=None`` leaves the field out of the body."""
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt(hold)}],
        **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        **extra,
    }
    return gateway.request("POST", "/v1/messages", body, key=key, headers={**ANTHROPIC_VERSION, **(headers or {})})


def count_tokens(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/messages/count_tokens",
        {"model": model, "messages": [{"role": "user", "content": prompt()}]},
        key=key,
        headers=ANTHROPIC_VERSION,
    )


def responses(gateway: Gateway, model: str, key: str, *, hold: bool = False, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/responses", {"model": model, "input": prompt(hold), **extra}, key=key)


def text_completion(
    gateway: Gateway, model: str, key: str, *, hold: bool = False, **extra: JsonValue
) -> httpx.Response:
    return gateway.request("POST", "/v1/completions", {"model": model, "prompt": prompt(hold), **extra}, key=key)


def embedding(gateway: Gateway, model: str, key: str, *, hold: bool = False) -> httpx.Response:
    return gateway.request("POST", "/v1/embeddings", {"model": model, "input": prompt(hold)}, key=key)


def chat(gateway: Gateway, model: str, key: str, *, hold: bool = False, **extra: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt(hold)}], **extra},
        key=key,
    )


def transcription(gateway: Gateway, model: str, key: str, *, hold: bool = False) -> httpx.Response:
    """A multipart transcription; the ``prompt`` form field carries the hold marker when ``hold`` is set."""
    return gateway.request_multipart(
        "/v1/audio/transcriptions",
        {"model": model, "prompt": prompt(hold)},
        {"file": ("a.wav", WAV_BYTES, WAV_CONTENT_TYPE)},
        key=key,
    )


def image_edit(gateway: Gateway, model: str, key: str, *, n: int = 1, hold: bool = False) -> httpx.Response:
    return gateway.request_multipart(
        "/v1/images/edits",
        {"model": model, "prompt": prompt(hold), "n": str(n)},
        {"image": ("a.png", PNG_BYTES, "image/png")},
        key=key,
    )


def anthropic_error(response: httpx.Response, status: int) -> dict[str, JsonValue]:
    """Assert the Anthropic error envelope ``{"type": "error", "error": {...}}`` with ``status``; return ``error``."""
    assert response.status_code == status, f"{response.status_code} {response.text}"
    assert response.headers["content-type"].startswith("application/json"), dict(response.headers)
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert body["type"] == "error", response.text
    return object_value(body["error"])


def openai_error(response: httpx.Response, status: int) -> dict[str, JsonValue]:
    """Assert an OpenAI error envelope ``{"error": {...}}`` with ``status``; return ``error``."""
    assert response.status_code == status, f"{response.status_code} {response.text}"
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert "type" not in body or body.get("type") != "error", response.text
    return object_value(body["error"])


def message_of(error: Mapping[str, JsonValue]) -> str:
    return string_value(error["message"])


def redis_url() -> tuple[str, int]:
    return os.environ["REDIS_HOST"], int(os.environ["REDIS_PORT"])
