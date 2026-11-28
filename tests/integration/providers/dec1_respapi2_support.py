"""Shared rig for the `test_dec1_respapi2_*` files: `/v1/responses` across bridged providers, round 2.

Round 1 (`dec1_respapi_support.py`, imported unchanged) pinned the provider-agnostic bridge on three carriers: an
`openai/` deployment with `use_chat_completions_api: true`, DeepSeek and Anthropic. Round 2 adds the carriers the
round 1 index listed as not covered, each served by the same owned loopback peer:

    GEMINI    `gemini/gemini-2.5-flash` (Google AI Studio) with `api_base` `<peer>/v1beta`, so the peer sees
              `POST /v1beta/models/gemini-2.5-flash:generateContent` (or `:streamGenerateContent`).
    BEDROCK   `bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0` with a bearer `api_key`, so the peer sees
              `POST /model/<id>/converse`.
    GROQ      `groq/llama-3.3-70b-versatile` with `api_base` `<peer>/v1`, so the peer sees
              `POST /v1/chat/completions`.

None of them has a native Responses config (`litellm/utils.py:10709` lists the providers that do), so every
`/v1/responses` call on them goes through `litellm/responses/litellm_completion_transformation/`. The peer
answers each path with a canned reply that says `REPLY_TEXT` with 11 input and 7 output tokens, so every
deployment priced at 0.001 and 0.002 per token costs `CALL_COST` (0.025) per call. Nothing is sent to a real
provider host: every `api_base` is the loopback peer.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Final

from integration._support.client import Gateway, Scenario, eventually, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_respapi_support import (
    COMPLETION_TOKENS,
    LANDING_SECONDS,
    NO_RETRIES,
    PRICES,
    PROMPT_TOKENS,
    PROVIDER_KEY,
    REPLY_TEXT,
    Call,
    Peer,
    default_respond,
    json_reply,
)
from pydantic import JsonValue

GEMINI_BACKEND: Final = "gemini-2.5-flash"
GEMINI: Final = f"gemini/{GEMINI_BACKEND}"
GEMINI_PATH: Final = f"/v1beta/models/{GEMINI_BACKEND}:generateContent"
GEMINI_STREAM_PATH: Final = f"/v1beta/models/{GEMINI_BACKEND}:streamGenerateContent"
BEDROCK_BACKEND: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK: Final = f"bedrock/{BEDROCK_BACKEND}"
GROQ_BACKEND: Final = "llama-3.3-70b-versatile"
GROQ: Final = f"groq/{GROQ_BACKEND}"
WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def gemini_usage(prompt: int = PROMPT_TOKENS, candidates: int = COMPLETION_TOKENS) -> dict[str, JsonValue]:
    return {"promptTokenCount": prompt, "candidatesTokenCount": candidates, "totalTokenCount": prompt + candidates}


def gemini_answer(
    *, parts: Sequence[Mapping[str, JsonValue]] | None = None, finish: str = "STOP"
) -> dict[str, JsonValue]:
    """One Gemini candidate saying `REPLY_TEXT` (or `parts`) with the canned usage."""
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [dict(part) for part in parts] if parts is not None else [{"text": REPLY_TEXT}],
                },
                "finishReason": finish,
                "index": 0,
            }
        ],
        "usageMetadata": gemini_usage(),
        "modelVersion": GEMINI_BACKEND,
    }


def gemini_stream(pieces: Sequence[str] = ("dec1 ", "gemini ", "reply"), finish: str = "STOP") -> Reply:
    """One SSE frame per piece; the last frame carries `finishReason` and the canned usage."""
    frames: Final[list[dict[str, JsonValue]]] = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": piece}]}, "index": 0}]} for piece in pieces
    ]
    frames[-1] = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": pieces[-1]}]}, "index": 0, "finishReason": finish}
        ],
        "usageMetadata": gemini_usage(),
        "modelVersion": GEMINI_BACKEND,
    }
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\r\n\r\n".encode() for frame in frames),
    )


def converse_answer(
    *, content: Sequence[Mapping[str, JsonValue]] | None = None, stop_reason: str = "end_turn"
) -> dict[str, JsonValue]:
    """A Bedrock Converse answer saying `REPLY_TEXT` (or `content`) with the canned usage."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [dict(block) for block in content] if content is not None else [{"text": REPLY_TEXT}],
            }
        },
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": PROMPT_TOKENS,
            "outputTokens": COMPLETION_TOKENS,
            "totalTokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        },
        "metrics": {"latencyMs": 1},
    }


def every_respond(call: Call) -> Reply:
    """The canned answer for whichever provider path the proxy called."""
    if call.method == "POST" and ":streamGenerateContent" in call.path:
        return gemini_stream()
    if call.method == "POST" and call.path.endswith(":generateContent"):
        return json_reply(gemini_answer())
    if call.method == "POST" and call.path.endswith("/converse"):
        return json_reply(converse_answer())
    return default_respond(call)


def gemini(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    """A Google AI Studio deployment whose base is `<peer>/v1beta`."""
    return scenario.model(
        model=GEMINI, api_base=f"{provider.url}/v1beta", api_key=PROVIDER_KEY, **NO_RETRIES, **PRICES, **parameters
    )


def bedrock(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    """A Bedrock Converse deployment with bearer key auth at the peer."""
    return scenario.model(
        model=BEDROCK,
        api_base=provider.url,
        api_key=PROVIDER_KEY,
        aws_region_name="us-east-1",
        **NO_RETRIES,
        **PRICES,
        **parameters,
    )


def groq(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    """A Groq deployment whose base is `<peer>/v1`."""
    return scenario.model(
        model=GROQ, api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY, **NO_RETRIES, **PRICES, **parameters
    )


def gemini_texts(call: Call) -> list[tuple[str, str]]:
    """`(role, joined text)` for every Gemini `contents` entry the proxy sent."""
    return [
        (
            str(object_value(entry)["role"]),
            "".join(str(object_value(part).get("text", "")) for part in list_value(object_value(entry)["parts"])),
        )
        for entry in list_value(call.body["contents"])
    ]


def converse_texts(call: Call) -> list[tuple[str, str]]:
    """`(role, joined text)` for every Converse `messages` entry the proxy sent."""
    return [
        (
            str(object_value(entry)["role"]),
            "".join(str(object_value(block).get("text", "")) for block in list_value(object_value(entry)["content"])),
        )
        for entry in list_value(call.body["messages"])
    ]


def key_spend(gateway: Gateway, key: str, at_least: float) -> float:
    """The key's stored spend once it reaches `at_least`."""
    digest: Final = sha256(key.encode()).hexdigest()
    rows: Final = eventually(
        lambda: gateway.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest,)),
        lambda found: len(found) == 1 and float(str(found[0]["spend"])) >= at_least - 1e-9,
        seconds=LANDING_SECONDS,
    )
    return float(str(rows[0]["spend"]))


def unique_text() -> str:
    return f"dec1-respapi2-{uuid.uuid4().hex}"
