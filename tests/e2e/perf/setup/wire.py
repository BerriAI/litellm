"""What crosses the process boundary between the test and the Locust runner: the wire plan (one
entry per planned request, with the shape the runner expands into the body right before sending,
so a 300k-token request costs a few hundred bytes in the plan file) and the record each request
leaves behind."""

from __future__ import annotations

import hashlib
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

Endpoint = Literal["chat_completions", "responses", "messages"]


class TokenSource(BaseModel):
    """How the runner mints a JWT for a Keycloak user, so a run longer than the realm's access
    token lifespan keeps authenticating as that user."""

    token_url: str
    client_id: str
    username: str
    password: str = Field(repr=False)


class WireShape(BaseModel):
    """`RequestShape` on the wire: the token counts the runner turns into text."""

    model_config = ConfigDict(frozen=True)

    system_tokens: int
    history_turns: int
    turn_tokens: int
    user_tokens: int
    tools: bool
    output_tokens: int


class WireRequest(BaseModel):
    seq: int
    marker: str
    caller: str
    endpoint: Endpoint
    path: str
    model: str
    shape: WireShape
    authorization: str | None = Field(default=None, repr=False)
    token_source: TokenSource | None = None
    stream: bool
    expected_text: str
    blocked: bool


class WireRecord(BaseModel):
    seq: int
    marker: str
    caller: str
    endpoint: Endpoint
    stream: bool
    blocked: bool
    started_at: float
    status: int
    header_s: float
    first_byte_s: float | None
    first_token_s: float | None
    total_s: float
    text: str
    ok: bool
    mock_delay_s: float | None
    call_id: str | None
    error: str | None


_WORDS: Final = (
    "customer",
    "order",
    "invoice",
    "shipment",
    "returns",
    "policy",
    "account",
    "balance",
    "ticket",
    "status",
    "update",
    "region",
    "warehouse",
    "support",
    "agent",
    "summary",
    "quarter",
    "report",
    "renewal",
    "contract",
    "delivery",
    "address",
    "payment",
    "refund",
)
_WORDS_PER_TOKEN: Final = 0.47
_TOOLS: Final[tuple[tuple[str, str, dict[str, object]], ...]] = (
    (
        "get_weather",
        "Current weather for a city",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    ),
    (
        "search_orders",
        "Find a customer's orders, optionally by status",
        {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "status": {"type": "string"}},
            "required": ["customer_id"],
        },
    ),
)


def filler_text(seed: str, tokens: int) -> str:
    """About `tokens` tokens of plain prose, the same text for the same seed on every run."""
    count: Final = max(1, round(tokens * _WORDS_PER_TOKEN))
    start: Final = int.from_bytes(hashlib.sha1(seed.encode()).digest()[:4], "big")
    return " ".join(_WORDS[(start + index * 5) % len(_WORDS)] for index in range(count))


def _openai_tools() -> list[dict[str, object]]:
    return [
        {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
        for name, description, parameters in _TOOLS
    ]


def _responses_tools() -> list[dict[str, object]]:
    return [
        {"type": "function", "name": name, "description": description, "parameters": parameters}
        for name, description, parameters in _TOOLS
    ]


def _anthropic_tools() -> list[dict[str, object]]:
    return [
        {"name": name, "description": description, "input_schema": parameters}
        for name, description, parameters in _TOOLS
    ]


def _turns(item: WireRequest) -> list[dict[str, object]]:
    """The history as alternating user and assistant turns, then the final user message carrying
    the marker."""
    earlier: Final[list[dict[str, object]]] = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": filler_text(f"{item.marker}:turn{index}", item.shape.turn_tokens),
        }
        for index in range(2 * item.shape.history_turns)
    ]
    prompt: Final = f"{item.marker} {filler_text(f'{item.marker}:user', item.shape.user_tokens)}"
    return [*earlier, {"role": "user", "content": prompt}]


def request_body(item: WireRequest) -> dict[str, object]:
    """The request in the endpoint's own wire format, sized by the planned shape."""
    system: Final = filler_text(f"{item.marker}:system", item.shape.system_tokens) if item.shape.system_tokens else ""
    if item.endpoint == "responses":
        return {
            "model": item.model,
            "input": _turns(item),
            "max_output_tokens": item.shape.output_tokens,
            "stream": item.stream,
            **({"instructions": system} if system else {}),
            **({"tools": _responses_tools()} if item.shape.tools else {}),
        }
    if item.endpoint == "messages":
        return {
            "model": item.model,
            "messages": _turns(item),
            "max_tokens": item.shape.output_tokens,
            "stream": item.stream,
            **({"system": system} if system else {}),
            **({"tools": _anthropic_tools()} if item.shape.tools else {}),
        }
    return {
        "model": item.model,
        "messages": [*([{"role": "system", "content": system}] if system else []), *_turns(item)],
        "max_tokens": item.shape.output_tokens,
        "stream": item.stream,
        **({"tools": _openai_tools()} if item.shape.tools else {}),
    }
