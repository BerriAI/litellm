"""What crosses the process boundary between the test and the Locust runner: the wire plan (one
entry per planned request, ready to send) and the record each request leaves behind."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Endpoint = Literal["chat_completions", "responses", "messages"]


class TokenSource(BaseModel):
    """How the runner mints a JWT for a Keycloak user, so a run longer than the realm's access
    token lifespan keeps authenticating as that user."""

    token_url: str
    client_id: str
    username: str
    password: str = Field(repr=False)


class WireRequest(BaseModel):
    seq: int
    marker: str
    caller: str
    endpoint: Endpoint
    path: str
    body: dict[str, object]
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
