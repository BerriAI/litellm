"""What crosses the process boundary between the test and the Locust runner: the wire plan (one
entry per planned request, with the shape the runner expands into the body right before sending,
so a 300k-token request costs a few hundred bytes in the plan file) and the record each request
leaves behind."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from setup.corpus.session import Endpoint as CorpusEndpoint
from setup.corpus.session import request_body as corpus_body

Endpoint = Literal["chat_completions", "responses", "messages"]
_PAYLOAD: Final = TypeAdapter(dict[str, object])


class TokenSource(BaseModel):
    """How the runner mints a JWT for a Keycloak user, so a run longer than the realm's access
    token lifespan keeps authenticating as that user."""

    token_url: str
    client_id: str
    username: str
    password: str = Field(repr=False)


class WireShape(BaseModel):
    """`RequestShape` on the wire: what the runner hands the corpus to build the body."""

    model_config = ConfigDict(frozen=True)

    name: str
    input_tokens: int
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
    response_id: str | None = None


def response_id(payload: Mapping[str, object]) -> str | None:
    nested: Final = payload.get("response") or payload.get("message")
    source: Final = _PAYLOAD.validate_python(nested) if isinstance(nested, Mapping) else payload
    identifier: Final = source.get("id")
    return identifier if isinstance(identifier, str) else None


class PlanProgress:
    def __init__(self, total: int) -> None:
        self.total: Final = total
        self.issued = 0
        self.completed = 0
        self.stopped = False

    def take(self, *, stop: bool) -> bool:
        self.stopped = self.stopped or stop
        if self.stopped or self.issued >= self.total:
            return False
        self.issued += 1
        return True

    @property
    def drained(self) -> bool:
        return (self.stopped or self.issued == self.total) and self.completed == self.issued


_CORPUS_ENDPOINT: Final[Mapping[Endpoint, CorpusEndpoint]] = MappingProxyType(
    {"chat_completions": "chat", "responses": "responses", "messages": "messages"}
)


def request_body(item: WireRequest) -> dict[str, object]:
    """The request in the endpoint's own wire format: the planned shape's coding-agent session, the marker in
    the final user message, built right before sending so the plan file stays small."""
    return corpus_body(
        endpoint=_CORPUS_ENDPOINT[item.endpoint],
        model=item.model,
        marker=item.marker,
        input_tokens=item.shape.input_tokens,
        tools=item.shape.tools,
        output_tokens=item.shape.output_tokens,
        stream=item.stream,
    )
