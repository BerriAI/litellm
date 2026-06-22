"""Spend-tracking e2e client: a Gateway plus the spend-specific read endpoints.

Generic proxy operations (keys, customers, chat/embed, route probing, SpendLogs
polling) come from the shared Gateway, DI'd in (composition, not inheritance).
This client adds only the spend surface: /spend/calculate, /spend/tags,
key-spend polling, and the route probes the breadth test uses.

Re-exports unwrap / is_ok / unique_marker / SpendLogRow so the tests import their
helpers from one place.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from e2e_config import unique_marker
from e2e_http import (
    NetworkError,
    NoBody,
    ProbeResult,
    RateLimitedError,
    Result,
    StreamingResponse,
    Success,
    UnauthorizedError,
    UnknownApiError,
    ValidationError,
    is_ok,
    unwrap,
)
from e2e_gateway import Gateway, build_gateway
from models import (
    ChatBody,
    ChatMessage,
    ChatMetadata,
    ChatResponse,
    DateRangeParams,
    EmbedBody,
    EmbedResponse,
    OpenAPISchema,
    SpendCalculateBody,
    SpendCalculateResponse,
    SpendLogRow,
    TagSpend,
    TagSpends,
)

__all__ = [
    "SpendClient",
    "SpendTagsError",
    "TagSpendPoll",
    "build_client",
    "reset_spend_logs",
    "unique_marker",
    "unwrap",
    "is_ok",
    "SpendLogRow",
    "ProbeResult",
]


class SpendTagsError(Exception):
    """A /spend/tags call that did not return 200. Carries the real HTTP status
    and body so a server error surfaces verbatim instead of as a missing tag."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"/spend/tags returned HTTP {status}: {body[:500]}")


def _spend_tags_error(result: Result[TagSpends]) -> SpendTagsError:
    match result:
        case Success():
            raise AssertionError("not an error result")
        case UnauthorizedError():
            return SpendTagsError(401, "unauthorized")
        case RateLimitedError(retry_after_seconds=retry, body=body):
            return SpendTagsError(429, f"rate_limited (retry_after={retry}): {body}")
        case UnknownApiError(status_code=status, body=body):
            return SpendTagsError(status, body)
        case NetworkError(message=message):
            return SpendTagsError(-1, f"network error: {message}")
        case ValidationError(message=message):
            return SpendTagsError(-2, f"response did not match TagSpends: {message}")


@dataclass(frozen=True, slots=True)
class TagSpendPoll:
    """Outcome of polling /spend/tags for one tag: the last matching entry (if any),
    whether the endpoint ever returned 200, and a human-readable description of the
    last observed HTTP outcome for failure messages."""

    entry: TagSpend | None
    saw_ok: bool
    last_status: str


def reset_spend_logs() -> None:
    """Truncate LiteLLM_SpendLogs for a clean slate. No proxy endpoint deletes
    spend logs (/global/spend/reset keeps them), so go to the DB directly. Uses
    DATABASE_URL (default: the local docker postgres on its mapped host port; note
    the in-container `@db` host isn't resolvable from the host, so default to
    localhost).
    """
    import psycopg

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm",
    )
    with psycopg.connect(url) as conn:
        _ = conn.execute('TRUNCATE TABLE "LiteLLM_SpendLogs"')


def _chat_body(
    model: str,
    content: str,
    *,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    user: str | None = None,
    stream: bool = False,
) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=content)],
        max_tokens=max_tokens,
        stream=stream,
        user=user,
        metadata=ChatMetadata(tags=tags) if tags else None,
    )


@dataclass(frozen=True, slots=True)
class SpendClient:
    gateway: Gateway

    def chat(
        self,
        key: str,
        model: str,
        content: str,
        *,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        user: str | None = None,
    ) -> Result[ChatResponse]:
        return self.gateway.chat(
            key, _chat_body(model, content, max_tokens=max_tokens, tags=tags, user=user)
        )

    def chat_stream(
        self, key: str, model: str, content: str, *, max_tokens: int | None = None
    ) -> StreamingResponse:
        return self.gateway.chat_stream(
            key, _chat_body(model, content, max_tokens=max_tokens, stream=True)
        )

    def embed(self, key: str, model: str, content: str) -> Result[EmbedResponse]:
        return self.gateway.embed(key, EmbedBody(model=model, input=content))

    def poll_logs_for_key(
        self,
        key: str,
        *,
        min_rows: int = 1,
        predicate: Callable[[list[SpendLogRow]], bool] | None = None,
    ) -> list[SpendLogRow]:
        return self.gateway.poll_logs_for_key(
            key, min_rows=min_rows, predicate=predicate
        )

    def calculate_spend(self, model: str, content: str) -> float:
        return unwrap(
            self.gateway.transport.post(
                "/spend/calculate",
                headers=self.gateway.transport.master,
                json=SpendCalculateBody(
                    model=model, messages=[ChatMessage(role="user", content=content)]
                ),
                response_type=SpendCalculateResponse,
            )
        ).cost

    def spend_by_tags(self) -> list[TagSpend]:
        """Tagged aggregates from /spend/tags. A non-200 is a real server failure,
        not "no tags": raise SpendTagsError with the actual status and body so it
        surfaces instead of being swallowed into an empty list."""
        result = self.gateway.transport.get(
            "/spend/tags",
            headers=self.gateway.transport.master,
            params=NoBody(),
            response_type=TagSpends,
        )
        match result:
            case Success(data=data):
                return list(data.root)
            case _:
                raise _spend_tags_error(result)

    def poll_tag_spend(self, tag: str, *, minimum: float = 0.0) -> TagSpendPoll:
        """Poll /spend/tags until the tag's aggregate reaches `minimum`. A hard
        server error (any non-200) propagates immediately as SpendTagsError rather
        than being polled into a timeout; eventual consistency only manifests as a
        200 whose payload does not yet carry the tag, so only that case waits."""
        deadline = time.monotonic() + self.gateway.poll_timeout
        entry: TagSpend | None = None
        while time.monotonic() < deadline:
            tags = self.spend_by_tags()
            matches = [t for t in tags if t.individual_request_tag == tag]
            if matches:
                entry = matches[0]
                if (entry.total_spend or 0.0) >= minimum:
                    return TagSpendPoll(entry=entry, saw_ok=True, last_status="HTTP 200")
            time.sleep(self.gateway.poll_interval)
        return TagSpendPoll(entry=entry, saw_ok=True, last_status="HTTP 200")

    def poll_key_spend(self, key: str, *, minimum: float = 0.0) -> float:
        deadline = time.monotonic() + self.gateway.poll_timeout
        spend = 0.0
        while time.monotonic() < deadline:
            spend = self.gateway.key_info(key).spend or 0.0
            if spend > minimum:
                return spend
            time.sleep(self.gateway.poll_interval)
        return spend

    def probe(self, path: str, *, params: DateRangeParams) -> ProbeResult:
        return self.gateway.transport.probe(path, params=params)

    def openapi(self) -> OpenAPISchema:
        return unwrap(
            self.gateway.transport.get(
                "/openapi.json",
                headers=self.gateway.transport.master,
                params=NoBody(),
                response_type=OpenAPISchema,
            )
        )


def build_client() -> SpendClient:
    return SpendClient(gateway=build_gateway())
