"""Client for the logging e2e suite.

Two jobs live here. The first is driving traffic through the proxy and scraping
its Prometheus ``/metrics`` endpoint (plaintext, so it goes through
``transport.probe``). The second is verifying Datadog delivery end to end: the
proxy ships every request's StandardLoggingPayload to the Datadog logs intake on
its ``datadog`` success/failure callback, and ``DatadogClient`` reads those events
back out through the Datadog Logs Search API to prove they actually landed.

The read-back is a real external call, so it still goes through the shared
``HttpTransport`` (the only sanctioned path to ``requests``) - just pointed at the
Datadog API host instead of the proxy. Verification is a poll, not a push: the
proxy batches and flushes asynchronously (every 5s or at ``DD_MAX_BATCH_SIZE``)
and Datadog then indexes for search, so a test drives traffic and waits for the
marker it stamped to become searchable rather than "flushing" anything itself.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, REQUEST_TIMEOUT
from e2e_gateway import Gateway, build_gateway
from e2e_http import Headers, NoBody, Result, StreamingResponse, Success, unwrap
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody
from transport import HttpTransport


class DatadogSearchHeaders(Headers):
    """Auth + content type for the Datadog Logs API. Datadog authenticates reads
    with an API key plus an *application* key (writes/intake need only the API
    key), sent as their own headers rather than a bearer token."""

    dd_api_key: str = Field(serialization_alias="DD-API-KEY")
    dd_application_key: str = Field(serialization_alias="DD-APPLICATION-KEY")
    content_type: str = Field(default="application/json", serialization_alias="Content-Type")


class DatadogSearchFilter(BaseModel):
    query: str
    from_: str = Field(default="now-15m", serialization_alias="from")
    to: str = "now"


class DatadogSearchPage(BaseModel):
    limit: int = 100


class DatadogSearchBody(BaseModel):
    """POST /api/v2/logs/events/search body. ``sort=-timestamp`` returns newest
    first so a small ``page.limit`` still sees the events a test just produced."""

    filter: DatadogSearchFilter
    page: DatadogSearchPage = DatadogSearchPage()
    sort: str = "-timestamp"


class DatadogLogAttributes(BaseModel):
    message: str | None = None
    status: str | None = None
    service: str | None = None
    tags: list[str] = []
    timestamp: str | None = None


class DatadogLogEvent(BaseModel):
    id: str | None = None
    attributes: DatadogLogAttributes | None = None


class DatadogSearchResponse(BaseModel):
    data: list[DatadogLogEvent] = []


DD_ERROR = "error"


class ResponsesBody(BaseModel):
    """Minimal POST /v1/responses body: the fields a logging test needs to drive a
    real completion through the Responses API and get it shipped to Datadog."""

    model: str
    input: str


@dataclass(frozen=True, slots=True)
class DatadogClient:
    """Reads litellm's shipped logs back out of Datadog to prove delivery.

    Wraps an ``HttpTransport`` aimed at the Datadog API host (``api.<DD_SITE>``);
    every call still flows through the shared e2e_http layer, so no test touches
    ``requests``. Built from ``DD_API_KEY`` / ``DD_SITE`` / ``DD_APP_KEY`` in the
    environment (see ``build_datadog_client``)."""

    transport: HttpTransport
    api_key: str
    app_key: str
    poll_timeout: float = POLL_TIMEOUT
    poll_interval: float = POLL_INTERVAL

    def _headers(self) -> DatadogSearchHeaders:
        return DatadogSearchHeaders(dd_api_key=self.api_key, dd_application_key=self.app_key)

    def search(self, query: str, *, limit: int = 100, window: str = "now-15m") -> list[DatadogLogEvent]:
        """Every log event Datadog currently returns for ``query`` (free-text over
        the log message plus facets like ``status:error``). Never raises: a failed
        read yields an empty list so the caller keeps polling to its deadline."""
        result: Result[DatadogSearchResponse] = self.transport.post(
            "/api/v2/logs/events/search",
            headers=self._headers(),
            json=DatadogSearchBody(
                filter=DatadogSearchFilter(query=query, from_=window),
                page=DatadogSearchPage(limit=limit),
            ),
            response_type=DatadogSearchResponse,
        )
        match result:
            case Success(data=payload):
                return payload.data
            case _:
                return []

    def poll_for_events(self, query: str, *, min_count: int = 1, window: str = "now-15m") -> list[DatadogLogEvent]:
        """Poll the search API until at least ``min_count`` events match ``query``
        or the deadline passes; returns whatever was seen on the last read."""
        deadline = time.monotonic() + self.poll_timeout
        events: list[DatadogLogEvent] = []
        while time.monotonic() < deadline:
            events = self.search(query, window=window)
            if len(events) >= min_count:
                return events
            time.sleep(self.poll_interval)
        return events


@dataclass(frozen=True, slots=True)
class LoggingClient:
    gateway: Gateway
    datadog: DatadogClient | None

    def key_with_alias(self, alias: str, *, models: list[str]) -> str:
        return self.gateway.generate_key(KeyGenerateBody(key_alias=alias, models=models, user_id=f"e2e-{alias}"))

    def delete_key(self, key: str) -> None:
        self.gateway.delete_key(key)

    def chat_result(self, key: str, model: str, text: str) -> Result[ChatResponse]:
        """The raw tagged-union outcome, so a test can assert on a failure instead
        of turning it into one."""
        return self.gateway.chat(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=64,
            ),
        )

    def chat(self, key: str, model: str, text: str) -> ChatResponse:
        return unwrap(self.chat_result(key, model, text))

    def responses(self, key: str, model: str, text: str) -> StreamingResponse:
        """Drive POST /v1/responses. Returns the raw outcome (the Responses body is
        provider-native), so a test asserts on status and then verifies the log
        reached Datadog rather than parsing the completion here."""
        return self.gateway.transport.send(
            "/v1/responses",
            headers=self.gateway.transport.bearer(key),
            json=ResponsesBody(model=model, input=text),
        )

    def scrape_metrics(self) -> str:
        return self.gateway.probe("/metrics", params=NoBody()).body


def _datadog_api_base(dd_site: str) -> str:
    """The Datadog API host for a site: ``us5.datadoghq.com`` -> the
    ``api.us5.datadoghq.com`` reads host. Tolerates a site given with a scheme or
    an already-``api.``-prefixed host."""
    host = dd_site.strip().removeprefix("https://").removeprefix("http://").strip("/")
    host = host if host.startswith("api.") else f"api.{host}"
    return f"https://{host}"


def build_datadog_client() -> DatadogClient | None:
    """A ``DatadogClient`` when the read-back credentials are all present, else
    ``None`` so the suite skips. ``DD_APP_KEY`` is required on top of the
    ``DD_API_KEY`` / ``DD_SITE`` the proxy ships with, because the Logs Search API
    rejects reads that carry only an API key."""
    api_key = os.getenv("DD_API_KEY")
    app_key = os.getenv("DD_APP_KEY")
    dd_site = os.getenv("DD_SITE")
    if not (api_key and app_key and dd_site):
        return None
    return DatadogClient(
        transport=HttpTransport(
            base_url=_datadog_api_base(dd_site),
            master_key="",
            request_timeout=REQUEST_TIMEOUT,
        ),
        api_key=api_key,
        app_key=app_key,
    )


def build_logging_client() -> LoggingClient:
    return LoggingClient(gateway=build_gateway(), datadog=build_datadog_client())
