"""Spend-tracking e2e client: a ProxyClient plus the spend-specific read endpoints.

Generic proxy operations (keys, customers, chat/embed, route probing, SpendLogs
polling) come from the shared ProxyClient, DI'd in (composition, not inheritance).
This client adds only the spend surface: /spend/calculate, /spend/tags,
key-spend polling, and the route probes the breadth test uses.

Re-exports unwrap / is_ok / unique_marker / SpendLogRow so the tests import their
helpers from one place.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

from e2e_config import unique_marker
from e2e_http import (
    FileUploadForm,
    Headers,
    NoBody,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
    is_ok,
    unwrap,
)
from models import (
    AnthropicMessagesBody,
    ChatBody,
    ChatMessage,
    ChatMetadata,
    ChatResponse,
    DateRangeParams,
    EmbedBody,
    EmbedResponse,
    KeyGenerateBody,
    KeyGenerateResponse,
    OpenAPISchema,
    SpendCalculateBody,
    SpendCalculateResponse,
    SpendLogRow,
    SpendLogsPage,
    SpendLogsPageParams,
    SpendTagsResponse,
    TagSpend,
    UserDeleteBody,
    UserDeleteResponse,
    UserNewBody,
    UserNewResponse,
    UserRole,
)
from proxy_client import Converged, ProxyClient, await_converged
from pydantic import BaseModel, Field

__all__ = [
    "BatchCreateBody",
    "CallbackLogMetadata",
    "CallbackLogPayload",
    "BatchObject",
    "DailyActivityKeyBreakdown",
    "FileObject",
    "ProbeResult",
    "ResponseIdentity",
    "SpendClient",
    "SpendLogRow",
    "StreamingResponse",
    "build_client",
    "is_ok",
    "unique_marker",
    "unwrap",
]


class GeminiApiKeyHeaders(Headers):
    x_goog_api_key: str = Field(serialization_alias="x-goog-api-key")
    content_type: str = Field(default="application/json", serialization_alias="Content-Type")


class GeminiPart(BaseModel):
    text: str


class GeminiContent(BaseModel):
    parts: list[GeminiPart]


class GeminiGenerationConfig(BaseModel):
    maxOutputTokens: int


class GeminiGenerateBody(BaseModel):
    contents: list[GeminiContent]
    generationConfig: GeminiGenerationConfig


class ResponsesBody(BaseModel):
    model: str
    input: str
    cache: dict[str, bool] | None = {"no-cache": True}


class QueuedChatBody(ChatBody):
    priority: int = 0


class ResponseIdentity(BaseModel):
    id: str | None = None


class HealthParams(BaseModel):
    model: str


class ModelQuery(BaseModel):
    model: str


class FileObject(BaseModel):
    id: str


class BatchCreateBody(BaseModel):
    input_file_id: str
    endpoint: str = "/v1/chat/completions"
    completion_window: str = "24h"
    model: str
    metadata: dict[str, str]


class BatchObject(BaseModel):
    id: str
    status: str


class ProviderQuery(BaseModel):
    provider: str


class CallbackLogMetadata(BaseModel):
    user_api_key_hash: str
    user_api_key_alias: str
    user_api_key_user_id: str


class CallbackLogPayload(BaseModel):
    id: str
    litellm_call_id: str
    model: str
    call_type: str = "acompletion"
    start_time: float = Field(serialization_alias="startTime")
    end_time: float = Field(serialization_alias="endTime")
    response_cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metadata: CallbackLogMetadata


class CallbackLogRecord(BaseModel):
    status: str = "success"
    standard_logging_payload: CallbackLogPayload


class CallbackLogsRequest(BaseModel):
    records: list[CallbackLogRecord]


class CallbackLogsResponse(BaseModel):
    processed: int
    failed: int


class DailyActivityParams(BaseModel):
    start_date: str
    end_date: str
    api_key: str


class DailyActivityKeyMetadata(BaseModel):
    key_alias: str | None = None
    team_id: str | None = None
    user_email: str | None = None


class DailyActivityKeyMetrics(BaseModel):
    api_requests: int = 0


class DailyActivityKeyBreakdown(BaseModel):
    metrics: DailyActivityKeyMetrics
    metadata: DailyActivityKeyMetadata


class DailyActivityBreakdown(BaseModel):
    api_keys: dict[str, DailyActivityKeyBreakdown] = {}


class DailyActivityRow(BaseModel):
    date: str
    breakdown: DailyActivityBreakdown


class DailyActivityResponse(BaseModel):
    results: list[DailyActivityRow] = []


def _chat_body(
    model: str,
    content: str,
    *,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    user: str | None = None,
    stream: bool = False,
    cache: dict[str, bool] | None = {"no-cache": True},
) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=content)],
        max_tokens=max_tokens,
        stream=stream,
        user=user,
        metadata=ChatMetadata(tags=tags) if tags else None,
        cache=cache,
    )


@dataclass(frozen=True, slots=True)
class SpendClient:
    proxy: ProxyClient

    def chat(
        self,
        key: str,
        model: str,
        content: str,
        *,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        user: str | None = None,
        cache: dict[str, bool] | None = {"no-cache": True},
    ) -> Result[ChatResponse]:
        return self.proxy.chat(
            key,
            _chat_body(model, content, max_tokens=max_tokens, tags=tags, user=user, cache=cache),
        )

    def chat_stream(
        self, key: str, model: str, content: str, *, max_tokens: int | None = None
    ) -> StreamingResponse:
        return self.proxy.chat_stream(
            key, _chat_body(model, content, max_tokens=max_tokens, stream=True)
        )

    def messages_stream(
        self, key: str, model: str, content: str, *, max_tokens: int
    ) -> StreamingResponse:
        return self.proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
                stream=True,
            ),
        )

    def embed(self, key: str, model: str, content: str) -> Result[EmbedResponse]:
        return self.proxy.embed(key, EmbedBody(model=model, input=content))

    def poll_logs_for_key(
        self,
        key: str,
        *,
        min_rows: int = 1,
        predicate: Callable[[list[SpendLogRow]], bool] | None = None,
    ) -> list[SpendLogRow]:
        return self.proxy.poll_logs_for_key(
            key, min_rows=min_rows, predicate=predicate
        )

    def calculate_spend(self, model: str, content: str) -> float:
        return unwrap(
            self.proxy.transport.post(
                "/spend/calculate",
                headers=self.proxy.transport.master,
                json=SpendCalculateBody(
                    model=model, messages=[ChatMessage(role="user", content=content)]
                ),
                response_type=SpendCalculateResponse,
            )
        ).cost

    def spend_by_tags(self) -> list[TagSpend]:
        result = self.proxy.transport.get(
            "/spend/tags",
            headers=self.proxy.transport.master,
            params=NoBody(),
            response_type=SpendTagsResponse,
        )
        match result:
            case Success(data=data):
                return data.root
            case _:
                return []

    def poll_tag_spend(self, tag: str, *, minimum: float = 0.0) -> TagSpend | None:
        """Poll /spend/tags until the tag's aggregate reaches `minimum`; last seen."""
        deadline = time.monotonic() + self.proxy.poll_timeout
        entry: TagSpend | None = None
        while time.monotonic() < deadline:
            matches = [
                t for t in self.spend_by_tags() if t.individual_request_tag == tag
            ]
            if matches:
                entry = matches[0]
                if (entry.total_spend or 0.0) >= minimum:
                    return entry
            time.sleep(self.proxy.poll_interval)
        return entry

    def poll_key_spend(self, key: str, *, minimum: float = 0.0) -> float:
        deadline = time.monotonic() + self.proxy.poll_timeout
        spend = 0.0
        while time.monotonic() < deadline:
            spend = self.proxy.key_info(key).spend or 0.0
            if spend > minimum:
                return spend
            time.sleep(self.proxy.poll_interval)
        return spend

    def spend_logs_page(
        self, *, api_key: str | None, page: int, page_size: int
    ) -> SpendLogsPage:
        """One page of /spend/logs/v2 over a window wide enough to contain every
        row this test run wrote (the endpoint requires explicit dates)."""
        now = datetime.now(timezone.utc)
        fmt = "%Y-%m-%d %H:%M:%S"
        return unwrap(
            self.proxy.transport.get(
                "/spend/logs/v2",
                headers=self.proxy.transport.master,
                params=SpendLogsPageParams(
                    start_date=(now - timedelta(days=1)).strftime(fmt),
                    end_date=(now + timedelta(days=1)).strftime(fmt),
                    page=page,
                    page_size=page_size,
                    api_key=api_key,
                ),
                response_type=SpendLogsPage,
            )
        )

    def probe(self, path: str, *, params: DateRangeParams) -> ProbeResult:
        return self.proxy.transport.probe(path, params=params)

    def create_user(self, *, email: str, role: UserRole, user_id: str) -> str:
        return unwrap(
            self.proxy.transport.post(
                "/user/new",
                headers=self.proxy.transport.master,
                json=UserNewBody(user_email=email, user_role=role, user_id=user_id),
                response_type=UserNewResponse,
            )
        ).user_id

    def delete_user(self, user_id: str) -> None:
        _ = unwrap(
            self.proxy.transport.post(
                "/user/delete",
                headers=self.proxy.transport.master,
                json=UserDeleteBody(user_ids=[user_id]),
                response_type=UserDeleteResponse,
            )
        )

    def generate_key_record(self, body: KeyGenerateBody) -> KeyGenerateResponse:
        return unwrap(
            self.proxy.transport.post(
                "/key/generate",
                headers=self.proxy.transport.master,
                json=body,
                response_type=KeyGenerateResponse,
            )
        )

    def send_chat(self, key: str, model: str, content: str, *, max_tokens: int) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=_chat_body(model, content, max_tokens=max_tokens),
        )

    def send_queued_chat(self, key: str, model: str, content: str, *, max_tokens: int) -> StreamingResponse:
        return self.proxy.transport.send(
            "/queue/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=QueuedChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
            ),
        )

    def send_messages(self, key: str, model: str, content: str, *, max_tokens: int) -> StreamingResponse:
        return self.proxy.transport.send(
            "/v1/messages",
            headers=self.proxy.transport.bearer(key),
            json=AnthropicMessagesBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
            ),
        )

    def send_responses(self, key: str, model: str, content: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/v1/responses",
            headers=self.proxy.transport.bearer(key),
            json=ResponsesBody(model=model, input=content),
        )

    def send_embed(self, key: str, model: str, content: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/embeddings",
            headers=self.proxy.transport.bearer(key),
            json=EmbedBody(model=model, input=content),
        )

    def send_gemini_generate(self, key: str, model: str, content: str, *, max_tokens: int) -> StreamingResponse:
        return self.proxy.transport.send(
            f"/gemini/v1beta/models/{model}:generateContent",
            headers=GeminiApiKeyHeaders(x_goog_api_key=key),
            json=GeminiGenerateBody(
                contents=[GeminiContent(parts=[GeminiPart(text=content)])],
                generationConfig=GeminiGenerationConfig(maxOutputTokens=max_tokens),
            ),
        )

    def upload_batch_file(self, key: str, model: str, content: bytes) -> FileObject:
        return unwrap(
            self.proxy.transport.upload(
                "/v1/files",
                headers=self.proxy.transport.bearer(key),
                form=FileUploadForm(purpose="batch"),
                filename="key_attribution.jsonl",
                content=content,
                params=ModelQuery(model=model),
                response_type=FileObject,
            )
        )

    def create_batch(self, key: str, body: BatchCreateBody) -> BatchObject:
        return unwrap(
            self.proxy.transport.post(
                "/v1/batches",
                headers=self.proxy.transport.bearer(key),
                json=body,
                response_type=BatchObject,
            )
        )

    def retrieve_batch(self, key: str, batch_id: str, *, provider: str) -> BatchObject:
        return unwrap(
            self.proxy.transport.get(
                f"/v1/batches/{batch_id}",
                headers=self.proxy.transport.bearer(key),
                params=ProviderQuery(provider=provider),
                response_type=BatchObject,
            )
        )

    def replay_callback_log(self, key: str, payload: CallbackLogPayload) -> CallbackLogsResponse:
        return unwrap(
            self.proxy.transport.post(
                "/v1/rust_control_plane/logs",
                headers=self.proxy.transport.bearer(key),
                json=CallbackLogsRequest(records=[CallbackLogRecord(standard_logging_payload=payload)]),
                response_type=CallbackLogsResponse,
            )
        )

    def health(self, model: str) -> ProbeResult:
        return self.proxy.transport.probe("/health", params=HealthParams(model=model))

    def daily_activity_for_key(self, token: str, *, start: datetime, end: datetime) -> DailyActivityKeyBreakdown | None:
        response: Final = unwrap(
            self.proxy.transport.get(
                "/user/daily/activity",
                headers=self.proxy.transport.master,
                params=DailyActivityParams(
                    start_date=start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"),
                    api_key=token,
                ),
                response_type=DailyActivityResponse,
            )
        )
        return next(
            (row.breakdown.api_keys[token] for row in response.results if token in row.breakdown.api_keys),
            None,
        )

    def poll_daily_activity_for_key(
        self, token: str, *, start: datetime, end: datetime, min_requests: int
    ) -> DailyActivityKeyBreakdown | None:
        outcome: Final = await_converged(
            lambda: self.daily_activity_for_key(token, start=start, end=end),
            converged=lambda found: found is not None and found.metrics.api_requests >= min_requests,
            timeout=self.proxy.poll_timeout,
            interval=self.proxy.poll_interval,
            now=time.monotonic,
            sleep=time.sleep,
        )
        return outcome.result if isinstance(outcome, Converged) else outcome.last_result

    def openapi(self) -> OpenAPISchema:
        return unwrap(
            self.proxy.transport.get(
                "/openapi.json",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=OpenAPISchema,
            )
        )


def build_client(proxy: ProxyClient) -> SpendClient:
    return SpendClient(proxy=proxy)
