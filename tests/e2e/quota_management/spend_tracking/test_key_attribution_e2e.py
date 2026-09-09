"""Every spend row a live proxy writes joins its virtual key (MAT-180).

One virtual key with an alias, owned by a user with an email, drives every spend
write path a key can reach: /chat/completions, /queue/chat/completions,
/v1/messages, /v1/responses, /embeddings, the Gemini native passthrough, a batch
input file upload, and a batch create. Each row those calls write must carry
`api_key` equal to the key's LiteLLM_VerificationToken.token (the sha256 hash
/key/generate returns as `token`), which is the join /spend/logs?api_key= and
/user/daily/activity rely on to report key_alias and user_email. A row keyed by a
re-hashed token (v1.99.0's regression, #39568 and #39572) shows up as a
key-hash-* row with no alias and no email in the customer's usage exports.

The health-check service account writes rows too; those must stay keyed by the
literal service-account name, never by a hash of it. The batch cost row is
written by the CheckBatchCost poller once the batch completes, up to an hour
later, so it rides a cross-run baton like the batches suite: each run submits a
one-line marker batch whose metadata records the token it expects on the cost
row, and asserts on the newest completed marker from any run (a cold start with
no completed marker is a documented vacuous pass, never a skip).

/spend/logs carries no email field, so the email assertion lives on
/user/daily/activity alone; /spend/logs is held to the alias in metadata.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Iterator

import pytest

from models import ChatMessage, KeyGenerateBody
from proxy_client import Converged, await_converged
from spend_e2e_client import (
    BatchCreateBody,
    BatchObject,
    DailyActivityKeyBreakdown,
    ResponseIdentity,
    SpendClient,
    SpendLogRow,
    StreamingResponse,
    unique_marker,
)
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

CHAT_MODEL: Final = "gemini-2.5-flash"
MESSAGES_MODEL: Final = "claude-haiku-4-5"
RESPONSES_MODEL: Final = "openai-responses-codex"
EMBED_MODEL: Final = "openai-text-embedding-3-small"
BATCH_MODEL: Final = "openai-gpt-4o-mini"
BATCH_BACKEND_MODEL: Final = "gpt-4o-mini"
HEALTH_SERVICE_ACCOUNT: Final = "litellm-internal-health-check"
BATON_MARKER_KEY: Final = "litellm_e2e_suite"
BATON_MARKER_VALUE: Final = "key-attribution-baton"
BATON_EXPECTED_TOKEN_KEY: Final = "expected_api_key"
BATON_POLL_SECONDS: Final = 300.0
BATON_POLL_INTERVAL_SECONDS: Final = 10.0
BATON_LIST_LIMIT: Final = 100
MAX_TOKENS: Final = 8
WRITE_PATHS: Final = (
    "chat_completions",
    "queue_chat_completions",
    "messages",
    "responses",
    "embeddings",
    "gemini_passthrough",
    "batch_file_upload",
    "batch_create",
)


class BatchLineBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int


class BatchLine(BaseModel):
    custom_id: str
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: BatchLineBody


@dataclass(frozen=True, slots=True)
class AttributedKey:
    key: str
    token: str
    alias: str
    email: str
    user_id: str


@dataclass(frozen=True, slots=True)
class WritePath:
    name: str
    request_id: str


@dataclass(frozen=True, slots=True)
class DrivenKey:
    identity: AttributedKey
    paths: tuple[WritePath, ...]
    started_at: datetime


def _body_id(name: str, sent: StreamingResponse) -> WritePath:
    assert sent.ok, f"{name} failed with {sent.status_code}: {sent.body[:300]}"
    response_id: Final = ResponseIdentity.model_validate_json(sent.body).id
    assert response_id, f"{name} answered without a response id: {sent.body[:300]}"
    return WritePath(name=name, request_id=response_id)


def _call_id(name: str, sent: StreamingResponse) -> WritePath:
    assert sent.ok, f"{name} failed with {sent.status_code}: {sent.body[:300]}"
    assert sent.call_id, f"{name} answered without an x-litellm-call-id header"
    return WritePath(name=name, request_id=sent.call_id)


def _batch_jsonl(marker: str) -> bytes:
    line: Final = BatchLine(
        custom_id=marker,
        body=BatchLineBody(
            model=BATCH_BACKEND_MODEL,
            messages=[ChatMessage(role="user", content=f"Reply with the word ok. {marker}")],
            max_tokens=MAX_TOKENS,
        ),
    )
    return f"{line.model_dump_json()}\n".encode()


def _drive_batch(client: SpendClient, identity: AttributedKey, marker: str) -> tuple[WritePath, WritePath]:
    uploaded: Final = client.upload_batch_file(identity.key, BATCH_MODEL, _batch_jsonl(marker))
    created: Final = client.create_batch(
        identity.key,
        BatchCreateBody(
            input_file_id=uploaded.id,
            model=BATCH_MODEL,
            metadata={
                BATON_MARKER_KEY: BATON_MARKER_VALUE,
                BATON_EXPECTED_TOKEN_KEY: identity.token,
                "run": marker,
            },
        ),
    )
    return (
        WritePath(name="batch_file_upload", request_id=uploaded.id),
        WritePath(name="batch_create", request_id=created.id),
    )


def _drive_every_write_path(client: SpendClient, identity: AttributedKey) -> tuple[WritePath, ...]:
    marker: Final = unique_marker()
    prompt: Final = f"Reply with the word ok. {marker}"
    key: Final = identity.key
    return (
        _body_id("chat_completions", client.send_chat(key, CHAT_MODEL, prompt, max_tokens=MAX_TOKENS)),
        _body_id("queue_chat_completions", client.send_queued_chat(key, CHAT_MODEL, prompt, max_tokens=MAX_TOKENS)),
        _body_id("messages", client.send_messages(key, MESSAGES_MODEL, prompt, max_tokens=MAX_TOKENS)),
        _body_id("responses", client.send_responses(key, RESPONSES_MODEL, prompt)),
        _call_id("embeddings", client.send_embed(key, EMBED_MODEL, prompt)),
        _call_id("gemini_passthrough", client.send_gemini_generate(key, CHAT_MODEL, prompt, max_tokens=MAX_TOKENS)),
        *_drive_batch(client, identity, marker),
    )


def _completed_baton(listed: list[BatchObject]) -> BatchObject | None:
    return max(
        (
            batch
            for batch in listed
            if batch.status == "completed" and (batch.metadata or {}).get(BATON_MARKER_KEY) == BATON_MARKER_VALUE
        ),
        key=lambda batch: batch.created_at or 0,
        default=None,
    )


def _newest_completed_baton(client: SpendClient, key: str) -> BatchObject | None:
    outcome: Final = await_converged(
        lambda: _completed_baton(client.list_batches(key, BATCH_MODEL, limit=BATON_LIST_LIMIT)),
        converged=lambda completed: completed is not None,
        timeout=BATON_POLL_SECONDS,
        interval=BATON_POLL_INTERVAL_SECONDS,
        now=time.monotonic,
        sleep=time.sleep,
    )
    return outcome.result if isinstance(outcome, Converged) else None


def _health_rows_between(client: SpendClient, started_at: datetime) -> list[SpendLogRow]:
    return [
        row
        for row in client.proxy.spend_logs_window(
            start=started_at - timedelta(minutes=1), end=datetime.now(timezone.utc) + timedelta(minutes=1)
        )
        if HEALTH_SERVICE_ACCOUNT in (row.request_tags or [])
    ]


def _health_rows_since(client: SpendClient, started_at: datetime) -> list[SpendLogRow]:
    outcome: Final = await_converged(
        lambda: _health_rows_between(client, started_at),
        converged=lambda rows: bool(rows),
        timeout=client.proxy.poll_timeout,
        interval=client.proxy.poll_interval,
        now=time.monotonic,
        sleep=time.sleep,
    )
    return outcome.result if isinstance(outcome, Converged) else outcome.last_result


class TestKeyAttribution:
    @pytest.fixture(scope="class")
    def driven(self, client: SpendClient) -> Iterator[DrivenKey]:
        marker: Final = unique_marker()
        user_id: Final = client.create_user(
            email=f"key-attribution-{marker}@example.com",
            role="proxy_admin",
            user_id=f"key-attribution-{marker}",
        )
        record: Final = client.generate_key_record(
            KeyGenerateBody(models=[], user_id=user_id, key_alias=f"key-attribution-{marker}")
        )
        assert record.token, "/key/generate answered without the key's token hash"
        assert record.key_alias, "/key/generate dropped the key alias"
        identity: Final = AttributedKey(
            key=record.key,
            token=record.token,
            alias=record.key_alias,
            email=f"key-attribution-{marker}@example.com",
            user_id=user_id,
        )
        started_at: Final = datetime.now(timezone.utc)
        try:
            yield DrivenKey(
                identity=identity,
                paths=_drive_every_write_path(client, identity),
                started_at=started_at,
            )
        finally:
            client.proxy.delete_key(identity.key)
            client.delete_user(identity.user_id)

    @pytest.mark.covers(
        "quota_management.spend_tracking.key_attribution.joins_key",
        exercised_on=["chat_completions", "messages", "responses", "embeddings", "batches", "files", "google_native"],
    )
    def test_every_write_path_row_joins_the_key(self, client: SpendClient, driven: DrivenKey) -> None:
        assert tuple(path.name for path in driven.paths) == WRITE_PATHS
        found: Final = tuple(
            (path, client.proxy.poll_logs_for_request_id(path.request_id)) for path in driven.paths
        )
        unwritten: Final = [path.name for path, rows in found if not rows]
        assert not unwritten, f"write paths that produced no spend row within the poll window: {unwritten}"
        unjoined: Final = [
            (path.name, row.call_type, row.api_key)
            for path, rows in found
            for row in rows
            if row.api_key != driven.identity.token
        ]
        assert not unjoined, (
            "spend rows whose api_key does not join LiteLLM_VerificationToken.token "
            f"{driven.identity.token}: {unjoined}"
        )
        unaliased: Final = [
            (path.name, row.call_type, row.metadata.user_api_key_alias if row.metadata else None)
            for path, rows in found
            for row in rows
            if row.metadata is None or row.metadata.user_api_key_alias != driven.identity.alias
        ]
        assert not unaliased, f"spend rows written without key alias {driven.identity.alias!r}: {unaliased}"

    @pytest.mark.covers(
        "quota_management.spend_tracking.key_attribution.reports_alias_and_email",
        exercised_on=["chat_completions", "messages", "responses", "embeddings", "batches", "files", "google_native"],
    )
    def test_spend_logs_by_key_return_every_row_with_the_alias(self, client: SpendClient, driven: DrivenKey) -> None:
        expected_ids: Final = frozenset(path.request_id for path in driven.paths)
        rows: Final = client.poll_logs_for_key(
            driven.identity.key,
            min_rows=len(driven.paths),
            predicate=lambda found: expected_ids <= frozenset(row.request_id or "" for row in found),
        )
        missing: Final = expected_ids - frozenset(row.request_id or "" for row in rows)
        assert not missing, (
            f"/spend/logs?api_key= does not return {len(missing)} of {len(expected_ids)} rows for the key: "
            f"{sorted(path.name for path in driven.paths if path.request_id in missing)}"
        )
        aliases: Final = frozenset(row.metadata.user_api_key_alias if row.metadata else None for row in rows)
        assert aliases == {driven.identity.alias}, f"/spend/logs rows carry aliases {sorted(map(str, aliases))}"

    @pytest.mark.covers(
        "quota_management.spend_tracking.key_attribution.reports_alias_and_email",
        exercised_on=["chat_completions", "messages", "responses", "embeddings", "batches", "files", "google_native"],
    )
    def test_user_daily_activity_reports_alias_and_email(self, client: SpendClient, driven: DrivenKey) -> None:
        breakdown: Final[DailyActivityKeyBreakdown | None] = client.poll_daily_activity_for_key(
            driven.identity.token,
            start=driven.started_at - timedelta(days=1),
            end=datetime.now(timezone.utc) + timedelta(days=1),
            min_requests=len(driven.paths),
        )
        assert breakdown is not None, (
            f"/user/daily/activity?api_key={driven.identity.token} has no api_keys breakdown: "
            "the key's rows did not aggregate under its token"
        )
        assert breakdown.metrics.api_requests >= len(driven.paths), (
            f"/user/daily/activity counts {breakdown.metrics.api_requests} requests for the key, "
            f"expected at least {len(driven.paths)}"
        )
        assert breakdown.metadata.key_alias == driven.identity.alias, f"key_alias={breakdown.metadata.key_alias!r}"
        assert breakdown.metadata.user_email == driven.identity.email, f"user_email={breakdown.metadata.user_email!r}"

    @pytest.mark.covers(
        "quota_management.spend_tracking.key_attribution.health_rows_keep_service_account",
        exercised_on=["chat_completions"],
    )
    def test_health_check_rows_keep_the_service_account_key(self, client: SpendClient) -> None:
        started_at: Final = datetime.now(timezone.utc)
        probe: Final = client.health(CHAT_MODEL)
        assert probe.healthy, f"/health?model={CHAT_MODEL} answered {probe.status_code}: {probe.body[:300]}"
        rows: Final = _health_rows_since(client, started_at)
        assert rows, f"/health?model={CHAT_MODEL} wrote no {HEALTH_SERVICE_ACCOUNT}-tagged spend row"
        rehashed: Final = [(row.request_id, row.api_key) for row in rows if row.api_key != HEALTH_SERVICE_ACCOUNT]
        assert not rehashed, f"health-check rows keyed by something other than {HEALTH_SERVICE_ACCOUNT!r}: {rehashed}"

    @pytest.mark.covers(
        "quota_management.spend_tracking.key_attribution.batch_cost_joins_key",
        exercised_on=["batches"],
    )
    def test_completed_batch_cost_row_joins_the_key(self, client: SpendClient, driven: DrivenKey) -> None:
        completed: Final = _newest_completed_baton(client, driven.identity.key)
        if completed is None:
            return
        expected_token: Final = (completed.metadata or {}).get(BATON_EXPECTED_TOKEN_KEY)
        assert expected_token, f"marker batch {completed.id} lost its expected token metadata: {completed.metadata}"
        fetched: Final = client.retrieve_batch(driven.identity.key, completed.id)
        assert fetched.status == "completed", f"listed-completed marker retrieved as {fetched.status!r}"
        rows: Final = client.proxy.poll_logs_for_request_id(
            f"{fetched.id}_batch_cost",
            predicate=lambda found: any((row.spend or 0) > 0 for row in found),
        )
        priced: Final = [row for row in rows if (row.spend or 0) > 0]
        assert priced, f"completed batch {fetched.id} has no positive-cost spend row under {fetched.id}_batch_cost"
        unjoined: Final = [(row.call_type, row.api_key) for row in priced if row.api_key != expected_token]
        assert not unjoined, f"batch cost rows whose api_key does not join the key's token {expected_token}: {unjoined}"
