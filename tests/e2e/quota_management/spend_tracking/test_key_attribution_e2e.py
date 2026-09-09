"""Every spend row a live proxy writes joins its virtual key (MAT-180).

One virtual key with an alias, owned by a user with an email, drives every spend
write path a key can reach: /chat/completions, /queue/chat/completions,
/v1/messages, /v1/responses, /embeddings, the Gemini native passthrough, a batch
input file upload, a batch create, and a replayed callback log (POST
/v1/rust_control_plane/logs, the writer an external gateway feeds). Each row those calls write must carry
`api_key` equal to the key's LiteLLM_VerificationToken.token (the sha256 hash
/key/generate returns as `token`), which is the join /spend/logs?api_key= and
/user/daily/activity rely on to report key_alias and user_email. A row keyed by a
re-hashed token (v1.99.0's regression, #39568 and #39572) shows up as a
key-hash-* row with no alias and no email in the customer's usage exports.

The health-check service account writes rows too; those must stay keyed by the
literal service-account name, never by a hash of it. A batch's cost row is
written by the retrieve that first sees the batch in a terminal state, so the
batch the run creates is one OpenAI fails at validation within seconds (its one
line targets /v1/embeddings under a /v1/chat/completions batch), and the test
retrieves it by its raw provider id with the same key until it is failed. A raw
id is never owned by the CheckBatchCost poller, so that retrieve prices the batch
inline against the retrieving key and its {provider_batch_id}_batch_cost row
must join the key's token with its alias. A completed batch with a positive
cost is out of a single run's reach (OpenAI's completion window is 24h, and a
stack booted fresh per run lists no earlier run's batches), so the poller's own
row is not asserted here.

/spend/logs carries no email field, so the email assertion lives on
/user/daily/activity alone; /spend/logs is held to the alias in metadata.
"""

import base64
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from models import KeyGenerateBody
from proxy_client import Converged, await_converged
from pydantic import BaseModel
from spend_e2e_client import (
    BatchCreateBody,
    BatchObject,
    CallbackLogMetadata,
    CallbackLogPayload,
    DailyActivityKeyBreakdown,
    ResponseIdentity,
    SpendClient,
    SpendLogRow,
    StreamingResponse,
    unique_marker,
)

pytestmark = pytest.mark.e2e

CHAT_MODEL: Final = "gemini-2.5-flash"
MESSAGES_MODEL: Final = "claude-haiku-4-5"
RESPONSES_MODEL: Final = "openai-responses-codex"
EMBED_MODEL: Final = "openai-text-embedding-3-small"
BATCH_MODEL: Final = "openai-gpt-4o-mini"
BATCH_BACKEND_MODEL: Final = "gpt-4o-mini"
BATCH_PROVIDER: Final = "openai"
HEALTH_SERVICE_ACCOUNT: Final = "litellm-internal-health-check"
BATCH_TERMINAL_STATUSES: Final = frozenset({"completed", "failed", "cancelled", "expired"})
FAILED_BATCH_POLL_SECONDS: Final = 120.0
FAILED_BATCH_POLL_INTERVAL_SECONDS: Final = 5.0
MAX_TOKENS: Final = 8
REPLAY_RESPONSE_COST: Final = 0.0001
REPLAY_PROMPT_TOKENS: Final = 5
REPLAY_COMPLETION_TOKENS: Final = 1
WRITE_PATHS: Final = (
    "chat_completions",
    "queue_chat_completions",
    "messages",
    "responses",
    "embeddings",
    "gemini_passthrough",
    "batch_file_upload",
    "batch_create",
    "callback_replay",
)


class EmbeddingLineBody(BaseModel):
    model: str
    input: str


class EmbeddingLine(BaseModel):
    custom_id: str
    method: str = "POST"
    url: str = "/v1/embeddings"
    body: EmbeddingLineBody


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


def _endpoint_mismatched_jsonl(marker: str) -> bytes:
    line: Final = EmbeddingLine(custom_id=marker, body=EmbeddingLineBody(model=BATCH_BACKEND_MODEL, input=marker))
    return f"{line.model_dump_json()}\n".encode()


def _drive_batch(client: SpendClient, identity: AttributedKey, marker: str) -> tuple[WritePath, WritePath]:
    uploaded: Final = client.upload_batch_file(identity.key, BATCH_MODEL, _endpoint_mismatched_jsonl(marker))
    created: Final = client.create_batch(
        identity.key,
        BatchCreateBody(
            input_file_id=uploaded.id,
            model=BATCH_MODEL,
            metadata={"run": marker},
        ),
    )
    return (
        WritePath(name="batch_file_upload", request_id=uploaded.id),
        WritePath(name="batch_create", request_id=created.id),
    )


def _drive_callback_replay(client: SpendClient, identity: AttributedKey, marker: str) -> WritePath:
    request_id: Final = f"callback-replay-{marker}"
    finished_at: Final = time.time()
    replayed: Final = client.replay_callback_log(
        identity.key,
        CallbackLogPayload(
            id=request_id,
            litellm_call_id=request_id,
            model=CHAT_MODEL,
            start_time=finished_at - 1,
            end_time=finished_at,
            response_cost=REPLAY_RESPONSE_COST,
            prompt_tokens=REPLAY_PROMPT_TOKENS,
            completion_tokens=REPLAY_COMPLETION_TOKENS,
            total_tokens=REPLAY_PROMPT_TOKENS + REPLAY_COMPLETION_TOKENS,
            metadata=CallbackLogMetadata(
                user_api_key_hash=identity.token,
                user_api_key_alias=identity.alias,
                user_api_key_user_id=identity.user_id,
            ),
        ),
    )
    assert replayed.processed == 1 and replayed.failed == 0, f"callback replay rejected the payload: {replayed}"
    return WritePath(name="callback_replay", request_id=request_id)


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
        _drive_callback_replay(client, identity, marker),
    )


def _provider_batch_id(unified_batch_id: str) -> str:
    encoded: Final = unified_batch_id.removeprefix("batch_")
    decoded: Final = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    return decoded.removeprefix("litellm:").split(";", 1)[0]


def _driven_batch_id(driven: DrivenKey) -> str:
    return next(path.request_id for path in driven.paths if path.name == "batch_create")


def _await_terminal_batch(client: SpendClient, key: str, provider_batch_id: str) -> BatchObject:
    outcome: Final = await_converged(
        lambda: client.retrieve_batch(key, provider_batch_id, provider=BATCH_PROVIDER),
        converged=lambda batch: batch.status in BATCH_TERMINAL_STATUSES,
        timeout=FAILED_BATCH_POLL_SECONDS,
        interval=FAILED_BATCH_POLL_INTERVAL_SECONDS,
        now=time.monotonic,
        sleep=time.sleep,
    )
    return outcome.result if isinstance(outcome, Converged) else outcome.last_result


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
        exercised_on=[
            "chat_completions",
            "messages",
            "responses",
            "embeddings",
            "batches",
            "files",
            "google_native",
            "rust_control_plane",
        ],
    )
    def test_every_write_path_row_joins_the_key(self, client: SpendClient, driven: DrivenKey) -> None:
        assert tuple(path.name for path in driven.paths) == WRITE_PATHS
        found: Final = tuple((path, client.proxy.poll_logs_for_request_id(path.request_id)) for path in driven.paths)
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
        exercised_on=[
            "chat_completions",
            "messages",
            "responses",
            "embeddings",
            "batches",
            "files",
            "google_native",
            "rust_control_plane",
        ],
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
        exercised_on=[
            "chat_completions",
            "messages",
            "responses",
            "embeddings",
            "batches",
            "files",
            "google_native",
            "rust_control_plane",
        ],
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
        "quota_management.spend_tracking.key_attribution.retrieve_batch_cost_joins_retrieving_key",
        exercised_on=["batches"],
    )
    def test_terminal_batch_cost_row_joins_the_retrieving_key(self, client: SpendClient, driven: DrivenKey) -> None:
        provider_batch_id: Final = _provider_batch_id(_driven_batch_id(driven))
        fetched: Final = _await_terminal_batch(client, driven.identity.key, provider_batch_id)
        assert fetched.status == "failed", (
            f"endpoint-mismatched batch {provider_batch_id} is {fetched.status!r} after "
            f"{FAILED_BATCH_POLL_SECONDS:.0f}s, so its terminal cost row cannot be asserted"
        )
        cost_request_id: Final = f"{provider_batch_id}_batch_cost"
        rows: Final = client.proxy.poll_logs_for_request_id(cost_request_id)
        assert rows, f"retrieving failed batch {provider_batch_id} wrote no cost row under {cost_request_id}"
        call_types: Final = tuple(sorted({row.call_type or "" for row in rows}))
        assert call_types == ("aretrieve_batch",), f"cost rows under {cost_request_id} carry call types {call_types}"
        unjoined: Final = [
            (row.call_type, row.api_key, row.metadata.user_api_key_alias if row.metadata else None)
            for row in rows
            if row.api_key != driven.identity.token
            or row.metadata is None
            or row.metadata.user_api_key_alias != driven.identity.alias
        ]
        assert not unjoined, (
            f"batch cost rows that do not join the retrieving key's token {driven.identity.token} "
            f"with alias {driven.identity.alias!r}: {unjoined}"
        )
