from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from itertools import chain
from pathlib import Path
from threading import Event
from types import MappingProxyType
from typing import Final, LiteralString
from uuid import uuid4

import httpx
import psutil
import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows, write_rows
from integration._support.process import OwnedProxy, group_members, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

from tests.integration.observability._s3_v2_support import SURFACES, call_surface, surface_reply
from tests.integration.observability._ternary_support import (
    CsvRow,
    MultipartRecord,
    MultipartSink,
    bound_sink_server,
    non_loopback_ipv4,
    row_by_alias,
    running_sink,
    ternary_environment,
    uploads_for_alias,
    vantage_environment,
    write_proxy_config,
)

_ROW_INSERT: Final[LiteralString] = """
INSERT INTO "LiteLLM_DailyUserSpend" (
    id, user_id, date, api_key, model, model_group, custom_llm_provider,
    endpoint, mcp_namespaced_tool_name,
    prompt_tokens, completion_tokens, spend,
    api_requests, successful_requests, failed_requests, updated_at
) VALUES (
    gen_random_uuid()::text, %s, %s, %s, %s, %s, 'openai',
    NULL, NULL,
    %s, %s, %s,
    %s, %s, %s, %s::timestamptz
)
"""
_ROW_DELETE: Final[LiteralString] = 'DELETE FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s AND model LIKE %s'
_BULK_INSERT: Final[LiteralString] = """
INSERT INTO "LiteLLM_DailyUserSpend" (
    id, user_id, date, api_key, model, model_group, custom_llm_provider,
    endpoint, mcp_namespaced_tool_name,
    prompt_tokens, completion_tokens, spend,
    api_requests, successful_requests, failed_requests, updated_at
)
SELECT
    gen_random_uuid()::text, %s, %s, %s, %s || '-' || sequence::text, %s, 'openai',
    NULL, NULL,
    11, 4, 0.0001,
    1, 1, 0, now()
FROM generate_series(0, %s::int - 1) AS sequence
"""
_TOKEN_TAGS: Final = frozenset(
    {"prompt_tokens", "completion_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"}
)
_REQUEST_BODY_ADAPTER: Final = TypeAdapter(dict[str, object])
_MESSAGE_BODY_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class TernaryRig:
    owned: OwnedProxy
    provider: Wire
    sink: MultipartSink
    upstream_gate: Event


@pytest.fixture(scope="module")
def ternary_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TernaryRig]:
    sink: Final = MultipartSink()
    directory: Final = tmp_path_factory.mktemp("ternary-focus")
    config: Final = write_proxy_config(directory, ("ternary",))
    upstream_gate: Final = Event()
    upstream_gate.set()

    def upstream_reply(request: Request) -> Reply:
        reply: Final = surface_reply(request)
        if upstream_gate.is_set() or not request.body:
            return reply
        chunks: Final = reply.chunks if reply.chunks is not None else (reply.body,)
        return Reply(
            status=reply.status,
            body=reply.body,
            content_type=reply.content_type,
            chunks=chunks,
            gate_after_first=upstream_gate,
            headers=reply.headers,
        )

    with (
        running_sink(sink) as active_sink,
        wire_server(upstream_reply) as provider,
        gateway_from_environment() as gateway,
        owned_proxy_process(
            gateway,
            directory,
            ternary_environment(active_sink.url),
            config=config,
            workers=2,
        ) as owned,
    ):
        yield TernaryRig(owned, provider, active_sink, upstream_gate)


def _models(scenario: Scenario, provider: Wire) -> tuple[str, str]:
    openai_model: Final = scenario.model(
        model="openai/gpt-4o-mini",
        api_base=f"{provider.url}/v1",
        api_key="synthetic-provider-key",
    )
    anthropic_model: Final = scenario.model(
        model="anthropic/claude-sonnet-4-5-20250929",
        api_base=provider.url,
        api_key="synthetic-provider-key",
    )
    return openai_model, anthropic_model


def _request_key(scenario: Scenario, alias: str, models: tuple[str, str]) -> str:
    return scenario.key(key_alias=alias, models=list(models))


def _chat(gateway: Gateway, model: str, key: str, identity: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": identity}]},
        key=key,
    )


def _database_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _seed_row(
    api_key: str,
    model: str,
    group: str,
    *,
    row_date: date,
    updated_at: datetime,
    prompt_tokens: str = "11",
    completion_tokens: str = "4",
) -> None:
    write_rows(
        _ROW_INSERT,
        (
            api_key,
            row_date.isoformat(),
            api_key,
            model,
            group,
            prompt_tokens,
            completion_tokens,
            "0.0001",
            "1",
            "1",
            "0",
            updated_at.astimezone(timezone.utc).isoformat(),
        ),
    )


def _delete_seeded_rows(api_key: str, marker: str) -> None:
    write_rows(_ROW_DELETE, (_database_key(api_key), f"{marker}%"))


def _readiness_response(client: httpx.Client, url: str) -> httpx.Response | None:
    try:
        return client.get(url)
    except httpx.TransportError:
        return None


def _seed_chunk_rows(api_key: str, marker: str, count: int) -> None:
    write_rows(
        _BULK_INSERT,
        (
            _database_key(api_key),
            datetime.now(timezone.utc).date().isoformat(),
            _database_key(api_key),
            marker,
            marker,
            str(count),
        ),
    )


def _marker_upload_groups(
    records: tuple[MultipartRecord, ...], alias: str, marker: str
) -> tuple[tuple[str, tuple[MultipartRecord, ...]], ...]:
    matching: Final = tuple(
        record
        for record in records
        if any(
            row.tags.get("api_key_alias") == alias and row.values.get("ChargeDescription", "").startswith(f"{marker}-")
            for row in record.rows
        )
    )
    upload_ids: Final = tuple(dict.fromkeys(record.headers["x-ternary-upload-id"] for record in matching))
    return tuple(
        (
            upload_id,
            tuple(record for record in matching if record.headers["x-ternary-upload-id"] == upload_id),
        )
        for upload_id in upload_ids
    )


def _rows_for_marker(records: tuple[MultipartRecord, ...], alias: str, marker: str) -> tuple[CsvRow, ...]:
    return tuple(
        chain.from_iterable(
            tuple(
                row
                for row in row_by_alias(record, alias)
                if row.values.get("ChargeDescription", "").startswith(f"{marker}-")
            )
            for record in records
        )
    )


def _request_identity(request: Request) -> str:
    body: Final = _REQUEST_BODY_ADAPTER.validate_json(request.body)
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    first_message: Final = _MESSAGE_BODY_ADAPTER.validate_python(messages[0])
    identity: Final = first_message["content"]
    assert isinstance(identity, str)
    return identity


def _first_upload_with_aliases(records: tuple[MultipartRecord, ...], aliases: frozenset[str]) -> MultipartRecord | None:
    return next(
        (
            record
            for record in records
            if frozenset(row.tags.get("api_key_alias", "") for row in record.rows) >= aliases
        ),
        None,
    )


def _records_received_or_empty(sink: MultipartSink) -> tuple[MultipartRecord, ...]:
    return eventually(
        lambda: sink.records,
        lambda records: bool(records),
        seconds=3,
        return_last_on_timeout=True,
    )


def _scripted_cache_reply(request: Request) -> Reply:
    identity: Final = _request_identity(request)
    result: Final = (
        {
            "id": identity,
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 4,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
            },
        }
        if request.target.endswith("/messages")
        else {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
                "cache_creation_input_tokens": 5,
            },
        }
    )
    return Reply(body=json.dumps(result).encode())


@pytest.mark.parametrize("surface", SURFACES, ids=SURFACES)
def test_ternary_exports_each_request_surface_with_its_key_alias(ternary_rig: TernaryRig, surface: str) -> None:
    marker: Final = f"r01-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    with ternary_rig.owned.gateway.scenario() as scenario:
        models: Final = _models(scenario, ternary_rig.provider)
        key: Final = _request_key(scenario, alias, models)
        caller_id, _ = call_surface(
            ternary_rig.owned.gateway,
            surface,
            models[0],
            models[1],
            key,
            marker,
        )
        if surface == "responses":
            assert caller_id.startswith("resp_"), caller_id
        else:
            assert caller_id == marker
        upstream_requests: Final = ternary_rig.provider.drain()
        assert any(marker in request.body.decode() for request in upstream_requests), (
            f"{surface} did not reach the scripted upstream with its marker"
        )

        uploads: Final = eventually(
            lambda: uploads_for_alias(ternary_rig.sink.records, alias),
            lambda values: bool(values),
            seconds=45,
        )
        upload, row = uploads[0]
    assert upload.path == "/external-cost-sources/v1/conn%3Aaudit%401/focus"
    assert upload.headers["authorization"] == "Bearer synthetic-ternary-api-key"
    assert re.fullmatch(r"[0-9a-f]{32}", upload.headers["x-ternary-upload-id"])
    assert upload.headers["x-ternary-chunk-index"] == "0"
    assert upload.headers["x-ternary-chunk-total"] == "1"
    assert upload.field_name == "csv"
    assert row.tags["api_key_alias"] == alias
    assert row.tags["prompt_tokens"] == "11"
    assert row.tags["completion_tokens"] == "4"


@pytest.mark.parametrize("surface", ("messages", "chat"), ids=("messages", "chat"))
def test_ternary_cache_usage_tags_keep_read_and_creation_counts(ternary_rig: TernaryRig, surface: str) -> None:
    marker: Final = f"r02-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    with (
        wire_server(_scripted_cache_reply) as provider,
        ternary_rig.owned.gateway.scenario() as scenario,
    ):
        models: Final = _models(scenario, provider)
        key: Final = _request_key(scenario, alias, models)
        caller_id, _ = call_surface(
            ternary_rig.owned.gateway,
            surface,
            models[0],
            models[1],
            key,
            marker,
        )
        assert caller_id == marker
        uploads: Final = eventually(
            lambda: uploads_for_alias(ternary_rig.sink.records, alias),
            lambda values: bool(values),
            seconds=45,
        )
    tags: Final = uploads[0][1].tags
    assert tags["cache_read_input_tokens"] == "3"
    assert tags["cache_creation_input_tokens"] == "5"


def test_ternary_day_window_filters_date_and_updated_at_rows(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r03-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    now: Final = datetime.now(timezone.utc)
    today: Final = now.date()
    yesterday: Final = today - timedelta(days=1)
    older: Final = today - timedelta(days=3)
    with ternary_rig.owned.gateway.scenario() as scenario:
        key: Final = _request_key(scenario, alias, _models(scenario, ternary_rig.provider))
        try:
            _seed_row(_database_key(key), f"{marker}-today", marker, row_date=today, updated_at=now)
            _seed_row(_database_key(key), f"{marker}-yesterday", marker, row_date=yesterday, updated_at=now)
            _seed_row(_database_key(key), f"{marker}-old-date", marker, row_date=older, updated_at=now)
            _seed_row(
                _database_key(key),
                f"{marker}-stale-update",
                marker,
                row_date=today,
                updated_at=now - timedelta(days=3),
            )
            uploads: Final = eventually(
                lambda: uploads_for_alias(ternary_rig.sink.records, alias),
                lambda values: bool(values),
                seconds=45,
            )
            observed_models: Final = frozenset(row.values["ChargeDescription"] for _, row in uploads)
            assert observed_models == frozenset({f"{marker}-today", f"{marker}-yesterday"})
        finally:
            _delete_seeded_rows(key, marker)


def test_ternary_repeated_ticks_resend_identical_rows_with_new_upload_ids(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r04-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    with ternary_rig.owned.gateway.scenario() as scenario:
        models: Final = _models(scenario, ternary_rig.provider)
        key: Final = _request_key(scenario, alias, models)
        response: Final = _chat(ternary_rig.owned.gateway, models[0], key, marker)
        assert response.status_code == 200, response.text
        uploads: Final = eventually(
            lambda: uploads_for_alias(ternary_rig.sink.records, alias),
            lambda values: len({record.headers["x-ternary-upload-id"] for record, _ in values}) >= 2,
            seconds=45,
        )
    first, second = uploads[:2]
    assert first[0].headers["x-ternary-upload-id"] != second[0].headers["x-ternary-upload-id"]
    assert first[1] == second[1]


@pytest.mark.parametrize("status", (403, 404), ids=("403", "404"))
def test_ternary_recovers_from_sink_error_without_restarting_proxy(ternary_rig: TernaryRig, status: int) -> None:
    marker: Final = f"r{5 if status == 403 else 6}-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    ternary_rig.sink.set_status_override(status)
    try:
        with ternary_rig.owned.gateway.scenario() as scenario:
            models: Final = _models(scenario, ternary_rig.provider)
            key: Final = _request_key(scenario, alias, models)
            response: Final = _chat(ternary_rig.owned.gateway, models[0], key, marker)
            assert response.status_code == 200, response.text
            health: Final = ternary_rig.owned.gateway.client.get("/health/readiness")
            assert health.status_code == 200, health.text
            failed: Final = eventually(
                lambda: uploads_for_alias(ternary_rig.sink.records, alias),
                lambda values: any(record.status == status for record, _ in values),
                seconds=45,
            )
            failed_ids: Final = frozenset(
                record.headers["x-ternary-upload-id"] for record, _ in failed if record.status == status
            )
            ternary_rig.sink.set_status_override(None)
            recovered: Final = eventually(
                lambda: uploads_for_alias(ternary_rig.sink.records, alias),
                lambda values: any(
                    record.status == 200 and record.headers["x-ternary-upload-id"] not in failed_ids
                    for record, _ in values
                ),
                seconds=45,
            )
    finally:
        ternary_rig.sink.set_status_override(None)
    assert any(record.status == status for record, _ in failed)
    assert any(
        record.status == 200 and record.headers["x-ternary-upload-id"] not in failed_ids for record, _ in recovered
    )


def test_ternary_does_not_follow_sink_redirects(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r07-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    with running_sink(MultipartSink()) as redirected:
        ternary_rig.sink.set_redirect(f"{redirected.url}/should-not-receive")
        try:
            with ternary_rig.owned.gateway.scenario() as scenario:
                models: Final = _models(scenario, ternary_rig.provider)
                key: Final = _request_key(scenario, alias, models)
                response: Final = _chat(ternary_rig.owned.gateway, models[0], key, marker)
                assert response.status_code == 200, response.text
                upload: Final = eventually(
                    lambda: uploads_for_alias(ternary_rig.sink.records, alias),
                    lambda values: any(record.status == 307 for record, _ in values),
                    seconds=45,
                )
                assert any(record.status == 307 for record, _ in upload)
                redirected_records: Final = eventually(
                    lambda: redirected.records,
                    lambda records: bool(records),
                    seconds=3,
                    return_last_on_timeout=True,
                )
                assert redirected_records == ()
        finally:
            ternary_rig.sink.set_redirect(None)


def test_ternary_chunks_one_hundred_thousand_and_one_seeded_rows(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r08-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    seeded_count: Final = 100_001
    with ternary_rig.owned.gateway.scenario() as scenario:
        key: Final = _request_key(scenario, alias, _models(scenario, ternary_rig.provider))
        try:
            _seed_chunk_rows(key, marker, seeded_count)
            groups: Final = eventually(
                lambda: _marker_upload_groups(ternary_rig.sink.records, alias, marker),
                lambda values: any(
                    sum(len(_rows_for_marker((record,), alias, marker)) for record in chunks) >= seeded_count
                    for _, chunks in values
                ),
                seconds=80,
                return_last_on_timeout=True,
            )
            completed: Final = next(
                (
                    (upload_id, chunks)
                    for upload_id, chunks in groups
                    if sum(len(_rows_for_marker((record,), alias, marker)) for record in chunks) >= seeded_count
                ),
                None,
            )
            group_chunk_details: Final = tuple(
                (
                    upload_id,
                    tuple(
                        (
                            record.headers["x-ternary-chunk-index"],
                            len(_rows_for_marker((record,), alias, marker)),
                            len(record.content),
                            record.status,
                        )
                        for record in chunks
                    ),
                )
                for upload_id, chunks in groups[:3]
            )
            assert completed is not None, (
                f"no upload id delivered all seeded rows; first upload chunk details: {group_chunk_details!r}"
            )
            upload_id, chunks = completed
            records: Final = chunks
            indices: Final = tuple(int(record.headers["x-ternary-chunk-index"]) for record in records)
            totals: Final = frozenset(int(record.headers["x-ternary-chunk-total"]) for record in records)
            rows: Final = _rows_for_marker(chunks, alias, marker)
            model_names: Final = tuple(row.values["ChargeDescription"] for row in rows)
            assert indices == tuple(range(len(records)))
            assert totals == {len(records)}
            assert len(records) >= 2
            assert len(
                {(record.headers["x-ternary-upload-id"], index) for record, index in zip(records, indices)}
            ) == len(records)
            assert all(record.headers["x-ternary-upload-id"] == upload_id for record in records)
            assert all(record.status == 200 for record in records)
            assert all(len(_rows_for_marker((record,), alias, marker)) <= 100_000 for record in records)
            assert all(len(record.content) <= 30 * 1024 * 1024 for record in records)
            assert all(record.filename.endswith(f".part{index + 1}") for index, record in enumerate(records))
            assert len(model_names) == seeded_count
            assert len(set(model_names)) == seeded_count
            assert frozenset(model_names) == frozenset(f"{marker}-{index}" for index in range(seeded_count))
        finally:
            _delete_seeded_rows(key, marker)


def test_ternary_stops_after_chunk_two_fails_then_retries_with_a_fresh_id(ternary_rig: TernaryRig) -> None:
    identity: Final = f"r09-{uuid4().hex[:8]}"
    marker: Final = f"{identity}-{'x' * 50_000}"
    alias: Final = f"{identity}-alias"
    seeded_count: Final = 350
    ternary_rig.sink.set_fail_chunk_once(1)
    with ternary_rig.owned.gateway.scenario() as scenario:
        key: Final = _request_key(scenario, alias, _models(scenario, ternary_rig.provider))
        try:
            _seed_chunk_rows(key, marker, seeded_count)
            partial_groups: Final = eventually(
                lambda: _marker_upload_groups(ternary_rig.sink.records, alias, marker),
                lambda groups: any(
                    any(record.status == 500 and record.headers["x-ternary-chunk-index"] == "1" for record in chunks)
                    for _, chunks in groups
                ),
                seconds=80,
            )
            failed_id, failed_chunks = next(
                (upload_id, chunks)
                for upload_id, chunks in partial_groups
                if any(record.status == 500 and record.headers["x-ternary-chunk-index"] == "1" for record in chunks)
            )
            failed_indices: Final = tuple(int(record.headers["x-ternary-chunk-index"]) for record in failed_chunks)
            assert failed_indices == (0, 1)
            assert tuple(record.status for record in failed_chunks) == (200, 500)
            assert all(int(record.headers["x-ternary-chunk-index"]) < 2 for record in failed_chunks)
            retried: Final = eventually(
                lambda: _marker_upload_groups(ternary_rig.sink.records, alias, marker),
                lambda groups: any(
                    upload_id != failed_id
                    and len(chunks) >= 2
                    and frozenset(int(record.headers["x-ternary-chunk-index"]) for record in chunks)
                    == frozenset(range(int(chunks[0].headers["x-ternary-chunk-total"])))
                    for upload_id, chunks in groups
                ),
                seconds=80,
            )
            retry_id, retry_chunks = next(
                (
                    (upload_id, chunks)
                    for upload_id, chunks in retried
                    if upload_id != failed_id
                    and len(chunks) >= 2
                    and frozenset(int(record.headers["x-ternary-chunk-index"]) for record in chunks)
                    == frozenset(range(int(chunks[0].headers["x-ternary-chunk-total"])))
                )
            )
            final_groups: Final = _marker_upload_groups(ternary_rig.sink.records, alias, marker)
            final_failed_chunks: Final = next(chunks for upload_id, chunks in final_groups if upload_id == failed_id)
            assert tuple(int(record.headers["x-ternary-chunk-index"]) for record in final_failed_chunks) == (0, 1)
            failed_total: Final = int(final_failed_chunks[0].headers["x-ternary-chunk-total"])
            assert failed_total >= 3
            assert frozenset(int(record.headers["x-ternary-chunk-total"]) for record in final_failed_chunks) == {
                failed_total
            }
            assert retry_id != failed_id
            retry_total: Final = int(retry_chunks[0].headers["x-ternary-chunk-total"])
            retry_indices: Final = tuple(int(record.headers["x-ternary-chunk-index"]) for record in retry_chunks)
            assert retry_indices == tuple(range(retry_total))
            assert retry_total >= 3
            assert all(record.status == 200 for record in retry_chunks)
            assert frozenset(int(record.headers["x-ternary-chunk-total"]) for record in retry_chunks) == {retry_total}
            retry_models: Final = tuple(
                row.values["ChargeDescription"] for row in _rows_for_marker(retry_chunks, alias, marker)
            )
            assert len(retry_models) == seeded_count
            assert len(set(retry_models)) == seeded_count
            assert frozenset(retry_models) == frozenset(f"{marker}-{index}" for index in range(seeded_count))
        finally:
            _delete_seeded_rows(key, marker)
            ternary_rig.sink.set_fail_chunk_once(None)


def test_ternary_oversized_row_blocks_exports_until_seed_cleanup(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r10-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    huge_group: Final = "x" * (31 * 1024 * 1024)
    now: Final = datetime.now(timezone.utc)
    with ternary_rig.owned.gateway.scenario() as scenario:
        models: Final = _models(scenario, ternary_rig.provider)
        key: Final = _request_key(scenario, alias, models)
        response: Final = _chat(ternary_rig.owned.gateway, models[0], key, marker)
        assert response.status_code == 200, response.text
        try:
            _seed_row(_database_key(key), f"{marker}-normal", marker, row_date=now.date(), updated_at=now)
            _seed_row(
                _database_key(key),
                f"{marker}-oversized",
                huge_group,
                row_date=now.date(),
                updated_at=now,
            )
            before: Final = ternary_rig.sink.records
            response_during_block: Final = _chat(ternary_rig.owned.gateway, models[0], key, f"{marker}-health")
            assert response_during_block.status_code == 200, response_during_block.text
            quiet: Final = eventually(
                lambda: ternary_rig.sink.records,
                lambda records: len(records) > len(before),
                seconds=3,
                return_last_on_timeout=True,
            )
            assert len(quiet) == len(before), "sink received a request while the oversized row was present"
            write_rows(
                'DELETE FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s AND model=%s',
                (_database_key(key), f"{marker}-oversized"),
            )
            resumed: Final = eventually(
                lambda: uploads_for_alias(ternary_rig.sink.records, alias),
                lambda values: any(row.values.get("ChargeDescription") == f"{marker}-normal" for _, row in values),
                seconds=45,
            )
            assert any(row.values.get("ChargeDescription") == f"{marker}-normal" for _, row in resumed)
        finally:
            write_rows(
                'DELETE FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s AND model=%s',
                (_database_key(key), f"{marker}-oversized"),
            )
            _delete_seeded_rows(key, marker)


@contextmanager
def _special_proxy(
    gateway: Gateway,
    directory: Path,
    sink_url: str,
    *,
    callbacks: tuple[str, ...] = ("ternary",),
    environment: Mapping[str, str] | None = None,
    remove_environment: tuple[str, ...] = (),
    redis_db: int | None = None,
) -> Generator[OwnedProxy, None, None]:
    config: Final = write_proxy_config(directory, callbacks, redis_db=redis_db)
    default_environment: Final = ternary_environment(sink_url)
    environment_values: Final = {**default_environment, **(environment or {})}
    overrides: Final = MappingProxyType(
        {name: value for name, value in environment_values.items() if name not in remove_environment}
    )
    with owned_proxy_process(
        gateway,
        directory,
        overrides,
        config=config,
        remove_environment=remove_environment,
        workers=2,
    ) as owned:
        yield owned


@pytest.mark.parametrize(
    ("variable", "empty"),
    (
        pytest.param("TERNARY_API_KEY", True, id="empty-api-key"),
        pytest.param("TERNARY_API_KEY", False, id="missing-api-key"),
        pytest.param("TERNARY_CONNECTION_ID", True, id="empty-connection-id"),
        pytest.param("TERNARY_CONNECTION_ID", False, id="missing-connection-id"),
        pytest.param("TERNARY_BASE_URL", True, id="empty-base-url"),
        pytest.param("TERNARY_BASE_URL", False, id="missing-base-url"),
    ),
)
def test_ternary_missing_or_empty_environment_rejects_uploads(
    gateway: Gateway,
    tmp_path: Path,
    variable: str,
    empty: bool,
) -> None:
    with running_sink(MultipartSink()) as sink:
        excluded: Final = frozenset() if empty else frozenset({variable})
        base_environment: Final = ternary_environment(sink.url, excluded=excluded)
        overrides: Final = MappingProxyType({**base_environment, **({variable: ""} if empty else {})})
        remove: Final = tuple(excluded)
        with _special_proxy(gateway, tmp_path, sink.url, environment=overrides, remove_environment=remove) as owned:
            log_text: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: variable in text,
                seconds=30,
            )
            response: Final = owned.gateway.client.get("/health/readiness")
            assert response.status_code == 200, response.text
            assert _records_received_or_empty(sink) == ()
            assert variable in log_text


def test_ternary_rejects_non_loopback_http_sink(gateway: Gateway, tmp_path: Path) -> None:
    address: Final = non_loopback_ipv4()
    sink: Final = MultipartSink()
    with bound_sink_server(sink, address) as bound, wire_server(surface_reply) as provider:
        environment: Final = MappingProxyType({**ternary_environment(bound.url), "TERNARY_BASE_URL": bound.url})
        with _special_proxy(gateway, tmp_path, bound.url, environment=environment) as owned:
            with owned.gateway.scenario() as scenario:
                models: Final = _models(scenario, provider)
                key: Final = _request_key(scenario, f"r12-{uuid4().hex[:8]}", models)
                response: Final = _chat(owned.gateway, models[0], key, f"r12-{uuid4().hex[:8]}")
                assert response.status_code == 200, response.text
                log_text: Final = eventually(
                    lambda: owned.log.read_text(),
                    lambda text: "base_url must be an HTTPS URL" in text,
                    seconds=30,
                )
                readiness: Final = owned.gateway.client.get("/health/readiness")
                assert readiness.status_code == 200, readiness.text
                assert _records_received_or_empty(sink) == ()
                assert "base_url must be an HTTPS URL" in log_text


@pytest.mark.parametrize(
    "connection_id",
    ("/", "audit/id", "..", "bad id"),
    ids=("bare-slash", "embedded-slash", "dot-dot", "whitespace"),
)
def test_ternary_rejects_unsafe_connection_id(gateway: Gateway, tmp_path: Path, connection_id: str) -> None:
    with running_sink(MultipartSink()) as sink, wire_server(surface_reply) as provider:
        environment: Final = MappingProxyType(
            {
                **ternary_environment(sink.url),
                "TERNARY_CONNECTION_ID": connection_id,
            }
        )
        with _special_proxy(gateway, tmp_path, sink.url, environment=environment, redis_db=15) as owned:
            with owned.gateway.scenario() as scenario:
                models: Final = _models(scenario, provider)
                key: Final = _request_key(scenario, f"r13-{uuid4().hex[:8]}", models)
                response: Final = _chat(owned.gateway, models[0], key, f"r13-{uuid4().hex[:8]}")
                assert response.status_code == 200, response.text
                spend_rows: Final = eventually(
                    lambda: read_rows(
                        'SELECT api_requests FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s',
                        (_database_key(key),),
                    ),
                    lambda rows: bool(rows),
                    seconds=15,
                )
                assert spend_rows
                log_text: Final = eventually(
                    lambda: owned.log.read_text(),
                    lambda text: "connection_id must not contain" in text,
                    seconds=30,
                )
                readiness: Final = owned.gateway.client.get("/health/readiness")
                assert readiness.status_code == 200, readiness.text
                assert _records_received_or_empty(sink) == ()
                assert "connection_id must not contain" in log_text


@pytest.mark.parametrize(
    ("frequency", "interval", "expected_error"),
    (
        pytest.param("hourly", "1", "Unsupported TERNARY_EXPORT_FREQUENCY", id="unsupported-frequency"),
        pytest.param("interval", "0", "TERNARY_EXPORT_INTERVAL_SECONDS must be a positive integer", id="zero-interval"),
        pytest.param(
            "interval", "invalid", "TERNARY_EXPORT_INTERVAL_SECONDS must be a positive integer", id="invalid-interval"
        ),
    ),
)
def test_ternary_invalid_frequency_keeps_proxy_serving_without_uploads(
    gateway: Gateway,
    tmp_path: Path,
    frequency: str,
    interval: str,
    expected_error: str,
) -> None:
    with running_sink(MultipartSink()) as sink, wire_server(surface_reply) as provider:
        environment: Final = MappingProxyType(
            {
                **ternary_environment(sink.url, interval_seconds=interval),
                "TERNARY_EXPORT_FREQUENCY": frequency,
            }
        )
        with _special_proxy(gateway, tmp_path, sink.url, environment=environment) as owned:
            readiness: Final = owned.gateway.client.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
            with owned.gateway.scenario() as scenario:
                models: Final = _models(scenario, provider)
                marker: Final = f"r14-{uuid4().hex[:8]}"
                key: Final = _request_key(scenario, f"{marker}-alias", models)
                response: Final = _chat(owned.gateway, models[0], key, marker)
                assert response.status_code == 200, response.text
                upstream_requests: Final = provider.drain()
                assert any(marker in request.body.decode() for request in upstream_requests), (
                    f"chat request did not reach the scripted upstream with {marker}"
                )
            log_text: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: "ValueError:" in text and expected_error in text,
                seconds=30,
            )
            assert "ValueError:" in log_text, log_text
            assert expected_error in log_text, log_text
            assert _records_received_or_empty(sink) == ()


def test_ternary_two_workers_share_the_redis_upload_lock(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r15-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    ternary_rig.sink.reset_peak_in_flight()
    ternary_rig.sink.set_delay(3)
    try:
        with ternary_rig.owned.gateway.scenario() as scenario:
            models: Final = _models(scenario, ternary_rig.provider)
            key: Final = _request_key(scenario, alias, models)
            response: Final = _chat(ternary_rig.owned.gateway, models[0], key, marker)
            assert response.status_code == 200, response.text
            records: Final = eventually(
                lambda: tuple(
                    record for record, _ in uploads_for_alias(ternary_rig.sink.records, alias) if record.status == 200
                ),
                lambda values: len(values) >= 2,
                seconds=40,
            )
            health: Final = ternary_rig.owned.gateway.client.get("/health/readiness")
            assert health.status_code == 200, health.text
    finally:
        ternary_rig.sink.set_delay(0)
        eventually(lambda: ternary_rig.sink.in_flight, lambda count: count == 0, seconds=20)
    pairs: Final = tuple(
        (record.headers["x-ternary-upload-id"], record.headers["x-ternary-chunk-index"]) for record in records
    )
    assert ternary_rig.sink.peak_in_flight == 1
    assert len(set(pairs)) == len(pairs)


def _surface_jobs(
    scenario: Scenario, models: tuple[str, str], marker: str, per_surface: int
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        chain.from_iterable(
            tuple(
                (
                    surface,
                    f"{marker}-{surface}-{index}",
                    _request_key(scenario, f"{marker}-{surface}-{index}", models),
                )
                for index in range(per_surface)
            )
            for surface in SURFACES
        )
    )


def test_ternary_sink_restart_recovers_all_mixed_surface_aliases(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r16-{uuid4().hex[:8]}"
    with ternary_rig.owned.gateway.scenario() as scenario:
        models: Final = _models(scenario, ternary_rig.provider)
        jobs: Final = _surface_jobs(scenario, models, marker, 6)
        expected_aliases: Final = frozenset(alias for _, alias, _ in jobs)
        ternary_rig.sink.stop()
        previous_record_count: Final = len(ternary_rig.sink.records)
        with ThreadPoolExecutor(max_workers=36) as pool:
            answers: Final = tuple(
                pool.map(
                    lambda job: (
                        job[1],
                        call_surface(
                            ternary_rig.owned.gateway,
                            job[0],
                            models[0],
                            models[1],
                            job[2],
                            job[1],
                        )[0],
                    ),
                    jobs,
                )
            )
            assert frozenset(alias for alias, _ in answers) == expected_aliases
            assert len(ternary_rig.sink.records) == previous_record_count
        ternary_rig.sink.start()
        record: Final = eventually(
            lambda: _first_upload_with_aliases(
                tuple(upload for upload in ternary_rig.sink.records[previous_record_count:] if upload.status == 200),
                expected_aliases,
            ),
            lambda value: value is not None,
            seconds=80,
        )
        assert record is not None
    observed_aliases: Final = tuple(
        row.tags.get("api_key_alias", "") for row in record.rows if row.tags.get("api_key_alias") in expected_aliases
    )
    assert frozenset(observed_aliases) == expected_aliases
    assert len(observed_aliases) == len(expected_aliases)
    unexpected_rows: Final = tuple(
        (
            row.tags.get("api_key_alias", ""),
            row.values.get("ConsumedQuantity"),
            row.tags.get("prompt_tokens"),
            row.tags.get("completion_tokens"),
        )
        for row in record.rows
        if row.tags.get("api_key_alias") in expected_aliases
        and (
            Decimal(row.values.get("ConsumedQuantity") or "NaN") != Decimal("1")
            or row.tags.get("prompt_tokens") != "11"
            or row.tags.get("completion_tokens") != "4"
        )
    )
    assert not unexpected_rows, f"unexpected usage fields for burst aliases: {unexpected_rows!r}"


def _worker_is_alive(process: psutil.Process) -> bool:
    try:
        return process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_ternary_worker_kill_keeps_serving_and_delivers_successful_aliases(ternary_rig: TernaryRig) -> None:
    marker: Final = f"r17-{uuid4().hex[:8]}"
    aliases: Final = tuple(f"{marker}-{index}" for index in range(12))
    fast_aliases: Final = aliases[:4]
    burst_aliases: Final = aliases[4:]
    with ternary_rig.owned.gateway.scenario() as scenario:
        ternary_rig.provider.drain()
        models: Final = _models(scenario, ternary_rig.provider)
        keys: Final = tuple(_request_key(scenario, alias, models) for alias in aliases)
        probe_alias: Final = f"{marker}-after-kill"
        probe_key: Final = _request_key(scenario, probe_alias, models)

        def send(job: tuple[str, str]) -> tuple[str, bool, str]:
            alias, key = job
            try:
                response: Final = _chat(ternary_rig.owned.gateway, models[0], key, alias)
            except httpx.TransportError as error:
                return alias, False, str(error)
            return alias, response.status_code == 200, response.text

        fast_jobs: Final = tuple(zip(fast_aliases, keys[:4]))
        burst_jobs: Final = tuple(zip(burst_aliases, keys[4:]))
        fast_record_start: Final = len(ternary_rig.sink.records)
        with ThreadPoolExecutor(max_workers=12) as pool:
            fast_answers: Final = tuple(pool.map(send, fast_jobs))
            assert all(ok for _, ok, _ in fast_answers), fast_answers
            assert len(tuple(request for request in ternary_rig.provider.drain() if request.method == "POST")) == len(
                fast_aliases
            )
            expected_fast_aliases: Final = frozenset(fast_aliases)
            fast_delivery: Final = eventually(
                lambda: _first_upload_with_aliases(
                    ternary_rig.sink.records[fast_record_start:],
                    expected_fast_aliases,
                ),
                lambda upload: upload is not None,
                seconds=12,
            )
            assert fast_delivery is not None
            fast_rows: Final = tuple(
                row for row in fast_delivery.rows if row.tags.get("api_key_alias") in expected_fast_aliases
            )
            assert frozenset(row.tags["api_key_alias"] for row in fast_rows) == expected_fast_aliases
            assert len(fast_rows) == len(expected_fast_aliases)
            burst_record_start: Final = len(ternary_rig.sink.records)
            ternary_rig.upstream_gate.clear()
            try:
                burst_futures: Final = tuple(pool.submit(send, job) for job in burst_jobs)
                eventually(
                    lambda: ternary_rig.provider.received.qsize(),
                    lambda count: count >= 2,
                    seconds=20,
                )
                worker_processes: Final = tuple(
                    process
                    for process in group_members(ternary_rig.owned.process.pid)
                    if process.pid != ternary_rig.owned.process.pid and "spawn_main" in " ".join(process.cmdline())
                )
                assert len(worker_processes) >= 2, "two proxy workers were not running"
                assert any(not future.done() for future in burst_futures), "worker kill did not occur during the burst"
                worker_processes[0].kill()
            finally:
                ternary_rig.upstream_gate.set()
            eventually(lambda: _worker_is_alive(worker_processes[0]), lambda running: not running, seconds=5)
            assert _worker_is_alive(worker_processes[1]), "the surviving proxy worker exited after its peer was killed"
            burst_answers: Final = tuple(future.result() for future in burst_futures)
        readiness_url: Final = str(ternary_rig.owned.gateway.client.base_url.join("/health/readiness"))
        with httpx.Client(
            base_url=str(ternary_rig.owned.gateway.client.base_url),
            timeout=10,
            trust_env=False,
        ) as client:
            readiness: Final = eventually(
                lambda: _readiness_response(client, readiness_url),
                lambda response: response is not None and response.status_code == 200,
                seconds=10,
            )
            assert readiness is not None
            assert readiness.status_code == 200, readiness.text
            survivor_gateway: Final = Gateway(
                client,
                ternary_rig.owned.gateway.key,
                ternary_rig.owned.gateway.upstream_url,
            )
            probe_response: Final = _chat(survivor_gateway, models[0], probe_key, probe_alias)
            assert probe_response.status_code == 200, probe_response.text
        successful: Final = frozenset(alias for alias, ok, _ in burst_answers if ok) | {probe_alias}
        delivery: Final = eventually(
            lambda: _first_upload_with_aliases(
                ternary_rig.sink.records[burst_record_start:],
                successful,
            ),
            lambda value: value is not None,
            seconds=30,
            return_last_on_timeout=True,
        )
        later_records: Final = ternary_rig.sink.records[burst_record_start:]
        aliases_per_upload: Final = tuple(
            frozenset(
                row.tags.get("api_key_alias", "") for row in record.rows if row.tags.get("api_key_alias") in successful
            )
            for record in later_records
        )
        observed_aliases: Final = frozenset(chain.from_iterable(aliases_per_upload))
        upload_alias_counts: Final = tuple(len(aliases) for aliases in aliases_per_upload)
        assert delivery is not None, (
            f"no upload contained all successful aliases; "
            f"missing={sorted(successful - observed_aliases)!r}; observed={sorted(observed_aliases)!r}; "
            f"upload_alias_counts={upload_alias_counts!r}"
        )
    rows: Final = tuple(row for row in delivery.rows if row.tags.get("api_key_alias") in successful)
    row_aliases: Final = tuple(row.tags["api_key_alias"] for row in rows)
    assert frozenset(row_aliases) == successful
    assert len(row_aliases) == len(successful)


def test_ternary_sigterm_restart_delivers_each_successful_alias_once(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"r18-{uuid4().hex[:8]}"
    aliases: Final = tuple(f"{marker}-{index}" for index in range(18))
    fast_aliases: Final = aliases[:12]
    burst_aliases: Final = aliases[12:]
    release_upstream: Final = Event()
    sink: Final = MultipartSink(delay_seconds=3)
    directory: Final = tmp_path
    config: Final = write_proxy_config(directory, ("ternary",))

    def provider_reply(request: Request) -> Reply:
        if not request.body:
            return surface_reply(request)
        reply: Final = surface_reply(request)
        identity: Final = _request_identity(request)
        return (
            Reply(
                status=reply.status,
                content_type=reply.content_type,
                chunks=(reply.body, b""),
                gate_after_first=release_upstream,
                headers=reply.headers,
            )
            if identity in burst_aliases
            else reply
        )

    with (
        running_sink(sink),
        wire_server(provider_reply) as provider,
        gateway.scenario() as scenario,
    ):
        models: Final = _models(scenario, provider)
        keys: Final = tuple(_request_key(scenario, alias, models) for alias in aliases)
        overrides: Final = ternary_environment(sink.url)
        with owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as first:

            def send(job: tuple[str, str]) -> tuple[str, bool]:
                alias, key = job
                try:
                    response: Final = _chat(first.gateway, models[0], key, alias)
                except httpx.TransportError:
                    return alias, False
                return alias, response.status_code == 200

            with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
                fast_jobs: Final = tuple(zip(fast_aliases, keys[:12]))
                fast_futures: Final = tuple(pool.submit(send, job) for job in fast_jobs)
                fast_answers: Final = tuple(future.result() for future in fast_futures)
                eventually(lambda: sink.in_flight, lambda count: count > 0, seconds=40)
                burst_jobs: Final = tuple(zip(burst_aliases, keys[12:]))
                burst_futures: Final = tuple(pool.submit(send, job) for job in burst_jobs)
                try:
                    eventually(
                        lambda: provider.received.qsize(),
                        lambda count: count >= len(aliases),
                        seconds=20,
                    )
                    assert any(not future.done() for future in burst_futures)
                    first.process.terminate()
                finally:
                    release_upstream.set()
                burst_answers: Final = tuple(future.result() for future in burst_futures)
                answers: Final = (*fast_answers, *burst_answers)
                first.process.wait(timeout=30)
        successful: Final = frozenset(alias for alias, ok in answers if ok)
        assert successful, "no caller received 200 before SIGTERM"
        previous_record_count: Final = len(sink.records)
        with owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as restarted:
            delivery: Final = eventually(
                lambda: sink.records[previous_record_count:],
                lambda records: (
                    _first_upload_with_aliases(
                        tuple(record for record in records if record.status == 200),
                        successful,
                    )
                    is not None
                ),
                seconds=60,
            )
            completed: Final = _first_upload_with_aliases(
                tuple(record for record in delivery if record.status == 200),
                successful,
            )
            assert completed is not None
            observed: Final = tuple(
                row.tags["api_key_alias"] for row in completed.rows if row.tags.get("api_key_alias") in successful
            )
            assert frozenset(observed) == successful
            assert len(observed) == len(successful)
            health: Final = restarted.gateway.client.get("/health/readiness")
            assert health.status_code == 200, health.text
    sink.set_delay(0)


def test_vantage_callback_keeps_its_csv_path_and_tags(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"r19-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    with (
        running_sink(MultipartSink()) as sink,
        wire_server(surface_reply) as provider,
    ):
        config: Final = write_proxy_config(tmp_path, ("vantage",))
        environment: Final = vantage_environment(sink.url)
        with owned_proxy_process(gateway, tmp_path, environment, config=config, workers=2) as owned:
            with owned.gateway.scenario() as scenario:
                models: Final = _models(scenario, provider)
                key: Final = _request_key(scenario, alias, models)
                response: Final = _chat(owned.gateway, models[0], key, marker)
                assert response.status_code == 200, response.text
                uploads: Final = eventually(
                    lambda: uploads_for_alias(sink.records, alias),
                    lambda values: bool(values),
                    seconds=45,
                )
                upload, row = uploads[0]
                assert upload.path == "/v2/integrations/synthetic-vantage-token/costs.csv"
                assert upload.headers["authorization"] == "Bearer synthetic-vantage-api-key"
                assert not _TOKEN_TAGS.intersection(row.tags)


def test_vantage_and_ternary_callbacks_receive_only_their_own_paths(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"r20-{uuid4().hex[:8]}"
    alias: Final = f"{marker}-alias"
    with (
        running_sink(MultipartSink()) as ternary_sink,
        running_sink(MultipartSink()) as vantage_sink,
        wire_server(surface_reply) as provider,
    ):
        config: Final = write_proxy_config(tmp_path, ("vantage", "ternary"))
        environment: Final = MappingProxyType(
            {
                **ternary_environment(ternary_sink.url),
                **vantage_environment(vantage_sink.url),
            }
        )
        with owned_proxy_process(gateway, tmp_path, environment, config=config, workers=2) as owned:
            with owned.gateway.scenario() as scenario:
                models: Final = _models(scenario, provider)
                key: Final = _request_key(scenario, alias, models)
                response: Final = _chat(owned.gateway, models[0], key, marker)
                assert response.status_code == 200, response.text
                ternary_uploads: Final = eventually(
                    lambda: uploads_for_alias(ternary_sink.records, alias),
                    lambda values: bool(values),
                    seconds=45,
                )
                vantage_uploads: Final = eventually(
                    lambda: uploads_for_alias(vantage_sink.records, alias),
                    lambda values: bool(values),
                    seconds=45,
                )
                ternary_record, ternary_row = ternary_uploads[0]
                vantage_record, vantage_row = vantage_uploads[0]
                assert ternary_record.path == "/external-cost-sources/v1/conn%3Aaudit%401/focus"
                assert vantage_record.path == "/v2/integrations/synthetic-vantage-token/costs.csv"
                assert ternary_sink.records
                assert vantage_sink.records
                assert all(record.path.startswith("/external-cost-sources/v1/") for record in ternary_sink.records)
                assert all(
                    record.path == "/v2/integrations/synthetic-vantage-token/costs.csv"
                    for record in vantage_sink.records
                )
                assert _TOKEN_TAGS <= ternary_row.tags.keys()
                assert not _TOKEN_TAGS.intersection(vantage_row.tags)
