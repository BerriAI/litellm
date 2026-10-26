import json
import os
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from itertools import chain
from typing import Final
from uuid import uuid4

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration._support.prisma_clients import connected_proxy_prisma
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue, TypeAdapter

from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE

INPUT_COST_PER_TOKEN: Final = 0.001
OUTPUT_COST_PER_TOKEN: Final = 0.002
PROMPT_TOKENS: Final = 100
COMPLETION_TOKENS: Final = 50
BATCH_COST_SHARE: Final = 0.5

_INPUT_FILE: Final = JsonResponse(
    content_type="application/json",
    body={
        "id": "file-in-$REQUEST_ID",
        "object": "file",
        "purpose": "batch",
        "bytes": 100,
        "created_at": 1,
        "filename": "in.jsonl",
        "status": "processed",
    },
)


def _batch(status: str, output_file_id: str | None) -> dict[str, JsonValue]:
    return {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": status,
        "output_file_id": output_file_id,
        "error_file_id": None,
        "created_at": 1,
        "in_progress_at": 1,
        "completed_at": 1 if status == "completed" else None,
        "expires_at": 1,
        "request_counts": {"total": 1, "completed": 1 if status == "completed" else 0, "failed": 0},
        "metadata": None,
    }


def _accepting_routes() -> dict[str, JsonResponse | TextResponse]:
    return {
        "POST /files": _INPUT_FILE,
        "POST /batches": JsonResponse(content_type="application/json", body=_batch("validating", None)),
    }


def _gone_at_provider_routes(*, retryable: bool = False) -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            **_accepting_routes(),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(
                content_type="application/json",
                status=503 if retryable else 404,
                body={
                    "error": {
                        "message": "Temporary failure" if retryable else "No batch found with id 'batch-$REQUEST_ID'.",
                        "type": "invalid_request_error",
                        "param": "id",
                        "code": "batch_not_found",
                    }
                },
            ),
        },
    )


def _completed_routes(*, ready: bool = True) -> RoutedResponse:
    output_line: Final = {
        "id": "batch_req_1",
        "custom_id": "r1",
        "response": {
            "status_code": 200,
            "request_id": "$REQUEST_ID-1",
            "body": {
                "id": "chatcmpl-$REQUEST_ID-1",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            },
        },
        "error": None,
    }
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            **_accepting_routes(),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(
                content_type="application/json",
                body=_batch("completed", "file-out-$REQUEST_ID") if ready else _batch("in_progress", None),
            ),
            "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body=json.dumps(output_line, separators=(",", ":")) + "\n"
            ),
        },
    )


def _batch_was_polled(upstream: ScenarioHandle) -> bool:
    response: Final = httpx.get(f"{upstream.control_url}/__observations", timeout=15, trust_env=False)
    response.raise_for_status()
    observations: Final = TypeAdapter(dict[str, list[dict[str, JsonValue]]]).validate_json(response.content)
    return any(
        item.get("method") == "GET" and string_value(item["path"]).startswith(f"/{upstream.scenario_id}/batches/")
        for item in observations["requests"]
    )


def _missing_output_routes() -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            **_completed_routes().routes,
            "GET /files/file-out-$REQUEST_ID/content": JsonResponse(
                content_type="application/json",
                status=404,
                body={
                    "error": {
                        "message": "No file found with id 'file-out-$REQUEST_ID'.",
                        "type": "invalid_request_error",
                        "code": "file_not_found",
                    }
                },
            ),
        },
    )


def _scripted_deployment(scenario: Scenario, marker: str, routes: RoutedResponse) -> tuple[str, ScenarioHandle]:
    scenario_id: Final = f"poll-{marker}-{sha256(os.urandom(16)).hexdigest()[:12]}"
    handle: Final = register_scenario(scenario_id, routes)
    scenario.cleanups.callback(delete_scenario, handle)
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": f"poll-{marker}-{sha256(scenario_id.encode()).hexdigest()[:12]}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-scripted-provider",
                "api_base": handle.api_base(),
                "input_cost_per_token": INPUT_COST_PER_TOKEN,
                "output_cost_per_token": OUTPUT_COST_PER_TOKEN,
            },
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return string_value(created["model_name"]), handle


def _submitted_batch_id(gateway: Gateway, key: str, model_name: str) -> str:
    request_line: Final = {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model_name, "messages": [{"role": "user", "content": "poll starvation"}]},
    }
    file_response: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model_name},
        {"file": ("in.jsonl", (json.dumps(request_line) + "\n").encode(), "application/jsonl")},
        key=key,
    )
    assert file_response.is_success, file_response.text
    batch_response: Final = gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(file_response.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model_name,
        },
        key=key,
    )
    assert batch_response.is_success, batch_response.text
    return string_value(JSON_OBJECT.validate_json(batch_response.content)["id"])


def _managed_rows(batch_ids: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    placeholders: Final = ", ".join("%s" for _ in batch_ids)
    return read_rows(
        f'SELECT batch_processed FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id IN ({placeholders})',
        batch_ids,
    )


@pytest.mark.parametrize(
    ("retryable", "old_completed", "backlog"),
    [(False, False, 0), (False, True, 0), (True, False, 0), (True, True, 0), (True, False, 1000)],
)
@pytest.mark.timeout(180)
@pytest.mark.covers("quota_management.spend_tracking.batch_costs.uncostable_rows_retire_so_newer_batches_are_costed")
def test_uncostable_batches_do_not_starve_billable_batches(
    gateway: Gateway, retryable: bool, old_completed: bool, backlog: int
) -> None:
    with gateway.scenario() as scenario:
        owner: Final = f"batch-backlog-{uuid4().hex}"
        if backlog:
            _insert_scan_rows(owner, "old", backlog, datetime(2000, 1, 1, tzinfo=timezone.utc))
            scenario.cleanups.callback(
                write_rows, 'DELETE FROM "LiteLLM_ManagedObjectTable" WHERE created_by = %s', (owner,)
            )
        team_id: Final = scenario.team()
        key: Final = scenario.key(team_id=team_id)
        gone_batch_ids: Final = tuple(
            _submitted_batch_id(
                gateway,
                key,
                _scripted_deployment(scenario, f"gone{index}", _gone_at_provider_routes(retryable=retryable))[0],
            )
            for index in range(MAX_OBJECTS_PER_POLL_CYCLE)
        )
        missing_output_id: Final = _submitted_batch_id(
            gateway, key, _scripted_deployment(scenario, "missing-output", _missing_output_routes())[0]
        )
        model, upstream = _scripted_deployment(scenario, "costable", _completed_routes(ready=False))
        costable_batch_id: Final = _submitted_batch_id(gateway, key, model)
        for batch_id in (*gone_batch_ids, missing_output_id, costable_batch_id):
            scenario.cleanups.callback(
                write_rows,
                'DELETE FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id = %s',
                (batch_id,),
            )
        if old_completed:
            write_rows(
                'UPDATE "LiteLLM_ManagedObjectTable" SET status = %s, '
                "created_at = NOW() - INTERVAL '30 days' WHERE unified_object_id = %s",
                ("completed", costable_batch_id),
            )
        assert eventually(lambda: _batch_was_polled(upstream), bool, seconds=120)
        register_scenario(upstream.scenario_id, _completed_routes(), control_url=upstream.control_url)
        spend_rows: Final = eventually(
            lambda: read_rows(
                'SELECT call_type, status, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" '
                "WHERE api_key = %s AND call_type = %s",
                (sha256(key.encode()).hexdigest(), "aretrieve_batch"),
            ),
            lambda rows: len(rows) == 1,
            seconds=120,
        )
        assert spend_rows == [
            {
                "call_type": "aretrieve_batch",
                "status": "success",
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "spend": pytest.approx(
                    BATCH_COST_SHARE
                    * (PROMPT_TOKENS * INPUT_COST_PER_TOKEN + COMPLETION_TOKENS * OUTPUT_COST_PER_TOKEN)
                ),
            }
        ]
        assert _managed_rows((costable_batch_id,)) == [{"batch_processed": True}]
        assert eventually(
            lambda: _managed_rows((missing_output_id,)),
            lambda rows: rows == [{"batch_processed": True}],
            seconds=120,
        ) == [{"batch_processed": True}]
        expected_blockers: Final = [{"batch_processed": not retryable}] * MAX_OBJECTS_PER_POLL_CYCLE
        blockers: Final = eventually(
            lambda: _managed_rows(gone_batch_ids), lambda rows: rows == expected_blockers, seconds=120
        )
        assert blockers == expected_blockers

        expected_spend: Final = BATCH_COST_SHARE * (
            PROMPT_TOKENS * INPUT_COST_PER_TOKEN + COMPLETION_TOKENS * OUTPUT_COST_PER_TOKEN
        )
        for table, column, identity in (
            ("LiteLLM_VerificationToken", "token", sha256(key.encode()).hexdigest()),
            ("LiteLLM_TeamTable", "team_id", team_id),
        ):
            balance: Final = eventually(
                partial(read_rows, f'SELECT spend FROM "{table}" WHERE {column} = %s', (identity,)),
                lambda rows: len(rows) == 1 and rows[0]["spend"] == pytest.approx(expected_spend),
                seconds=70,
            )
            assert balance == [{"spend": pytest.approx(expected_spend)}]


def _insert_scan_rows(owner: str, group: str, size: int, created_at: datetime) -> tuple[str, ...]:
    prefix: Final = f"{owner}-{group}-"
    write_rows(
        'INSERT INTO "LiteLLM_ManagedObjectTable" '
        "(id, unified_object_id, model_object_id, file_object, file_purpose, status, created_by, created_at, updated_at) "
        "SELECT %s || lpad(n::text, 8, '0'), %s || n, %s || n, '{}'::jsonb, 'batch', 'completed', %s, "
        "%s::timestamptz, NOW() FROM generate_series(0, %s::int - 1) n",
        (prefix, prefix, prefix, owner, created_at.isoformat(), str(size)),
    )
    return tuple(f"{prefix}{index:08}" for index in range(size))


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [2 * MAX_OBJECTS_PER_POLL_CYCLE, 2 * MAX_OBJECTS_PER_POLL_CYCLE + 1, 1000])
async def test_arrivals_progress_while_retry_scan_advances_and_wraps(count: int) -> None:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    owner: Final = f"batch-scan-{uuid4().hex}"

    try:
        old_ids: Final = _insert_scan_rows(owner, "old", count, datetime(2000, 1, 1, tzinfo=timezone.utc))
        async with connected_proxy_prisma() as prisma:
            logging: Final = ProxyLogging(UserApiKeyCache())
            router: Final = Router()
            poller: Final = CheckBatchCost(logging, prisma, router)
            where: Final = {"created_by": owner, "batch_processed": False}
            first: Final = await poller._find_jobs(where)
            assert tuple(row.id for row in first) == old_ids[:MAX_OBJECTS_PER_POLL_CYCLE]

            recent_ids: Final = _insert_scan_rows(
                owner, "recent", MAX_OBJECTS_PER_POLL_CYCLE + 1, datetime.now(timezone.utc)
            )

            async def poll_with_arrival(turn: int) -> tuple[str, ...]:
                if turn % 4 == 0:
                    _insert_scan_rows(owner, f"arrival-{turn}", 1, datetime.now(timezone.utc))
                page: Final = await poller._find_jobs(where)
                assert len(page) <= MAX_OBJECTS_PER_POLL_CYCLE
                write_rows(
                    'UPDATE "LiteLLM_ManagedObjectTable" SET batch_processed = true '
                    "WHERE created_by = %s AND id = ANY(%s) AND id LIKE %s",
                    (owner, [row.id for row in page], f"{owner}-arrival-%"),
                )
                return tuple(row.id for row in page)

            pages: Final = tuple(
                [await poll_with_arrival(turn) for turn in range(4 * (count // MAX_OBJECTS_PER_POLL_CYCLE + 2))]
            )
            selected: Final = tuple(chain.from_iterable(pages))
            assert all(selected.count(identity) >= 2 for identity in recent_ids)
            selected_old: Final = tuple(identity for identity in selected if identity in old_ids)
            assert selected_old[: count - MAX_OBJECTS_PER_POLL_CYCLE] == old_ids[MAX_OBJECTS_PER_POLL_CYCLE:]
            assert set(old_ids).issubset(selected_old)

            latest_ids: Final = _insert_scan_rows(owner, "latest", 1, datetime.now(timezone.utc))
            followups: Final = tuple([await poller._find_jobs(where) for _ in range(12)])
            assert any(latest_ids[0] in tuple(row.id for row in page) for page in followups)

            restarted: Final = CheckBatchCost(logging, prisma, router)
            restart_page: Final = await restarted._find_jobs(where)
            assert tuple(row.id for row in restart_page) == old_ids[:MAX_OBJECTS_PER_POLL_CYCLE]
    finally:
        write_rows('DELETE FROM "LiteLLM_ManagedObjectTable" WHERE created_by = %s', (owner,))


@pytest.mark.asyncio
@pytest.mark.parametrize("steady_arrivals", [False, True])
@pytest.mark.parametrize("backlog", [0, 1000])
async def test_recent_batches_are_revisited_without_scanning_old_backlog(steady_arrivals: bool, backlog: int) -> None:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    owner: Final = f"batch-repeat-{uuid4().hex}"
    try:
        _insert_scan_rows(owner, "old", backlog, datetime(2000, 1, 1, tzinfo=timezone.utc))
        async with connected_proxy_prisma() as prisma:
            poller: Final = CheckBatchCost(ProxyLogging(UserApiKeyCache()), prisma, Router())
            where: Final = {"created_by": owner, "batch_processed": False}
            await poller._find_jobs(where)
            target: Final = _insert_scan_rows(owner, "target", 1, datetime.now(timezone.utc))[0]
            assert target in tuple(row.id for row in await poller._find_jobs(where))
            for cycle in range(3):
                if steady_arrivals:
                    _insert_scan_rows(owner, f"arrival-{cycle}", 1, datetime.now(timezone.utc))
                pages: Final = tuple([await poller._find_jobs(where) for _ in range(4)])
                assert any(target in tuple(row.id for row in page) for page in pages)
                assert all(len(page) <= MAX_OBJECTS_PER_POLL_CYCLE for page in pages)
            write_rows(
                'UPDATE "LiteLLM_ManagedObjectTable" SET batch_processed = true WHERE created_by = %s AND id != %s',
                (owner, target),
            )
            write_rows(
                'UPDATE "LiteLLM_ManagedObjectTable" SET created_at = %s WHERE id = %s',
                (datetime(2000, 1, 1, tzinfo=timezone.utc), target),
            )
            recovery: Final = tuple([await poller._find_jobs(where) for _ in range(4)])
            assert any(target in tuple(row.id for row in page) for page in recovery)

    finally:
        write_rows('DELETE FROM "LiteLLM_ManagedObjectTable" WHERE created_by = %s', (owner,))
