"""Contract: the rollups that ride on the spend writer today, ahead of the Spend Logs V2 port.

The V2 plan ("DB write and batching path") ports the row and delivery writer to Rust and leaves "piggy-backed
rollups" out of scope: they must move into the ledger projection before PR 4 removes the Python writer, or
"those tables stop filling". Rounds 1 and 2 pin the request row, the delivery and the daily spend tables. This file
pins the rollups no earlier round reads, so the move can be checked against what the Python writer does today:

1. `LiteLLM_DailyRequestErrors`: one counter per (UTC day, hashed key, team, user, model group, HTTP status) for
   every failure the spend writer sees, recorded on the ledger path even when the failure row itself is dropped
   (`update_database`, litellm/proxy/db/db_spend_update_writer.py:958; `RequestErrorAccumulator.record`,
   litellm/proxy/db/request_error_tracking.py:96).
2. `LiteLLM_SpendLogToolIndex` and `LiteLLM_DailyToolSpend`: one index row per (request, invoked tool) and one
   day row per tool that adds the request's whole spend and tokens, only for requests whose spend log is kept
   (`build_tool_usage_transaction` and `flush_tool_usage_transactions`, litellm/proxy/db/spend_log_tool_index.py:150
   and :202; gate at db_spend_update_writer.py:921).
3. `LiteLLM_DailyModelUsage` task types: the `task:<name>` request tag picks the `task_type` key column when the
   name is a known task, else `uncategorized` (`_model_task`, litellm/accounting/projection.py:345), and the
   success, failure and timing counters (`_model_contributions`, projection.py:263).

Every table is keyed by something unique to the test (the hashed key, a fresh tool name, a fresh model group),
so counts are exact even when other tests write the same tables at the same time.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.mcp import call_tool, mcp_peer, register_mcp, tool_names
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    ask,
    chat_model,
    failing,
    hashed,
    number,
    row_by_id,
    rows_of_key,
    scripted_upstream,
    served,
    served_id,
)
from tests.integration.spend.dec1_spendlogs2_support import error_information, failure_rows
from tests.integration.spend.dec1_spendlogs3_support import (
    TOOL_DAY_QUERY,
    TOOL_INDEX_QUERY,
    counters,
    fresh_tool_name,
    model_usage_reaches,
    model_usage_rows,
    request_error_rows,
    request_errors_reach,
    stays_empty,
    stored_json,
    tool_call_upstream,
    tool_day_row,
    tool_index_rows,
    utc_today,
)
from tests.integration.spend.dec1_spendlogs_support import owned_spend_proxy

pytestmark: Final = pytest.mark.shares_worker_state


def _status_by_prompt(request: Request) -> Reply:
    """A provider that fails with 400 when the prompt contains `status-400` and with 500 otherwise."""
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    status: Final = 400 if b"status-400" in request.body else 500
    return failing(status)(request)


@contextmanager
def _status_upstream() -> Iterator[Wire]:
    with wire_server(_status_by_prompt) as wire:
        yield wire


@contextmanager
def _failing_upstream() -> Iterator[Wire]:
    with wire_server(failing(500)) as wire:
        yield wire


# LiteLLM_DailyRequestErrors


def test_a_provider_failure_counts_one_request_error_under_the_key_and_status(gateway: Gateway) -> None:
    """A key without team or owner calls a deployment whose provider answers 500 (no retries). One
    `LiteLLM_DailyRequestErrors` row appears with today's UTC date, the key's sha256 hash, `team_id` '' and
    `user_id` '' (the accumulator stores missing ids as empty strings), the model group, `status_code` 500 and
    `failed_requests` 1 (request_error_tracking.py:96)."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        row: Final = request_error_rows(gateway, key, 1)[0]
        assert (row["date"], row["api_key"], row["team_id"], row["user_id"]) == (utc_today(), hashed(key), "", ""), row
        assert (row["model_group"], row["status_code"], row["failed_requests"]) == (model, 500, 1), row


def test_a_request_error_row_names_the_team_and_owner_of_the_key(gateway: Gateway) -> None:
    """The failing key belongs to a team and is owned by a user. Its request error row carries that `team_id`
    and that `user_id` (`payload["team_id"]` and `payload["user"]` of the spend row,
    request_error_tracking.py:101)."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team()
        owner: Final = scenario.user()
        key: Final = scenario.key(models=[model], team_id=team, user_id=owner)
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        row: Final = request_error_rows(gateway, key, 1)[0]
        assert (row["team_id"], row["user_id"]) == (team, owner), row


def test_two_failures_with_one_status_fold_into_one_counter(gateway: Gateway) -> None:
    """Two failed calls of the same key and model group with the same status 500 leave a single row with
    `failed_requests` 2: the counter key is (date, key, team, user, group, status), so repeats fold
    (`fold_counts`, request_error_tracking.py:119, and the upsert's `ON CONFLICT DO UPDATE`)."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        for _ in range(2):
            assert ask(gateway, model, key, num_retries=0).status_code == 500
        rows: Final = request_errors_reach(gateway, key, 2)
        assert [(row["status_code"], row["failed_requests"]) for row in rows] == [(500, 2)], rows


def test_failures_with_different_statuses_get_separate_counters(gateway: Gateway) -> None:
    """One call fails with 400 and one with 500 on the same key and model group. Two rows exist, ordered by
    status: (400, 1) and (500, 1), because `status_code` is part of the primary key of the table
    (schema.prisma `@@id([date, api_key, team_id, user_id, model_group, status_code])`)."""
    with _status_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(gateway, model, key, text=f"status-400 {uuid.uuid4().hex}", num_retries=0).status_code == 400
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        rows: Final = request_error_rows(gateway, key, 2)
        assert [(row["status_code"], row["failed_requests"]) for row in rows] == [(400, 1), (500, 1)], rows


def test_the_request_error_status_is_the_failure_rows_error_code(gateway: Gateway) -> None:
    """The status counted is the integer form of the failure row's `metadata.error_information.error_code`
    (`status_code_from_metadata`, request_error_tracking.py:62): for a provider 500 both read 500."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        failure: Final = failure_rows(gateway, key, 1)[0]
        counted: Final = request_error_rows(gateway, key, 1)[0]
        assert int(str(error_information(failure)["error_code"])) == counted["status_code"] == 500, (failure, counted)


def test_a_successful_call_writes_no_request_error_row(gateway: Gateway) -> None:
    """A served chat call leaves its spend row but never a `LiteLLM_DailyRequestErrors` row: `record` returns
    early for `success` (request_error_tracking.py:97), so the table only grows with failing callers."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        assert rows_of_key(gateway, key, 1)[0]["status"] == "success"
        stays_empty(gateway, 'SELECT api_key FROM "LiteLLM_DailyRequestErrors" WHERE api_key = %s', (hashed(key),))


def test_an_unknown_virtual_key_counts_a_401_request_error(gateway: Gateway) -> None:
    """A request with an `sk-` key that does not exist is refused with 401 and its failure row is hashed under
    that key (round 2). The same hash gets a request error row with `status_code` 401 and `failed_requests` 1.
    UNVERIFIED: the auth failure row's `error_information.error_code` is "401" (the row path is
    `async_post_call_failure_hook`, then `update_database`, db_spend_update_writer.py:958)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        unknown: Final = f"sk-dec1-sl3-unknown-{uuid.uuid4().hex}"
        assert ask(gateway, model, unknown).status_code == 401
        row: Final = request_error_rows(gateway, unknown, 1)[0]
        assert (row["status_code"], row["failed_requests"]) == (401, 1), row


def test_a_budget_refusal_counts_a_422_request_error(gateway: Gateway) -> None:
    """A key with `max_budget` 0 is refused with the default 422 budget error before the provider. Its request
    error row has `status_code` 422 and `failed_requests` 1. UNVERIFIED: `model_group` of the refusal row is the
    requested group, so it is not asserted here."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0)
        assert ask(gateway, model, key).status_code == 422
        row: Final = request_error_rows(gateway, key, 1)[0]
        assert (row["status_code"], row["failed_requests"]) == (422, 1), row


def test_disable_error_logs_still_counts_request_errors(gateway: Gateway, tmp_path: Path) -> None:
    """With `general_settings.disable_error_logs: true` a provider failure writes no failure row
    (`keep_spend_logs` is false for failures), but `_record_request_error_safely` runs on the ledger path
    regardless (db_spend_update_writer.py:958), so the key still gets a request error row with status 500."""
    with owned_spend_proxy(
        gateway, tmp_path, general_settings={"disable_error_logs": True}, upstream=_failing_upstream
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(owned, model, key, num_retries=0).status_code == 500
        row: Final = request_error_rows(owned, key, 1)[0]
        assert (row["status_code"], row["failed_requests"]) == (500, 1), row
        assert owned.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)) == []


def test_disable_spend_logs_still_counts_request_errors(gateway: Gateway, tmp_path: Path) -> None:
    """With `general_settings.disable_spend_logs: true` a provider failure leaves no request row but still a
    request error row with status 500 and `failed_requests` 1: the error rollup is an aggregate, like the daily
    tables, and does not depend on the row being kept (db_spend_update_writer.py:900-958)."""
    with owned_spend_proxy(
        gateway, tmp_path, general_settings={"disable_spend_logs": True}, upstream=_failing_upstream
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(owned, model, key, num_retries=0).status_code == 500
        row: Final = request_error_rows(owned, key, 1)[0]
        assert (row["status_code"], row["failed_requests"]) == (500, 1), row


# LiteLLM_SpendLogToolIndex and LiteLLM_DailyToolSpend


def test_a_tool_the_answer_calls_is_indexed_under_the_request_id(gateway: Gateway) -> None:
    """The provider answers with one tool call of a fresh tool name. `LiteLLM_SpendLogToolIndex` gets exactly one
    row with the response id as `request_id` and that `tool_name` (`response_tool_call_names`,
    spend_log_tool_index.py:46)."""
    tool: Final = fresh_tool_name()
    with tool_call_upstream(tool) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(gateway, model, key))
        rows: Final = tool_index_rows(gateway, request_id, 1)
        assert [(row["request_id"], row["tool_name"]) for row in rows] == [(request_id, tool)], rows


def test_the_tool_index_start_time_is_the_spend_rows_start_time(gateway: Gateway) -> None:
    """The index row's `start_time` is the request's `startTime` (both built from the spend payload's
    `startTime`, spend_log_tool_index.py:170), so a drill-down from a tool lands on the right row."""
    tool: Final = fresh_tool_name()
    with tool_call_upstream(tool) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(gateway, model, key))
        indexed: Final = tool_index_rows(gateway, request_id, 1)[0]
        stored: Final = gateway.rows(
            'SELECT "startTime"::text AS start_time FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)
        )
        assert stored and stored[0]["start_time"] == indexed["start_time"], (stored, indexed)


def test_each_called_tool_gets_its_own_day_row_with_the_whole_request_spend(gateway: Gateway) -> None:
    """The answer calls two fresh tools. The request is indexed under both, and each tool's
    `LiteLLM_DailyToolSpend` row for today has `spend` 0.02, `total_tokens` 15 and `request_count` 1: the whole
    request is attributed to every tool it called, not split (spend_log_tool_index.py:224-249)."""
    first: Final = fresh_tool_name()
    second: Final = fresh_tool_name()
    with tool_call_upstream(first, second) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(gateway, model, key))
        assert sorted(string_value(row["tool_name"]) for row in tool_index_rows(gateway, request_id, 2)) == sorted(
            (first, second)
        )
        for tool in (first, second):
            day = tool_day_row(gateway, tool)
            assert (day["date"], number(day["spend"])) == (utc_today(), pytest.approx(CHAT_COST)), day
            assert counters(day, "total_tokens", "request_count") == (15, 1), day


def test_a_tool_called_twice_in_one_answer_counts_once(gateway: Gateway) -> None:
    """The answer calls the same tool twice. The names are de-duplicated (`dict.fromkeys`,
    spend_log_tool_index.py:162): one index row, and the day row has `request_count` 1 and `spend` 0.02."""
    tool: Final = fresh_tool_name()
    with tool_call_upstream(tool, tool) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(gateway, model, key))
        assert len(tool_index_rows(gateway, request_id, 1)) == 1
        day: Final = tool_day_row(gateway, tool)
        assert (counters(day, "request_count"), number(day["spend"])) == ((1,), pytest.approx(CHAT_COST)), day


def test_two_requests_calling_one_tool_add_up_on_its_day_row(gateway: Gateway) -> None:
    """Two requests whose answers call the same fresh tool leave one day row with `request_count` 2,
    `total_tokens` 30 and `spend` 0.04 (upsert increments, spend_log_tool_index.py:251)."""
    tool: Final = fresh_tool_name()
    with tool_call_upstream(tool) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        for _ in range(2):
            served(ask(gateway, model, key))
        day: Final = tool_day_row(gateway, tool, requests=2)
        assert counters(day, "request_count", "total_tokens") == (2, 30), day
        assert number(day["spend"]) == pytest.approx(2 * CHAT_COST), day


def test_a_tool_declared_but_not_called_is_not_indexed(gateway: Gateway) -> None:
    """The request declares a fresh tool in `tools` but the provider answers with plain text. The request row
    lands and no index row exists for it: declared but not invoked tools are excluded (module docstring,
    spend_log_tool_index.py:1-6)."""
    tool: Final = fresh_tool_name()
    declared: Final[list[JsonValue]] = [
        {"type": "function", "function": {"name": tool, "parameters": {"type": "object", "properties": {}}}}
    ]
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        request_id: Final = served_id(ask(gateway, model, key, tools=declared))
        row_by_id(gateway, request_id)
        stays_empty(gateway, TOOL_INDEX_QUERY, (request_id,))
        assert gateway.rows(TOOL_DAY_QUERY, (tool,)) == []


def test_disable_spend_logs_skips_the_tool_rollup(gateway: Gateway, tmp_path: Path) -> None:
    """With `disable_spend_logs: true` a request whose answer calls a fresh tool leaves neither an index row nor a
    day row for that tool: the tool rollup is only enqueued when the spend log is kept
    (db_spend_update_writer.py:921), unlike the daily spend tables, which still fill. A port must keep the
    difference or decide it on purpose."""
    tool: Final = fresh_tool_name()
    with owned_spend_proxy(
        gateway, tmp_path, general_settings={"disable_spend_logs": True}, upstream=lambda: tool_call_upstream(tool)
    ) as (owned, scenario, wire):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(owned, model, key))
        eventually(
            lambda: owned.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),)),
            lambda found: bool(found) and number(found[0]["spend"]) > 0,
            seconds=LANDING_SECONDS,
        )
        stays_empty(owned, TOOL_DAY_QUERY, (tool,))
        assert owned.rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (hashed(key),)) == []


def test_an_mcp_tool_call_is_indexed_under_its_namespaced_name(gateway: Gateway) -> None:
    """A REST call of the MCP `add` tool writes a `call_mcp_tool` row; the tool index holds one row for that
    request whose `tool_name` is the namespaced name, and the tool's day row has `request_count` 1 and `spend`
    0 (unpriced tool). The MCP name comes from `mcp_namespaced_tool_name` (spend_log_tool_index.py:157)."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "sl3" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        full: Final = tool_names(gateway, key, identity)["add"]
        assert call_tool(gateway, key, identity, full, {"a": 2, "b": 3}).status_code == 200
        row: Final = eventually(
            lambda: gateway.rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
                (hashed(key), "call_mcp_tool"),
            ),
            lambda found: len(found) == 1,
            seconds=LANDING_SECONDS,
        )[0]
        indexed: Final = tool_index_rows(gateway, string_value(row["request_id"]), 1)
        assert [entry["tool_name"] for entry in indexed] == [full], indexed
        day: Final = tool_day_row(gateway, full)
        assert (counters(day, "request_count"), number(day["spend"])) == ((1,), 0.0), day


# LiteLLM_DailyModelUsage


def test_a_known_task_tag_files_model_usage_under_that_task_type(gateway: Gateway) -> None:
    """A call tagged `task:summarization` (a task in litellm/proxy/model_insights_tasks.json) leaves one
    `LiteLLM_DailyModelUsage` row for its model group with `task_type` `summarization` (`_model_task`,
    projection.py:345, `MODEL_INSIGHTS_TASK_TAG_PREFIX` "task:", litellm/constants.py:1934)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": ["task:summarization"]}))
        rows: Final = model_usage_rows(gateway, model, 1)
        assert rows[0]["task_type"] == "summarization", rows


def test_an_unknown_task_tag_falls_back_to_uncategorized(gateway: Gateway) -> None:
    """A call tagged `task:dec1-not-a-task` (not in the task list) is filed under `task_type` `uncategorized`
    (`MODEL_INSIGHTS_DEFAULT_TASK`, litellm/constants.py:1933): the column is bounded to the known task names."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": ["task:dec1-not-a-task"]}))
        rows: Final = model_usage_rows(gateway, model, 1)
        assert rows[0]["task_type"] == "uncategorized", rows


def test_two_task_types_on_one_model_group_get_separate_rows(gateway: Gateway) -> None:
    """One call tagged `task:translation` and one untagged on the same model group leave two rows keyed by
    `task_type`: `translation` and `uncategorized`, each with `request_count` 1 and `spend` 0.02 (`task_type` is
    part of the primary key, schema.prisma `@@id([date, model_group, model, custom_llm_provider, task_type])`)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key, metadata={"tags": ["task:translation"]}))
        served(ask(gateway, model, key))
        rows: Final = model_usage_rows(gateway, model, 2)
        assert [(row["task_type"], row["request_count"]) for row in rows] == [
            ("translation", 1),
            ("uncategorized", 1),
        ], rows
        assert all(number(row["spend"]) == pytest.approx(CHAT_COST) for row in rows), rows


def test_model_usage_is_keyed_by_the_rows_model_and_provider(gateway: Gateway) -> None:
    """The model usage row of a served call has the request row's `model` (`gpt-4o-mini`) and
    `custom_llm_provider` (`openai`) as key columns, and the token counters 10 and 5 (projection.py:276-281)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        served(ask(gateway, model, key))
        row: Final = model_usage_rows(gateway, model, 1)[0]
        assert (row["model"], row["custom_llm_provider"], row["date"]) == ("gpt-4o-mini", "openai", utc_today()), row
        assert counters(row, "prompt_tokens", "completion_tokens") == (10, 5), row


def test_a_served_call_is_timed_in_model_usage_with_the_rows_duration(gateway: Gateway) -> None:
    """A successful call counts `request_count` 1, `successful_requests` 1, `timed_requests` 1 and adds the
    request row's `request_duration_ms` to `total_response_time_ms` (projection.py:270-290)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key)))
        usage: Final = model_usage_reaches(gateway, model, 1)[0]
        assert counters(usage, "request_count", "successful_requests", "failed_requests", "timed_requests") == (
            1,
            1,
            0,
            1,
        ), usage
        assert usage["total_response_time_ms"] == row["request_duration_ms"], (usage, row)


def test_a_failed_call_counts_in_model_usage_without_timing(gateway: Gateway) -> None:
    """A provider failure counts `request_count` 1 and `failed_requests` 1 but `timed_requests` 0 and
    `total_response_time_ms` 0: only successful caller requests are timed (projection.py:270)."""
    with wire_server(failing(500)) as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        assert ask(gateway, model, key, num_retries=0).status_code == 500
        usage: Final = model_usage_reaches(gateway, model, 1)[0]
        assert counters(
            usage, "request_count", "successful_requests", "failed_requests", "timed_requests", "total_response_time_ms"
        ) == (1, 0, 1, 0, 0), usage


def test_task_tags_also_stay_on_the_request_row(gateway: Gateway) -> None:
    """The `task:` tag that picks the model usage `task_type` is a plain request tag: the spend row's
    `request_tags` still lists `task:code_review` (tags reach the row through the StandardLoggingPayload,
    spend_tracking_utils.py:795-799)."""
    with scripted_upstream() as wire, gateway.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model])
        row: Final = row_by_id(gateway, served_id(ask(gateway, model, key, metadata={"tags": ["task:code_review"]})))
        tags: Final = stored_json(row["request_tags"])
        assert isinstance(tags, list) and "task:code_review" in tags, row
