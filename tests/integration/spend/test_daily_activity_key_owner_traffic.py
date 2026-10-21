import json
import os
import threading
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, string_value
from integration._support.daily_activity import (
    AGGREGATED_USER_ACTIVITY,
    DAY,
    ROUTES,
    USER_SPEND,
    Route,
    activity_of_key,
    assert_key_owner_and_totals,
    assert_key_reported,
    daily_rows,
    key_metadata,
    key_no_key_table_holds,
    purge_key_from_the_key_tables,
    seeded_metrics,
    seeded_row,
    user_row,
    user_with_an_email,
)
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

REQUESTS_OF_KEY: Final = (
    'SELECT COALESCE(SUM(api_requests), 0)::int AS requests FROM "LiteLLM_DailyUserSpend" '
    "WHERE api_key=%s AND user_id=%s"
)
NAMED_SPEND_LOGS_OF_KEY: Final = (
    'SELECT COUNT(*)::int AS named FROM "LiteLLM_SpendLogs" '
    "WHERE api_key=%s AND NULLIF(metadata->>'user_api_key_alias', '') IS NOT NULL"
)
UNIFIED_ENDPOINTS: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
REQUESTS_OF_A_BURST: Final = 21
READS_DURING_A_BURST: Final = 30
TOKEN_LIMIT_DISCOVERY: Final = ("GET", "/v1/models")
TOOL_CALL: Final = "call_integration_usage"
ANSWER: Final = "One request cost $0.25"
SUMMARY_OF_ONE_SEEDED_ROW: Final = "\n".join(
    (
        "Total Spend: $0.2500",
        "Total Requests: 1",
        "Successful: 1 | Failed: 0",
        "Total Tokens: 15",
        "",
        "Top Models by Spend:",
        "  - gpt-4o-mini: $0.2500 (1 reqs, 15 tokens)",
        "",
        "Top Providers by Spend:",
        "  - openai: $0.2500 (1 reqs)",
    )
)


def _chat_completion() -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _response() -> dict[str, JsonValue]:
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _provider(request: Request) -> Reply:
    body: Final = _response() if request.target.endswith("/responses") else _chat_completion()
    return Reply(body=json.dumps(body).encode())


def _usage_tool_call() -> dict[str, JsonValue]:
    call: Final[dict[str, JsonValue]] = {
        "id": TOOL_CALL,
        "type": "function",
        "function": {
            "name": "get_usage_data",
            "arguments": json.dumps({"start_date": DAY, "end_date": DAY}),
        },
    }
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _streamed_chunk(delta: dict[str, JsonValue], finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": "chatcmpl-integration-usage",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _usage_analyst(request: Request) -> Reply:
    if json.loads(request.body).get("stream"):
        return Reply(
            chunks=(
                _streamed_chunk({"role": "assistant", "content": ANSWER}, None),
                _streamed_chunk({}, "stop"),
                b"data: [DONE]\n\n",
            ),
            content_type="text/event-stream",
        )
    return Reply(body=json.dumps(_usage_tool_call()).encode())


def _sent_for_callers(requests: Iterable[Request]) -> tuple[Request, ...]:
    return tuple(request for request in requests if (request.method, request.target) != TOKEN_LIMIT_DISCOVERY)


def _priced_model(scenario: Scenario, provider_url: str) -> str:
    return scenario.model(
        api_base=f"{provider_url}/v1", input_cost_per_token=0.001, output_cost_per_token=0.002, num_retries=0
    )


def _request_body(endpoint: str, model: str, prompt: str) -> dict[str, JsonValue]:
    if endpoint == "/v1/chat/completions":
        return {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if endpoint == "/v1/messages":
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]}
    return {"model": model, "input": prompt}


def _activity_on_route(gateway: Gateway, route: Route, api_key: str, entity: str) -> httpx.Response:
    filters: Final = {} if route.entity_filter is None else {route.entity_filter: entity}
    return activity_of_key(gateway, route.path, api_key, **filters)


def _prompt() -> str:
    return f"daily activity owner {uuid.uuid4().hex}"


def _totals_of_requests(requests: int) -> dict[str, float]:
    return {
        "total_spend": 0.02 * requests,
        "total_prompt_tokens": 10 * requests,
        "total_completion_tokens": 5 * requests,
        "total_tokens": 15 * requests,
        "total_api_requests": requests,
        "total_successful_requests": requests,
        "total_failed_requests": 0,
    }


def _activity_around_today(gateway: Gateway, api_key: str) -> httpx.Response:
    today: Final = datetime.now(UTC).date()
    return gateway.request(
        "GET",
        AGGREGATED_USER_ACTIVITY,
        params={
            "start_date": str(today - timedelta(days=1)),
            "end_date": str(today + timedelta(days=1)),
            "timezone": "0",
            "api_key": api_key,
        },
    )


def _wait_for_requests(api_key: str, user: str, requests: int) -> None:
    eventually(
        lambda: read_rows(REQUESTS_OF_KEY, (api_key, user)),
        lambda rows: rows[0]["requests"] == requests,
        seconds=70,
    )


def _wait_for_named_spend_logs(api_key: str, requests: int) -> None:
    eventually(
        lambda: read_rows(NAMED_SPEND_LOGS_OF_KEY, (api_key,)),
        lambda rows: rows[0]["named"] == requests,
        seconds=70,
    )


def _cli_session_token(user: str, team: str) -> str:
    cli_user: Final = LiteLLM_UserTable(user_id=user, user_role="internal_user", teams=[team], models=[])
    return ExperimentalUIJWTToken.get_cli_jwt_auth_token(user_info=cli_user, team_id=team, team_alias="cli-team")


def test_key_used_on_every_unified_endpoint_is_reported_with_its_own_alias_and_user(gateway: Gateway) -> None:
    chat_prompt, messages_prompt, responses_prompt = _prompt(), _prompt(), _prompt()
    with wire_server(_provider) as wire, gateway.scenario() as scenario:
        model: Final = _priced_model(scenario, wire.url)
        owner, email = user_with_an_email(scenario)
        alias: Final = f"integration-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(user_id=owner, key_alias=alias, models=[model])
        stored: Final = sha256(key.encode()).hexdigest()
        prompts: Final = (chat_prompt, messages_prompt, responses_prompt)
        answers: Final = tuple(
            gateway.request("POST", endpoint, _request_body(endpoint, model, prompt), key=key)
            for endpoint, prompt in zip(UNIFIED_ENDPOINTS, prompts, strict=True)
        )
        assert [answer.status_code for answer in answers] == [200, 200, 200], [answer.text for answer in answers]
        received: Final = _sent_for_callers(wire.drain())
        assert [request.target for request in received] == ["/v1/chat/completions", "/v1/responses", "/v1/responses"]
        assert [json.loads(request.body)["model"] for request in received] == ["gpt-4o-mini"] * 3
        assert json.loads(received[0].body)["messages"] == [{"role": "user", "content": chat_prompt}]
        assert messages_prompt in received[1].body.decode()
        assert json.loads(received[2].body)["input"] == responses_prompt
        _wait_for_requests(stored, owner, 3)
        assert_key_owner_and_totals(
            _activity_around_today(gateway, stored),
            stored,
            key_metadata(alias=alias, user=owner, email=email, exists=True),
            _totals_of_requests(3),
        )


def test_key_purged_from_the_key_tables_is_reported_with_the_alias_its_spend_logs_name(gateway: Gateway) -> None:
    prompts: Final = (_prompt(), _prompt(), _prompt())
    with wire_server(_provider) as wire, gateway.scenario() as scenario:
        model: Final = _priced_model(scenario, wire.url)
        owner, email = user_with_an_email(scenario)
        alias: Final = f"integration-alias-{uuid.uuid4().hex}"
        generated: Final = gateway.post("/key/generate", {"user_id": owner, "key_alias": alias, "models": [model]})
        key: Final = string_value(generated["key"])
        stored: Final = sha256(key.encode()).hexdigest()
        try:
            answers: Final = tuple(
                gateway.request("POST", endpoint, _request_body(endpoint, model, prompt), key=key)
                for endpoint, prompt in zip(UNIFIED_ENDPOINTS, prompts, strict=True)
            )
            assert [answer.status_code for answer in answers] == [200, 200, 200], [answer.text for answer in answers]
            received: Final = _sent_for_callers(wire.drain())
            assert [request.target for request in received] == [
                "/v1/chat/completions",
                "/v1/responses",
                "/v1/responses",
            ]
            _wait_for_requests(stored, owner, 3)
            _wait_for_named_spend_logs(stored, 3)
        finally:
            purge_key_from_the_key_tables(stored)
        assert_key_owner_and_totals(
            _activity_around_today(gateway, stored),
            stored,
            key_metadata(alias=alias, user=owner, email=email, exists=False),
            _totals_of_requests(3),
        )


def test_cli_session_spend_is_reported_with_the_user_and_team_of_the_session(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"))
    prompt: Final = _prompt()
    with wire_server(_provider) as wire, gateway.scenario() as scenario:
        model: Final = _priced_model(scenario, wire.url)
        owner, email = user_with_an_email(scenario)
        team: Final = scenario.team(models=[model], members_with_roles=[{"role": "user", "user_id": owner}])
        answer: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
            key=_cli_session_token(owner, team),
        )
        assert answer.status_code == 200, answer.text
        received: Final = _sent_for_callers(wire.drain())
        assert [request.target for request in received] == ["/v1/chat/completions"]
        assert json.loads(received[0].body) == {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        stored: Final = f"cli-session-{owner}"
        _wait_for_requests(stored, owner, 1)
        assert_key_owner_and_totals(
            _activity_around_today(gateway, stored),
            stored,
            key_metadata(alias=stored, team=team, user=owner, email=email),
            _totals_of_requests(1),
        )


@pytest.mark.timeout(300)
def test_usage_ai_chat_hands_the_model_the_usage_summary_without_any_key_owner(
    gateway: Gateway, tmp_path: Path
) -> None:
    question: Final = f"what did we spend {uuid.uuid4().hex}"
    owner: Final = f"integration-{uuid.uuid4().hex}"
    ownerless_key: Final = f"integration-ownerless-{uuid.uuid4().hex}"
    with (
        scratch_database() as scratch_url,
        wire_server(_usage_analyst) as wire,
        owned_proxy(
            gateway,
            tmp_path,
            {
                "DATABASE_URL": scratch_url,
                "OPENAI_API_BASE": f"{wire.url}/v1",
                "OPENAI_BASE_URL": f"{wire.url}/v1",
                "OPENAI_API_KEY": "integration-provider-key",
            },
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as candidate,
    ):
        candidate.post("/user/new", {"user_id": owner, "user_email": f"{owner}@example.com", "auto_create_key": False})
        with daily_rows((user_row(owner, ownerless_key, DAY),), database_url=scratch_url):
            answer: Final = candidate.request(
                "POST",
                "/usage/ai/chat",
                {"messages": [{"role": "user", "content": question}], "model": "openai/gpt-4o-mini"},
            )
        assert answer.status_code == 200, answer.text
        tool_call: Final = {
            "type": "tool_call",
            "tool_name": "get_usage_data",
            "tool_label": "global usage data",
            "arguments": {"start_date": DAY, "end_date": DAY},
        }
        events: Final = [
            json.loads(line.removeprefix("data: ")) for line in answer.text.splitlines() if line.startswith("data: ")
        ]
        assert events == [
            {"type": "status", "message": "Thinking..."},
            {**tool_call, "status": "running"},
            {**tool_call, "status": "complete"},
            {"type": "status", "message": "Analyzing results..."},
            {"type": "chunk", "content": ANSWER},
            {"type": "done"},
        ], answer.text
        asked, analysed = wire.drain()
        assert [asked.target, analysed.target] == ["/v1/chat/completions", "/v1/chat/completions"]
        assert json.loads(asked.body)["messages"][-1] == {"role": "user", "content": question}
        assert json.loads(analysed.body)["messages"][-1] == {
            "role": "tool",
            "tool_call_id": TOOL_CALL,
            "content": SUMMARY_OF_ONE_SEEDED_ROW,
        }
        assert owner not in analysed.body.decode()
        assert ownerless_key not in analysed.body.decode()


@pytest.mark.timeout(300)
def test_owner_is_reported_on_every_route_while_a_burst_of_requests_waits_on_the_provider(gateway: Gateway) -> None:
    released: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()

    def held_provider(request: Request) -> Reply:
        if (request.method, request.target) == TOKEN_LIMIT_DISCOVERY:
            return _provider(request)
        held.put(request.target)
        assert released.wait(timeout=120), "The burst was never released"
        return _provider(request)

    api_key: Final = key_no_key_table_holds()
    entity: Final = f"integration-entity-{uuid.uuid4().hex}"
    prompts: Final = tuple(_prompt() for _ in range(REQUESTS_OF_A_BURST))
    entity_columns: Final = {route.table: route.entity_column for route in ROUTES if route.table != USER_SPEND}
    with (
        wire_server(held_provider) as wire,
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.client.base_url, timeout=180, trust_env=False) as patient,
        ThreadPoolExecutor(max_workers=REQUESTS_OF_A_BURST) as traffic,
        ThreadPoolExecutor(max_workers=READS_DURING_A_BURST) as readers,
    ):
        model: Final = _priced_model(scenario, wire.url)
        owner, email = user_with_an_email(scenario)
        key: Final = scenario.key(models=[model])
        rows: Final = (
            user_row(owner, api_key, DAY),
            *(seeded_row(table, column, entity, api_key, DAY) for table, column in entity_columns.items()),
        )
        try:
            with daily_rows(rows):
                burst: Final = tuple(
                    traffic.submit(
                        patient.post,
                        UNIFIED_ENDPOINTS[index % len(UNIFIED_ENDPOINTS)],
                        json=_request_body(UNIFIED_ENDPOINTS[index % len(UNIFIED_ENDPOINTS)], model, prompt),
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    for index, prompt in enumerate(prompts)
                )
                eventually(held.qsize, lambda waiting: waiting >= REQUESTS_OF_A_BURST, seconds=60)
                reads: Final = tuple(
                    readers.submit(_activity_on_route, gateway, ROUTES[index % len(ROUTES)], api_key, entity)
                    for index in range(READS_DURING_A_BURST)
                )
                activity: Final = tuple(read.result() for read in reads)
                still_waiting: Final = [call.done() for call in burst]
        finally:
            released.set()
        answers: Final = tuple(call.result() for call in burst)
        received: Final = tuple(request.body.decode() for request in _sent_for_callers(wire.drain()))
    assert still_waiting == [False] * REQUESTS_OF_A_BURST
    assert [answer.status_code for answer in answers] == [200] * REQUESTS_OF_A_BURST, [
        answer.text for answer in answers
    ]
    assert [sum(prompt in body for body in received) for prompt in prompts] == [1] * REQUESTS_OF_A_BURST
    assert len(received) == REQUESTS_OF_A_BURST, len(received)
    for response in activity:
        assert_key_reported(response, api_key, DAY, key_metadata(user=owner, email=email), seeded_metrics(1))
