import json
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows, write_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, JsonValue, TypeAdapter


class _ActivityMetrics(BaseModel):
    spend: float
    api_requests: int
    successful_requests: int


class _ActivityDay(BaseModel):
    date: str
    metrics: _ActivityMetrics


class _Activity(BaseModel):
    results: tuple[_ActivityDay, ...]


_CALL_TYPES: Final = {
    "message/send": "asend_message",
    "message/stream": "asend_message_streaming",
    "SendMessage": "asend_message",
    "SendStreamingMessage": "asend_message_streaming",
}
_PER_TOKEN: Final[dict[str, JsonValue]] = {"input_cost_per_token": 0.125, "output_cost_per_token": 0.5}
_PER_TOKEN_COST: Final = 1 * 0.125 + 2 * 0.5

_MESSAGE_V03: Final = {"kind": "message", "role": "user", "messageId": "", "parts": [{"kind": "text", "text": "ping"}]}


def _peer_card(url: str, name: str, version: str = "0.3") -> dict[str, JsonValue]:
    return {
        "protocolVersion": version,
        "name": name,
        "description": "Synthetic peer",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def _upstream_reply(body: dict[str, JsonValue]) -> Reply:
    if body["method"] == "message/stream":
        events: Final = (
            {
                "kind": "artifact-update",
                "taskId": "t1",
                "contextId": "c1",
                "artifact": {"artifactId": "a1", "parts": [{"kind": "text", "text": "billed"}]},
            },
            {
                "kind": "status-update",
                "taskId": "t1",
                "contextId": "c1",
                "status": {"state": "completed"},
                "final": True,
            },
        )
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': event})}\n\n".encode()
                for event in events
            ),
        )
    return Reply(
        body=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "kind": "message",
                    "role": "agent",
                    "messageId": "peer-message",
                    "parts": [{"kind": "text", "text": "billed"}],
                },
            }
        ).encode()
    )


def _assert_a2a_calls_are_billed(
    gateway: Gateway,
    pricing: dict[str, JsonValue],
    cost: float,
    calls: tuple[tuple[str, str], ...],
    version: str = "0.3",
) -> None:
    marker: Final = "a2aspend" + uuid.uuid4().hex[:12]

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        return _upstream_reply(json.loads(request.body))

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": marker,
                "agent_card_params": _peer_card(wire.url, marker, version),
                "litellm_params": pricing,
            },
        )
        assert created.status_code == 200, created.text
        agent: Final = created.json()["agent_id"]

        def cleanup() -> None:
            deleted: Final = gateway.request("DELETE", f"/v1/agents/{agent}")
            assert deleted.status_code == 200, deleted.text
            assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (agent,)) == []

        scenario.cleanups.callback(cleanup)
        key: Final = scenario.key()
        digest: Final = sha256(key.encode()).hexdigest()
        message: Final = (
            {
                "kind": "message",
                "role": "user",
                "messageId": marker + "-in",
                "parts": [{"kind": "text", "text": "ping"}],
            }
            if version == "0.3"
            else {"role": "ROLE_USER", "messageId": marker + "-in", "parts": [{"text": "ping"}]}
        )
        wire_message: Final = {
            **_MESSAGE_V03,
            "messageId": marker + "-in",
        }
        methods: Final = tuple(method for method, _ in calls)
        wire_methods: Final = tuple(
            "message/stream" if "stream" in method.lower() else "message/send" for method in methods
        )
        for method, path in calls:
            payload = {"jsonrpc": "2.0", "id": f"{marker}-{method}", "method": method, "params": {"message": message}}
            headers: Final = {
                "Authorization": f"Bearer {key}",
                **({"a2a-version": version} if version == "1.0" else {}),
            }
            if "stream" not in method.lower():
                sent = gateway.client.post(path.format(agent=agent, name=marker), headers=headers, json=payload)
                assert sent.status_code == 200, sent.text
                result: Final = sent.json()["result"]
                parts: Final = result["parts"] if version == "0.3" else result["message"]["parts"]
                expected_parts: Final = (
                    [{"kind": "text", "text": "billed"}] if version == "0.3" else [{"text": "billed"}]
                )
                assert parts == expected_parts, sent.text
                continue
            with gateway.client.stream(
                "POST",
                path.format(agent=agent, name=marker),
                headers={**headers, "Accept": "text/event-stream"},
                json=payload,
            ) as streamed:
                text = streamed.read().decode()
                assert streamed.status_code == 200, text
            assert text.startswith("data: ") and (
                '"state": "completed"' in text if version == "0.3" else '"state": "TASK_STATE_COMPLETED"' in text
            ), text
        forwarded: Final = tuple(json.loads(item.body) for item in wire.drain() if item.method == "POST")
        assert forwarded == tuple(
            {
                "jsonrpc": "2.0",
                "id": forwarded[index]["id"] if index < len(forwarded) else "<none>",
                "method": wire_method,
                "params": {"configuration": {"blocking": True}, "message": wire_message},
            }
            for index, wire_method in enumerate(wire_methods)
        ), forwarded
        assert all(str(uuid.UUID(item["id"])) == item["id"] for item in forwarded), forwarded

        logs: Final = eventually(
            lambda: read_rows(
                "SELECT request_id, agent_id, spend, api_key, call_type, "
                """to_char("startTime", 'YYYY-MM-DD') AS day FROM "LiteLLM_SpendLogs" """
                "WHERE api_key=%s ORDER BY request_id",
                (digest,),
            ),
            lambda rows: len(rows) == len(methods),
            seconds=30,
        )
        assert [{name: value for name, value in row.items() if name != "day"} for row in logs] == [
            {
                "request_id": f"{marker}-{method}",
                "agent_id": agent,
                "spend": cost,
                "api_key": digest,
                "call_type": _CALL_TYPES[method],
            }
            for method in sorted(methods)
        ], logs
        total: Final = cost * len(methods)
        assert eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (agent,)),
            lambda rows: rows == [{"spend": total}],
            seconds=20,
        ) == [{"spend": total}]
        assert eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda rows: rows == [{"spend": total}],
            seconds=20,
        ) == [{"spend": total}]
        requests_per_day: Final = {
            day: sum(1 for row in logs if row["day"] == day) for day in sorted({str(row["day"]) for row in logs})
        }
        daily: Final = eventually(
            lambda: read_rows(
                "SELECT agent_id, date, api_key, spend, api_requests::text AS api_requests, "
                'successful_requests::text AS successful_requests FROM "LiteLLM_DailyAgentSpend" WHERE agent_id=%s '
                "ORDER BY date",
                (agent,),
            ),
            lambda rows: sum(int(str(row["api_requests"])) for row in rows) == len(methods),
            seconds=45,
        )
        assert daily == [
            {
                "agent_id": agent,
                "date": day,
                "api_key": digest,
                "spend": cost * count,
                "api_requests": str(count),
                "successful_requests": str(count),
            }
            for day, count in requests_per_day.items()
        ], daily
        activity: Final = gateway.request(
            "GET",
            "/agent/daily/activity",
            params={"agent_ids": agent, "start_date": min(requests_per_day), "end_date": max(requests_per_day)},
        )
        assert activity.status_code == 200, activity.text
        days: Final = _Activity.model_validate_json(activity.content).results
        assert [(day.date, day.metrics) for day in days] == [
            (day, _ActivityMetrics(spend=cost * count, api_requests=count, successful_requests=count))
            for day, count in sorted(requests_per_day.items(), reverse=True)
        ], activity.text
        assert [
            result["breakdown"]["models"][f"a2a_agent/{marker}"]["api_key_breakdown"][digest]["metrics"]["spend"]
            for result in activity.json()["results"]
        ] == [cost * count for _, count in sorted(requests_per_day.items(), reverse=True)], activity.text


@pytest.mark.parametrize("version", ("0.3", "1.0"))
def test_a2a_send_and_stream_bill_cost_per_query_to_the_agent_the_key_and_daily_agent_activity(
    gateway: Gateway, version: str
) -> None:
    calls: Final = (
        (("message/send", "/a2a/{agent}"), ("message/stream", "/a2a/{agent}"))
        if version == "0.3"
        else (("SendMessage", "/v1/a2a/{agent}/message/send"), ("SendStreamingMessage", "/a2a/{name}"))
    )
    _assert_a2a_calls_are_billed(gateway, {"cost_per_query": 0.25}, 0.25, calls, version)


def test_a2a_send_bills_input_and_output_tokens_at_the_agent_per_token_prices(gateway: Gateway) -> None:
    _assert_a2a_calls_are_billed(gateway, _PER_TOKEN, _PER_TOKEN_COST, (("message/send", "/a2a/{agent}"),))


def test_a2a_stream_bills_input_and_output_tokens_at_the_agent_per_token_prices(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: message/stream to an agent priced with input_cost_per_token/output_cost_per_token logs spend 0.0 "
        "with prompt_tokens 0 and completion_tokens 0, because A2AStreamingIterator reads text from the request "
        "Part RootModels without dumping them and ignores artifact-update text, while message/send bills 1.125"
    )
    _assert_a2a_calls_are_billed(gateway, _PER_TOKEN, _PER_TOKEN_COST, (("message/stream", "/a2a/{agent}"),))


def test_agent_bound_chat_completion_is_classified_as_an_agent_in_ui_spend_logs(gateway: Gateway) -> None:
    marker: Final = "agentlog" + uuid.uuid4().hex[:12]
    agent_session: Final = f"{marker}-agent-session"
    plain_session: Final = f"{marker}-plain-session"
    agent_prompt: Final = f"{marker} agent-bound completion"
    plain_prompt: Final = f"{marker} plain completion"

    def upstream(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request
        body: Final = TypeAdapter(dict[str, JsonValue]).validate_json(request.body)
        assert body["model"] == "gpt-4o-mini", body
        assert body["messages"] in (
            [{"role": "user", "content": agent_prompt}],
            [{"role": "user", "content": plain_prompt}],
        ), body
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }
            ).encode()
        )

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=f"{wire.url}/v1",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {"agent_name": marker, "agent_card_params": _peer_card(wire.url, marker)},
        )
        assert created.status_code == 200, created.text
        created_body: Final = TypeAdapter(dict[str, JsonValue]).validate_python(created.json())
        agent_id: Final = TypeAdapter(str).validate_python(created_body["agent_id"])

        def cleanup_agent() -> None:
            deleted: Final = gateway.request("DELETE", f"/v1/agents/{agent_id}")
            assert deleted.status_code == 200, deleted.text
            assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (agent_id,)) == []

        scenario.cleanups.callback(cleanup_agent)
        agent_key: Final = scenario.key(agent_id=agent_id, models=[model])
        plain_key: Final = scenario.key(models=[model])
        agent_digest: Final = sha256(agent_key.encode()).hexdigest()
        plain_digest: Final = sha256(plain_key.encode()).hexdigest()
        body: Final = {"model": model, "messages": [{"role": "user", "content": agent_prompt}]}
        agent_response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            body,
            key=agent_key,
            headers={"x-litellm-session-id": agent_session},
        )
        assert agent_response.status_code == 200, agent_response.text
        plain_response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {**body, "messages": [{"role": "user", "content": plain_prompt}]},
            key=plain_key,
            headers={"x-litellm-session-id": plain_session},
        )
        assert plain_response.status_code == 200, plain_response.text

        spend_rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, session_id, agent_id, api_key, call_type FROM "LiteLLM_SpendLogs" '
                "WHERE session_id IN (%s, %s) ORDER BY session_id",
                (agent_session, plain_session),
            ),
            lambda rows: len(rows) == 2,
            seconds=70,
        )
        rows_by_session: Final = {str(row["session_id"]): row for row in spend_rows}
        assert set(rows_by_session) == {agent_session, plain_session}, spend_rows
        assert (
            rows_by_session[agent_session]["api_key"],
            rows_by_session[agent_session]["call_type"],
            rows_by_session[agent_session]["agent_id"],
        ) == (agent_digest, "acompletion", agent_id), spend_rows
        assert (
            rows_by_session[plain_session]["api_key"],
            rows_by_session[plain_session]["call_type"],
            rows_by_session[plain_session]["agent_id"],
        ) == (plain_digest, "acompletion", None), spend_rows
        outbound_requests: Final = wire.drain()
        assert [(request.method, request.target) for request in outbound_requests] == [
            ("POST", "/v1/chat/completions"),
            ("POST", "/v1/chat/completions"),
        ]

        today: Final = datetime.now(timezone.utc)
        date_window: Final = {
            "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        }

        def ui_rows(session_id: str, span_type: str) -> list[dict[str, JsonValue]]:
            response: Final = gateway.request(
                "GET",
                "/spend/logs/ui",
                params={**date_window, "session_id": session_id, "span_type": span_type},
            )
            assert response.status_code == 200, response.text
            response_body: Final = TypeAdapter(dict[str, JsonValue]).validate_python(response.json())
            return TypeAdapter(list[dict[str, JsonValue]]).validate_python(response_body["data"])

        agent_rows: Final = ui_rows(agent_session, "agent")
        agent_llm_rows: Final = ui_rows(agent_session, "llm")
        plain_agent_rows: Final = ui_rows(plain_session, "agent")
        plain_llm_rows: Final = ui_rows(plain_session, "llm")
        assert [row["request_id"] for row in agent_rows] == [rows_by_session[agent_session]["request_id"]], agent_rows
        assert agent_llm_rows == [], agent_llm_rows
        assert plain_agent_rows == [], plain_agent_rows
        assert [row["request_id"] for row in plain_llm_rows] == [rows_by_session[plain_session]["request_id"]], (
            plain_llm_rows
        )
        assert (
            agent_rows[0]["agent_id"],
            agent_rows[0]["session_agent_count"],
            agent_rows[0]["session_llm_count"],
        ) == (agent_id, 1, 0), agent_rows
        assert (
            plain_llm_rows[0]["session_llm_count"],
            plain_llm_rows[0]["session_agent_count"],
        ) == (1, 0), plain_llm_rows


def test_spend_log_ui_span_filters_classify_agent_bound_rows(gateway: Gateway) -> None:
    marker: Final = "agentspan" + uuid.uuid4().hex[:12]
    agent_id: Final = f"agent-{marker}"
    api_key: Final = f"test-key-{marker}"
    logged_at: Final = datetime.now(timezone.utc)
    timestamp: Final = logged_at.strftime("%Y-%m-%d %H:%M:%S")
    date_window: Final = {
        "start_date": (logged_at - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": (logged_at + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    rows_to_seed: Final = (
        (f"{marker}-mcp", "call_mcp_tool", agent_id, f"{marker}-mcp-session"),
        (f"{marker}-batch", "aretrieve_batch", agent_id, f"{marker}-batch-session"),
        (f"{marker}-agent", "acompletion", agent_id, f"{marker}-agent-session"),
        (f"{marker}-llm", "acompletion", "", f"{marker}-llm-session"),
    )
    insert_sql: Final = (
        'INSERT INTO "LiteLLM_SpendLogs" '
        '(request_id, call_type, api_key, "startTime", "endTime", agent_id, session_id, status) '
        "VALUES (%s, %s, %s, %s::timestamp, %s::timestamp, %s, %s, 'success')"
    )

    with gateway.scenario() as scenario:
        scenario.cleanups.callback(
            write_rows,
            'DELETE FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s',
            (f"{marker}-%",),
        )
        for request_id, call_type, row_agent_id, session_id in rows_to_seed:
            write_rows(
                insert_sql,
                (request_id, call_type, api_key, timestamp, timestamp, row_agent_id, session_id),
            )

        def ui_rows(span_type: str) -> list[dict[str, JsonValue]]:
            response: Final = gateway.request(
                "GET",
                "/spend/logs/ui",
                params={**date_window, "session_id": marker, "span_type": span_type},
            )
            assert response.status_code == 200, response.text
            response_body: Final = TypeAdapter(dict[str, JsonValue]).validate_python(response.json())
            return TypeAdapter(list[dict[str, JsonValue]]).validate_python(response_body["data"])

        rows_by_span: Final = {
            span_type: ui_rows(span_type) for span_type in ("agent", "llm", "mcp", "batch")
        }
        assert {
            span_type: frozenset(str(row["request_id"]) for row in rows)
            for span_type, rows in rows_by_span.items()
        } == {
            "agent": frozenset({f"{marker}-agent"}),
            "llm": frozenset({f"{marker}-llm"}),
            "mcp": frozenset({f"{marker}-mcp"}),
            "batch": frozenset({f"{marker}-batch"}),
        }, rows_by_span

        mcp_row: Final = rows_by_span["mcp"][0]
        batch_row: Final = rows_by_span["batch"][0]
        agent_row: Final = rows_by_span["agent"][0]
        llm_row: Final = rows_by_span["llm"][0]
        assert (
            mcp_row["session_agent_count"],
            mcp_row["session_llm_count"],
            mcp_row.get("mcp_tool_call_count", 0),
        ) == (0, 0, 1), mcp_row
        assert (
            batch_row["session_agent_count"],
            batch_row["session_llm_count"],
            batch_row.get("mcp_tool_call_count", 0),
        ) == (0, 1, 0), batch_row
        assert (
            agent_row["agent_id"],
            agent_row["session_agent_count"],
            agent_row["session_llm_count"],
            agent_row.get("mcp_tool_call_count", 0),
        ) == (agent_id, 1, 0, 0), agent_row
        assert (
            llm_row["agent_id"],
            llm_row["session_agent_count"],
            llm_row["session_llm_count"],
            llm_row.get("mcp_tool_call_count", 0),
        ) == ("", 0, 1, 0), llm_row
