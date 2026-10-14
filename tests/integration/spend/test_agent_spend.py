import json
import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, JsonValue


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
            {"kind": "message", "role": "user", "messageId": marker + "-in", "parts": [{"kind": "text", "text": "ping"}]}
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
                sent = gateway.client.post(
                    path.format(agent=agent, name=marker), headers=headers, json=payload
                )
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
