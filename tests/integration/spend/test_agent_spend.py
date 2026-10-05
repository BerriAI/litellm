import json
import uuid
from hashlib import sha256
from typing import Final

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


def _peer_card(url: str, name: str) -> dict[str, JsonValue]:
    return {
        "protocolVersion": "0.3",
        "name": name,
        "description": "Synthetic peer",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def test_a2a_send_and_stream_bill_cost_per_query_to_the_agent_the_key_and_daily_agent_activity(
    gateway: Gateway,
) -> None:
    marker: Final = "a2aspend" + uuid.uuid4().hex[:12]

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        body: Final = json.loads(request.body)
        if body["method"] == "message/stream":
            event: Final = {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "kind": "status-update",
                    "taskId": "t1",
                    "contextId": "c1",
                    "status": {"state": "completed"},
                    "final": True,
                },
            }
            return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(event)}\n\n".encode(),))
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

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": marker,
                "agent_card_params": _peer_card(wire.url, marker),
                "litellm_params": {"cost_per_query": 0.25},
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
        message: Final = {
            "kind": "message",
            "role": "user",
            "messageId": marker + "-in",
            "parts": [{"kind": "text", "text": "ping"}],
        }
        sent: Final = gateway.client.post(
            f"/a2a/{agent}",
            headers={"Authorization": f"Bearer {key}"},
            json={"jsonrpc": "2.0", "id": marker + "-send", "method": "message/send", "params": {"message": message}},
        )
        assert sent.status_code == 200, sent.text
        assert sent.json()["result"]["parts"] == [{"kind": "text", "text": "billed"}], sent.text
        with gateway.client.stream(
            "POST",
            f"/a2a/{agent}",
            headers={"Authorization": f"Bearer {key}", "Accept": "text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": marker + "-stream",
                "method": "message/stream",
                "params": {"message": message},
            },
        ) as streamed:
            body: Final = streamed.read().decode()
            assert streamed.status_code == 200, body
        assert body.startswith("data: ") and '"state": "completed"' in body, body
        assert len(tuple(item for item in wire.drain() if item.method == "POST")) == 2

        logs: Final = eventually(
            lambda: read_rows(
                "SELECT request_id, agent_id, spend, api_key, call_type, "
                """to_char("startTime", 'YYYY-MM-DD') AS day FROM "LiteLLM_SpendLogs" """
                "WHERE api_key=%s ORDER BY request_id",
                (digest,),
            ),
            lambda rows: len(rows) == 2,
            seconds=60,
        )
        assert [{name: value for name, value in row.items() if name != "day"} for row in logs] == [
            {
                "request_id": marker + "-send",
                "agent_id": agent,
                "spend": 0.25,
                "api_key": digest,
                "call_type": "asend_message",
            },
            {
                "request_id": marker + "-stream",
                "agent_id": agent,
                "spend": 0.25,
                "api_key": digest,
                "call_type": "asend_message_streaming",
            },
        ], logs
        assert eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (agent,)),
            lambda rows: rows == [{"spend": 0.5}],
            seconds=60,
        ) == [{"spend": 0.5}]
        assert eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda rows: rows == [{"spend": 0.5}],
            seconds=60,
        ) == [{"spend": 0.5}]
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
            lambda rows: sum(int(str(row["api_requests"])) for row in rows) == 2,
            seconds=90,
        )
        assert daily == [
            {
                "agent_id": agent,
                "date": day,
                "api_key": digest,
                "spend": 0.25 * count,
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
            (day, _ActivityMetrics(spend=0.25 * count, api_requests=count, successful_requests=count))
            for day, count in sorted(requests_per_day.items(), reverse=True)
        ], activity.text
        assert [
            result["breakdown"]["models"][f"a2a_agent/{marker}"]["api_key_breakdown"][digest]["metrics"]["spend"]
            for result in activity.json()["results"]
        ] == [0.25 * count for _, count in sorted(requests_per_day.items(), reverse=True)], activity.text
