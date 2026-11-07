import json
import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from integration.compatibility._a2a_peers import legacy_send, peer, register, sse_frames
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
                "WHERE api_key=%s ORDER BY call_type",
                (digest,),
            ),
            lambda rows: len(rows) == len(methods),
            seconds=30,
        )
        assert [
            {name: value for name, value in row.items() if name not in {"day", "request_id"}}
            for row in logs
        ] == [
            {
                "agent_id": agent,
                "spend": cost,
                "api_key": digest,
                "call_type": _CALL_TYPES[method],
            }
            for method in sorted(methods)
        ], logs
        request_ids: Final = tuple(str(row["request_id"]) for row in logs)
        assert len(set(request_ids)) == len(methods), logs
        assert all(str(uuid.UUID(request_id)) == request_id for request_id in request_ids), logs
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
    _assert_a2a_calls_are_billed(gateway, _PER_TOKEN, _PER_TOKEN_COST, (("message/stream", "/a2a/{agent}"),))


def _assert_reused_jsonrpc_id_has_distinct_spend_identity(
    gateway: Gateway,
    *,
    version: str,
    method: str,
) -> None:
    marker: Final = "a2aidentity" + uuid.uuid4().hex
    dialect: Final = "v1" if version == "1.0" else "langgraph"
    params: Final = None if version == "1.0" else {"a2a_protocol_version": "0.3"}

    with peer(marker, dialect) as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, marker, agent.url, params)
        user_ids: Final = tuple(scenario.user(user_id=f"{marker}-user-{index}") for index in range(2))
        alias_user_pairs: Final = tuple(
            (f"{marker}-key-{index}", user_id)
            for index, user_id in enumerate(user_ids)
        )
        keys: Final = tuple(
            scenario.key(user_id=user_id, key_alias=alias)
            for alias, user_id in alias_user_pairs
        )
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        body: Final = (
            legacy_send("1", method=method)
            if version == "0.3"
            else {
                "jsonrpc": "2.0",
                "id": "1",
                "method": method,
                "params": {
                    "message": {
                        "messageId": "client-message",
                        "parts": [{"text": "reused id"}],
                        "role": "ROLE_USER",
                    }
                },
            }
        )

        responses: Final = tuple(
            gateway.request(
                "POST",
                f"/a2a/{identity}",
                body,
                key=key,
                headers={"a2a-version": version},
            )
            for key in keys
        )
        assert all(response.status_code == 200 for response in responses), responses
        if "stream" in method.lower():
            frames_by_response: Final = tuple(sse_frames(response.text) for response in responses)
            assert all(
                frames and all(frame["id"] == "1" for frame in frames) and all("error" not in frame for frame in frames)
                for frames in frames_by_response
            ), responses
        else:
            response_bodies: Final = tuple(response.json() for response in responses)
            assert all(
                response_body["id"] == "1" and "error" not in response_body for response_body in response_bodies
            ), responses

        assert agent.wire is not None
        upstream_requests: Final = tuple(item for item in agent.wire.drain() if item.method == "POST")
        upstream: Final = tuple(json.loads(item.body) for item in upstream_requests)
        assert len(upstream) == 2, upstream
        trace_ids: Final = tuple(
            str(item.headers["x-litellm-trace-id"])
            for item in upstream_requests
        )

        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend.request_id, spend.api_key, spend.call_type, token.key_alias, token.user_id '
                'FROM "LiteLLM_SpendLogs" AS spend '
                'JOIN "LiteLLM_VerificationToken" AS token ON spend.api_key=token.token '
                "WHERE spend.api_key=%s OR spend.api_key=%s ORDER BY token.key_alias",
                (digests[0], digests[1]),
            ),
            lambda found: len(found) == 2,
            seconds=30,
            return_last_on_timeout=True,
        )
        assert len(rows) == 2, rows
        assert tuple((str(row["key_alias"]), str(row["user_id"])) for row in rows) == tuple(
            sorted(alias_user_pairs)
        ), rows
        request_ids: Final = tuple(str(row["request_id"]) for row in rows)
        response_call_ids: Final = tuple(response.headers.get("x-litellm-call-id") for response in responses)
        assert all(response_call_id is not None for response_call_id in response_call_ids), responses
        assert response_call_ids == request_ids, (response_call_ids, rows)
        assert len(set(request_ids)) == 2 and all(request_id != "1" for request_id in request_ids), rows
        assert all(str(uuid.UUID(request_id)) == request_id for request_id in request_ids), rows
        assert len(trace_ids) == 2, trace_ids
        assert len(set(trace_ids)) == 2 and all(trace_id != "1" for trace_id in trace_ids), trace_ids


@pytest.mark.parametrize(
    ("version", "method"),
    (
        ("1.0", "SendMessage"),
        ("1.0", "SendStreamingMessage"),
        ("0.3", "message/send"),
        ("0.3", "message/stream"),
    ),
)
def test_a2a_reused_jsonrpc_ids_have_distinct_litellm_spend_ids(
    gateway: Gateway,
    version: str,
    method: str,
) -> None:
    _assert_reused_jsonrpc_id_has_distinct_spend_identity(gateway, version=version, method=method)
