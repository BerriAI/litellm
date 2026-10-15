import json
import math
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.upstream import JsonResponse, SseResponse, delete_scenario, register_scenario
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

Surface = Literal["chat", "chat-stream", "messages", "messages-stream", "responses", "responses-stream"]

_SURFACES: Final[tuple[Surface, ...]] = (
    "chat",
    "chat-stream",
    "messages",
    "messages-stream",
    "responses",
    "responses-stream",
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])
_OBSERVATIONS: Final = TypeAdapter(dict[str, list[dict[str, JsonValue]]])
_LENS_TOKEN: Final = f"integration-lens-service-token-{uuid.uuid4().hex}"
_ANSWER: Final = "scripted answer"
_SPEND_PER_CALL: Final = 20 * 0.001 + 20 * 0.002


@dataclass(frozen=True, slots=True)
class _LensSink:
    rows: list[dict[str, JsonValue]]  # mutable-ok: appended by the sink's handler threads
    outages: threading.Semaphore

    def respond(self, request: Request) -> Reply:
        if request.method != "POST" or request.target != "/internal/spend":
            return Reply(body=b"{}")
        assert request.headers.get("authorization") == f"Bearer {_LENS_TOKEN}", request.headers
        if self.outages.acquire(blocking=False):
            return Reply(status=503, body=b'{"error":"lens unavailable"}')
        self.rows.extend(_JSON_OBJECTS.validate_json(request.body))  # mutable-ok: see above
        return Reply(status=204, body=b"")

    def rows_for(self, model: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(row for row in tuple(self.rows) if row.get("model_group") == model)


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    lens: _LensSink


def _config(directory: Path, flag: bool | None) -> Path:
    configuration: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    path: Final = directory / f"disable-end-user-cost-tracking-{flag}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **configuration,
                "general_settings": {
                    **object_value(configuration["general_settings"]),
                    "disable_responses_id_security": True,
                },
                "litellm_settings": {
                    **object_value(configuration["litellm_settings"]),
                    "disable_end_user_cost_tracking": flag,
                },
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("disable-end-user-cost-tracking")
    sink: Final = _LensSink(rows=[], outages=threading.Semaphore(0))
    with (
        wire_server(sink.respond) as lens,
        gateway_from_environment() as shared,
        owned_proxy_process(
            shared,
            directory,
            {"LITELLM_LENS_URL": lens.url, "LITELLM_LENS_SERVICE_TOKEN": _LENS_TOKEN},
            config=_config(directory, True),
            workers=2,
        ) as proxy,
    ):
        yield _Rig(proxy, sink)


def _root(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _observed(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    return _OBSERVATIONS.validate_json(response.content)["requests"]


def _anthropic_reply(stream: bool) -> JsonResponse | SseResponse:
    if not stream:
        return JsonResponse(
            content_type="application/json",
            body={
                "id": "msg_$UNIQUE_ID",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": _ANSWER}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 20},
            },
        )
    return SseResponse(
        content_type="text/event-stream",
        frames=(
            "event: message_start\ndata: "
            '{"type":"message_start","message":{"id":"msg_$UNIQUE_ID","type":"message",'
            '"role":"assistant","model":"claude-sonnet-5-5","content":[],"stop_reason":null,'
            '"stop_sequence":null,"usage":{"input_tokens":20,"output_tokens":0}}}',
            "event: content_block_start\ndata: "
            '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "event: content_block_delta\ndata: "
            f'{{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{_ANSWER}"}}}}',
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
            "event: message_delta\ndata: "
            '{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
            '"usage":{"output_tokens":20}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ),
    )


def _completed_response(status: str) -> dict[str, JsonValue]:
    return {
        "id": "resp_$UNIQUE_ID",
        "object": "response",
        "created_at": 1759700000,
        "status": status,
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": _ANSWER, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 20,
            "total_tokens": 40,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _responses_reply(stream: bool) -> JsonResponse | SseResponse:
    if not stream:
        return JsonResponse(content_type="application/json", body=_completed_response("completed"))
    created: Final = {**_completed_response("in_progress"), "output": [], "usage": None}
    return SseResponse(
        content_type="text/event-stream",
        frames=(
            "event: response.created\ndata: "
            + json.dumps({"type": "response.created", "sequence_number": 0, "response": created}),
            "event: response.output_text.delta\ndata: "
            + json.dumps(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": _ANSWER,
                    "logprobs": [],
                }
            ),
            "event: response.completed\ndata: "
            + json.dumps(
                {"type": "response.completed", "sequence_number": 2, "response": _completed_response("completed")}
            ),
        ),
    )


def _scripted_model(scenario: Scenario, reply: JsonResponse | SseResponse, provider_model: str, suffix: str) -> str:
    handle: Final = register_scenario(f"sc-{uuid.uuid4().hex}", reply)
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(
        model=provider_model,
        api_base=f"{handle.api_base()}{suffix}",
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    )


def _model(scenario: Scenario, surface: Surface) -> str:
    match surface:
        case "chat" | "chat-stream":
            return scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        case "messages" | "messages-stream":
            return _scripted_model(
                scenario, _anthropic_reply(surface == "messages-stream"), "anthropic/claude-sonnet-5-5", ""
            )
        case "responses" | "responses-stream":
            return _scripted_model(
                scenario, _responses_reply(surface == "responses-stream"), "openai/gpt-4o-mini", "/v1"
            )


def _openai_client(gateway: Gateway, key: str) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{_root(gateway)}/v1", api_key=key, max_retries=0, http_client=httpx.Client(trust_env=False)
    )


def _anthropic_client(gateway: Gateway, key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=_root(gateway), api_key=key, max_retries=0, http_client=httpx.Client(trust_env=False)
    )


def _chat(gateway: Gateway, model: str, key: str, end_user: str, text: str) -> str:
    with _openai_client(gateway, key) as client:
        completion: Final = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": text}], user=end_user
        )
    assert completion.choices[0].finish_reason == "stop", completion
    assert completion.usage is not None and completion.usage.total_tokens == 40, completion.usage
    return completion.id


def _chat_stream(gateway: Gateway, model: str, key: str, end_user: str, text: str) -> str:
    with _openai_client(gateway, key) as client:
        chunks: Final = tuple(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": text}],
                stream=True,
                stream_options={"include_usage": True},
                extra_headers={"x-litellm-end-user-id": end_user},
            )
        )
    finishes: Final = tuple(choice.finish_reason for chunk in chunks for choice in chunk.choices)
    assert "stop" in finishes, chunks
    return chunks[0].id


def _messages(gateway: Gateway, model: str, key: str, end_user: str, text: str) -> str:
    with _anthropic_client(gateway, key) as client:
        message: Final = client.messages.create(
            model=model, max_tokens=64, messages=[{"role": "user", "content": text}], metadata={"user_id": end_user}
        )
    assert [block.type for block in message.content] == ["text"], message
    assert message.content[0].type == "text" and message.content[0].text == _ANSWER, message
    assert message.stop_reason == "end_turn", message
    return message.id


def _messages_stream(gateway: Gateway, model: str, key: str, end_user: str, text: str) -> str:
    with (
        _anthropic_client(gateway, key) as client,
        client.messages.stream(
            model=model, max_tokens=64, messages=[{"role": "user", "content": text}], metadata={"user_id": end_user}
        ) as stream,
    ):
        message: Final = stream.get_final_message()
    assert message.content[0].type == "text" and message.content[0].text == _ANSWER, message
    assert message.stop_reason == "end_turn", message
    return message.id


def _responses(gateway: Gateway, model: str, key: str, end_user: str, text: str) -> str:
    with _openai_client(gateway, key) as client:
        response: Final = client.responses.create(model=model, input=text, safety_identifier=end_user)
    assert response.output_text == _ANSWER, response
    return response.id


def _responses_stream(gateway: Gateway, model: str, key: str, end_user: str, text: str) -> str:
    with _openai_client(gateway, key) as client:
        events: Final = tuple(client.responses.create(model=model, input=text, safety_identifier=end_user, stream=True))
    completed: Final = tuple(event for event in events if event.type == "response.completed")
    assert len(completed) == 1, events
    assert completed[0].type == "response.completed" and completed[0].response.output_text == _ANSWER, completed
    return completed[0].response.id


_SENDERS: Final[Mapping[Surface, Callable[[Gateway, str, str, str, str], str]]] = MappingProxyType(
    {
        "chat": _chat,
        "chat-stream": _chat_stream,
        "messages": _messages,
        "messages-stream": _messages_stream,
        "responses": _responses,
        "responses-stream": _responses_stream,
    }
)


def _provider_body(surface: Surface, end_user: str, text: str) -> dict[str, JsonValue]:
    match surface:
        case "chat":
            return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}], "user": end_user}
        case "chat-stream":
            return {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": text}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        case "messages" | "messages-stream":
            return {
                "model": "claude-sonnet-5-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": text}],
                "metadata": {"user_id": end_user},
                "stream": surface == "messages-stream",
            }
        case "responses" | "responses-stream":
            return {
                "model": "gpt-4o-mini",
                "input": text,
                "safety_identifier": end_user,
                **({"stream": True} if surface == "responses-stream" else {}),
            }


def _spend_rows(model: str, count: int) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        eventually(
            lambda: read_rows(
                'SELECT request_id, end_user, spend, cache_hit, status FROM "LiteLLM_SpendLogs" '
                'WHERE model_group = %s ORDER BY "startTime"',
                (model,),
            ),
            lambda rows: len(rows) >= count,
            seconds=70,
        )
    )


def _team_requests_flushed(team: str, count: int) -> None:
    eventually(
        lambda: read_rows(
            'SELECT COALESCE(SUM(api_requests), 0)::int AS requests FROM "LiteLLM_DailyTeamSpend" WHERE team_id = %s',
            (team,),
        ),
        lambda rows: int(str(rows[0]["requests"])) >= count,
        seconds=70,
    )


def _end_user_rows(end_users: tuple[str, ...]) -> tuple[tuple[dict[str, JsonValue], ...], ...]:
    placeholders: Final = ", ".join("%s" for _ in end_users)
    return (
        tuple(
            read_rows(
                f'SELECT end_user_id FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id IN ({placeholders})', end_users
            )
        ),
        tuple(read_rows(f'SELECT user_id FROM "LiteLLM_EndUserTable" WHERE user_id IN ({placeholders})', end_users)),
    )


def _lens_rows(rig: _Rig, model: str, count: int) -> tuple[dict[str, JsonValue], ...]:
    return eventually(lambda: rig.lens.rows_for(model), lambda rows: len(rows) >= count, seconds=70)


def _assert_untracked_lens_rows(rows: tuple[dict[str, JsonValue], ...]) -> None:
    assert [row["end_user"] for row in rows] == [""] * len(rows), rows
    metadata: Final = tuple(_JSON_OBJECT.validate_json(str(row["metadata"])) for row in rows)
    assert [entry.get("user_api_key_end_user_id") for entry in metadata] == [None] * len(rows), metadata


def _forget_customers(scenario: Scenario, end_users: tuple[str, ...]) -> None:
    scenario.cleanups.callback(scenario.gateway.request, "POST", "/customer/delete", {"user_ids": list(end_users)})


@pytest.mark.parametrize("surface", _SURFACES)
def test_flag_on_keeps_end_user_out_of_spend_rows_and_lens_export_with_cache_twin(rig: _Rig, surface: Surface) -> None:
    gateway: Final = rig.proxy.gateway
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        team: Final = scenario.team()
        model: Final = _model(scenario, surface)
        key: Final = scenario.key(models=[model], team_id=team)
        end_user: Final = f"integration-untracked-end-user-{uuid.uuid4().hex}"
        _forget_customers(scenario, (end_user,))
        text: Final = f"untracked end user {uuid.uuid4().hex}"

        first: Final = _SENDERS[surface](gateway, model, key, end_user, text)
        _spend_rows(model, 1)
        twin: Final = _SENDERS[surface](gateway, model, key, end_user, text)

        observed: Final = _observed(upstream)
        assert [object_value(request)["body"] for request in observed] == [_provider_body(surface, end_user, text)], (
            observed
        )
        assert twin == first, (first, twin)
        rows: Final = _spend_rows(model, 2)
        assert len(rows) == 2, rows
        assert rows[0]["request_id"] == first, (rows, first)
        assert str(rows[1]["request_id"]).startswith(f"{first}_cache_hit"), (rows, first)
        assert [row["end_user"] for row in rows] == ["", ""], rows
        assert [row["status"] for row in rows] == ["success", "success"], rows
        assert [str(row["cache_hit"]) == "True" for row in rows] == [False, True], rows
        assert math.isclose(float(str(rows[0]["spend"])), _SPEND_PER_CALL), rows
        assert float(str(rows[1]["spend"])) == 0.0, rows
        _team_requests_flushed(team, 2)
        assert _end_user_rows((end_user,)) == ((), ()), end_user
        lens_rows: Final = _lens_rows(rig, model, 2)
        assert sorted((bool(row["cache_hit"]), str(row["request_id"]).split("_cache_hit")[0]) for row in lens_rows) == [
            (False, first),
            (True, first),
        ], lens_rows
        _assert_untracked_lens_rows(lens_rows)


@pytest.mark.parametrize("status", (401, 404, 500))
def test_flag_on_failed_request_creates_no_customer_and_logs_no_end_user(rig: _Rig, status: int) -> None:
    gateway: Final = rig.proxy.gateway
    with (
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        gateway.scenario() as scenario,
    ):
        provider_model: Final = f"untracked-failure-{uuid.uuid4().hex}"
        configured: Final = upstream.post(f"/__scripts/{provider_model}", json={"statuses": [status] * 4})
        assert configured.status_code == 200, configured.text
        scenario.cleanups.callback(upstream.delete, f"/__scripts/{provider_model}")
        _observed(upstream)
        team: Final = scenario.team()
        model: Final = scenario.model(model=f"openai/{provider_model}")
        key: Final = scenario.key(models=[model], team_id=team)
        end_user: Final = f"integration-untracked-end-user-{uuid.uuid4().hex}"
        _forget_customers(scenario, (end_user,))
        text: Final = f"untracked failure {uuid.uuid4().hex}"

        failed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": text}], "user": end_user},
            key=key,
        )

        assert failed.status_code == status, failed.text
        assert "Controlled provider failure" in failed.text, failed.text
        observed: Final = _observed(upstream)
        assert [object_value(request)["body"] for request in observed] == [
            {"model": provider_model, "messages": [{"role": "user", "content": text}]}
        ], observed
        rows: Final = _spend_rows(model, 1)
        assert [(row["status"], row["end_user"]) for row in rows] == [("failure", "")], rows
        _team_requests_flushed(team, 1)
        assert _end_user_rows((end_user,)) == ((), ()), end_user
        assert gateway.request("GET", "/customer/info", params={"end_user_id": end_user}).status_code in (400, 404)


@pytest.mark.parametrize(
    ("user", "status"),
    (
        pytest.param(12345, 200, id="int"),
        pytest.param(["integration-list-end-user"], 200, id="list"),
        pytest.param("", 200, id="empty"),
        pytest.param(f"integration-5kb-end-user-{'u' * 5120}", 200, id="5kb"),
    ),
)
def test_flag_on_malformed_end_user_values_are_served_and_not_attributed(
    rig: _Rig, user: JsonValue, status: int
) -> None:
    gateway: Final = rig.proxy.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], team_id=team)
        served: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"sad {uuid.uuid4().hex}"}], "user": user},
            key=key,
        )
        assert served.status_code == status, served.text
        rows: Final = _spend_rows(model, 1)
        assert [row["end_user"] for row in rows] == [""], rows
        _team_requests_flushed(team, 1)
        if isinstance(user, str) and user:
            assert _end_user_rows((user,)) == ((), ()), user
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_flag_on_header_sent_twice_and_body_user_are_none_attributed(rig: _Rig) -> None:
    gateway: Final = rig.proxy.gateway
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        team: Final = scenario.team()
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], team_id=team)
        end_users: Final = tuple(f"integration-untracked-end-user-{index}-{uuid.uuid4().hex}" for index in range(3))
        _forget_customers(scenario, end_users)
        text: Final = f"precedence {uuid.uuid4().hex}"

        served: Final = gateway.client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": text}], "user": end_users[2]},
            headers=[
                ("Authorization", f"Bearer {key}"),
                ("x-litellm-end-user-id", end_users[0]),
                ("x-litellm-end-user-id", end_users[1]),
            ],
        )

        assert served.status_code == 200, served.text
        assert [object_value(request)["body"] for request in _observed(upstream)] == [
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}], "user": end_users[2]}
        ]
        rows: Final = _spend_rows(model, 1)
        assert [row["end_user"] for row in rows] == [""], rows
        _team_requests_flushed(team, 1)
        assert _end_user_rows(end_users) == ((), ()), end_users
        lens_rows: Final = _lens_rows(rig, model, 1)
        _assert_untracked_lens_rows(lens_rows)


def test_flag_on_still_refuses_an_end_user_over_budget_at_auth(rig: _Rig) -> None:
    gateway: Final = rig.proxy.gateway
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        end_user: Final = f"integration-budgeted-end-user-{uuid.uuid4().hex}"
        created: Final = gateway.request(
            "POST", "/customer/new", {"user_id": end_user, "max_budget": 0.5, "spend": 1.0}
        )
        assert created.status_code == 200, created.text
        _forget_customers(scenario, (end_user,))
        _observed(upstream)

        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"budget {uuid.uuid4().hex}"}], "user": end_user},
            key=key,
        )

        assert denied.status_code == 422, denied.text
        error: Final = object_value(_JSON_OBJECT.validate_json(denied.content)["error"])
        assert error["type"] == "budget_exceeded", error
        assert str(error["message"]).startswith(
            f"ExceededBudget: End User={end_user} over budget. Spend=1.0, Budget=0.5"
        ), error
        assert _observed(upstream) == []


def test_flag_on_management_views_report_no_end_user(rig: _Rig) -> None:
    gateway: Final = rig.proxy.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], team_id=team)
        end_user: Final = f"integration-untracked-end-user-{uuid.uuid4().hex}"
        _forget_customers(scenario, (end_user,))
        response_id: Final = _chat(gateway, model, key, end_user, f"management {uuid.uuid4().hex}")
        _spend_rows(model, 1)
        _team_requests_flushed(team, 1)
        today: Final = date.today().isoformat()

        logs: Final = gateway.request("GET", "/spend/logs", params={"request_id": response_id})
        activity: Final = gateway.request(
            "GET",
            "/customer/daily/activity",
            params={"end_user_ids": end_user, "start_date": today, "end_date": today},
        )
        info: Final = gateway.request("GET", "/customer/info", params={"end_user_id": end_user})

        assert logs.status_code == 200, logs.text
        assert [(row["request_id"], row["end_user"]) for row in _JSON_OBJECTS.validate_json(logs.content)] == [
            (response_id, "")
        ], logs.text
        assert activity.status_code == 200, activity.text
        assert _JSON_OBJECT.validate_json(activity.content)["results"] == [], activity.text
        assert info.status_code in (400, 404), info.text


@pytest.mark.parametrize("flag", (False, None), ids=("explicit-false", "null"))
def test_flag_off_or_null_still_attributes_the_end_user(gateway: Gateway, tmp_path: Path, flag: bool | None) -> None:
    with (
        owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, flag)) as tracked,
        tracked.scenario() as scenario,
    ):
        team: Final = scenario.team()
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], team_id=team)
        end_user: Final = f"integration-tracked-end-user-{uuid.uuid4().hex}"
        _forget_customers(scenario, (end_user,))

        response_id: Final = _chat(tracked, model, key, end_user, f"tracked {uuid.uuid4().hex}")

        rows: Final = _spend_rows(model, 1)
        assert [(row["request_id"], row["end_user"]) for row in rows] == [(response_id, end_user)], rows
        daily: Final = eventually(
            lambda: read_rows(
                'SELECT spend, api_requests FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id = %s', (end_user,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert math.isclose(float(str(daily[0]["spend"])), _SPEND_PER_CALL), daily
        assert int(str(daily[0]["api_requests"])) == 1, daily


@dataclass(frozen=True, slots=True)
class _BurstCall:
    surface: Surface
    end_user: str
    text: str


def _burst_calls(label: str) -> tuple[_BurstCall, ...]:
    return tuple(
        _BurstCall(surface, f"integration-burst-end-user-{label}-{uuid.uuid4().hex}", f"burst {uuid.uuid4().hex}")
        for surface in _SURFACES * 2
    )


def _burst(
    rig: _Rig, models: Mapping[Surface, str], key: str, calls: tuple[_BurstCall, ...]
) -> tuple[tuple[Surface, str], ...]:
    def send(call: _BurstCall) -> tuple[Surface, str]:
        return call.surface, _SENDERS[call.surface](
            rig.proxy.gateway, models[call.surface], key, call.end_user, call.text
        )

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return tuple(pool.map(send, calls))


def _assert_landed_once(rig: _Rig, model: str, ids: list[str]) -> None:
    rows: Final = _spend_rows(model, len(ids))
    assert sorted(str(row["request_id"]) for row in rows) == ids, (model, rows, ids)
    assert [row["end_user"] for row in rows] == [""] * len(rows), rows
    lens_rows: Final = _lens_rows(rig, model, len(ids))
    assert sorted(str(row["request_id"]) for row in lens_rows) == ids, (model, lens_rows, ids)
    _assert_untracked_lens_rows(lens_rows)


def _assert_burst_landed_once(rig: _Rig, models: Mapping[Surface, str], sent: tuple[tuple[Surface, str], ...]) -> None:
    for surface, model in models.items():
        _assert_landed_once(rig, model, sorted(response_id for kind, response_id in sent if kind == surface))


def test_flag_on_burst_lands_each_row_once_through_sink_outage_and_worker_kill(rig: _Rig) -> None:
    gateway: Final = rig.proxy.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        models: Final[Mapping[Surface, str]] = MappingProxyType(
            {surface: _model(scenario, surface) for surface in _SURFACES}
        )
        key: Final = scenario.key(models=list(models.values()), team_id=team)
        calls: Final = _burst_calls("before-kill") + _burst_calls("after-kill")
        _forget_customers(scenario, tuple(call.end_user for call in calls))
        rig.lens.outages.release(2)

        before_kill: Final = _burst(rig, models, key, calls[: len(_SURFACES) * 2])
        _assert_burst_landed_once(rig, models, before_kill)
        assert not rig.lens.outages.acquire(blocking=False), "the Lens sink outage was never hit"
        victim: Final = min(rig.proxy.worker_pids())
        replacement: Final = rig.proxy.kill_worker(victim)
        assert replacement != victim, (victim, replacement)
        after_kill: Final = _burst(rig, models, key, calls[len(_SURFACES) * 2 :])
        _assert_burst_landed_once(rig, models, before_kill + after_kill)

        _team_requests_flushed(team, len(calls))
        assert _end_user_rows(tuple(call.end_user for call in calls)) == ((), ()), calls
