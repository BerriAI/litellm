import asyncio
import json
import uuid
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse
from pydantic import JsonValue

PROMPT_TOKENS: Final = 20
COMPLETION_TOKENS: Final = 20


def _register_match_embeddings(prefix: str) -> ScenarioHandle:
    return register_scenario(
        f"{prefix}-{uuid.uuid4().hex}",
        JsonResponse(
            content_type="application/json",
            body={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0] + [0.0] * 1535}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        ),
    )


def _match_embeddings(scenario) -> str:
    handle: Final[ScenarioHandle] = _register_match_embeddings("audit-match")
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(model="openai/text-embedding-3-small", mode="embedding", api_base=handle.api_base())


def _nomatch_embeddings(scenario) -> str:
    return scenario.model(model="openai/text-embedding-3-small", mode="embedding")


def _semantic_router(scenario, *, default_model: str, route_model: str, embedding_model: str) -> str:
    alias: Final = f"audit-router-{uuid.uuid4().hex}"
    routes: Final = json.dumps(
        {"routes": [{"name": route_model, "utterances": ["fix this python stack trace"], "score_threshold": 0.3}]}
    )
    created_name: Final = scenario.model(
        model=f"auto_router/{alias}",
        auto_router_config=routes,
        auto_router_default_model=default_model,
        auto_router_embedding_model=embedding_model,
    )
    return created_name


def _passthrough_scenario(scenario) -> ScenarioHandle:
    anthropic_body: Final = JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "audit-pass",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS},
        },
    )
    responses_body: Final = JsonResponse(
        content_type="application/json",
        body={
            "id": "resp_$REQUEST_ID",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "audit-pass",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS, "total_tokens": 40},
        },
    )
    handle: Final = register_scenario(
        "v1",
        RoutedResponse(
            content_type="application/x-routed",
            routes={
                "POST /messages": anthropic_body,
                "POST /v1/messages": anthropic_body,
                "POST /responses": responses_body,
                "POST /v1/responses": responses_body,
            },
        ),
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _semantic_setup(scenario, *, embeddings) -> tuple[str, str, str]:
    default_model: Final = scenario.model(model="openai/gpt-4o-mini")
    route_model: Final = scenario.model(model="openai/gpt-4o-mini")
    embedding_model: Final = embeddings(scenario)
    alias: Final = _semantic_router(
        scenario, default_model=default_model, route_model=route_model, embedding_model=embedding_model
    )
    return alias, default_model, route_model


def _chat_body(alias: str, session_id, text: str = "fix this python stack trace") -> dict[str, JsonValue]:
    return {
        "model": alias,
        "messages": [{"role": "user", "content": text}],
        "litellm_session_id": session_id,
    }


def _probe_bodies(alias: str, marker: str, *, all_endpoints: bool) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    chat: Final = (
        "/v1/chat/completions",
        _chat_body(alias, f"audit-probe-{marker}-{uuid.uuid4().hex}", text=f"probe {uuid.uuid4().hex}"),
    )
    extras: Final = (
        (
            (
                "/v1/messages",
                {
                    "model": alias,
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": f"probe {uuid.uuid4().hex}"}],
                    "litellm_session_id": f"audit-probe-{marker}-{uuid.uuid4().hex}",
                },
            ),
            (
                "/v1/responses",
                {
                    "model": alias,
                    "input": f"probe {uuid.uuid4().hex}",
                    "litellm_session_id": f"audit-probe-{marker}-{uuid.uuid4().hex}",
                },
            ),
        )
        if all_endpoints
        else ()
    )
    return (chat, *extras)


def _ready(gateway: Gateway, alias: str, marker: str, *, all_endpoints: bool = False) -> None:
    proxy_url: Final = str(gateway.client.base_url).rstrip("/")
    headers: Final = {"Authorization": f"Bearer {gateway.key}", "Connection": "close"}

    def probe() -> int:
        def round_ok() -> bool:
            return all(
                httpx.post(f"{proxy_url}{endpoint}", json=body, headers=headers, timeout=30).status_code == 200
                for _ in range(3)
                for endpoint, body in _probe_bodies(alias, marker, all_endpoints=all_endpoints)
            )

        return sum(1 for _ in range(3) if round_ok())

    count: Final = eventually(probe, lambda value: value == 3, seconds=90)
    assert count == 3, count


def _completed_chat(gateway: Gateway, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    return object_value(response.json())


def _forget_scenario(handle: ScenarioHandle) -> None:
    try:
        delete_scenario(handle)
    except httpx.HTTPStatusError as error:
        assert error.response.status_code == 404, error


def _burst_body(alias: str, session_id: str, index: int) -> dict[str, JsonValue]:
    if index % 3 == 0:
        return {
            **_chat_body(alias, session_id, text=f"fix this python stack trace {index}"),
            "stream": index % 2 == 0,
            "stream_options": {"include_usage": True},
        }
    if index % 3 == 1:
        return {
            "model": alias,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": f"fix this python stack trace {index}"}],
            "litellm_session_id": session_id,
        }
    return {
        "model": alias,
        "input": f"fix this python stack trace {index}",
        "litellm_session_id": session_id,
    }


def _decision_row(gateway: Gateway, session_id: str, alias: str) -> dict:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, model, model_group, status, metadata->'routing_decision' AS decision"
            ' FROM "LiteLLM_SpendLogs" WHERE session_id=%s AND model_group=%s'
            " AND status='success' AND metadata->>'routing_decision' IS NOT NULL",
            (session_id, alias),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _rollup_row(session_id: str, turns: int) -> dict:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT session_id, router_name, router_type, turns, spend, tier_turns"
            ' FROM "LiteLLM_AutoRouterSession" WHERE session_id=%s',
            (session_id,),
        ),
        lambda values: len(values) == 1 and values[0]["turns"] == turns,
        seconds=70,
    )
    return rows[0]


def _decision_of(row: dict) -> dict:
    value: Final = row["decision"]
    return json.loads(value) if isinstance(value, str) else value


def _benchmarks_group(gateway: Gateway, alias: str) -> dict | None:
    groups: Final = gateway.get("/auto_router/benchmarks")["groups"]
    assert isinstance(groups, list)
    for entry in groups:
        group: Final = object_value(entry)
        if group["router_name"] == alias:
            return group
    return None


@pytest.mark.covers("other.routing.auto_router.semantic_match_turn_records_tier_decision_and_session_row")
def test_a_matched_semantic_route_records_a_tier_decision_and_a_session_turn(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias, default_model, route_model = _semantic_setup(scenario, embeddings=_match_embeddings)
        _ready(gateway, alias, uuid.uuid4().hex)
        session_id: Final = f"audit-h1-{uuid.uuid4().hex}"

        body: Final = _completed_chat(gateway, _chat_body(alias, session_id))
        assert string_value(body["id"])

        row: Final = _decision_row(gateway, session_id, alias)
        assert row["status"] == "success", row
        assert row["model"] == "openai/gpt-4o-mini" and row["model_group"] == alias, row
        assert _decision_of(row) == {
            "router_model_name": alias,
            "router_type": "semantic",
            "routed_model": route_model,
            "tier": route_model,
        }, row

        session: Final = _rollup_row(session_id, 1)
        assert session["router_name"] == alias and session["router_type"] == "semantic", session
        assert session["tier_turns"] == {route_model: 1}, session


@pytest.mark.covers("other.routing.auto_router.semantic_stream_fallback_turn_records_default_fallback")
def test_a_streamed_semantic_fallback_records_a_default_fallback_decision(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias, default_model, route_model = _semantic_setup(scenario, embeddings=_nomatch_embeddings)
        session_id: Final = f"audit-h2-{uuid.uuid4().hex}"
        _ready(gateway, alias, uuid.uuid4().hex)
        client: Final = openai.OpenAI(base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key)

        stream: Final = client.chat.completions.create(
            model=alias,
            messages=[{"role": "user", "content": "fix this python stack trace"}],
            stream=True,
            extra_body={"litellm_session_id": session_id, "stream_options": {"include_usage": True}},
        )
        identifiers: Final = {chunk.id for chunk in stream if chunk.id}
        response_id: Final = next(iter(identifiers))
        assert string_value(response_id)

        row: Final = _decision_row(gateway, session_id, alias)
        assert row["request_id"] == response_id and row["status"] == "success", (row, response_id)
        assert _decision_of(row) == {
            "router_model_name": alias,
            "router_type": "semantic",
            "routed_model": default_model,
            "cause": "default_fallback",
        }, row

        session: Final = _rollup_row(session_id, 1)
        assert session["router_name"] == alias and session["router_type"] == "semantic", session
        assert session["tier_turns"] == {}, session


@pytest.mark.covers("other.routing.auto_router.semantic_anthropic_messages_turn_records_match_decision")
def test_an_anthropic_messages_turn_records_a_semantic_decision(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        _passthrough_scenario(scenario)
        alias, default_model, route_model = _semantic_setup(scenario, embeddings=_match_embeddings)
        session_id: Final = f"audit-h3-{uuid.uuid4().hex}"
        _ready(gateway, alias, uuid.uuid4().hex, all_endpoints=True)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url).rstrip("/"), api_key=gateway.key)

        message: Final = client.messages.create(
            model=alias,
            max_tokens=64,
            messages=[{"role": "user", "content": "fix this python stack trace"}],
            extra_body={"litellm_session_id": session_id},
        )
        assert message.id, message

        row: Final = _decision_row(gateway, session_id, alias)
        assert row["status"] == "success", row
        assert _decision_of(row) == {
            "router_model_name": alias,
            "router_type": "semantic",
            "routed_model": route_model,
            "tier": route_model,
        }, row

        session: Final = _rollup_row(session_id, 1)
        assert session["router_name"] == alias and session["router_type"] == "semantic", session


@pytest.mark.covers("other.routing.auto_router.semantic_responses_turn_records_fallback_decision")
def test_a_responses_turn_records_a_semantic_fallback_decision(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        _passthrough_scenario(scenario)
        alias, default_model, route_model = _semantic_setup(scenario, embeddings=_nomatch_embeddings)
        session_id: Final = f"audit-h4-{uuid.uuid4().hex}"
        _ready(gateway, alias, uuid.uuid4().hex, all_endpoints=True)
        client: Final = openai.AsyncOpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key
        )

        async def turn() -> dict[str, JsonValue]:
            response = await client.responses.create(
                model=alias,
                input="fix this python stack trace",
                extra_body={"litellm_session_id": session_id},
            )
            return {"id": response.id}

        body: Final = asyncio.run(turn())
        assert str(body["id"]).startswith("resp_"), body

        row: Final = _decision_row(gateway, session_id, alias)
        assert row["status"] == "success", row
        assert _decision_of(row) == {
            "router_model_name": alias,
            "router_type": "semantic",
            "routed_model": default_model,
            "cause": "default_fallback",
        }, row

        session: Final = _rollup_row(session_id, 1)
        assert session["router_name"] == alias and session["router_type"] == "semantic", session


@pytest.mark.covers(
    "other.routing.auto_router.semantic_stream_turns_feed_benchmarks_and_session_views",
    "other.routing.auto_router.semantic_benchmarks_lists_the_router_idle_then_with_traffic",
    "other.routing.auto_router.semantic_session_endpoint_reports_the_recorded_turns",
)
def test_streamed_semantic_turns_feed_the_benchmarks_and_session_views(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        default_model: Final = scenario.model(model="openai/gpt-4o-mini")
        route_model: Final = scenario.model(
            model="openai/gpt-4o-mini", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
        alias: Final = _semantic_router(
            scenario,
            default_model=default_model,
            route_model=route_model,
            embedding_model=_match_embeddings(scenario),
        )
        session_id: Final = f"audit-h5-{uuid.uuid4().hex}"

        idle: Final = _benchmarks_group(gateway, alias)
        assert idle is not None, "configured semantic router is missing from /auto_router/benchmarks"
        assert idle["router_type"] == "semantic" and idle["tier_turns"] == {}, idle
        assert (idle["sessions"], idle["turns"]) == (0, 0) and float(str(idle["spend"])) == 0.0, idle

        marker: Final = uuid.uuid4().hex
        _ready(gateway, alias, marker)
        client: Final = openai.AsyncOpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key
        )

        async def turn(number: int) -> str:
            stream = await client.chat.completions.create(
                model=alias,
                messages=[{"role": "user", "content": f"fix this python stack trace {number}"}],
                stream=True,
                extra_body={"litellm_session_id": session_id, "stream_options": {"include_usage": True}},
            )
            identifiers: Final = {chunk.id async for chunk in stream if chunk.id}
            return next(iter(identifiers))

        for number in range(2):
            assert string_value(asyncio.run(turn(number)))

        session: Final = _rollup_row(session_id, 2)
        assert session["router_name"] == alias and session["router_type"] == "semantic", session
        assert session["tier_turns"] == {route_model: 2}, session
        per_turn: Final = PROMPT_TOKENS * 0.001 + COMPLETION_TOKENS * 0.002
        assert float(session["spend"]) == pytest.approx(2 * per_turn), session

        def measured_group() -> tuple[int, int, dict] | None:
            probes: Final = read_rows(
                'SELECT session_id, turns FROM "LiteLLM_AutoRouterSession" WHERE session_id LIKE %s',
                (f"audit-probe-{marker}-%",),
            )
            group: Final = _benchmarks_group(gateway, alias)
            if group is None:
                return None
            return (
                int(group["turns"]) - int(sum(int(row["turns"]) for row in probes)),
                int(group["sessions"]) - len(probes),
                group["tier_turns"],
            )

        group_turns, group_sessions, group_tiers = eventually(
            measured_group,
            lambda value: value is not None and value[0] == 2 and value[1] == 1,
            seconds=70,
        )
        group: Final = _benchmarks_group(gateway, alias)
        assert group is not None and group["router_type"] == "semantic", group
        assert float(str(group["spend"])) > 0, group
        assert group_tiers == {route_model: int(group["turns"])}, group
        assert group["saved_spend"] is None and group["baseline_spend"] is None and group["saved_pct"] is None, group

        view: Final = gateway.request("GET", "/auto_router/session", params={"session_id": session_id})
        assert view.status_code == 200, view.text
        detail: Final = object_value(view.json())
        assert detail["router_type"] == "semantic" and detail["turns"] == 2, detail


@pytest.mark.covers("other.routing.auto_router.semantic_broken_embedding_call_falls_back_and_is_counted")
def test_a_broken_embedding_call_falls_back_and_is_counted(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        default_model: Final = scenario.model(model="openai/gpt-4o-mini")
        route_model: Final = scenario.model(model="openai/gpt-4o-mini")
        handle: Final[ScenarioHandle] = _register_match_embeddings("audit-s1-emb")
        scenario.cleanups.callback(_forget_scenario, handle)
        embedding_model: Final = scenario.model(
            model="openai/text-embedding-3-small", mode="embedding", api_base=handle.api_base()
        )
        alias: Final = _semantic_router(
            scenario,
            default_model=default_model,
            route_model=route_model,
            embedding_model=embedding_model,
        )
        session_id: Final = f"audit-s1-{uuid.uuid4().hex}"

        _ready(gateway, alias, uuid.uuid4().hex)
        delete_scenario(handle)

        body: Final = _completed_chat(
            gateway, _chat_body(alias, session_id, text="debug a null pointer in this c program")
        )
        assert string_value(body["id"])

        row: Final = _decision_row(gateway, session_id, alias)
        assert row["status"] == "success" and row["model"] == "openai/gpt-4o-mini", row
        assert _decision_of(row) == {
            "router_model_name": alias,
            "router_type": "semantic",
            "routed_model": default_model,
            "cause": "default_fallback",
        }, row

        session: Final = _rollup_row(session_id, 1)
        assert session["router_name"] == alias and session["router_type"] == "semantic", session


@pytest.mark.covers("other.routing.auto_router.semantic_session_id_edge_values_never_break_the_turn")
def test_session_id_edge_values_never_break_a_semantic_turn(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias, default_model, route_model = _semantic_setup(scenario, embeddings=_match_embeddings)
        marker: Final = uuid.uuid4().hex
        _ready(gateway, alias, marker)

        attempts: Final = (
            ("int-session-id", _chat_body(alias, 7)),
            ("list-session-id", _chat_body(alias, ["a", "b"])),
            ("empty-session-id", _chat_body(alias, "")),
            ("long-session-id", _chat_body(alias, f"audit-s2-{marker}-" + "x" * 5000)),
            ("first-duplicate", _chat_body(alias, f"audit-s2-dup-{marker}")),
            ("second-duplicate", _chat_body(alias, f"audit-s2-dup-{marker}")),
            ("missing-session-id", {"model": alias, "messages": [{"role": "user", "content": "fix this stack trace"}]}),
        )
        for label, body in attempts:
            response: Final = gateway.request("POST", "/v1/chat/completions", body)
            assert response.status_code < 500, f"{label}: {response.status_code} {response.text}"
            if label in {"int-session-id", "list-session-id"}:
                assert response.status_code in (200, 400), f"{label}: {response.status_code} {response.text}"
            else:
                assert response.status_code == 200, f"{label}: {response.status_code} {response.text}"

        control: Final = _completed_chat(gateway, _chat_body(alias, f"audit-s2-control-{marker}"))
        assert string_value(control["id"])

        row: Final = _decision_row(gateway, f"audit-s2-control-{marker}", alias)
        assert row["status"] == "success", row
        assert read_rows('SELECT session_id FROM "LiteLLM_AutoRouterSession" WHERE session_id=%s', ("",)) == []


@pytest.mark.covers("other.routing.auto_router.semantic_unauthenticated_turn_records_nothing")
def test_an_unauthenticated_semantic_turn_records_nothing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias, default_model, route_model = _semantic_setup(scenario, embeddings=_match_embeddings)
        _ready(gateway, alias, uuid.uuid4().hex)
        session_id: Final = f"audit-s3-{uuid.uuid4().hex}"

        response: Final = gateway.client.post("/v1/chat/completions", json=_chat_body(alias, session_id))
        assert response.status_code == 401, response.text
        assert read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE session_id=%s', (session_id,)) == []
        assert read_rows('SELECT session_id FROM "LiteLLM_AutoRouterSession" WHERE session_id=%s', (session_id,)) == []


@pytest.mark.covers("other.routing.auto_router.semantic_failed_turn_is_not_counted_and_the_next_success_is")
def test_a_failed_semantic_turn_is_not_counted_but_the_next_success_is(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        default_model: Final = scenario.model(model="openai/gpt-4o-mini")
        wire_model: Final = f"audit-route-{uuid.uuid4().hex}"
        route_model: Final = scenario.model(model=f"openai/{wire_model}", model_info={"allowed_fails": 1000})
        alias: Final = _semantic_router(
            scenario,
            default_model=default_model,
            route_model=route_model,
            embedding_model=_match_embeddings(scenario),
        )
        _ready(gateway, alias, uuid.uuid4().hex)

        scripted: Final = httpx.post(
            f"{gateway.upstream_url}/__scripts/{wire_model}", json={"statuses": [500] * 10}, timeout=15
        )
        assert scripted.status_code == 200, scripted.text
        failed_session: Final = f"audit-s4-fail-{uuid.uuid4().hex}"
        failure: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(alias, failed_session, text="audit s4 failing turn"),
        )
        assert failure.status_code == 500 and "error" in failure.text.lower(), failure.text
        cleared: Final = httpx.delete(f"{gateway.upstream_url}/__scripts/{wire_model}", timeout=15)
        assert cleared.status_code == 200, cleared.text

        success_session: Final = f"audit-s4-ok-{uuid.uuid4().hex}"
        body: Final = _completed_chat(gateway, _chat_body(alias, success_session, text="audit s4 success turn"))
        assert string_value(body["id"])

        row: Final = _decision_row(gateway, success_session, alias)
        assert row["status"] == "success", row
        assert _decision_of(row)["tier"] == route_model, row
        session: Final = _rollup_row(success_session, 1)
        assert session["router_name"] == alias, session
        assert (
            read_rows('SELECT session_id FROM "LiteLLM_AutoRouterSession" WHERE session_id=%s', (failed_session,)) == []
        )


@pytest.mark.covers("other.routing.auto_router.semantic_rollup_counts_only_the_auto_routed_turn_in_a_shared_session")
def test_a_shared_session_counts_only_the_auto_routed_turn(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias, default_model, route_model = _semantic_setup(scenario, embeddings=_match_embeddings)
        plain_model: Final = scenario.model(model="openai/gpt-4o-mini")
        session_id: Final = f"audit-e1-{uuid.uuid4().hex}"

        _ready(gateway, alias, uuid.uuid4().hex)
        routed: Final = _completed_chat(gateway, _chat_body(alias, session_id))
        assert string_value(routed["id"])
        plain: Final = _completed_chat(gateway, _chat_body(plain_model, session_id))
        assert string_value(plain["id"])

        plain_row: Final = eventually(
            lambda: read_rows(
                "SELECT model_group, metadata->'routing_decision' AS decision"
                ' FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (plain["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )[0]
        assert plain_row["model_group"] == plain_model and plain_row["decision"] is None, plain_row

        session: Final = _rollup_row(session_id, 1)
        assert session["router_name"] == alias and session["router_type"] == "semantic", session


@pytest.mark.covers("other.routing.auto_router.complexity_router_decisions_are_unchanged_by_the_semantic_fix")
def test_complexity_router_decisions_are_unchanged_by_the_semantic_fix(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        target: Final = scenario.model(model="openai/gpt-4o-mini")
        alias: Final = scenario.model(
            model="auto_router/complexity_router",
            complexity_router_default_model=target,
            complexity_router_config={
                "tiers": {"SIMPLE": target, "MEDIUM": target, "COMPLEX": target, "REASONING": target}
            },
        )
        session_id: Final = f"audit-e2-{uuid.uuid4().hex}"
        _ready(gateway, alias, uuid.uuid4().hex)

        body: Final = _completed_chat(gateway, _chat_body(alias, session_id, "what is 2+2"))
        assert string_value(body["id"])

        row: Final = _decision_row(gateway, session_id, alias)
        decision: Final = _decision_of(row)
        assert decision["router_type"] == "complexity" and decision["router_model_name"] == alias, row
        assert decision["tier"] and decision["routed_model"] == target, row
        assert {
            "router_model_name",
            "router_type",
            "routed_model",
            "cause",
            "tier",
        } <= set(decision), row

        session: Final = _rollup_row(session_id, 1)
        assert session["router_name"] == alias and session["router_type"] == "complexity", session


@pytest.mark.covers("other.routing.auto_router.semantic_mixed_endpoint_burst_records_every_turn_exactly_once")
def test_a_mixed_endpoint_burst_records_every_turn_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        _passthrough_scenario(scenario)
        default_model: Final = scenario.model(model="openai/gpt-4o-mini")
        wire_model: Final = f"audit-route-{uuid.uuid4().hex}"
        route_model: Final = scenario.model(model=f"openai/{wire_model}", model_info={"allowed_fails": 1000})
        alias: Final = _semantic_router(
            scenario,
            default_model=default_model,
            route_model=route_model,
            embedding_model=_match_embeddings(scenario),
        )
        marker: Final = uuid.uuid4().hex
        _ready(gateway, alias, marker, all_endpoints=True)

        statuses: Final = [503] * 5 + [200] * 300
        scripted: Final = httpx.post(
            f"{gateway.upstream_url}/__scripts/{wire_model}", json={"statuses": statuses}, timeout=15
        )
        assert scripted.json() == {"configured": len(statuses)}, scripted.text

        proxy_url: Final = str(gateway.client.base_url).rstrip("/")

        async def burst() -> dict[str, int]:
            async with httpx.AsyncClient(timeout=60, trust_env=False) as client:

                async def call(index: int) -> tuple[str, int]:
                    session_id: Final = f"audit-c1-{marker}-{index}"
                    endpoint: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")[index % 3]
                    response: Final = await client.post(
                        f"{proxy_url}{endpoint}",
                        json=_burst_body(alias, session_id, index),
                        headers={"Authorization": f"Bearer {gateway.key}"},
                    )
                    await response.aread()
                    return session_id, response.status_code

                return dict(await asyncio.gather(*(call(index) for index in range(30))))

        outcomes: Final = asyncio.run(burst())
        ok_ids: Final = sorted(sid for sid, status in outcomes.items() if status == 200)
        failed_ids: Final = sorted(sid for sid, status in outcomes.items() if status != 200)
        assert len(ok_ids) == 25 and len(failed_ids) == 5, outcomes

        placeholders: Final = ",".join(["%s"] * 30)
        decision_rows: Final = eventually(
            lambda: read_rows(
                "SELECT session_id, status, metadata->'routing_decision' AS decision"
                f' FROM "LiteLLM_SpendLogs" WHERE session_id IN ({placeholders}) AND model_group=%s'
                " AND status='success' AND metadata->>'routing_decision' IS NOT NULL",
                (*ok_ids, *failed_ids, alias),
            ),
            lambda values: len(values) == len(ok_ids),
            seconds=70,
        )
        by_session: Final = {row["session_id"]: row for row in decision_rows}
        assert set(by_session) == set(ok_ids), (sorted(by_session), ok_ids)
        for session_id in ok_ids:
            row: Final = by_session[session_id]
            assert row["status"] == "success" and _decision_of(row)["tier"] == route_model, row

        rollup_rows: Final = eventually(
            lambda: read_rows(
                f'SELECT session_id, turns FROM "LiteLLM_AutoRouterSession" WHERE session_id IN ({placeholders})',
                tuple(ok_ids + failed_ids),
            ),
            lambda values: len(values) == 25,
            seconds=70,
        )
        assert {row["session_id"] for row in rollup_rows} == set(ok_ids), rollup_rows
        assert all(row["turns"] == 1 for row in rollup_rows), rollup_rows

        def warmed_group() -> int:
            warmup_turns: Final = sum(
                int(row["turns"])
                for row in read_rows(
                    'SELECT turns FROM "LiteLLM_AutoRouterSession" WHERE session_id LIKE %s',
                    (f"audit-probe-{marker}-%",),
                )
            )
            group: Final = _benchmarks_group(gateway, alias)
            return int(group["turns"]) - warmup_turns if group is not None else -1

        turns: Final = eventually(warmed_group, lambda value: value == 25, seconds=70)
        assert turns == 25, turns
