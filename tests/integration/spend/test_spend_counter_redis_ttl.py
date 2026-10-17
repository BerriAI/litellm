import asyncio
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.spend_counters import (
    counter_ttl,
    counter_value,
    key_counter,
    outlives,
    rig_cache,
)
from pydantic import JsonValue
from redis import Redis

RATES: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
ONE_CALL_COST: Final = 0.06
ANTHROPIC_HEADERS: Final = {"anthropic-version": "2023-06-01"}


@dataclass(frozen=True, slots=True)
class Entities:
    model: str
    key: str
    team: str
    user: str
    organization: str
    end_user: str
    tag: str

    def counters(self) -> Mapping[str, str]:
        return {
            "key": key_counter(self.key),
            "team": f"spend:team:{self.team}",
            "team_member": f"spend:team_member:{self.user}:{self.team}",
            "user": f"spend:user:{self.user}",
            "end_user": f"spend:end_user:{self.end_user}",
            "tag": f"spend:tag:{self.tag}",
            "org": f"spend:org:{self.organization}",
        }


def _entities(gateway: Gateway, scenario: Scenario) -> Entities:
    model: Final = scenario.model(**RATES)
    models: Final = [model]
    organization: Final = scenario.organization(max_budget=50.0)
    team: Final = scenario.team(models=models, max_budget=20.0, organization_id=organization)
    user: Final = scenario.user(max_budget=20.0)
    added: Final = gateway.request(
        "POST",
        "/team/member_add",
        {"team_id": team, "member": {"user_id": user, "role": "user"}, "max_budget_in_team": 20.0},
    )
    assert added.status_code == 200, added.text
    tag: Final = f"integration-tag-{uuid.uuid4().hex}"
    key: Final = scenario.key(team_id=team, user_id=user, models=models, max_budget=5.0, tags=[tag])
    end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
    return Entities(model, key, team, user, organization, end_user, tag)


def _base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _chat_openai_sync(gateway: Gateway, entities: Entities, text: str) -> str:
    with openai.OpenAI(base_url=f"{_base(gateway)}/v1", api_key=entities.key, max_retries=0, timeout=60) as client:
        response: Final = client.chat.completions.create(
            model=entities.model, messages=[{"role": "user", "content": text}], user=entities.end_user
        )
    assert response.usage is not None and response.usage.total_tokens == 40, response
    return response.id


def _chat_openai_async_stream(gateway: Gateway, entities: Entities, text: str) -> str:
    async def run() -> str:
        async with openai.AsyncOpenAI(
            base_url=f"{_base(gateway)}/v1", api_key=entities.key, max_retries=0, timeout=60
        ) as client:
            stream: Final = await client.chat.completions.create(
                model=entities.model,
                messages=[{"role": "user", "content": text}],
                user=entities.end_user,
                stream=True,
                stream_options={"include_usage": True},
            )
            ids: Final = [chunk.id async for chunk in stream]
        assert len(set(ids)) == 1, ids
        return ids[0]

    return asyncio.run(run())


def _chat_httpx_stream(gateway: Gateway, entities: Entities, text: str) -> str:
    with gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": entities.model,
            "messages": [{"role": "user", "content": text}],
            "user": entities.end_user,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"Authorization": f"Bearer {entities.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        frames: Final = [line.removeprefix("data: ") for line in response.iter_lines() if line.startswith("data: ")]
    ids: Final = {json.loads(frame)["id"] for frame in frames if frame != "[DONE]"}
    assert len(ids) == 1, frames
    return ids.pop()


def _messages_anthropic_sync(gateway: Gateway, entities: Entities, text: str) -> str:
    with anthropic.Anthropic(base_url=_base(gateway), api_key=entities.key, max_retries=0, timeout=60) as client:
        message: Final = client.messages.create(
            model=entities.model,
            max_tokens=32,
            messages=[{"role": "user", "content": text}],
            metadata={"user_id": entities.end_user},
        )
    assert message.usage.output_tokens > 0, message
    return message.id


def _messages_anthropic_async_stream(gateway: Gateway, entities: Entities, text: str) -> str:
    async def run() -> str:
        async with anthropic.AsyncAnthropic(
            base_url=_base(gateway), api_key=entities.key, max_retries=0, timeout=60
        ) as client:
            async with client.messages.stream(
                model=entities.model,
                max_tokens=32,
                messages=[{"role": "user", "content": text}],
                metadata={"user_id": entities.end_user},
            ) as stream:
                async for _ in stream.text_stream:
                    pass
                final: Final = await stream.get_final_message()
        assert final.usage.output_tokens > 0, final
        return final.id

    return asyncio.run(run())


def _messages_httpx(gateway: Gateway, entities: Entities, text: str) -> str:
    response: Final = gateway.request(
        "POST",
        "/v1/messages",
        {
            "model": entities.model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": text}],
            "metadata": {"user_id": entities.end_user},
        },
        key=entities.key,
        headers=ANTHROPIC_HEADERS,
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _responses_httpx(gateway: Gateway, entities: Entities, text: str) -> str:
    response: Final = gateway.request(
        "POST",
        "/v1/responses",
        {"model": entities.model, "input": text, "user": entities.end_user},
        key=entities.key,
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _responses_httpx_stream(gateway: Gateway, entities: Entities, text: str) -> str:
    with gateway.client.stream(
        "POST",
        "/v1/responses",
        json={"model": entities.model, "input": text, "user": entities.end_user, "stream": True},
        headers={"Authorization": f"Bearer {entities.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        lines: Final = [line for line in response.iter_lines() if line.startswith("data: ")]
    events: Final = [json.loads(line.removeprefix("data: ")) for line in lines if line != "data: [DONE]"]
    completed: Final = [event for event in events if event.get("type") == "response.completed"]
    assert len(completed) == 1, [event.get("type") for event in events]
    return str(completed[0]["response"]["id"])


@dataclass(frozen=True, slots=True)
class Caller:
    call: Callable[[Gateway, Entities, str], str]
    upstream_path: str
    exact_cost: float | None


CALLERS: Final[Mapping[str, Caller]] = {
    "chat-openai-sync": Caller(_chat_openai_sync, "/v1/chat/completions", ONE_CALL_COST),
    "chat-openai-async-stream": Caller(_chat_openai_async_stream, "/v1/chat/completions", ONE_CALL_COST),
    "chat-httpx-stream": Caller(_chat_httpx_stream, "/v1/chat/completions", ONE_CALL_COST),
    "messages-anthropic-sync": Caller(_messages_anthropic_sync, "/v1/responses", ONE_CALL_COST),
    "messages-anthropic-async-stream": Caller(_messages_anthropic_async_stream, "/v1/responses", ONE_CALL_COST),
    "messages-httpx": Caller(_messages_httpx, "/v1/responses", ONE_CALL_COST),
    "responses-httpx": Caller(_responses_httpx, "/v1/responses", ONE_CALL_COST),
    "responses-httpx-stream": Caller(_responses_httpx_stream, "/v1/responses", ONE_CALL_COST),
}


def _spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, spend, cache_hit FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"',
        (sha256(key.encode()).hexdigest(),),
    )


def _one_spend_row(key: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: _spend_rows(key), lambda values: len(values) == 1, seconds=70)
    return rows[0]


def _observed(cache: Redis, counters: Mapping[str, str]) -> Mapping[str, tuple[float | None, int]]:
    return {name: (counter_value(cache, counter), counter_ttl(cache, counter)) for name, counter in counters.items()}


def _assert_every_counter_outlives(
    cache: Redis, counters: Mapping[str, str], spend: float
) -> Mapping[str, tuple[float | None, int]]:
    observed: Final = eventually(
        lambda: _observed(cache, counters),
        lambda states: all(
            value is not None and value == pytest.approx(spend, abs=1e-9) for value, _ in states.values()
        ),
        seconds=20,
    )
    for name, (_, ttl) in observed.items():
        assert outlives(ttl), (name, observed)
    return observed


@pytest.mark.parametrize("caller", sorted(CALLERS))
def test_every_counter_scope_outlives_the_batch_write(gateway: Gateway, caller: str) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        rig_cache() as cache,
    ):
        entities: Final = _entities(gateway, scenario)
        text: Final = f"counter ttl {caller} {uuid.uuid4().hex}"
        upstream.get("/__observations").raise_for_status()
        response_id: Final = CALLERS[caller].call(gateway, entities, text)
        received: Final = upstream.get("/__observations").json()["requests"]
        paths: Final = [request["path"] for request in received if text in json.dumps(request["body"])]
        assert len(paths) == 1 and paths[0].endswith(CALLERS[caller].upstream_path), received
        row: Final = _one_spend_row(entities.key)
        spend: Final = float(str(row["spend"]))
        assert spend == pytest.approx(CALLERS[caller].exact_cost or spend) and spend > 0, row
        assert response_id, row
        _assert_every_counter_outlives(cache, entities.counters(), spend)


def test_a_response_cache_hit_keeps_the_counter_and_its_ttl(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        rig_cache() as cache,
    ):
        entities: Final = _entities(gateway, scenario)
        body: Final = {
            "model": entities.model,
            "messages": [{"role": "user", "content": f"cache twin {uuid.uuid4().hex}"}],
            "user": entities.end_user,
        }
        first: Final = gateway.request("POST", "/v1/chat/completions", body, key=entities.key)
        assert first.status_code == 200, first.text
        row: Final = _one_spend_row(entities.key)
        spend: Final = float(str(row["spend"]))
        before: Final = _assert_every_counter_outlives(cache, entities.counters(), spend)
        upstream.get("/__observations").raise_for_status()
        second: Final = gateway.request("POST", "/v1/chat/completions", body, key=entities.key)
        assert second.status_code == 200, second.text
        assert second.json()["id"] == first.json()["id"], (first.text, second.text)
        assert upstream.get("/__observations").json()["requests"] == []
        rows: Final = eventually(lambda: _spend_rows(entities.key), lambda values: len(values) == 2, seconds=70)
        assert str(rows[1]["cache_hit"]) == "True" and float(str(rows[1]["spend"])) == 0.0, rows
        after: Final = _assert_every_counter_outlives(cache, entities.counters(), spend)
        assert {name: value for name, (value, _) in after.items()} == pytest.approx(
            {name: value for name, (value, _) in before.items()}, abs=1e-9
        ), (before, after)


def _exhausted_key(gateway: Gateway, scenario: Scenario) -> tuple[str, str]:
    model: Final = scenario.model(**RATES)
    key: Final = scenario.key(models=[model], max_budget=ONE_CALL_COST)
    served: Final = gateway.chat(model, key=key, text=f"exhaust {uuid.uuid4().hex}")
    assert served["usage"]["total_tokens"] == 40, served
    eventually(
        lambda: read_rows(
            'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
        ),
        lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= ONE_CALL_COST,
        seconds=70,
    )
    return model, key


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"after {uuid.uuid4().hex}"}]},
        key=key,
    )


def _assert_denied(response: httpx.Response) -> None:
    assert response.status_code == 422, response.text
    assert response.json()["error"]["type"] == "budget_exceeded", response.text


def test_a_stale_low_counter_is_repaired_with_the_long_ttl(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(
            models=[model], budget_limits=[{"max_budget": ONE_CALL_COST, "budget_duration": "1d"}]
        )
        counter: Final = f"{key_counter(key)}:window:1d"
        assert gateway.chat(model, key=key, text=f"exhaust window {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        _one_spend_row(key)
        assert counter_value(cache, counter) == pytest.approx(ONE_CALL_COST)

        served: Final[list[int]] = []

        def stale_then_request() -> int:
            assert cache.set(counter, "0.01")
            code: Final = _chat(gateway, model, key).status_code
            served.append(code)
            return code

        assert eventually(stale_then_request, lambda code: code == 422, seconds=30) == 422
        assert set(served) == {200, 422}, served
        repaired: Final = (counter_value(cache, counter), counter_ttl(cache, counter))
        floor: Final = repaired[0]
        assert floor is not None and ONE_CALL_COST - 1e-9 <= floor <= served.count(200) * ONE_CALL_COST + 1e-9, (
            repaired,
            served,
        )
        assert outlives(repaired[1]), repaired


def test_a_lost_counter_is_reseeded_from_the_db_with_the_long_ttl(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model, key = _exhausted_key(gateway, scenario)
        counter: Final = key_counter(key)
        assert cache.delete(counter) == 1
        _assert_denied(_chat(gateway, model, key))
        reseeded: Final = eventually(
            lambda: (counter_value(cache, counter), counter_ttl(cache, counter)),
            lambda state: state[0] is not None and state[0] >= ONE_CALL_COST and outlives(state[1]),
            seconds=15,
        )
        assert reseeded[0] == pytest.approx(ONE_CALL_COST), reseeded


def test_key_update_spend_sets_the_counter_with_the_long_ttl(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        counter: Final = key_counter(key)
        assert gateway.chat(model, key=key, text=f"before update {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        _one_spend_row(key)
        updated: Final = gateway.request("POST", "/key/update", {"key": key, "spend": 1.25})
        assert updated.status_code == 200, updated.text
        assert (counter_value(cache, counter), outlives(counter_ttl(cache, counter))) == (1.25, True), updated.text
        served: Final = _chat(gateway, model, key)
        assert served.status_code == 200, served.text
        eventually(
            lambda: counter_value(cache, counter),
            lambda value: value is not None and value == pytest.approx(1.25 + ONE_CALL_COST),
            seconds=15,
        )
        assert outlives(counter_ttl(cache, counter))


def test_key_reset_spend_sets_the_counter_with_the_long_ttl(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model, key = _exhausted_key(gateway, scenario)
        counter: Final = key_counter(key)
        _assert_denied(_chat(gateway, model, key))
        reset: Final = gateway.request("POST", f"/key/{key}/reset_spend", {"reset_to": 0.0})
        assert reset.status_code == 200, reset.text
        assert reset.json()["key_hash"] == sha256(key.encode()).hexdigest(), reset.text
        assert (counter_value(cache, counter), outlives(counter_ttl(cache, counter))) == (0.0, True), reset.text
        served: Final = _chat(gateway, model, key)
        assert served.status_code == 200, served.text


def test_team_member_reset_spend_sets_the_counter_with_the_long_ttl(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(**RATES)
        team: Final = scenario.team(models=[model])
        user: Final = scenario.user()
        added: Final = gateway.request(
            "POST",
            "/team/member_add",
            {"team_id": team, "member": {"user_id": user, "role": "user"}, "max_budget_in_team": ONE_CALL_COST},
        )
        assert added.status_code == 200, added.text
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        counter: Final = f"spend:team_member:{user}:{team}"
        assert gateway.chat(model, key=key, text=f"member {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s', (team, user)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= ONE_CALL_COST,
            seconds=70,
        )
        _assert_denied(_chat(gateway, model, key))
        reset: Final = gateway.request("POST", f"/team/{team}/member/{user}/reset_spend", {"reset_to": 0.0})
        assert reset.status_code == 200, reset.text
        assert reset.json()["previous_spend"] == pytest.approx(ONE_CALL_COST), reset.text
        assert (counter_value(cache, counter), outlives(counter_ttl(cache, counter))) == (0.0, True), reset.text
        served: Final = _chat(gateway, model, key)
        assert served.status_code == 200, served.text


def test_malformed_spend_writes_are_rejected_and_leave_the_counter_alone(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        bystander: Final = scenario.key(models=[model], max_budget=5.0)
        counter: Final = key_counter(key)
        assert gateway.chat(model, key=key, text=f"before bad writes {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        _one_spend_row(key)
        before: Final = (counter_value(cache, counter), counter_ttl(cache, counter))
        assert before[0] == pytest.approx(ONE_CALL_COST) and outlives(before[1]), before
        rejected: Final = tuple(
            gateway.request(*attempt)
            for attempt in (
                ("POST", "/key/update", {"key": key, "spend": "a" * 5000}),
                ("POST", "/key/update", {"key": key, "spend": [1.0]}),
                ("POST", "/key/update", {"key": key, "spend": {"spend": 1.0}}),
                ("POST", f"/key/{key}/reset_spend", {"reset_to": "abc"}),
                ("POST", f"/key/{key}/reset_spend", {"reset_to": True}),
                ("POST", f"/key/{key}/reset_spend", {"reset_to": [0.0]}),
                ("POST", f"/key/{key}/reset_spend", {"reset_to": ONE_CALL_COST * 10}),
            )
        )
        for response in rejected:
            assert response.status_code in (400, 422), response.text
            assert "error" in response.json() or "detail" in response.json(), response.text
        assert (counter_value(cache, counter), counter_ttl(cache, counter) <= before[1]) == (before[0], True)
        assert outlives(counter_ttl(cache, counter))
        served: Final = _chat(gateway, model, bystander)
        assert served.status_code == 200, served.text


def _landed_chat(gateway: Gateway, model: str, key: str, index: int) -> str:
    served: Final = gateway.chat(model, key=key, text=f"repeat {index} {uuid.uuid4().hex}")
    eventually(lambda: _spend_rows(key), lambda rows: len(rows) == index + 1, seconds=70)
    return str(served["id"])


def test_repeated_requests_log_once_each_and_the_counter_matches_their_sum(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, rig_cache() as cache:
        model: Final = scenario.model(**RATES)
        key: Final = scenario.key(models=[model], max_budget=5.0)
        counter: Final = key_counter(key)
        ids: Final = tuple(_landed_chat(gateway, model, key, index) for index in range(3))
        assert len(set(ids)) == 3, ids
        rows: Final = _spend_rows(key)
        assert len(rows) == 3, rows
        assert sorted(str(row["request_id"]) for row in rows) == sorted(ids), rows
        total: Final = sum(float(str(row["spend"])) for row in rows)
        assert total == pytest.approx(3 * ONE_CALL_COST), rows
        eventually(
            lambda: counter_value(cache, counter),
            lambda value: value is not None and value == pytest.approx(total),
            seconds=15,
        )
        assert outlives(counter_ttl(cache, counter))
