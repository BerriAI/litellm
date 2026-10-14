import asyncio
import calendar
import json
import math
import os
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    held,
    object_value,
)
from integration._support.database import write_rows
from integration._support.process import graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.upstream import delete_scenario, register_scenario
from integration.authorization._hidden_alias_budget import (
    BUDGET_EXCEEDED,
    CHAT_REPLY,
    anthropic_client,
    async_anthropic_client,
    async_openai_client,
    base_url,
    fresh_chat,
    landed_all_once,
    landed_once,
    openai_client,
    settle,
    spend_marker,
    upstream_hits,
    upstream_requests,
)
from integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse
from pydantic import JsonValue, TypeAdapter
from redis import Redis

pytestmark: Final = pytest.mark.timeout(240)

DAY: Final = 86400
HOUR: Final = 3600
REPLY: Final = "window reply"
PRICES: Final[Mapping[str, JsonValue]] = {"input_cost_per_token": 0.0001, "output_cost_per_token": 0.0002}
OWNED_PROXY_SECONDS: Final = 2 * graceful_stop_seconds() + 120
REDIS_TEXT: Final = TypeAdapter(str | None)
REDIS_INT: Final = TypeAdapter(int)

_RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "id": "msg_$UNIQUE_ID",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": REPLY, "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40},
}
_RESPONSE_EVENTS: Final[tuple[Mapping[str, JsonValue], ...]] = (
    {
        "type": "response.created",
        "sequence_number": 0,
        "response": {**_RESPONSE, "status": "in_progress", "output": []},
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_$UNIQUE_ID",
        "output_index": 0,
        "content_index": 0,
        "delta": REPLY,
    },
    {"type": "response.completed", "sequence_number": 2, "response": _RESPONSE},
)


@dataclass(frozen=True, slots=True)
class BudgetRig:
    gateway: Gateway
    peer: Gateway
    redis: Redis
    chat: str
    responses: str
    responses_stream: str

    @property
    def replicas(self) -> tuple[Gateway, Gateway]:
        return (self.gateway, self.peer)


@dataclass(frozen=True, slots=True)
class Window:
    start_key: str
    counter_key: str


@dataclass(frozen=True, slots=True)
class MonthlyWindow:
    months: int
    started: datetime

    @property
    def period(self) -> str:
        return f"{self.months}mo"

    @property
    def ends(self) -> datetime:
        return months_later(self.started, self.months)

    def differs_from_a_window_opened_at(self, now: datetime) -> bool:
        return self.ends - self.started != months_later(now, self.months) - now

    def open_at(self, now: datetime) -> bool:
        return self.ends > now + timedelta(hours=1)


def months_later(moment: datetime, months: int) -> datetime:
    year, month_index = divmod(moment.year * 12 + moment.month - 1 + months, 12)
    return moment.replace(
        year=year, month=month_index + 1, day=min(moment.day, calendar.monthrange(year, month_index + 1)[1])
    )


def monthly_window_in_another_span() -> MonthlyWindow:
    now: Final = datetime.fromtimestamp(math.floor(time.time()), tz=timezone.utc)
    candidates: Final = (
        MonthlyWindow(months, now - timedelta(days=days_back)) for months, days_back in product((1, 2, 3), range(1, 93))
    )
    return next(window for window in candidates if window.differs_from_a_window_opened_at(now) and window.open_at(now))


def hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def key_window(key: str, model: str, period: str) -> Window:
    return Window(
        start_key=f"virtual_key_budget_start_time:{hashed(key)}:{model}:{period}",
        counter_key=f"virtual_key_spend:{hashed(key)}:{model}:{period}",
    )


def budget(model: str, period: str, limit: float = 100) -> dict[str, JsonValue]:
    return {model: {"budget_limit": limit, "time_period": period}}


def unused(limit: float, period: str) -> dict[str, JsonValue]:
    return {"current_spend": 0.0, "budget_limit": limit, "time_period": period, "budget_reset_at": None}


def reset_at(started_at: float, seconds: float) -> str:
    return datetime.fromtimestamp(started_at + seconds, tz=timezone.utc).isoformat()


def redis_text(redis: Redis, key: str) -> str | None:
    return REDIS_TEXT.validate_python(redis.get(key))


def redis_float(redis: Redis, key: str) -> float:
    return float(str(redis_text(redis, key)))


def opened_at(redis: Redis, window: Window) -> float:
    started: Final = eventually(
        lambda: (redis_text(redis, window.start_key), redis_text(redis, window.counter_key)),
        lambda found: found[0] is not None and found[1] is not None and float(found[1]) > 0,
        seconds=30,
    )
    return float(str(started[0]))


def expires_at(redis: Redis, key: str) -> float:
    remaining: Final = REDIS_INT.validate_python(redis.pttl(key))
    assert remaining > 0, f"{key} has no expiry: {remaining}"
    return time.time() + remaining / 1000


def key_usage(candidate: Gateway, key: str) -> Mapping[str, JsonValue]:
    return object_value(object_value(candidate.get("/key/info", {"key": key})["info"])["model_max_budget_usage"])


def model_usage(candidate: Gateway, key: str, model: str) -> Mapping[str, JsonValue]:
    return object_value(key_usage(candidate, key)[model])


def resets_on(candidates: tuple[Gateway, ...], key: str, model: str) -> tuple[JsonValue, ...]:
    return tuple(model_usage(candidate, key, model)["budget_reset_at"] for candidate in candidates)


def reported_everywhere(candidates: tuple[Gateway, ...], key: str, model: str, expected: str | None) -> None:
    eventually(
        lambda: resets_on(candidates, key, model),
        lambda found: found == (expected,) * len(candidates),
        seconds=30,
    )


def assert_spend_matches_counter(rig: BudgetRig, key: str, model: str, window: Window) -> None:
    eventually(
        lambda: (
            tuple(model_usage(candidate, key, model)["current_spend"] for candidate in rig.replicas),
            round(redis_float(rig.redis, window.counter_key), 4),
        ),
        lambda found: found[0] == (found[1],) * len(rig.replicas) and found[1] > 0,
        seconds=30,
    )


def assert_daily_window_reported(rig: BudgetRig, key: str, model: str) -> float:
    window: Final = key_window(key, model, "1d")
    started_at: Final = opened_at(rig.redis, window)
    reported_everywhere(rig.replicas, key, model, reset_at(started_at, DAY))
    assert_spend_matches_counter(rig, key, model, window)
    return started_at


def assert_window_kept(rig: BudgetRig, key: str, model: str, started_at: float) -> None:
    window: Final = key_window(key, model, "1d")
    assert redis_text(rig.redis, window.start_key) == json.dumps(started_at), model
    reported_everywhere(rig.replicas, key, model, reset_at(started_at, DAY))
    assert_spend_matches_counter(rig, key, model, window)


def assert_charged_once(rig: BudgetRig, key: str, marker: str, response_id: str | None) -> None:
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
    row: Final = landed_once(key, marker)
    assert response_id is None or row["request_id"] == response_id, row
    assert float(str(row["spend"])) > 0, row


def responses_group(scenario: Scenario, gateway: Gateway, name: str, response: JsonResponse | SseResponse) -> str:
    handle: Final = register_scenario(name, response, control_url=gateway.upstream_url)
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario.model(api_base=f"{handle.api_base()}/v1", **PRICES)


def fresh_responses(candidate: Gateway, model: str, key: str, marker: str, stream: bool) -> httpx.Response:
    return httpx.post(
        f"{base_url(candidate)}/v1/responses",
        json={"model": model, "input": marker, "stream": stream},
        headers={"Authorization": f"Bearer {key}", **spend_marker(marker)},
        timeout=60,
        trust_env=False,
    )


def settle_served_on(rig: BudgetRig, candidate: Gateway) -> None:
    admin: Final = rig.gateway.key
    settle(lambda marker: fresh_chat(candidate, rig.chat, admin, marker), 200, burst=8)
    settle(lambda marker: fresh_responses(candidate, rig.responses, admin, marker, False), 200, burst=8)
    settle(lambda marker: fresh_responses(candidate, rig.responses_stream, admin, marker, True), 200, burst=8)


@pytest.fixture(scope="module")
def rig() -> Iterator[BudgetRig]:
    suffix: Final = uuid.uuid4().hex[:12]
    with (
        gateway_from_environment() as gateway,
        httpx.Client(base_url=os.environ["INTEGRATION_PEER_URL"], timeout=15, trust_env=False) as peer_client,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as redis,
        gateway.scenario() as scenario,
    ):
        built: Final = BudgetRig(
            gateway=gateway,
            peer=Gateway(peer_client, gateway.key, gateway.upstream_url),
            redis=redis,
            chat=scenario.model(**PRICES),
            responses=responses_group(
                scenario,
                gateway,
                f"reset-at-json-{suffix}",
                JsonResponse(content_type="application/json", body=_RESPONSE),
            ),
            responses_stream=responses_group(
                scenario,
                gateway,
                f"reset-at-sse-{suffix}",
                SseResponse(
                    content_type="text/event-stream",
                    frames=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}" for event in _RESPONSE_EVENTS),
                ),
            ),
        )
        for candidate in built.replicas:
            settle_served_on(built, candidate)
        yield built


def test_reset_is_null_on_both_replicas_until_the_first_charge(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        assert [model_usage(candidate, key, rig.chat) for candidate in rig.replicas] == [unused(100.0, "1d")] * 2


def test_openai_chat_charge_reports_the_reset_on_key_info_and_v2_key_info(rig: BudgetRig) -> None:
    marker: Final = "chat-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        with openai_client(rig.gateway, key) as client:
            completion: Final = client.chat.completions.create(
                model=rig.chat, messages=[{"role": "user", "content": marker}], extra_headers=spend_marker(marker)
            )
        assert completion.choices[0].message.content == CHAT_REPLY, completion
        assert_charged_once(rig, key, marker, completion.id)
        started_at: Final = assert_daily_window_reported(rig, key, rig.chat)
        listed: Final = rig.peer.post("/v2/key/info", {"keys": [key]})["info"]
        assert isinstance(listed, list) and len(listed) == 1, listed
        entry: Final = object_value(object_value(object_value(listed[0])["model_max_budget_usage"])[rig.chat])
        assert entry["budget_reset_at"] == reset_at(started_at, DAY), entry


async def test_streamed_openai_chat_charge_reports_the_reset(rig: BudgetRig) -> None:
    marker: Final = "chat-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        async with async_openai_client(rig.gateway, key) as client:
            stream: Final = await client.chat.completions.create(
                model=rig.chat,
                messages=[{"role": "user", "content": marker}],
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=spend_marker(marker),
            )
            chunks: Final = tuple([chunk async for chunk in stream])
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == CHAT_REPLY
        assert_charged_once(rig, key, marker, chunks[0].id)
        assert_daily_window_reported(rig, key, rig.chat)


def test_anthropic_messages_charge_reports_the_reset(rig: BudgetRig) -> None:
    marker: Final = "messages-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.responses, "1d"))
        with anthropic_client(rig.gateway, key) as client:
            message: Final = client.messages.create(
                model=rig.responses,
                max_tokens=16,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            )
        assert [block.text for block in message.content if block.type == "text"] == [REPLY], message
        assert_charged_once(rig, key, marker, message.id)
        assert_daily_window_reported(rig, key, rig.responses)


async def test_streamed_anthropic_messages_charge_reports_the_reset(rig: BudgetRig) -> None:
    marker: Final = "messages-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.responses_stream, "1d"))
        async with (
            async_anthropic_client(rig.gateway, key) as client,
            client.messages.stream(
                model=rig.responses_stream,
                max_tokens=16,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            ) as stream,
        ):
            text: Final = "".join([piece async for piece in stream.text_stream])
            final: Final = await stream.get_final_message()
        assert text == REPLY, final
        assert_charged_once(rig, key, marker, final.id)
        assert_daily_window_reported(rig, key, rig.responses_stream)


def test_openai_responses_charge_reports_the_reset(rig: BudgetRig) -> None:
    marker: Final = "responses-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.responses, "1d"))
        with openai_client(rig.gateway, key) as client:
            response: Final = client.responses.create(
                model=rig.responses, input=marker, extra_headers=spend_marker(marker)
            )
        assert response.output_text == REPLY, response
        assert_charged_once(rig, key, marker, response.id)
        assert_daily_window_reported(rig, key, rig.responses)


async def test_streamed_openai_responses_charge_reports_the_reset(rig: BudgetRig) -> None:
    marker: Final = "responses-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.responses_stream, "1d"))
        async with async_openai_client(rig.gateway, key) as client:
            stream: Final = await client.responses.create(
                model=rig.responses_stream, input=marker, stream=True, extra_headers=spend_marker(marker)
            )
            events: Final = tuple([event async for event in stream])
        assert events[-1].type == "response.completed", events
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == REPLY
        assert_charged_once(rig, key, marker, None)
        assert_daily_window_reported(rig, key, rig.responses_stream)


def test_raw_chat_on_the_peer_reports_the_reset_on_both_replicas(rig: BudgetRig) -> None:
    marker: Final = "raw-peer-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        response: Final = fresh_chat(rig.peer, rig.chat, key, marker)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == CHAT_REPLY, response.text
        assert_charged_once(rig, key, marker, response.json()["id"])
        assert_daily_window_reported(rig, key, rig.chat)


def user_resets(rig: BudgetRig, user: str) -> tuple[JsonValue, ...]:
    def entry(usage: JsonValue) -> JsonValue:
        return object_value(object_value(usage)[rig.chat])["budget_reset_at"]

    return (
        *(
            entry(object_value(candidate.get("/user/info", {"user_id": user})["user_info"])["model_max_budget_usage"])
            for candidate in rig.replicas
        ),
        entry(rig.peer.get("/v2/user/info", {"user_id": user})["model_max_budget_usage"]),
    )


def test_user_info_and_v2_user_info_report_the_user_window_reset(rig: BudgetRig) -> None:
    marker: Final = "user-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        user: Final = scenario.user(model_max_budget=budget(rig.chat, "1d"))
        key: Final = scenario.key(user_id=user)
        response: Final = fresh_chat(rig.gateway, rig.chat, key, marker)
        assert response.status_code == 200, response.text
        assert_charged_once(rig, key, marker, response.json()["id"])
        started_at: Final = opened_at(
            rig.redis,
            Window(
                start_key=f"user_model_budget_start_time:{user}:{rig.chat}:1d",
                counter_key=f"user_model_spend:{user}:{rig.chat}:1d",
            ),
        )
        expected: Final = reset_at(started_at, DAY)
        eventually(lambda: user_resets(rig, user), lambda found: found == (expected,) * 3, seconds=30)


def team_resets(rig: BudgetRig, team: str) -> tuple[JsonValue, ...]:
    return tuple(
        object_value(
            object_value(
                object_value(candidate.get("/team/info", {"team_id": team})["team_info"])["model_max_budget_usage"]
            )[rig.chat]
        )["budget_reset_at"]
        for candidate in rig.replicas
    )


def test_team_info_reports_the_team_window_reset(rig: BudgetRig) -> None:
    marker: Final = "team-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(model_max_budget=budget(rig.chat, "1h"))
        key: Final = scenario.key(team_id=team)
        response: Final = fresh_chat(rig.gateway, rig.chat, key, marker)
        assert response.status_code == 200, response.text
        assert_charged_once(rig, key, marker, response.json()["id"])
        started_at: Final = opened_at(
            rig.redis,
            Window(
                start_key=f"team_model_budget_start_time:{team}:{rig.chat}:1h",
                counter_key=f"team_model_spend:{team}:{rig.chat}:1h",
            ),
        )
        expected: Final = reset_at(started_at, HOUR)
        eventually(lambda: team_resets(rig, team), lambda found: found == (expected,) * 2, seconds=30)


@dataclass(frozen=True, slots=True)
class Attempt:
    at: float
    status: int


def attempt(candidate: Gateway, model: str, key: str) -> Attempt:
    response: Final = fresh_chat(candidate, model, key, "after-reset-" + uuid.uuid4().hex)
    return Attempt(at=time.time(), status=response.status_code)


def test_a_used_up_minute_budget_is_refused_until_the_reported_reset(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1m", limit=0.000000001))
        first: Final = fresh_chat(rig.gateway, rig.chat, key, "spend-" + uuid.uuid4().hex)
        assert first.status_code == 200, first.text
        started_at: Final = opened_at(rig.redis, key_window(key, rig.chat, "1m"))
        resets: Final = started_at + 60
        reported_everywhere(rig.replicas, key, rig.chat, reset_at(started_at, 60))
        refused: Final = eventually(
            lambda: tuple(attempt(candidate, rig.chat, key) for candidate in rig.replicas),
            lambda found: all(tried.status == BUDGET_EXCEEDED for tried in found) and found[0].at >= resets - 2,
            seconds=90,
        )
        allowed: Final = eventually(
            lambda: attempt(rig.gateway, rig.chat, key),
            lambda found: found.status != BUDGET_EXCEEDED,
            seconds=30,
        )
        assert allowed.status == 200, allowed
        assert resets - 1.5 <= allowed.at <= resets + 5, (allowed, resets, refused)


def config_with(directory: Path, name: str, **sections: JsonValue) -> Path:
    loaded: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / name
    path.write_text(yaml.safe_dump({**loaded, **sections}))
    return path


@pytest.mark.timeout(OWNED_PROXY_SECONDS)
def test_without_redis_the_reset_comes_from_the_worker_memory(rig: BudgetRig, tmp_path: Path) -> None:
    with (
        rig.gateway.scenario() as scenario,
        owned_proxy(
            rig.gateway,
            tmp_path,
            {},
            config=config_with(tmp_path, "no_redis_config.yaml", litellm_settings={}),
            remove_environment=("REDIS_HOST", "REDIS_PORT"),
        ) as owned,
    ):
        settle(lambda marker: fresh_chat(owned, rig.chat, owned.key, marker), 200, burst=4)
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        before: Final = time.time()
        response: Final = fresh_chat(owned, rig.chat, key, "no-redis-" + uuid.uuid4().hex)
        assert response.status_code == 200, response.text
        reported: Final = eventually(
            lambda: model_usage(owned, key, rig.chat)["budget_reset_at"],
            lambda found: found is not None,
            seconds=30,
        )
        resets: Final = datetime.fromisoformat(str(reported)).timestamp()
        assert before + DAY <= resets <= time.time() + DAY, (before, reported)
        assert redis_text(rig.redis, key_window(key, rig.chat, "1d").start_key) is None


def assert_cache_hit_kept_the_window(rig: BudgetRig, key: str, window: Window, started_at: float) -> None:
    spent: Final = redis_float(rig.redis, window.counter_key)
    held(lambda: redis_float(rig.redis, window.counter_key), lambda found: found == spent, holding=3)
    assert redis_text(rig.redis, window.start_key) == json.dumps(started_at)
    assert resets_on(rig.replicas, key, rig.chat) == (reset_at(started_at, DAY),) * 2


def test_a_cached_chat_reply_keeps_the_window_start_and_reset(rig: BudgetRig) -> None:
    content: Final = "cache-twin-" + uuid.uuid4().hex
    first_marker: Final = "first-" + uuid.uuid4().hex
    second_marker: Final = "second-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        with openai_client(rig.gateway, key) as client:
            first: Final = client.chat.completions.create(
                model=rig.chat,
                messages=[{"role": "user", "content": content}],
                extra_headers=spend_marker(first_marker),
            )
        landed_once(key, first_marker)
        started_at: Final = assert_daily_window_reported(rig, key, rig.chat)
        with openai_client(rig.peer, key) as client:
            second: Final = client.chat.completions.with_raw_response.create(
                model=rig.chat,
                messages=[{"role": "user", "content": content}],
                extra_headers=spend_marker(second_marker),
            )
        assert second.parse().id == first.id, second.parse()
        assert "x-litellm-cache-key" in second.headers, dict(second.headers)
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), content) == 1
        assert float(str(landed_once(key, second_marker)["spend"])) == 0.0
        assert_cache_hit_kept_the_window(rig, key, key_window(key, rig.chat, "1d"), started_at)


async def streamed_chat(candidate: Gateway, model: str, key: str, content: str, marker: str) -> tuple[str, str, bool]:
    async with async_openai_client(candidate, key) as client:
        raw: Final = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            stream=True,
            extra_headers=spend_marker(marker),
        )
        chunks: Final = tuple([chunk async for chunk in raw.parse()])
    text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    return chunks[0].id, text, "x-litellm-cache-key" in raw.headers


async def test_a_cached_streamed_chat_reply_keeps_the_window_start_and_reset(rig: BudgetRig) -> None:
    content: Final = "stream-cache-twin-" + uuid.uuid4().hex
    first_marker: Final = "first-" + uuid.uuid4().hex
    second_marker: Final = "second-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        first: Final = await streamed_chat(rig.gateway, rig.chat, key, content, first_marker)
        assert first[1] == CHAT_REPLY, first
        landed_once(key, first_marker)
        started_at: Final = assert_daily_window_reported(rig, key, rig.chat)
        second: Final = await streamed_chat(rig.peer, rig.chat, key, content, second_marker)
        assert second == (first[0], CHAT_REPLY, True), (first, second)
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), content) == 1
        assert float(str(landed_once(key, second_marker)["spend"])) == 0.0
        assert_cache_hit_kept_the_window(rig, key, key_window(key, rig.chat, "1d"), started_at)


MALFORMED: Final = (
    pytest.param("time_period", 5, id="period-int"),
    pytest.param("time_period", ["1d"], id="period-list"),
    pytest.param("time_period", "", id="period-empty"),
    pytest.param("time_period", "9" * 5119 + "h", id="period-5kb-hours"),
    pytest.param("time_period", "x" * 5120, id="period-5kb-text"),
    pytest.param("time_period", None, id="period-null"),
    pytest.param("budget_limit", ["1"], id="limit-list"),
    pytest.param("budget_limit", "", id="limit-empty"),
    pytest.param("budget_limit", "x" * 5120, id="limit-5kb"),
)


def reported_reset(info: httpx.Response, model: str) -> JsonValue:
    usage: Final = object_value(info.json()["info"]).get("model_max_budget_usage")
    entry: Final = usage.get(model) if isinstance(usage, dict) else None
    return entry.get("budget_reset_at") if isinstance(entry, dict) else None


def reported_resets(rig: BudgetRig, key: str) -> tuple[tuple[int, JsonValue], ...]:
    infos: Final = tuple(candidate.request("GET", "/key/info", params={"key": key}) for candidate in rig.replicas)
    return tuple(
        (info.status_code, reported_reset(info, rig.chat) if info.status_code == 200 else None) for info in infos
    )


def store_budget(key: str, model_max_budget: Mapping[str, JsonValue]) -> None:
    write_rows(
        'UPDATE "LiteLLM_VerificationToken" SET model_max_budget = %s::jsonb WHERE token = %s',
        (json.dumps(model_max_budget), hashed(key)),
    )


def charged_with_stored_entry(rig: BudgetRig, scenario: Scenario, entry: Mapping[str, JsonValue]) -> str:
    key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
    store_budget(key, {rig.chat: dict(entry)})
    marker: Final = "stored-entry-" + uuid.uuid4().hex
    response: Final = fresh_chat(rig.gateway, rig.chat, key, marker)
    assert response.status_code == 200, response.text
    landed_once(key, marker)
    return key


@pytest.mark.parametrize(("field", "value"), MALFORMED)
def test_a_stored_malformed_budget_entry_never_breaks_reporting_or_other_keys(
    rig: BudgetRig, field: str, value: JsonValue
) -> None:
    with rig.gateway.scenario() as scenario:
        control: Final = scenario.key()
        key: Final = charged_with_stored_entry(rig, scenario, {"budget_limit": 100, "time_period": "1d", field: value})
        held(lambda: reported_resets(rig, key), lambda found: found == ((200, None),) * 2, holding=3)
        listed: Final = rig.peer.request("POST", "/v2/key/info", {"keys": [key]})
        assert listed.status_code == 200, listed.text
        served: Final = fresh_chat(rig.peer, rig.chat, control, "control-" + uuid.uuid4().hex)
        assert served.status_code == 200, served.text


def test_a_stored_entry_with_a_null_limit_still_reports_its_window_reset(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = charged_with_stored_entry(rig, scenario, {"budget_limit": None, "time_period": "1d"})
        assert_daily_window_reported(rig, key, rig.chat)


def test_the_last_of_two_entries_for_one_model_wins(rig: BudgetRig) -> None:
    first: Final = json.dumps({"budget_limit": 1, "time_period": "1h"})
    last: Final = json.dumps({"budget_limit": 2, "time_period": "1d"})
    model: Final = json.dumps(rig.chat)
    with rig.gateway.scenario() as scenario:
        created: Final = rig.gateway.client.post(
            "/key/generate",
            content=f'{{"model_max_budget": {{{model}: {first}, {model}: {last}}}}}'.encode(),
            headers={"Authorization": f"Bearer {rig.gateway.key}", "Content-Type": "application/json"},
        )
        assert created.status_code == 200, created.text
        key: Final = str(created.json()["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        assert [model_usage(candidate, key, rig.chat) for candidate in rig.replicas] == [unused(2.0, "1d")] * 2
        response: Final = fresh_chat(rig.gateway, rig.chat, key, "duplicate-" + uuid.uuid4().hex)
        assert response.status_code == 200, response.text
        assert_daily_window_reported(rig, key, rig.chat)


def test_a_window_ending_after_year_9999_reports_no_reset(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "99999999h"))
        response: Final = fresh_chat(rig.gateway, rig.chat, key, "far-" + uuid.uuid4().hex)
        assert response.status_code == 200, response.text
        window: Final = key_window(key, rig.chat, "99999999h")
        opened_at(rig.redis, window)
        assert_spend_matches_counter(rig, key, rig.chat, window)
        assert resets_on(rig.replicas, key, rig.chat) == (None, None)


def test_a_request_to_another_model_opens_no_window(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.responses, "1d"))
        marker: Final = "other-model-" + uuid.uuid4().hex
        response: Final = fresh_chat(rig.gateway, rig.chat, key, marker)
        assert response.status_code == 200, response.text
        landed_once(key, marker)
        assert [model_usage(candidate, key, rig.responses) for candidate in rig.replicas] == [unused(100.0, "1d")] * 2
        assert redis_text(rig.redis, key_window(key, rig.responses, "1d").start_key) is None


def seed_start(redis: Redis, start_key: str, started_at: float, ends_at: float) -> None:
    assert redis.set(start_key, json.dumps(started_at), ex=math.ceil(ends_at - time.time()))


def charge_on_both_replicas(rig: BudgetRig, key: str, markers: tuple[str, ...]) -> None:
    statuses: Final = tuple(
        fresh_chat(rig.replicas[index % 2], rig.chat, key, marker).status_code for index, marker in enumerate(markers)
    )
    assert statuses == (200,) * len(markers), statuses


def test_a_monthly_window_ends_one_calendar_month_after_its_seeded_start(rig: BudgetRig) -> None:
    monthly: Final = monthly_window_in_another_span()
    started_at: Final = monthly.started.timestamp()
    ends_at: Final = monthly.ends.timestamp()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, monthly.period))
        window: Final = key_window(key, rig.chat, monthly.period)
        seed_start(rig.redis, window.start_key, started_at, ends_at)
        first: Final = "monthly-" + uuid.uuid4().hex
        charge_on_both_replicas(rig, key, (first,))
        spent: Final = float(str(landed_once(key, first)["spend"]))
        eventually(
            lambda: redis_float(rig.redis, window.counter_key),
            lambda found: math.isclose(found, spent, rel_tol=1e-9),
            seconds=30,
        )
        assert abs(expires_at(rig.redis, window.counter_key) - ends_at) < 5, (monthly, ends_at)
        assert resets_on(rig.replicas, key, rig.chat) == (monthly.ends.isoformat(),) * 2
        markers: Final = tuple(f"monthly-again-{index}-{uuid.uuid4().hex}" for index in range(3))
        charge_on_both_replicas(rig, key, markers)
        total: Final = spent + sum(float(str(row["spend"])) for row in landed_all_once(key, frozenset(markers)))
        eventually(
            lambda: redis_float(rig.redis, window.counter_key),
            lambda found: math.isclose(found, total, rel_tol=1e-9),
            seconds=30,
        )
        assert redis_text(rig.redis, window.start_key) == json.dumps(started_at)
        assert abs(expires_at(rig.redis, window.counter_key) - ends_at) < 5, (monthly, ends_at)
        reported_everywhere(rig.replicas, key, rig.chat, monthly.ends.isoformat())


def test_two_budgets_on_one_key_each_report_their_own_reset(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget={**budget(rig.chat, "1d"), **budget(rig.responses, "1h")})
        chat: Final = fresh_chat(rig.gateway, rig.chat, key, "daily-" + uuid.uuid4().hex)
        assert chat.status_code == 200, chat.text
        hourly: Final = fresh_responses(rig.peer, rig.responses, key, "hourly-" + uuid.uuid4().hex, False)
        assert hourly.status_code == 200, hourly.text
        daily_start: Final = opened_at(rig.redis, key_window(key, rig.chat, "1d"))
        hourly_start: Final = opened_at(rig.redis, key_window(key, rig.responses, "1h"))
        reported_everywhere(rig.replicas, key, rig.chat, reset_at(daily_start, DAY))
        reported_everywhere(rig.replicas, key, rig.responses, reset_at(hourly_start, HOUR))


def test_an_ended_window_reports_no_reset_until_the_next_charge_opens_one(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1m"))
        window: Final = key_window(key, rig.chat, "1m")
        ended_start: Final = math.floor(time.time()) - 120.0
        seed_start(rig.redis, window.start_key, ended_start, time.time() + HOUR)
        assert resets_on(rig.replicas, key, rig.chat) == (None, None)
        response: Final = fresh_chat(rig.gateway, rig.chat, key, "reopen-" + uuid.uuid4().hex)
        assert response.status_code == 200, response.text
        reopened: Final = eventually(
            lambda: float(redis_text(rig.redis, window.start_key) or ended_start),
            lambda found: found > ended_start,
            seconds=30,
        )
        reported_everywhere(rig.replicas, key, rig.chat, reset_at(reopened, 60))


BURST_KINDS: Final[tuple[str, ...]] = (
    "chat",
    "chat-stream",
    "responses",
    "responses-stream",
    "messages",
    "messages-stream",
)


def burst_body(rig: BudgetRig, kind: str, marker: str) -> tuple[str, dict[str, JsonValue]]:
    stream: Final = kind.endswith("-stream")
    match kind.removesuffix("-stream"):
        case "chat":
            return "/v1/chat/completions", {
                "model": rig.chat,
                "messages": [{"role": "user", "content": marker}],
                "stream": stream,
            }
        case "responses":
            return "/v1/responses", {
                "model": rig.responses_stream if stream else rig.responses,
                "input": marker,
                "stream": stream,
            }
        case _:
            return "/v1/messages", {
                "model": rig.responses_stream if stream else rig.responses,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": marker}],
                "stream": stream,
            }


async def burst_request(client: httpx.AsyncClient, rig: BudgetRig, kind: str, key: str, marker: str) -> int:
    path, body = burst_body(rig, kind, marker)
    async with client.stream(
        "POST", path, json=body, headers={"Authorization": f"Bearer {key}", **spend_marker(marker)}
    ) as response:
        await response.aread()
        return response.status_code


async def mixed_burst(rig: BudgetRig, key: str, markers: tuple[str, ...]) -> tuple[int, ...]:
    async with (
        httpx.AsyncClient(base_url=base_url(rig.gateway), timeout=90, trust_env=False) as gateway,
        httpx.AsyncClient(base_url=base_url(rig.peer), timeout=90, trust_env=False) as peer,
    ):
        return tuple(
            await asyncio.gather(
                *(
                    burst_request((gateway, peer)[index % 2], rig, BURST_KINDS[index % len(BURST_KINDS)], key, marker)
                    for index, marker in enumerate(markers)
                )
            )
        )


async def test_a_mixed_burst_over_both_replicas_lands_once_and_keeps_one_reset(rig: BudgetRig) -> None:
    with rig.gateway.scenario() as scenario:
        models: Final = (rig.chat, rig.responses, rig.responses_stream)
        key: Final = scenario.key(model_max_budget={model: budget(model, "1d")[model] for model in models})
        opening: Final = (
            fresh_chat(rig.gateway, rig.chat, key, "open-" + uuid.uuid4().hex),
            fresh_responses(rig.peer, rig.responses, key, "open-" + uuid.uuid4().hex, False),
            fresh_responses(rig.gateway, rig.responses_stream, key, "open-" + uuid.uuid4().hex, True),
        )
        assert [response.status_code for response in opening] == [200] * 3, [response.text for response in opening]
        starts: Final = {model: opened_at(rig.redis, key_window(key, model, "1d")) for model in models}
        markers: Final = tuple(f"burst-{index}-{uuid.uuid4().hex}" for index in range(30))
        statuses: Final = await mixed_burst(rig, key, markers)
        assert statuses == (200,) * len(markers), statuses
        landed_all_once(key, frozenset(markers))
        for model in models:
            assert_window_kept(rig, key, model, starts[model])


@pytest.mark.timeout(OWNED_PROXY_SECONDS)
def test_a_killed_worker_leaves_the_reset_and_start_unchanged(rig: BudgetRig, tmp_path: Path) -> None:
    with (
        rig.gateway.scenario() as scenario,
        owned_proxy_process(rig.gateway, tmp_path, {}, workers=2) as owned,
    ):
        settle(lambda marker: fresh_chat(owned.gateway, rig.chat, owned.gateway.key, marker), 200, burst=8)
        workers: Final = eventually(owned.worker_pids, lambda found: len(found) == 2, seconds=30)
        key: Final = scenario.key(model_max_budget=budget(rig.chat, "1d"))
        first: Final = "before-kill-" + uuid.uuid4().hex
        assert fresh_chat(owned.gateway, rig.chat, key, first).status_code == 200
        landed_once(key, first)
        started_at: Final = opened_at(rig.redis, key_window(key, rig.chat, "1d"))
        expected: Final = reset_at(started_at, DAY)
        reported_everywhere((owned.gateway, *rig.replicas), key, rig.chat, expected)
        markers: Final = tuple(f"during-kill-{index}-{uuid.uuid4().hex}" for index in range(6))
        with owned.replacing_worker(min(workers)) as kill:
            kill()
            statuses: Final = tuple(fresh_chat(owned.gateway, rig.chat, key, marker).status_code for marker in markers)
        assert statuses == (200,) * len(markers), statuses
        landed_all_once(key, frozenset(markers))
        assert_window_kept(rig, key, rig.chat, started_at)
        assert {resets_on((owned.gateway,), key, rig.chat) for _ in range(8)} == {(expected,)}


ROUTER_TAG: Final = "monthly-tag-" + uuid.uuid4().hex[:12]
ROUTER_DEPLOYMENT: Final = "monthly-deployment-" + uuid.uuid4().hex[:12]
ROUTER_MODEL: Final = "monthly-router-" + uuid.uuid4().hex[:12]
ROUTER_START_KEYS: Final[tuple[str, ...]] = (
    "provider_budget_start_time:openai",
    f"deployment_budget_start_time:{ROUTER_DEPLOYMENT}",
    f"tag_budget_start_time:{ROUTER_TAG}",
)


@dataclass(frozen=True, slots=True)
class RouterRig:
    gateway: Gateway
    redis: Redis
    monthly: MonthlyWindow

    @property
    def counters(self) -> tuple[str, ...]:
        period: Final = self.monthly.period
        return (
            f"provider_spend:openai:{period}",
            f"deployment_spend:{ROUTER_DEPLOYMENT}:{period}",
            f"tag_spend:{ROUTER_TAG}:{period}",
        )


def router_config(directory: Path, upstream_url: str, period: str) -> Path:
    limits: Final[dict[str, JsonValue]] = {"budget_limit": 100, "time_period": period}
    return config_with(
        directory,
        "router_budget_config.yaml",
        model_list=[
            {
                "model_name": ROUTER_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{upstream_url}/v1",
                    "api_key": "integration-provider-key",
                    "max_budget": 100,
                    "budget_duration": period,
                    **PRICES,
                },
                "model_info": {"id": ROUTER_DEPLOYMENT},
            }
        ],
        router_settings={"disable_cooldowns": True, "provider_budget_config": {"openai": limits}},
        litellm_settings={"tag_budget_config": {ROUTER_TAG: limits}},
        general_settings={
            **object_value(
                JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))[
                    "general_settings"
                ]
            ),
            "coordination_redis": {"host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
        },
    )


def tagged_chat(candidate: Gateway, key: str, marker: str) -> httpx.Response:
    return fresh_chat(candidate, ROUTER_MODEL, key, marker, {"metadata": {"tags": [ROUTER_TAG]}})


@pytest.fixture(scope="module")
def router_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RouterRig]:
    directory: Final = tmp_path_factory.mktemp("router-budgets")
    monthly: Final = monthly_window_in_another_span()
    with (
        gateway_from_environment() as gateway,
        owned_redis(directory / "redis") as served,
        Redis(host=served.host, port=served.port, decode_responses=True) as redis,
    ):
        for start_key in ROUTER_START_KEYS:
            seed_start(redis, start_key, monthly.started.timestamp(), monthly.ends.timestamp())
        with owned_proxy(
            gateway,
            directory,
            {"REDIS_HOST": served.host, "REDIS_PORT": str(served.port)},
            config=router_config(directory, gateway.upstream_url, monthly.period),
        ) as owned:
            settle(lambda marker: tagged_chat(owned, owned.key, marker), 200, burst=2)
            yield RouterRig(gateway=owned, redis=redis, monthly=monthly)


def router_counters_charged(rig: RouterRig) -> tuple[float, ...]:
    return eventually(
        lambda: tuple(float(redis_text(rig.redis, counter) or 0) for counter in rig.counters),
        lambda found: all(spent > 0 for spent in found),
        seconds=30,
    )


@pytest.mark.timeout(OWNED_PROXY_SECONDS)
def test_router_budget_counters_expire_one_calendar_month_after_their_seeded_start(router_rig: RouterRig) -> None:
    router_counters_charged(router_rig)
    ends_at: Final = router_rig.monthly.ends.timestamp()
    expiries: Final = tuple(expires_at(router_rig.redis, counter) for counter in router_rig.counters)
    assert all(abs(expiry - ends_at) < 5 for expiry in expiries), (router_rig.monthly, expiries)


@pytest.mark.timeout(OWNED_PROXY_SECONDS)
def test_provider_budgets_report_the_reset_one_calendar_month_after_the_seeded_start(router_rig: RouterRig) -> None:
    router_counters_charged(router_rig)
    provider: Final = object_value(object_value(router_rig.gateway.get("/provider/budgets")["providers"])["openai"])
    reported: Final = datetime.fromisoformat(str(provider["budget_reset_at"])).timestamp()
    assert abs(reported - router_rig.monthly.ends.timestamp()) < 5, (router_rig.monthly, provider)


@pytest.mark.timeout(OWNED_PROXY_SECONDS)
def test_router_budget_starts_stay_put_across_more_charges(router_rig: RouterRig) -> None:
    before: Final = router_counters_charged(router_rig)
    markers: Final = tuple(f"router-again-{index}-{uuid.uuid4().hex}" for index in range(3))
    statuses: Final = tuple(
        tagged_chat(router_rig.gateway, router_rig.gateway.key, marker).status_code for marker in markers
    )
    assert statuses == (200,) * 3, statuses
    eventually(
        lambda: tuple(float(redis_text(router_rig.redis, counter) or 0) for counter in router_rig.counters),
        lambda found: all(spent > earlier for spent, earlier in zip(found, before, strict=True)),
        seconds=30,
    )
    started: Final = json.dumps(router_rig.monthly.started.timestamp())
    assert tuple(redis_text(router_rig.redis, start_key) for start_key in ROUTER_START_KEYS) == (started,) * 3
    ends_at: Final = router_rig.monthly.ends.timestamp()
    expiries: Final = tuple(expires_at(router_rig.redis, counter) for counter in router_rig.counters)
    assert all(abs(expiry - ends_at) < 5 for expiry in expiries), (router_rig.monthly, expiries)
