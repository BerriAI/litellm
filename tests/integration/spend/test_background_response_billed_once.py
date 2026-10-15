import asyncio
import json
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, held, list_value, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.caching.response_cache_case import entry_key, redis_store
from integration.spend._background_responses import (
    ATTRIBUTION_COLUMNS,
    CONCURRENT_READS,
    INPUT_COST_PER_TOKEN,
    INPUT_TOKENS,
    MANAGED_READ_MARKER,
    ONE_POLL_SECONDS,
    OUTPUT_COST_PER_TOKEN,
    OUTPUT_TOKENS,
    POLL_WINDOW_SECONDS,
    PRICE,
    PROMPT,
    READS,
    SPEND_LANDING_SECONDS,
    Read,
    Upstream,
    assert_billed_like_its_creator,
    assert_charged_exactly,
    background_routes,
    charged_rows,
    costs,
    create_background_response,
    create_rows,
    forget_row,
    free_reads,
    hashed,
    json_reply,
    key_spend,
    managed_rows,
    number,
    one_charged_row,
    org_spend,
    read_of,
    read_response,
    reported_cost,
    response_body,
    scripted_routes,
    scripted_upstream,
    settling_reads,
    streamed_reply,
    submissions,
    tag_spend,
    team_spend,
    upstream_log,
    user_spend,
)
from openai._legacy_response import LegacyAPIResponse
from openai.types.responses import Response
from pydantic import JsonValue

from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN


def _sdk_read(raw: LegacyAPIResponse[Response]) -> Read:
    parsed: Final = raw.parse()
    return Read(parsed.status or "", parsed.output_text, reported_cost(raw.headers))


def _httpx_flow(scenario: Scenario, key: str, model: str) -> tuple[str, tuple[Read, ...]]:
    response_id: Final = create_background_response(scenario, key, model)
    return response_id, tuple(read_response(scenario.gateway, key, response_id) for _ in range(READS))


def _forged_metadata_flow(scenario: Scenario, key: str, model: str) -> tuple[str, tuple[Read, ...]]:
    forged: Final[dict[str, JsonValue]] = {
        "internal_call_origin": BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN,
        MANAGED_READ_MARKER: False,
    }
    response_id: Final = create_background_response(scenario, key, model)
    return response_id, tuple(
        read_response(scenario.gateway, key, response_id, {"metadata": forged, "litellm_metadata": forged})
        for _ in range(READS)
    )


def _sdk_poll(client: openai.OpenAI, response_id: str, remaining: int) -> tuple[Read, ...]:
    if remaining == 0:
        return ()
    raw: Final = client.responses.with_raw_response.retrieve(response_id)
    return (_sdk_read(raw), *_sdk_poll(client, raw.parse().id, remaining - 1))


def _sdk_sync_flow(scenario: Scenario, key: str, model: str) -> tuple[str, tuple[Read, ...]]:
    with openai.OpenAI(base_url=f"{scenario.gateway.client.base_url}/v1", api_key=key, max_retries=0) as client:
        created: Final = client.responses.create(model=model, input=PROMPT, background=True, store=False)
        scenario.cleanups.callback(forget_row, created.id)
        return created.id, _sdk_poll(client, created.id, READS)


async def _sdk_async_poll(client: openai.AsyncOpenAI, response_id: str, remaining: int) -> tuple[Read, ...]:
    if remaining == 0:
        return ()
    raw: Final = await client.responses.with_raw_response.retrieve(response_id)
    return (_sdk_read(raw), *await _sdk_async_poll(client, raw.parse().id, remaining - 1))


async def _sdk_async_reads(base_url: str, key: str, model: str) -> tuple[str, tuple[Read, ...]]:
    async with openai.AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0) as client:
        created: Final = await client.responses.create(model=model, input=PROMPT, background=True, store=False)
        return created.id, await _sdk_async_poll(client, created.id, READS)


def _sdk_async_flow(scenario: Scenario, key: str, model: str) -> tuple[str, tuple[Read, ...]]:
    response_id, reads = asyncio.run(_sdk_async_reads(f"{scenario.gateway.client.base_url}/v1", key, model))
    scenario.cleanups.callback(forget_row, response_id)
    return response_id, reads


Flow = Callable[[Scenario, str, str], tuple[str, tuple[Read, ...]]]
FLOWS: Final[Mapping[str, Flow]] = {
    "httpx": _httpx_flow,
    "httpx_forged_metadata": _forged_metadata_flow,
    "openai_sync": _sdk_sync_flow,
    "openai_async": _sdk_async_flow,
}


def _team_member_key(scenario: Scenario, **fields: JsonValue) -> tuple[str, str, str]:
    team_id: Final = scenario.team(**fields)
    user_id: Final = scenario.member(team_id)
    return scenario.key(team_id=team_id, user_id=user_id), team_id, user_id


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    ("client", "terminal_status"),
    [
        ("httpx", "completed"),
        ("httpx", "incomplete"),
        ("httpx_forged_metadata", "completed"),
        ("openai_sync", "completed"),
        ("openai_async", "completed"),
    ],
)
def test_a_finished_background_response_bills_its_creator_once_however_often_it_is_read(
    gateway: Gateway, client: str, terminal_status: str
) -> None:
    with gateway.scenario() as scenario:
        key, team_id, user_id = _team_member_key(scenario)
        upstream: Final = scripted_upstream(scenario, background_routes(terminal_status))
        log: Final = upstream_log(gateway, upstream.handle)
        response_id, reads = FLOWS[client](scenario, key, upstream.model)
        assert reads in (settling_reads(terminal_status, READS), free_reads(terminal_status, READS))
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert_billed_like_its_creator(one_charged_row(upstream.model_id), key, team_id, user_id)
        assert team_spend(team_id) == pytest.approx(PRICE)
        assert submissions(next(log), upstream.handle) == (
            {"model": "gpt-4o-mini", "input": PROMPT, "background": True, "store": False},
        )
        assert [row["status"] for row in managed_rows(response_id)] == ["completed"]


def _org_team_member_key(scenario: Scenario) -> tuple[str, str, str, str]:
    organization_id: Final = scenario.organization()
    team_id: Final = scenario.team(organization_id=organization_id)
    user_id: Final = scenario.org_member(organization_id, "internal_user")
    scenario.gateway.post("/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}})
    return scenario.key(team_id=team_id, user_id=user_id), team_id, user_id, organization_id


def _billed_by_the_poll_alone(
    scenario: Scenario, key: str, tags: tuple[str, str]
) -> tuple[Upstream, dict[str, JsonValue]]:
    body_tag, header_tag = tags
    upstream: Final = scripted_upstream(scenario, background_routes("completed"))
    response_id: Final = create_background_response(
        scenario,
        key,
        upstream.model,
        body={
            "model": upstream.model,
            "input": PROMPT,
            "background": True,
            "store": False,
            "litellm_metadata": {"tags": [body_tag]},
        },
        headers={"x-litellm-tags": header_tag},
    )
    eventually(
        lambda: [row["status"] for row in managed_rows(response_id)],
        lambda statuses: statuses == ["completed"],
        seconds=ONE_POLL_SECONDS,
    )
    held(lambda: key_spend(key), costs(PRICE), holding=5, seconds=SPEND_LANDING_SECONDS)
    charged: Final = one_charged_row(upstream.model_id)
    create_row: Final = eventually(lambda: create_rows(upstream.model_id), lambda rows: len(rows) == 1)[0]
    assert {column: charged[column] for column in ATTRIBUTION_COLUMNS} == {
        column: create_row[column] for column in ATTRIBUTION_COLUMNS
    }
    assert set(list_value(charged["request_tags"])) == set(list_value(create_row["request_tags"])) >= set(tags)
    return upstream, charged


def _unique_tags() -> tuple[str, str]:
    return f"audit-body-{uuid.uuid4().hex[:8]}", f"audit-header-{uuid.uuid4().hex[:8]}"


@pytest.mark.timeout(240)
def test_a_background_response_nobody_reads_is_billed_once_to_its_creating_key_team_org_and_tags(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        key, team_id, user_id, organization_id = _org_team_member_key(scenario)
        tags: Final = _unique_tags()
        _, charged = _billed_by_the_poll_alone(scenario, key, tags)
        assert_billed_like_its_creator(charged, key, team_id, user_id)
        assert charged["organization_id"] == organization_id
        eventually(lambda: tuple(map(tag_spend, tags)), lambda spent: all(map(costs(PRICE), spent)))
        eventually(lambda: org_spend(organization_id), costs(PRICE))
        listed: Final = gateway.request(
            "GET", "/spend/logs", params={"request_id": string_value(charged["request_id"])}
        )
        assert listed.status_code == 200, listed.text
        assert [number(object_value(row)["spend"]) for row in list_value(listed.json())] == [pytest.approx(PRICE)]
        assert number(object_value(gateway.get("/key/info", {"key": key})["info"])["spend"]) == pytest.approx(PRICE)
        team_info: Final = object_value(gateway.get("/team/info", {"team_id": team_id})["team_info"])
        assert number(team_info["spend"]) == pytest.approx(PRICE)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("in_a_team", [False, True], ids=["no_team", "team"])
def test_a_background_response_nobody_reads_on_a_key_with_no_user_is_billed_once_to_the_key(
    gateway: Gateway, in_a_team: bool
) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(team_id=scenario.team()) if in_a_team else scenario.key()
        _, charged = _billed_by_the_poll_alone(scenario, key, _unique_tags())
        assert string_value(charged["request_id"]).endswith("_background_cost"), charged
        assert charged["api_key"] == hashed(key)
        assert number(charged["spend"]) == pytest.approx(PRICE)


@pytest.mark.timeout(240)
def test_a_background_response_is_free_until_it_finishes_and_a_failed_read_bills_nothing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream: Final = scripted_upstream(scenario, background_routes("in_progress"))
        response_id: Final = create_background_response(scenario, key, upstream.model)
        assert read_response(gateway, key, response_id) == Read("in_progress", "", 0.0)
        outage: Final = json_reply({"error": {"message": "The server had an error.", "type": "server_error"}}, 500)
        upstream.script(scripted_routes(json_reply(response_body("queued")), outage))
        failed: Final = gateway.request("GET", f"/v1/responses/{response_id}", key=key)
        assert failed.status_code == 500, failed.text
        assert "The server had an error." in failed.text
        assert [row["status"] for row in managed_rows(response_id)] == ["queued"]
        upstream.script(background_routes("completed"))
        assert tuple(read_response(gateway, key, response_id) for _ in range(2)) == settling_reads("completed", 2)
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert [row["status"] for row in managed_rows(response_id)] == ["completed"]


@pytest.mark.timeout(240)
def test_cancelling_a_background_response_bills_it_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        cancelled: Final = json_reply(response_body("cancelled"))
        upstream: Final = scripted_upstream(
            scenario, scripted_routes(json_reply(response_body("queued")), cancelled, cancel=cancelled)
        )
        response_id: Final = create_background_response(scenario, key, upstream.model)
        cancel: Final = gateway.request("POST", f"/v1/responses/{response_id}/cancel", key=key)
        reads: Final = (read_of(cancel), read_response(gateway, key, response_id))
        assert reads in (settling_reads("cancelled", 2), free_reads("cancelled", 2))
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert number(one_charged_row(upstream.model_id)["spend"]) == pytest.approx(PRICE)


@pytest.mark.timeout(120)
def test_a_foreground_response_bills_its_create_once_and_its_read_is_free(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        foreground: Final = json_reply(response_body("completed", background=False))
        upstream: Final = scripted_upstream(scenario, scripted_routes(foreground, foreground))
        created: Final = gateway.request(
            "POST", "/v1/responses", {"model": upstream.model, "input": PROMPT, "store": False}, key=key
        )
        assert read_of(created) == Read("completed", "pong", pytest.approx(PRICE))
        response_id: Final = string_value(object_value(created.json())["id"])
        assert read_response(gateway, key, response_id) == Read("completed", "pong", 0.0)
        held(lambda: key_spend(key), costs(PRICE), holding=10, seconds=SPEND_LANDING_SECONDS)
        assert managed_rows(response_id) == []


STREAMED_BACKGROUND_FIELDS: Final[Mapping[str, dict[str, JsonValue]]] = {
    "top_level": {"background": True},
    "extra_body": {"extra_body": {"background": True}},
}


def _streamed_response_id(gateway: Gateway, key: str, model: str, background: Mapping[str, JsonValue]) -> str:
    with gateway.client.stream(
        "POST",
        "/v1/responses",
        json={"model": model, "input": PROMPT, "store": False, "stream": True, **background},
        headers={"Authorization": f"Bearer {key}"},
    ) as stream:
        assert stream.status_code == 200, stream.read()
        events: Final = tuple(
            object_value(json.loads(line.removeprefix("data: ")))
            for line in stream.iter_lines()
            if line.startswith("data: {")
        )
    completed: Final = [event for event in events if event["type"] == "response.completed"]
    assert len(completed) == 1, events
    return string_value(object_value(completed[0]["response"])["id"])


@pytest.mark.timeout(240)
@pytest.mark.parametrize("background_in", STREAMED_BACKGROUND_FIELDS.keys())
def test_a_streamed_background_response_read_later_bills_its_stream_alone(gateway: Gateway, background_in: str) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream: Final = scripted_upstream(scenario, streamed_reply())
        response_id: Final = _streamed_response_id(
            gateway, key, upstream.model, STREAMED_BACKGROUND_FIELDS[background_in]
        )
        scenario.cleanups.callback(forget_row, response_id)
        upstream.script(background_routes("completed"))
        assert tuple(read_response(gateway, key, response_id) for _ in range(2)) == free_reads("completed", 2)
        assert_charged_exactly(lambda: key_spend(key), PRICE)


def _finished(scenario: Scenario, key: str) -> tuple[Upstream, str]:
    upstream: Final = scripted_upstream(scenario, background_routes("completed"))
    return upstream, create_background_response(scenario, key, upstream.model)


def _lose_the_row_then_finish(scenario: Scenario, key: str) -> tuple[Upstream, str]:
    upstream: Final = scripted_upstream(scenario, background_routes("queued"))
    response_id: Final = create_background_response(scenario, key, upstream.model)
    forget_row(response_id)
    upstream.script(background_routes("completed"))
    return upstream, response_id


@pytest.mark.timeout(240)
def test_a_finished_response_the_proxy_lost_track_of_bills_its_reader_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key, team_id, user_id = _team_member_key(scenario)
        upstream, response_id = _lose_the_row_then_finish(scenario, key)
        reads: Final = tuple(read_response(gateway, key, response_id) for _ in range(READS))
        assert reads == settling_reads("completed", READS)
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert_billed_like_its_creator(one_charged_row(upstream.model_id), key, team_id, user_id)
        assert [(row["status"], row["api_key"]) for row in managed_rows(response_id)] == [("completed", hashed(key))]


@pytest.mark.timeout(240)
@pytest.mark.parametrize("tracked", [True, False], ids=["tracked", "untracked"])
def test_concurrent_reads_of_a_finished_response_bill_it_once(gateway: Gateway, tracked: bool) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream, response_id = _finished(scenario, key) if tracked else _lose_the_row_then_finish(scenario, key)
        with ThreadPoolExecutor(max_workers=CONCURRENT_READS) as pool:
            reads: Final = tuple(pool.map(lambda _: read_response(gateway, key, response_id), range(CONCURRENT_READS)))
        assert [(read.status, read.text) for read in reads] == [("completed", "pong")] * CONCURRENT_READS
        assert [read.cost for read in reads if read.cost != 0.0] in ([], [pytest.approx(PRICE)])
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert len(charged_rows(upstream.model_id)) == 1
        assert [row["status"] for row in managed_rows(response_id)] == ["completed"]


@pytest.mark.timeout(240)
def test_another_key_of_the_same_owner_reads_for_free_and_the_creating_key_pays(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        creator, team_id, user_id = _team_member_key(scenario)
        reader: Final = scenario.key(team_id=team_id, user_id=user_id)
        upstream: Final = scripted_upstream(scenario, background_routes("completed"))
        response_id: Final = create_background_response(scenario, creator, upstream.model)
        reads: Final = tuple(read_response(gateway, reader, response_id) for _ in range(READS))
        assert reads in (settling_reads("completed", READS), free_reads("completed", READS))
        assert_charged_exactly(lambda: key_spend(creator), PRICE)
        assert key_spend(reader) == 0
        assert_billed_like_its_creator(one_charged_row(upstream.model_id), creator, team_id, user_id)


@pytest.mark.timeout(240)
def test_a_create_that_asks_for_background_inside_extra_body_is_billed_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream: Final = scripted_upstream(scenario, background_routes("completed"))
        log: Final = upstream_log(gateway, upstream.handle)
        response_id: Final = create_background_response(
            scenario,
            key,
            upstream.model,
            body={"model": upstream.model, "input": PROMPT, "store": False, "extra_body": {"background": True}},
        )
        reads: Final = tuple(read_response(gateway, key, response_id) for _ in range(READS))
        assert reads in (settling_reads("completed", READS), free_reads("completed", READS))
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert [object_value(body).get("background") for body in submissions(next(log), upstream.handle)] == [True]


@pytest.mark.timeout(240)
def test_a_response_whose_instructions_carry_a_nul_is_tracked_and_billed_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream: Final = scripted_upstream(scenario, background_routes("completed", instructions="be\u0000brief"))
        response_id: Final = create_background_response(scenario, key, upstream.model)
        reads: Final = tuple(read_response(gateway, key, response_id) for _ in range(READS))
        assert reads in (settling_reads("completed", READS), free_reads("completed", READS))
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        stored: Final = read_rows(
            "SELECT file_object->>'instructions' AS instructions FROM \"LiteLLM_ManagedObjectTable\" "
            "WHERE unified_object_id = %s",
            (response_id,),
        )
        assert stored == [{"instructions": "bebrief"}]


@pytest.mark.timeout(240)
def test_a_request_tag_carrying_a_nul_is_stored_without_it_and_lands_on_the_billed_row(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        suffix: Final = uuid.uuid4().hex[:8]
        upstream: Final = scripted_upstream(scenario, background_routes("completed"))
        response_id: Final = create_background_response(
            scenario,
            key,
            upstream.model,
            body={
                "model": upstream.model,
                "input": PROMPT,
                "background": True,
                "store": False,
                "litellm_metadata": {"tags": [f"audit-nul\u0000{suffix}"]},
            },
        )
        reads: Final = tuple(read_response(gateway, key, response_id) for _ in range(2))
        assert reads in (settling_reads("completed", 2), free_reads("completed", 2))
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert f"audit-nul{suffix}" in list_value(one_charged_row(upstream.model_id)["request_tags"])


@pytest.mark.timeout(240)
@pytest.mark.parametrize("managed_row", ["tracked", "lost"])
def test_a_read_retargeted_with_a_client_side_api_base_bills_the_real_usage_once(
    gateway: Gateway, managed_row: str
) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream: Final = scripted_upstream(
            scenario, background_routes("queued"), configurable_clientside_auth_params=["api_base"]
        )
        provider_id: Final = f"resp_{upstream.model}"
        cheap_body: Final = {
            **response_body("completed"),
            "id": provider_id,
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        decoy: Final = register_scenario(
            f"bgonce-decoy-{uuid.uuid4().hex[:12]}",
            scripted_routes(json_reply(cheap_body), json_reply(cheap_body), provider_id=provider_id),
        )
        scenario.cleanups.callback(delete_scenario, decoy)
        decoy_log: Final = upstream_log(gateway, decoy)
        response_id: Final = create_background_response(scenario, key, upstream.model)
        if managed_row == "lost":
            forget_row(response_id)
        upstream.script(background_routes("completed"))
        retargeted: Final = read_response(
            gateway, key, response_id, {"model": upstream.model, "api_base": decoy.api_base()}
        )
        assert (retargeted,) in (settling_reads("completed", 1), free_reads("completed", 1))
        assert [call["method"] for call in next(decoy_log)] == ["GET"]
        eventually(
            lambda: [row["status"] for row in managed_rows(response_id)],
            lambda statuses: statuses == ["completed"],
            seconds=ONE_POLL_SECONDS,
        )
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        charged: Final = one_charged_row(upstream.model_id)
        assert (charged["prompt_tokens"], charged["completion_tokens"]) == (INPUT_TOKENS, OUTPUT_TOKENS)


def _id_a_read_returned(gateway: Gateway, key: str, response_id: str) -> str:
    read: Final = gateway.request("GET", f"/v1/responses/{response_id}", None, key=key)
    assert read.status_code == 200, read.text
    return string_value(object_value(read.json())["id"])


@pytest.mark.timeout(240 + ONE_POLL_SECONDS)
@pytest.mark.parametrize("read_by", ["create_id", "id_a_read_returned"])
def test_a_row_stored_before_the_upgrade_bills_the_creating_team_and_user_once(gateway: Gateway, read_by: str) -> None:
    with gateway.scenario() as scenario:
        key, team_id, user_id = _team_member_key(scenario)
        upstream: Final = scripted_upstream(scenario, background_routes("queued"))
        created_id: Final = create_background_response(scenario, key, upstream.model)
        response_id: Final = created_id if read_by == "create_id" else _id_a_read_returned(gateway, key, created_id)
        write_rows(
            'UPDATE "LiteLLM_ManagedObjectTable" SET model_object_id = unified_object_id, api_key = NULL, '
            "request_tags = NULL WHERE unified_object_id = %s",
            (created_id,),
        )
        upstream.script(background_routes("completed"))
        reads: Final = tuple(read_response(gateway, key, response_id) for _ in range(READS))
        assert reads in (settling_reads("completed", READS), free_reads("completed", READS))
        held(lambda: team_spend(team_id), costs(PRICE), holding=ONE_POLL_SECONDS, seconds=SPEND_LANDING_SECONDS)
        assert user_spend(user_id) == pytest.approx(PRICE)
        charged: Final = one_charged_row(upstream.model_id)
        assert (charged["team_id"], charged["user"]) == (team_id, user_id)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("settled_status", ["completed", "stale_expired"])
def test_reads_of_a_response_whose_row_already_settled_are_free(gateway: Gateway, settled_status: str) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream: Final = scripted_upstream(scenario, background_routes("queued"))
        response_id: Final = create_background_response(scenario, key, upstream.model)
        write_rows(
            'UPDATE "LiteLLM_ManagedObjectTable" SET status = %s WHERE unified_object_id = %s',
            (settled_status, response_id),
        )
        upstream.script(background_routes("completed"))
        assert tuple(read_response(gateway, key, response_id) for _ in range(2)) == free_reads("completed", 2)
        assert_charged_exactly(lambda: key_spend(key), 0.0)
        assert charged_rows(upstream.model_id) == []


FOREIGN_IDS: Final[Mapping[str, Callable[[Upstream], str]]] = {
    "unknown": lambda _: f"resp_{uuid.uuid4().hex}",
    "five_kilobytes": lambda _: "resp_" + "a" * 5120,
    "raw_vendor_id": lambda upstream: f"resp_{upstream.model}",
}


@pytest.mark.timeout(240)
@pytest.mark.parametrize("foreign", FOREIGN_IDS.keys())
def test_a_read_of_an_id_the_proxy_never_issued_fails_without_tracking_or_billing_it(
    gateway: Gateway, foreign: str
) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream: Final = scripted_upstream(scenario, background_routes("completed"))
        response_id: Final = create_background_response(scenario, key, upstream.model)
        foreign_id: Final = FOREIGN_IDS[foreign](upstream)
        refused: Final = gateway.request("GET", f"/v1/responses/{foreign_id}", key=key)
        assert refused.status_code >= 400, refused.text
        assert managed_rows(foreign_id) == []
        assert (read_response(gateway, key, response_id),) in (
            settling_reads("completed", 1),
            free_reads("completed", 1),
        )
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert len(charged_rows(upstream.model_id)) == 1


@pytest.mark.timeout(240)
def test_a_background_create_answered_at_once_and_its_cached_twin_bill_the_create_alone(gateway: Gateway) -> None:
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        answered: Final = json_reply(response_body("completed"))
        upstream: Final = scripted_upstream(scenario, scripted_routes(answered, answered))
        log: Final = upstream_log(gateway, upstream.handle)
        first: Final = create_background_response(scenario, key, upstream.model)
        eventually(
            lambda: entry_key(store, f"msg_{upstream.model}"),
            lambda found: found is not None,
        )
        second: Final = create_background_response(scenario, key, upstream.model)
        assert len(submissions(next(log), upstream.handle)) == 1
        reads: Final = [read_response(gateway, key, response_id) for response_id in (first, second, first, second)]
        assert tuple(reads) == free_reads("completed", 4)
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        assert len(charged_rows(upstream.model_id)) == 1


def _end_user_spend(end_user: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id = %s', (end_user,))
    return number(rows[0]["spend"]) if rows else 0.0


def _charged_end_users(model_id: str) -> list[JsonValue]:
    rows: Final = read_rows('SELECT end_user FROM "LiteLLM_SpendLogs" WHERE model_id = %s AND spend > 0', (model_id,))
    return [row["end_user"] for row in rows]


@pytest.mark.timeout(240)
def test_a_background_response_bills_the_end_user_who_created_it_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        end_user: Final = f"audit-end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(gateway.request, "POST", "/end_user/delete", {"user_ids": [end_user]})
        upstream: Final = scripted_upstream(scenario, background_routes("completed"))
        response_id: Final = create_background_response(
            scenario,
            key,
            upstream.model,
            body={"model": upstream.model, "input": PROMPT, "background": True, "store": False, "user": end_user},
        )
        reads: Final = tuple(read_response(gateway, key, response_id) for _ in range(READS))
        assert reads in (settling_reads("completed", READS), free_reads("completed", READS))
        assert_charged_exactly(lambda: _end_user_spend(end_user), PRICE)
        assert key_spend(key) == pytest.approx(PRICE)
        assert _charged_end_users(upstream.model_id) == [end_user]


@pytest.mark.timeout(240)
def test_the_read_that_settles_a_background_response_splits_its_charge_into_input_and_output(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        _, response_id = _lose_the_row_then_finish(scenario, key)
        settling: Final = gateway.request("GET", f"/v1/responses/{response_id}", key=key)
        later: Final = tuple(read_response(gateway, key, response_id) for _ in range(READS - 1))
        assert read_of(settling) == Read("completed", "pong", pytest.approx(PRICE))
        split: Final = tuple(float(settling.headers[f"x-litellm-response-cost-{part}"]) for part in ("input", "output"))
        assert split == pytest.approx((INPUT_TOKENS * INPUT_COST_PER_TOKEN, OUTPUT_TOKENS * OUTPUT_COST_PER_TOKEN))
        assert later == free_reads("completed", READS - 1)
        assert_charged_exactly(lambda: key_spend(key), PRICE)


def _answered_at_once(scenario: Scenario, key: str) -> tuple[Upstream, str]:
    answered: Final = json_reply(response_body("completed"))
    upstream: Final = scripted_upstream(scenario, scripted_routes(answered, answered))
    return upstream, create_background_response(scenario, key, upstream.model)


def _stored_output(response_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT file_object->'output' AS output, position('pong' in file_object::text) > 0 AS holds_text "
        'FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id = %s',
        (response_id,),
    )


STORED_BY: Final[Mapping[str, Callable[[Scenario, str], tuple[Upstream, str]]]] = {
    "the_create": _answered_at_once,
    "a_read": _lose_the_row_then_finish,
}


@pytest.mark.timeout(240)
@pytest.mark.parametrize("stored_by", STORED_BY.keys())
def test_a_stored_background_response_keeps_none_of_the_model_output(gateway: Gateway, stored_by: str) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        _, response_id = STORED_BY[stored_by](scenario, key)
        assert read_response(gateway, key, response_id).text == "pong"
        assert _stored_output(response_id) == [{"output": [], "holds_text": False}]


def _model_usage(upstream: Upstream) -> tuple[float, int]:
    rows: Final = read_rows(
        "SELECT COALESCE(SUM(spend), 0) AS spend, COALESCE(SUM(request_count), 0)::bigint AS requests "
        'FROM "LiteLLM_DailyModelUsage" WHERE model_group IN (%s, %s)',
        (upstream.model, upstream.model_id),
    )
    requests: Final = rows[0]["requests"]
    assert isinstance(requests, int), rows
    return number(rows[0]["spend"]), requests


@pytest.mark.timeout(240)
def test_model_usage_counts_a_background_response_once_in_spend_and_only_client_calls_as_requests(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        upstream, response_id = _finished(scenario, key)
        reads: Final = tuple(read_response(gateway, key, response_id) for _ in range(READS))
        assert reads in (settling_reads("completed", READS), free_reads("completed", READS))
        assert_charged_exactly(lambda: key_spend(key), PRICE)
        client_calls: Final = 1 + READS
        held(
            lambda: _model_usage(upstream),
            lambda usage: costs(PRICE)(usage[0]) and usage[1] == client_calls,
            holding=POLL_WINDOW_SECONDS,
            seconds=POLL_WINDOW_SECONDS + SPEND_LANDING_SECONDS,
        )


BASE_MODEL: Final = "gpt-4o-mini"
BASE_MODEL_ONLY: Final[Mapping[str, JsonValue]] = {"base_model": BASE_MODEL}
BASE_MODEL_IN: Final[Mapping[str, tuple[Mapping[str, JsonValue] | None, Mapping[str, JsonValue]]]] = {
    "model_info": (BASE_MODEL_ONLY, {}),
    "litellm_params": (None, BASE_MODEL_ONLY),
}


def _unpriced_upstream(
    scenario: Scenario,
    unpriced: str,
    model_info: Mapping[str, JsonValue] | None,
    litellm_params: Mapping[str, JsonValue],
) -> Upstream:
    foreground: Final = json_reply(response_body("completed", background=False, model=unpriced))
    return scripted_upstream(
        scenario,
        scripted_routes(foreground, foreground),
        model_info=model_info,
        model=f"openai/{unpriced}",
        input_cost_per_token=None,
        output_cost_per_token=None,
        **litellm_params,
    )


@pytest.mark.timeout(240)
@pytest.mark.parametrize("base_model_in", BASE_MODEL_IN.keys())
def test_a_background_response_on_a_base_model_deployment_bills_what_a_foreground_response_costs(
    gateway: Gateway, base_model_in: str
) -> None:
    with gateway.scenario() as scenario:
        foreground_key: Final = scenario.key()
        background_key: Final = scenario.key()
        unpriced: Final = f"audit-unpriced-{uuid.uuid4().hex[:12]}"
        reference: Final = _unpriced_upstream(scenario, unpriced, BASE_MODEL_ONLY, {})
        created: Final = gateway.request(
            "POST", "/v1/responses", {"model": reference.model, "input": PROMPT, "store": False}, key=foreground_key
        )
        foreground_cost: Final = read_of(created).cost
        assert foreground_cost is not None and foreground_cost > 0, created.headers
        upstream: Final = _unpriced_upstream(scenario, unpriced, *BASE_MODEL_IN[base_model_in])
        upstream.script(background_routes("completed", model=unpriced))
        response_id: Final = create_background_response(scenario, background_key, upstream.model)
        reads: Final = tuple(read_response(gateway, background_key, response_id) for _ in range(READS))
        settled: Final = (
            Read("completed", "pong", pytest.approx(foreground_cost)),
            *free_reads("completed", READS - 1),
        )
        assert reads in (settled, free_reads("completed", READS))
        assert_charged_exactly(lambda: key_spend(background_key), foreground_cost)
        background_rows: Final = [
            row for row in charged_rows(upstream.model_id) if row["api_key"] == hashed(background_key)
        ]
        assert [number(row["spend"]) for row in background_rows] == [pytest.approx(foreground_cost)]


@pytest.mark.timeout(240)
def test_a_client_cannot_make_a_background_create_free_by_forging_the_managed_marker(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        answered: Final = json_reply(response_body("completed"))
        upstream: Final = scripted_upstream(scenario, scripted_routes(answered, answered))
        forged: Final[dict[str, JsonValue]] = {MANAGED_READ_MARKER: True}
        created: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": upstream.model,
                "input": PROMPT,
                "background": True,
                "store": False,
                "metadata": forged,
                "litellm_metadata": forged,
            },
            key=key,
        )
        assert read_of(created) == Read("completed", "pong", pytest.approx(PRICE))
        response_id: Final = string_value(object_value(created.json())["id"])
        scenario.cleanups.callback(forget_row, response_id)
        assert tuple(read_response(gateway, key, response_id) for _ in range(2)) == free_reads("completed", 2)
        assert_charged_exactly(lambda: key_spend(key), PRICE)
