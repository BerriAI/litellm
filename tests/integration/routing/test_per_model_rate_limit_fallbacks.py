from __future__ import annotations

import json
import re
import signal
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing.test_payment_required_mapping import OPENAI_MODEL, is_model_info_probe, model_list_reply
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

SETTING: Final = "disable_fallbacks_on_per_model_rate_limits"
FRESH_CONNECTION: Final = MappingProxyType({"Connection": "close"})
MARKER: Final = re.compile(rb"pmrl-[0-9a-f]{32}")
WORKER_STARTED: Final = re.compile(r"Started server process \[(\d+)\]")
STARTUP_COMPLETE: Final = "Application startup complete."
CONFIG_LIST: Final = TypeAdapter(list[dict[str, JsonValue]])
WORKER_PIDS: Final = TypeAdapter(tuple[int, ...])
BURST: Final = 8
TOKEN_LIMIT: Final = 100
INPUT_TOKEN_LIMIT: Final = 120
MAX_TOKENS: Final = 60
METERED_USAGE: Final = MappingProxyType({"prompt_tokens": 900, "completion_tokens": 100, "total_tokens": 1000})
MALFORMED_VALUES: Final[tuple[tuple[str, JsonValue], ...]] = (
    ("word", "sometimes"),
    ("empty", ""),
    ("five_kb", "x" * 5120),
    ("list", [True]),
    ("dict", {"enabled": True}),
    ("int", 2),
)


@dataclass(frozen=True, slots=True)
class Groups:
    primary: str
    fallback: str
    last: str
    metered: str
    gated_hard: str
    gated_soft: str
    capped: str

    def names(self) -> tuple[str, ...]:
        return (self.primary, self.fallback, self.last, self.metered, self.gated_hard, self.gated_soft, self.capped)


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    log: Path
    groups: Groups
    wires: Mapping[str, Wire]

    def traffic(self) -> dict[str, tuple[str, ...]]:
        return {group: markers_on(wire) for group, wire in self.wires.items()}

    def expected(self, served: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
        return {group: tuple(sorted(served.get(group, ()))) for group in self.wires}


def new_marker() -> str:
    return f"pmrl-{uuid.uuid4().hex}"


def new_groups() -> Groups:
    suffix: Final = uuid.uuid4().hex[:10]
    return Groups(
        primary=f"per-model-primary-{suffix}",
        fallback=f"per-model-fallback-{suffix}",
        last=f"per-model-last-{suffix}",
        metered=f"per-model-metered-{suffix}",
        gated_hard=f"per-model-gated-hard-{suffix}",
        gated_soft=f"per-model-gated-soft-{suffix}",
        capped=f"per-model-capped-{suffix}",
    )


def marker_in(request: Request) -> str:
    found: Final = MARKER.search(request.body)
    assert found is not None, (request.method, request.target, request.body[:300])
    return found.group(0).decode()


def markers_on(wire: Wire) -> tuple[str, ...]:
    return tuple(sorted(marker_in(request) for request in wire.drain() if not is_model_info_probe(request)))


def streamed(request: Request) -> bool:
    return JSON_OBJECT.validate_json(request.body).get("stream") is True


def healthy_reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    marker: Final = marker_in(request)
    if request.target.endswith("/responses"):
        return responses_reply(f"resp_{marker}", OPENAI_MODEL, f"served {marker}", stream=streamed(request))
    return chat_reply(f"chatcmpl-{marker}", OPENAI_MODEL, f"served {marker}", stream=streamed(request))


def metered_reply(request: Request) -> Reply:
    if is_model_info_probe(request):
        return model_list_reply()
    marker: Final = marker_in(request)
    body: Final = {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion",
        "created": 1,
        "model": OPENAI_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": f"served {marker}"}, "finish_reason": "stop"}
        ],
        "usage": dict(METERED_USAGE),
    }
    return Reply(body=json.dumps(body).encode())


def model_entry(group: str, wire: Wire, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {
            "model": OPENAI_MODEL,
            "api_base": wire.url + "/v1",
            "api_key": "synthetic-openai-key",
            **extra,
        },
    }


def proxy_config(
    directory: Path,
    *,
    model_list: Sequence[Mapping[str, JsonValue]],
    fallbacks: Sequence[Mapping[str, JsonValue]],
    general_settings: Mapping[str, JsonValue],
    callbacks: Sequence[str] = (),
) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    litellm_settings: Final = object_value(base["litellm_settings"])
    config: Final = {
        **base,
        "model_list": [dict(entry) for entry in model_list],
        "general_settings": {**object_value(base["general_settings"]), **general_settings},
        "litellm_settings": {**litellm_settings, **({"callbacks": list(callbacks)} if callbacks else {})},
        "router_settings": {
            **object_value(base["router_settings"]),
            "num_retries": 0,
            "fallbacks": [dict(entry) for entry in fallbacks],
        },
    }
    path: Final = directory / f"per-model-limits-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def enforcing_config(directory: Path, groups: Groups, wires: Mapping[str, Wire]) -> Path:
    return proxy_config(
        directory,
        model_list=(
            model_entry(groups.primary, wires[groups.primary]),
            model_entry(groups.fallback, wires[groups.fallback]),
            model_entry(groups.last, wires[groups.last]),
            model_entry(groups.metered, wires[groups.metered]),
            model_entry(groups.gated_hard, wires[groups.gated_hard], rpm=1),
            model_entry(groups.gated_soft, wires[groups.gated_soft], rpm=1),
            model_entry(groups.capped, wires[groups.capped]),
        ),
        fallbacks=(
            {groups.primary: [groups.fallback]},
            {groups.metered: [groups.fallback]},
            {groups.gated_hard: [groups.capped, groups.last]},
            {groups.gated_soft: [groups.fallback]},
        ),
        general_settings={SETTING: True},
        callbacks=("dynamic_rate_limiter_v3",),
    )


@pytest.fixture(scope="module")
def enforcing(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("per-model-limits")
    groups: Final = new_groups()
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        wires: Final = MappingProxyType(
            {
                group: stack.enter_context(wire_server(metered_reply if group == groups.metered else healthy_reply))
                for group in groups.names()
            }
        )
        owned: Final = stack.enter_context(
            owned_proxy_process(
                rig_gateway, directory, {}, config=enforcing_config(directory, groups, wires), workers=2
            )
        )
        yield Rig(owned.gateway, owned.log, groups, wires)


def chat(gateway: Gateway, key: str, group: str, content: str, **extra: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": group, "messages": [{"role": "user", "content": content}], **extra},
        key=key,
        headers=FRESH_CONNECTION,
    )


def responses_call(gateway: Gateway, key: str, group: str, marker: str, *, stream: bool) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/responses", {"model": group, "input": marker, "stream": stream}, key=key, headers=FRESH_CONNECTION
    )


def openai_sdk(gateway: Gateway, key: str) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        api_key=key,
        max_retries=0,
        default_headers=dict(FRESH_CONNECTION),
    )


def openai_async_sdk(gateway: Gateway, key: str) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=str(gateway.client.base_url).rstrip("/") + "/v1",
        api_key=key,
        max_retries=0,
        default_headers=dict(FRESH_CONNECTION),
    )


def anthropic_sdk(gateway: Gateway, key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url).rstrip("/"),
        api_key=key,
        max_retries=0,
        default_headers=dict(FRESH_CONNECTION),
    )


def anthropic_async_sdk(gateway: Gateway, key: str) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url).rstrip("/"),
        api_key=key,
        max_retries=0,
        default_headers=dict(FRESH_CONNECTION),
    )


def error_message(body: str) -> str:
    return string_value(object_value(JSON_OBJECT.validate_json(body)["error"])["message"])


def assert_limit_message(body: str, descriptor: str, group: str, kind: str, limit: int) -> None:
    message: Final = error_message(body)
    assert f"Rate limit exceeded for {descriptor}: " in message, message
    assert f":{group}. Limit type: {kind}. Current limit: {limit}, Remaining: " in message, message


def assert_refused(
    response: httpx.Response, descriptor: str, group: str, *, kind: str = "requests", limit: int = 1
) -> None:
    assert response.status_code == 429, (
        response.status_code,
        response.headers.get("x-litellm-model-group"),
        response.text,
    )
    assert_limit_message(response.text, descriptor, group, kind, limit)


def assert_served(response: httpx.Response, group: str, identity: str) -> None:
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-group"] == group, (
        response.headers.get("x-litellm-model-group"),
        response.text,
    )
    assert identity in response.text, response.text


def burst(gateway: Gateway, calls: Sequence[tuple[str, str]], group: str) -> tuple[httpx.Response, ...]:
    def send(call: tuple[str, str]) -> httpx.Response:
        return chat(gateway, call[0], group, call[1])

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return tuple(pool.map(send, calls))


def served_once_in_a_burst(rig: Rig, markers: Sequence[str], responses: Sequence[httpx.Response]) -> str:
    assert Counter(response.status_code for response in responses) == Counter({200: 1, 429: BURST - 1}), tuple(
        (response.status_code, response.headers.get("x-litellm-model-group"), response.text[:200])
        for response in responses
    )
    refusals: Final = tuple(response for response in responses if response.status_code == 429)
    for response in refusals:
        assert_refused(response, "model_per_key", rig.groups.primary)
    (served,) = (marker for marker, response in zip(markers, responses) if response.status_code == 200)
    return served


def worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text(errors="replace")
    return WORKER_PIDS.validate_python(WORKER_STARTED.findall(text)), text.count(STARTUP_COMPLETE)


def listed_setting(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    response: Final = gateway.request("GET", "/config/list", params={"config_type": "general_settings"})
    assert response.status_code == 200, response.text
    return tuple(entry for entry in CONFIG_LIST.validate_json(response.content) if entry["field_name"] == SETTING)


def update_setting(gateway: Gateway, value: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/config/field/update",
        {"field_name": SETTING, "field_value": value, "config_type": "general_settings"},
    )


def scope_key_fields(scenario: Scenario, scope: str, group: str) -> dict[str, JsonValue]:
    limit: Final[dict[str, JsonValue]] = {group: 1}
    if scope == "team":
        return {"team_id": scenario.team(model_rpm_limit=limit)}
    if scope == "organization":
        return {"team_id": scenario.team(organization_id=scenario.organization(model_rpm_limit=limit))}
    team: Final = scenario.team()
    return {"team_id": team, "project_id": scenario.project(team, model_rpm_limit=limit)}


def test_chat_openai_sdk_over_key_model_rpm_gets_429_instead_of_the_fallback(enforcing: Rig) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        client: Final = openai_sdk(enforcing.gateway, scenario.key(model_rpm_limit={primary: 1}))
        served: Final = client.chat.completions.with_raw_response.create(
            model=primary, messages=[{"role": "user", "content": first}]
        )
        assert served.headers["x-litellm-model-group"] == primary
        assert served.parse().id == f"chatcmpl-{first}"
        with pytest.raises(openai.RateLimitError) as refused:
            client.chat.completions.create(model=primary, messages=[{"role": "user", "content": second}])
        assert_limit_message(refused.value.response.text, "model_per_key", primary, "requests", 1)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


async def test_chat_stream_openai_async_sdk_over_key_model_rpm_gets_429_instead_of_the_fallback(
    enforcing: Rig,
) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        async with openai_async_sdk(enforcing.gateway, scenario.key(model_rpm_limit={primary: 1})) as client:
            stream: Final = await client.chat.completions.create(
                model=primary, messages=[{"role": "user", "content": first}], stream=True
            )
            identities: Final = {chunk.id async for chunk in stream}
            assert identities == {f"chatcmpl-{first}"}
            with pytest.raises(openai.RateLimitError) as refused:
                await client.chat.completions.create(
                    model=primary, messages=[{"role": "user", "content": second}], stream=True
                )
        assert_limit_message(refused.value.response.text, "model_per_key", primary, "requests", 1)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


def test_messages_anthropic_sdk_over_key_model_rpm_gets_429_instead_of_the_fallback(enforcing: Rig) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        client: Final = anthropic_sdk(enforcing.gateway, scenario.key(model_rpm_limit={primary: 1}))
        served: Final = client.messages.with_raw_response.create(
            model=primary, max_tokens=32, messages=[{"role": "user", "content": first}]
        )
        assert served.headers["x-litellm-model-group"] == primary
        block: Final = served.parse().content[0]
        assert isinstance(block, anthropic.types.TextBlock), block
        assert block.text == f"served {first}"
        with pytest.raises(anthropic.RateLimitError) as refused:
            client.messages.create(model=primary, max_tokens=32, messages=[{"role": "user", "content": second}])
        assert_limit_message(refused.value.response.text, "model_per_key", primary, "requests", 1)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


async def test_messages_stream_anthropic_async_sdk_over_key_model_rpm_gets_429_instead_of_the_fallback(
    enforcing: Rig,
) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        async with anthropic_async_sdk(enforcing.gateway, scenario.key(model_rpm_limit={primary: 1})) as client:
            async with client.messages.stream(
                model=primary, max_tokens=32, messages=[{"role": "user", "content": first}]
            ) as served:
                text: Final = await served.get_final_text()
            assert text == f"served {first}"
            with pytest.raises(anthropic.RateLimitError) as refused:
                async with client.messages.stream(
                    model=primary, max_tokens=32, messages=[{"role": "user", "content": second}]
                ) as rejected:
                    await rejected.get_final_text()
        assert_limit_message(refused.value.response.text, "model_per_key", primary, "requests", 1)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_responses_httpx_over_key_model_rpm_gets_429_instead_of_the_fallback(enforcing: Rig, stream: bool) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={primary: 1})
        assert_served(responses_call(enforcing.gateway, key, primary, first, stream=stream), primary, f"resp_{first}")
        assert_refused(responses_call(enforcing.gateway, key, primary, second, stream=stream), "model_per_key", primary)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


@pytest.mark.parametrize(
    ("scope", "descriptor"),
    (("team", "model_per_team"), ("organization", "model_per_organization"), ("project", "model_per_project")),
    ids=("team", "organization", "project"),
)
def test_chat_over_a_scope_model_rpm_gets_429_instead_of_the_fallback(
    enforcing: Rig, scope: str, descriptor: str
) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        fields: Final = scope_key_fields(scenario, scope, primary)
        first_key, second_key = scenario.key(**fields), scenario.key(**fields)
        assert_served(chat(enforcing.gateway, first_key, primary, first), primary, f"chatcmpl-{first}")
        assert_refused(chat(enforcing.gateway, second_key, primary, second), descriptor, primary)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


def test_chat_over_project_model_itpm_gets_429_instead_of_the_fallback(enforcing: Rig) -> None:
    enforcing.traffic()
    metered: Final = enforcing.groups.metered
    first, second = new_marker(), new_marker()
    filler: Final = " word" * 50
    with enforcing.gateway.scenario() as scenario:
        team: Final = scenario.team()
        key: Final = scenario.key(
            team_id=team, project_id=scenario.project(team, model_itpm_limit={metered: INPUT_TOKEN_LIMIT})
        )
        assert_served(chat(enforcing.gateway, key, metered, first + filler), metered, f"chatcmpl-{first}")
        assert_refused(
            chat(enforcing.gateway, key, metered, second + filler),
            "model_per_project_itpm",
            metered,
            kind="tokens",
            limit=INPUT_TOKEN_LIMIT,
        )
    assert enforcing.traffic() == enforcing.expected({metered: (first,)})


def test_chat_over_key_model_tpm_gets_429_instead_of_the_fallback(enforcing: Rig) -> None:
    enforcing.traffic()
    metered: Final = enforcing.groups.metered
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key(model_tpm_limit={metered: TOKEN_LIMIT})
        assert_served(chat(enforcing.gateway, key, metered, first, max_tokens=MAX_TOKENS), metered, f"chatcmpl-{first}")
        assert_refused(
            chat(enforcing.gateway, key, metered, second, max_tokens=MAX_TOKENS),
            "model_per_key",
            metered,
            kind="tokens",
            limit=TOKEN_LIMIT,
        )
    assert enforcing.traffic() == enforcing.expected({metered: (first,)})


def test_key_router_settings_fallbacks_are_skipped_on_a_per_model_429(enforcing: Rig) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key(
            model_rpm_limit={primary: 1}, router_settings={"fallbacks": [{primary: [enforcing.groups.last]}]}
        )
        assert_served(chat(enforcing.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        assert_refused(chat(enforcing.gateway, key, primary, second), "model_per_key", primary)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


def test_capacity_429_falls_back_but_a_capped_fallback_answers_429_instead_of_the_next_one(enforcing: Rig) -> None:
    enforcing.traffic()
    groups: Final = enforcing.groups
    warm, spend, refused = new_marker(), new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={groups.capped: 1})
        assert_served(chat(enforcing.gateway, key, groups.gated_hard, warm), groups.gated_hard, f"chatcmpl-{warm}")
        assert_served(chat(enforcing.gateway, key, groups.capped, spend), groups.capped, f"chatcmpl-{spend}")
        assert_refused(chat(enforcing.gateway, key, groups.gated_hard, refused), "model_per_key", groups.capped)
    assert enforcing.traffic() == enforcing.expected({groups.gated_hard: (warm,), groups.capped: (spend,)})


def test_model_capacity_429_without_a_per_model_limit_still_falls_back(enforcing: Rig) -> None:
    enforcing.traffic()
    groups: Final = enforcing.groups
    warm, moved = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key()
        assert_served(chat(enforcing.gateway, key, groups.gated_soft, warm), groups.gated_soft, f"chatcmpl-{warm}")
        assert_served(chat(enforcing.gateway, key, groups.gated_soft, moved), groups.fallback, f"chatcmpl-{moved}")
    assert enforcing.traffic() == enforcing.expected({groups.gated_soft: (warm,), groups.fallback: (moved,)})


def test_key_rpm_limit_without_a_model_scope_answers_429_on_both_the_primary_and_the_fallback(enforcing: Rig) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key(rpm_limit=1)
        assert_served(chat(enforcing.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        refused: Final = chat(enforcing.gateway, key, primary, second)
        assert refused.status_code == 429, refused.text
        assert error_message(refused.text).startswith("Rate limit exceeded for api_key: "), refused.text
        assert ". Limit type: requests. Current limit: 1, Remaining: 0." in refused.text, refused.text
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


def test_response_cache_hits_count_toward_key_model_rpm_and_the_next_request_gets_429(enforcing: Rig) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    marker: Final = new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={primary: 2})
        assert_served(chat(enforcing.gateway, key, primary, marker), primary, f"chatcmpl-{marker}")
        assert_served(chat(enforcing.gateway, key, primary, marker), primary, f"chatcmpl-{marker}")
        assert_refused(chat(enforcing.gateway, key, primary, marker), "model_per_key", primary, limit=2)
    assert enforcing.traffic() == enforcing.expected({primary: (marker,)})


def test_concurrent_burst_over_key_model_rpm_serves_one_and_refuses_the_rest_without_fallback(enforcing: Rig) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    markers: Final = tuple(new_marker() for _ in range(BURST))
    bystander: Final = new_marker()
    with enforcing.gateway.scenario() as scenario:
        limited: Final = scenario.key(model_rpm_limit={primary: 1})
        unlimited: Final = scenario.key()
        calls: Final = (*((limited, marker) for marker in markers), (unlimited, bystander))
        responses: Final = burst(enforcing.gateway, calls, primary)
    served: Final = served_once_in_a_burst(enforcing, markers, responses[:BURST])
    assert_served(responses[BURST], primary, f"chatcmpl-{bystander}")
    assert enforcing.traffic() == enforcing.expected({primary: (served, bystander)})


def test_yaml_owned_setting_refuses_a_database_write_and_keeps_enforcing(enforcing: Rig) -> None:
    enforcing.traffic()
    primary: Final = enforcing.groups.primary
    refused_write: Final = update_setting(enforcing.gateway, False)
    assert refused_write.status_code == 400, refused_write.text
    detail: Final = object_value(JSON_OBJECT.validate_json(refused_write.content)["detail"])
    assert detail["error"] == (
        f"general_settings key '{SETTING}' is set in the config file and cannot be changed here."
    ), detail
    assert detail["keys"] == [SETTING], detail
    listed: Final = listed_setting(enforcing.gateway)
    assert [(entry["field_type"], entry["field_value"], entry["editable"]) for entry in listed] == [
        ("Boolean", True, False)
    ], listed
    first, second = new_marker(), new_marker()
    with enforcing.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={primary: 1})
        assert_served(chat(enforcing.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        assert_refused(chat(enforcing.gateway, key, primary, second), "model_per_key", primary)
    assert enforcing.traffic() == enforcing.expected({primary: (first,)})


def test_worker_sigkill_and_replacement_keeps_refusing_per_model_overflow_without_fallback(enforcing: Rig) -> None:
    enforcing.traffic()
    pids, ready = eventually(
        lambda: worker_startups(enforcing.log), lambda found: len(found[0]) >= 2 and found[1] >= 2, seconds=60
    )
    victim: Final = next(pid for pid in pids if psutil.pid_exists(pid))
    psutil.Process(victim).send_signal(signal.SIGKILL)
    eventually(
        lambda: worker_startups(enforcing.log),
        lambda found: len(found[0]) > len(pids) and found[1] > ready,
        seconds=graceful_stop_seconds(),
    )
    primary: Final = enforcing.groups.primary
    markers: Final = tuple(new_marker() for _ in range(BURST))
    with enforcing.gateway.scenario() as scenario:
        limited: Final = scenario.key(model_rpm_limit={primary: 1})
        responses: Final = burst(enforcing.gateway, tuple((limited, marker) for marker in markers), primary)
    served: Final = served_once_in_a_burst(enforcing, markers, responses)
    assert enforcing.traffic() == enforcing.expected({primary: (served,)})


def test_config_list_reports_the_setting_as_an_unset_boolean(gateway: Gateway) -> None:
    listed: Final = listed_setting(gateway)
    assert [(entry["field_type"], entry["field_value"], entry["stored_in_db"]) for entry in listed] == [
        ("Boolean", None, None)
    ], listed


@pytest.mark.parametrize(
    "value", tuple(value for _, value in MALFORMED_VALUES), ids=tuple(name for name, _ in MALFORMED_VALUES)
)
def test_config_field_update_rejects_a_non_boolean_value_and_stores_nothing(gateway: Gateway, value: JsonValue) -> None:
    response: Final = update_setting(gateway, value)
    try:
        assert response.status_code == 400, response.text
        assert JSON_OBJECT.validate_json(response.content) == {
            "detail": {"error": f"Invalid type of field value={type(value)} passed in."}
        }, response.text
        listed: Final = listed_setting(gateway)
        assert [(entry["field_value"], entry["stored_in_db"]) for entry in listed] == [(None, None)], listed
    finally:
        if response.status_code == 200:
            gateway.post("/config/field/delete", {"field_name": SETTING, "config_type": "general_settings"})
