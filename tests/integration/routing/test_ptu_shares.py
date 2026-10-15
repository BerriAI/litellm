import asyncio
import json
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

import httpx
import pytest
import yaml
from anthropic import Anthropic
from integration._support.anthropic_sse import (
    LIFECYCLE,
    Attempts,
    delta_text,
    dropping_reply,
    event_types,
    message_id,
    message_json,
    message_stream,
    parse_sse,
    stream_reply,
)
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.openai_wire import chat_reply, openai_error, responses_reply
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter
from redis import Redis

if TYPE_CHECKING:
    from litellm.llms.azure.ptu_capacity import PTUCapacity

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_SHARED: Final = "ptu-shared"
_MIXED: Final = "ptu-mixed"
_PAIR: Final = "ptu-pair"
_CLAUDE: Final = "ptu-claude"
_SHARED_DEPLOYMENT: Final = "shared-gpt41"
_MIXED_SHARED_DEPLOYMENT: Final = "mixed-shared-gpt41"
_MIXED_OPEN_DEPLOYMENT: Final = "mixed-open-gpt41"
_PAIR_DEPLOYMENTS: Final = ("pair-a-gpt41", "pair-b-gpt41")
_CLAUDE_BACKEND: Final = "claude-ptu-backend"
_AZURE_KEY: Final = "synthetic-azure-key"
_ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
_UPSTREAM_URL_PLACEHOLDER: Final = "upstream-url-placeholder"
_ATTRIBUTION_ON: Final = MappingProxyType({"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "True"})
_ATTRIBUTION_OFF: Final = MappingProxyType({"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "False"})
_RESERVED: Final = "is reserved for the teams holding a PTU share of it"
_TEXT: Final = "served"
_SMALL_MAX_TOKENS: Final = 16
_SETTLE_USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"prompt_tokens": 5, "completion_tokens": 700, "total_tokens": 705}
)
_CACHED_USAGE: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "prompt_tokens": 1000,
        "completion_tokens": 10,
        "total_tokens": 1010,
        "prompt_tokens_details": {"cached_tokens": 800},
    }
)
_SHARE_ROLES: Final = (
    "holder",
    "settle",
    "cached",
    "refund",
    "twin",
    "oversize",
    "retry",
    "queue",
    "burst",
    "worker",
)
_ROLES: Final = (*_SHARE_ROLES, "outsider", "mixed", "pair", "messages")
_COUNTER: Final = TypeAdapter(int)
_CONFIG: Final = TypeAdapter(dict[str, JsonValue])

Surface = Literal["chat", "responses", "messages"]
Behavior = Literal["ok", "settle", "cached", "fail", "fail-once", "hold", "drop-once"]


def _gpt41() -> "PTUCapacity":
    from litellm.llms.azure.ptu_capacity import AZURE_PTU_CAPACITY

    return AZURE_PTU_CAPACITY["gpt-4.1"]


def _text_of(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_text_of(part) for part in content)
    if isinstance(content, dict):
        return _text_of(content.get("text", content.get("content", "")))
    return ""


def _prompt(body: Mapping[str, JsonValue]) -> str:
    messages: Final = body.get("messages")
    if isinstance(messages, list) and messages:
        return _text_of(messages[-1])
    return _text_of(body.get("input", ""))


def _deployment_of(request: Request, body: Mapping[str, JsonValue]) -> str:
    _, found, rest = request.target.partition("/openai/deployments/")
    return rest.split("/", 1)[0] if found else str(body.get("model"))


def _usage_reply(identity: str, model: str, usage: Mapping[str, JsonValue]) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": _TEXT}, "finish_reason": "stop"}],
                "usage": dict(usage),
            }
        ).encode()
    )


def _anthropic_reply(identity: str, behavior: str, attempt: int, stream: bool) -> Reply:
    full: Final = message_stream(identity, _CLAUDE_BACKEND, _TEXT)
    if behavior == "drop-once" and attempt == 1:
        return dropping_reply(full, abort_after=1)
    if stream:
        return stream_reply(full)
    return Reply(body=message_json(identity, _CLAUDE_BACKEND, _TEXT))


def _served_id(deployment: str, marker: str, attempt: int) -> str:
    return f"chatcmpl-{deployment}-{marker}-a{attempt}"


def _openai_reply(request: Request, behavior: str, identity: str, deployment: str, attempt: int, stream: bool) -> Reply:
    match behavior:
        case "settle":
            return _usage_reply(identity, deployment, _SETTLE_USAGE)
        case "cached":
            return _usage_reply(identity, deployment, _CACHED_USAGE)
        case "fail":
            return openai_error(500)
        case "fail-once" if attempt == 1:
            return openai_error(500)
        case _:
            pass
    if "/responses" in request.target:
        return responses_reply(identity, deployment, _TEXT, stream=stream)
    return chat_reply(identity, deployment, _TEXT, stream=stream)


def _respond(attempts: Attempts, release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        body: Final = JSON_OBJECT.validate_json(request.body)
        behavior, _, marker = _prompt(body).partition(":")
        deployment: Final = _deployment_of(request, body)
        attempt: Final = attempts.record(f"{deployment}:{marker}")
        identity: Final = _served_id(deployment, marker, attempt)
        stream: Final = body.get("stream") is True
        if behavior == "hold":
            release.wait(timeout=30)
        if request.target.startswith("/v1/messages"):
            return _anthropic_reply(identity, behavior, attempt, stream)
        return _openai_reply(request, behavior, identity, deployment, attempt, stream)

    return respond


def _ptu_terms(shares: Mapping[str, int]) -> dict[str, JsonValue]:
    return {
        "base_model": "azure/gpt-4.1",
        "ptu_count": sum(shares.values()),
        "cost_per_ptu_per_hour": 2.0,
        "ptu_effective_from": "2026-01-01T00:00:00Z",
        "ptu_shares": dict(shares),
    }


def _azure(model_name: str, deployment: str, model_info: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"azure/{deployment}",
            "api_base": _UPSTREAM_URL_PLACEHOLDER,
            "api_key": _AZURE_KEY,
            "api_version": "2025-03-01-preview",
        },
        "model_info": {"id": f"{deployment}-{uuid.uuid4().hex}", **model_info},
    }


def _model_list(teams: Mapping[str, str]) -> list[JsonValue]:
    one_each: Final = {teams[role]: 1 for role in _SHARE_ROLES}
    return [
        _azure(_SHARED, _SHARED_DEPLOYMENT, _ptu_terms(one_each)),
        _azure(_MIXED, _MIXED_SHARED_DEPLOYMENT, _ptu_terms({teams["mixed"]: 1})),
        _azure(_MIXED, _MIXED_OPEN_DEPLOYMENT, {"base_model": "azure/gpt-4.1"}),
        *(_azure(_PAIR, deployment, _ptu_terms({teams["pair"]: 1})) for deployment in _PAIR_DEPLOYMENTS),
        {
            "model_name": _CLAUDE,
            "litellm_params": {
                "model": f"anthropic/{_CLAUDE_BACKEND}",
                "api_base": _UPSTREAM_URL_PLACEHOLDER,
                "api_key": _ANTHROPIC_KEY,
            },
            "model_info": {"id": f"claude-{uuid.uuid4().hex}", **_ptu_terms({teams["messages"]: 1})},
        },
    ]


def _config(directory: Path, wire_url: str, model_list: list[JsonValue], name: str) -> Path:
    config: Final = {
        **_CONFIG.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())),
        "model_list": model_list,
        "router_settings": {"disable_cooldowns": True, "num_retries": 0, "ignore_invalid_deployments": True},
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config).replace(_UPSTREAM_URL_PLACEHOLDER, wire_url))
    return path


def _team_ids() -> Mapping[str, str]:
    run: Final = uuid.uuid4().hex[:12]
    return MappingProxyType({role: f"ptu-{role}-{run}" for role in _ROLES})


def _create_team(proxy: Gateway, team_id: str) -> str:
    proxy.post("/team/new", {"team_id": team_id, "team_alias": team_id})
    return string_value(proxy.post("/key/generate", {"team_id": team_id})["key"])


def _delete_teams(gateway: Gateway, team_ids: tuple[str, ...]) -> None:
    gateway.post("/team/delete", {"team_ids": list(team_ids)})


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    attempts: Attempts
    release: threading.Event
    teams: Mapping[str, str]
    keys: Mapping[str, str]

    @property
    def proxy(self) -> Gateway:
        return self.owned.gateway

    def chat(self, model: str, role: str | None, behavior: Behavior, marker: str, **extra: JsonValue) -> httpx.Response:
        return self.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "max_tokens": _SMALL_MAX_TOKENS,
                "messages": [{"role": "user", "content": f"{behavior}:{marker}"}],
                **extra,
            },
            key=self.keys[role] if role is not None else None,
        )

    def send(self, surface: Surface, model: str, role: str | None, marker: str) -> httpx.Response:
        key: Final = self.keys[role] if role is not None else None
        prompt: Final = f"ok:{marker}"
        match surface:
            case "chat":
                return self.chat(model, role, "ok", marker)
            case "responses":
                return self.proxy.request(
                    "POST",
                    "/v1/responses",
                    {"model": model, "input": prompt, "max_output_tokens": _SMALL_MAX_TOKENS},
                    key=key,
                )
            case "messages":
                return self.proxy.request(
                    "POST",
                    "/v1/messages",
                    {
                        "model": model,
                        "max_tokens": _SMALL_MAX_TOKENS,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    key=key,
                )

    def hits(self, deployment: str, marker: str) -> int:
        return self.attempts.count(f"{deployment}:{marker}")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    attempts: Final = Attempts()
    release: Final = threading.Event()
    teams: Final = _team_ids()
    directory: Final = tmp_path_factory.mktemp("ptu-shares")
    with gateway_from_environment() as gateway, wire_server(_respond(attempts, release)) as wire:
        config: Final = _config(directory, wire.url, _model_list(teams), "ptu-shares")
        try:
            with owned_proxy_process(gateway, directory, _ATTRIBUTION_ON, config=config, workers=2) as owned:
                keys: Final = MappingProxyType(
                    {role: _create_team(owned.gateway, team) for role, team in teams.items()}
                )
                yield _Rig(owned, attempts, release, teams, keys)
        finally:
            release.set()
            _delete_teams(gateway, tuple(teams.values()))


def _ptu_counter(team_id: str, model_group: str) -> int | None:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as client:
        keys: Final = tuple(client.scan_iter(match=f"*model_per_team_ptu:{team_id}:{model_group}}}:tokens"))
        assert len(keys) <= 1, keys
        raw: Final[object] = client.get(keys[0]) if keys else None
    return None if raw is None else _COUNTER.validate_python(raw)


def _settled_counter(team_id: str, model_group: str, expected: int) -> int | None:
    return eventually(
        lambda: _ptu_counter(team_id, model_group),
        lambda observed: observed == expected,
        seconds=15,
        return_last_on_timeout=True,
    )


def _error_message(response: httpx.Response) -> str:
    return string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])


@pytest.mark.parametrize("client", ("sync-sdk", "async-sdk-stream"))
def test_a_share_holder_is_served_chat_by_the_shared_deployment(rig: _Rig, client: str) -> None:
    marker: Final = uuid.uuid4().hex
    base_url: Final = f"{rig.proxy.client.base_url}/v1"
    if client == "sync-sdk":
        with OpenAI(api_key=rig.keys["holder"], base_url=base_url, max_retries=0, timeout=30) as sdk:
            completion: Final = sdk.chat.completions.create(
                model=_SHARED,
                max_tokens=_SMALL_MAX_TOKENS,
                messages=[{"role": "user", "content": f"ok:{marker}"}],
            )
        assert (completion.id, completion.choices[0].message.content) == (
            _served_id(_SHARED_DEPLOYMENT, marker, 1),
            _TEXT,
        ), completion

    async def stream() -> tuple[str, str]:
        async with AsyncOpenAI(api_key=rig.keys["holder"], base_url=base_url, max_retries=0, timeout=30) as sdk:
            chunks: Final = await sdk.chat.completions.create(
                model=_SHARED,
                max_tokens=_SMALL_MAX_TOKENS,
                messages=[{"role": "user", "content": f"ok:{marker}"}],
                stream=True,
            )
            received: Final = tuple([chunk async for chunk in chunks])
        return (
            received[0].id,
            "".join(choice.delta.content or "" for chunk in received for choice in chunk.choices),
        )

    if client == "async-sdk-stream":
        assert asyncio.run(stream()) == (_served_id(_SHARED_DEPLOYMENT, marker, 1), _TEXT)
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 1


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_a_share_holder_is_served_responses_by_the_shared_deployment(rig: _Rig, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.proxy.request(
        "POST",
        "/v1/responses",
        {"model": _SHARED, "input": f"ok:{marker}", "max_output_tokens": _SMALL_MAX_TOKENS, "stream": stream},
        key=rig.keys["holder"],
    )
    assert response.status_code == 200, response.text
    assert _served_id(_SHARED_DEPLOYMENT, marker, 1) in response.text, response.text
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 1


def _anthropic_text(sdk: Anthropic, marker: str, *, stream: bool) -> str:
    if stream:
        with sdk.messages.stream(
            model=_SHARED,
            max_tokens=_SMALL_MAX_TOKENS,
            messages=[{"role": "user", "content": f"ok:{marker}"}],
        ) as events:
            return "".join(events.text_stream)
    message: Final = sdk.messages.create(
        model=_SHARED,
        max_tokens=_SMALL_MAX_TOKENS,
        messages=[{"role": "user", "content": f"ok:{marker}"}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
def test_a_share_holder_is_served_messages_by_the_shared_deployment(rig: _Rig, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    with Anthropic(api_key=rig.keys["holder"], base_url=str(rig.proxy.client.base_url), max_retries=0) as sdk:
        received: Final = _anthropic_text(sdk, marker, stream=stream)
    assert received == _TEXT
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 1


@pytest.mark.parametrize("caller", ("outsider", "master-key"))
@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_a_caller_without_a_share_is_refused_the_shared_deployment(rig: _Rig, caller: str, surface: Surface) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.send(surface, _SHARED, "outsider" if caller == "outsider" else None, marker)
    assert response.status_code == 400, response.text
    assert f"Deployment {_SHARED} {_RESERVED}" in response.text, response.text
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 0


def test_a_mixed_group_serves_each_caller_from_its_own_deployment(rig: _Rig) -> None:
    served: Final = {
        caller: rig.chat(_MIXED, role, "ok", marker)
        for caller, role, marker in (
            ("holder", "mixed", uuid.uuid4().hex),
            ("outsider", "outsider", uuid.uuid4().hex),
            ("master", None, uuid.uuid4().hex),
        )
    }
    assert {caller: response.status_code for caller, response in served.items()} == dict.fromkeys(served, 200), {
        caller: response.text for caller, response in served.items()
    }
    deployments: Final = {
        caller: re.sub(
            r"^chatcmpl-(.+)-[0-9a-f]{32}-a1$", r"\1", string_value(JSON_OBJECT.validate_json(response.content)["id"])
        )
        for caller, response in served.items()
    }
    assert deployments == {
        "holder": _MIXED_SHARED_DEPLOYMENT,
        "outsider": _MIXED_OPEN_DEPLOYMENT,
        "master": _MIXED_OPEN_DEPLOYMENT,
    }


@pytest.mark.parametrize(("model_group", "role", "ptus"), ((_SHARED, "oversize", 1), (_PAIR, "pair", 2)))
def test_a_request_bigger_than_the_whole_share_is_refused_up_front(
    rig: _Rig, model_group: str, role: str, ptus: int
) -> None:
    capacity: Final = _gpt41()
    ceiling: Final = capacity.input_tpm_for(ptus)
    marker: Final = uuid.uuid4().hex
    response: Final = rig.chat(model_group, role, "ok", marker, max_tokens=ceiling)
    assert response.status_code == 400, response.text
    found: Final = re.search(
        rf"This request needs (\d+) normalized tokens, more than your team's PTU share of {model_group} "
        rf"allows in a minute \({ceiling}\), so it can never be admitted\. Lower max_tokens or shorten the prompt\.",
        _error_message(response),
    )
    assert found is not None, response.text
    assert int(found.group(1)) >= capacity.output_to_input_ratio * ceiling, response.text
    assert rig.hits(_SHARED_DEPLOYMENT, marker) + sum(rig.hits(name, marker) for name in _PAIR_DEPLOYMENTS) == 0
    assert _ptu_counter(rig.teams[role], model_group) in (None, 0)


def test_usage_settles_in_normalized_tokens_and_throttles_the_next_request(rig: _Rig) -> None:
    capacity: Final = _gpt41()
    ceiling: Final = capacity.input_tpm_for(1)
    team: Final = rig.teams["settle"]
    first: Final = uuid.uuid4().hex
    settled: Final = rig.chat(_SHARED, "settle", "settle", first)
    assert settled.status_code == 200, settled.text
    assert settled.json()["usage"]["completion_tokens"] == _SETTLE_USAGE["completion_tokens"], settled.text
    normalized: Final = round(
        int(str(_SETTLE_USAGE["prompt_tokens"]))
        + capacity.output_to_input_ratio * int(str(_SETTLE_USAGE["completion_tokens"]))
    )
    assert _settled_counter(team, _SHARED, normalized) == normalized
    max_tokens: Final = int((ceiling - normalized) / capacity.output_to_input_ratio) + 1
    second: Final = uuid.uuid4().hex
    throttled: Final = rig.chat(_SHARED, "settle", "ok", second, max_tokens=max_tokens)
    assert throttled.status_code == 429, throttled.text
    assert (
        f"Rate limit exceeded for model_per_team_ptu: {team}:{_SHARED}. Limit type: tokens. Current limit: {ceiling},"
        in _error_message(throttled)
    ), throttled.text
    assert int(str(_SETTLE_USAGE["total_tokens"])) + max_tokens * capacity.output_to_input_ratio < ceiling
    assert (rig.hits(_SHARED_DEPLOYMENT, first), rig.hits(_SHARED_DEPLOYMENT, second)) == (1, 0)


def test_cached_input_is_weighted_by_the_cached_ratio(rig: _Rig) -> None:
    capacity: Final = _gpt41()
    marker: Final = uuid.uuid4().hex
    response: Final = rig.chat(_SHARED, "cached", "cached", marker)
    assert response.status_code == 200, response.text
    cached: Final = 800
    expected: Final = round(
        (int(str(_CACHED_USAGE["prompt_tokens"])) - cached)
        + capacity.cached_input_ratio * cached
        + capacity.output_to_input_ratio * int(str(_CACHED_USAGE["completion_tokens"]))
    )
    assert _settled_counter(rig.teams["cached"], _SHARED, expected) == expected
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 1


def test_a_failed_upstream_call_refunds_the_reservation(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.chat(_SHARED, "refund", "fail", marker)
    assert response.status_code == 500, response.text
    assert "scripted 500" in response.text, response.text
    assert _settled_counter(rig.teams["refund"], _SHARED, 0) == 0
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 1


def test_a_response_cache_hit_settles_against_the_share_like_every_token_limit(rig: _Rig) -> None:
    capacity: Final = _gpt41()
    marker: Final = uuid.uuid4().hex
    first: Final = rig.chat(_SHARED, "twin", "settle", marker)
    normalized: Final = round(
        int(str(_SETTLE_USAGE["prompt_tokens"]))
        + capacity.output_to_input_ratio * int(str(_SETTLE_USAGE["completion_tokens"]))
    )
    assert _settled_counter(rig.teams["twin"], _SHARED, normalized) == normalized
    repeat: Final = rig.chat(_SHARED, "twin", "settle", marker)
    assert (first.status_code, repeat.status_code) == (200, 200), (first.text, repeat.text)
    assert repeat.json()["id"] == first.json()["id"] == _served_id(_SHARED_DEPLOYMENT, marker, 1)
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 1
    assert _settled_counter(rig.teams["twin"], _SHARED, 2 * normalized) == 2 * normalized


def test_a_cached_response_is_not_served_to_a_caller_without_a_share(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    cached: Final = rig.chat(_SHARED, "holder", "ok", marker)
    assert cached.status_code == 200, cached.text
    refused: Final = rig.chat(_SHARED, "outsider", "ok", marker)
    assert refused.status_code == 400, refused.text
    assert _RESERVED in refused.text, refused.text
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 1


def test_a_retry_after_an_upstream_error_stays_on_the_shared_deployment(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.chat(_SHARED, "retry", "fail-once", marker, num_retries=1)
    assert response.status_code == 200, response.text
    assert response.json()["id"] == _served_id(_SHARED_DEPLOYMENT, marker, 2), response.text
    assert rig.hits(_SHARED_DEPLOYMENT, marker) == 2


def test_the_priority_scheduler_serves_share_holders_and_refuses_everyone_else(rig: _Rig) -> None:
    served_marker: Final = uuid.uuid4().hex
    refused_marker: Final = uuid.uuid4().hex
    served: Final = rig.chat(_SHARED, "queue", "ok", served_marker, priority=1)
    refused: Final = rig.chat(_SHARED, "outsider", "ok", refused_marker, priority=1)
    assert served.status_code == 200, served.text
    assert served.json()["id"] == _served_id(_SHARED_DEPLOYMENT, served_marker, 1), served.text
    assert "x-litellm-request-prioritization-used" in served.headers, served.headers
    assert refused.status_code == 400, refused.text
    assert _RESERVED in refused.text, refused.text
    assert (rig.hits(_SHARED_DEPLOYMENT, served_marker), rig.hits(_SHARED_DEPLOYMENT, refused_marker)) == (1, 0)


def test_a_dropped_messages_stream_retries_on_the_shared_deployment(rig: _Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        {
            "model": _CLAUDE,
            "max_tokens": _SMALL_MAX_TOKENS,
            "stream": True,
            "num_retries": 1,
            "messages": [{"role": "user", "content": f"drop-once:{marker}"}],
        },
        key=rig.keys["messages"],
    )
    assert response.status_code == 200, response.text
    events: Final = parse_sse(response.text)
    assert event_types(events) == LIFECYCLE, events
    assert message_id(events) == _served_id(_CLAUDE_BACKEND, marker, 2), events
    assert delta_text(events) == _TEXT, events
    assert rig.hits(_CLAUDE_BACKEND, marker) == 2


def test_concurrent_requests_admit_only_what_fits_under_the_share(rig: _Rig) -> None:
    capacity: Final = _gpt41()
    ceiling: Final = capacity.input_tpm_for(1)
    max_tokens: Final = int(ceiling / (2.5 * capacity.output_to_input_ratio))
    markers: Final = tuple(uuid.uuid4().hex for _ in range(5))
    with ThreadPoolExecutor(max_workers=len(markers)) as pool:
        futures: Final[tuple[Future[httpx.Response], ...]] = tuple(
            pool.submit(rig.chat, _SHARED, "burst", "hold", marker, max_tokens=max_tokens) for marker in markers
        )
        eventually(
            lambda: (
                sum(future.done() for future in futures)
                + sum(rig.hits(_SHARED_DEPLOYMENT, marker) for marker in markers)
            ),
            lambda settled: settled >= len(markers),
            seconds=30,
        )
        rig.release.set()
        responses: Final = tuple(future.result(timeout=60) for future in futures)
    assert sorted(response.status_code for response in responses) == [200, 200, 429, 429, 429], tuple(
        response.text for response in responses
    )
    admitted: Final = tuple(
        marker for marker, response in zip(markers, responses, strict=True) if response.status_code == 200
    )
    assert tuple(rig.hits(_SHARED_DEPLOYMENT, marker) for marker in markers) == tuple(
        1 if marker in admitted else 0 for marker in markers
    )


@pytest.mark.timeout(4 * graceful_stop_seconds() + 240)
def test_a_replacement_worker_rebuilds_the_share_rules(rig: _Rig) -> None:
    victim: Final = min(rig.owned.worker_pids())
    replacement: Final = rig.owned.kill_worker(victim)
    assert replacement != victim
    answers: Final = tuple(
        (rig.chat(_SHARED, "worker", "ok", uuid.uuid4().hex), rig.chat(_SHARED, "outsider", "ok", uuid.uuid4().hex))
        for _ in range(6)
    )
    assert tuple((served.status_code, refused.status_code) for served, refused in answers) == ((200, 400),) * 6, tuple(
        (served.text, refused.text) for served, refused in answers
    )
    assert all(
        string_value(JSON_OBJECT.validate_json(served.content)["id"]).startswith(f"chatcmpl-{_SHARED_DEPLOYMENT}-")
        for served, _ in answers
    )


def test_with_attribution_off_the_access_rule_holds_without_a_ceiling(tmp_path: Path) -> None:
    attempts: Final = Attempts()
    teams: Final = _team_ids()
    capacity: Final = _gpt41()
    with gateway_from_environment() as gateway, wire_server(_respond(attempts, threading.Event())) as wire:
        config: Final = _config(tmp_path, wire.url, _model_list(teams), "ptu-shares-off")
        try:
            with owned_proxy_process(gateway, tmp_path, _ATTRIBUTION_OFF, config=config, workers=2) as owned:
                holder: Final = _create_team(owned.gateway, teams["oversize"])
                outsider: Final = _create_team(owned.gateway, teams["outsider"])
                served_marker: Final = uuid.uuid4().hex
                served: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": _SHARED,
                        "max_tokens": capacity.input_tpm_for(1),
                        "messages": [{"role": "user", "content": f"ok:{served_marker}"}],
                    },
                    key=holder,
                )
                refused: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": _SHARED, "messages": [{"role": "user", "content": f"ok:{uuid.uuid4().hex}"}]},
                    key=outsider,
                )
        finally:
            _delete_teams(gateway, (teams["oversize"], teams["outsider"]))
    assert served.status_code == 200, served.text
    assert served.json()["id"] == _served_id(_SHARED_DEPLOYMENT, served_marker, 1), served.text
    assert refused.status_code == 400, refused.text
    assert _RESERVED in refused.text, refused.text
    assert _ptu_counter(teams["oversize"], _SHARED) is None


def test_a_split_that_does_not_add_up_is_dropped_at_load_with_attribution_off(tmp_path: Path) -> None:
    attempts: Final = Attempts()
    holder: Final = _team_ids()["oversize"]
    marker: Final = uuid.uuid4().hex
    with gateway_from_environment() as gateway, wire_server(_respond(attempts, threading.Event())) as wire:
        split: Final = _azure(_SHARED, _SHARED_DEPLOYMENT, {**_ptu_terms({holder: 1}), "ptu_count": 2})
        config: Final = _config(tmp_path, wire.url, [split], "ptu-split-mismatch")
        try:
            with owned_proxy_process(gateway, tmp_path, _ATTRIBUTION_OFF, config=config, workers=2) as owned:
                key: Final = _create_team(owned.gateway, holder)
                refused: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": _SHARED, "messages": [{"role": "user", "content": f"ok:{marker}"}]},
                    key=key,
                )
                log: Final = owned.log.read_text()
        finally:
            _delete_teams(gateway, (holder,))
    assert refused.status_code == 400, refused.text
    assert _RESERVED not in refused.text, refused.text
    assert attempts.count(f"{_SHARED_DEPLOYMENT}:{marker}") == 0
    assert (
        f"PTU configuration on model '{_SHARED}' is invalid: ptu_shares must add up to ptu_count (1 of 2 allocated)"
        in log
    ), log[-4000:]
