import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    string_value,
)
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

Endpoint = Literal["chat", "messages", "responses"]

ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
UNTAGGED: Final = "prefix-untagged"
REGIONAL: Final = "prefix-regional"
BATCHED: Final = "prefix-batched"
FAIL_OPEN_BATCHED: Final = "prefix-fail-open-batched"
OUTAGE: Final = "prefix-outage"
DOWN: Final = "prefix-down"
USER_AGENT_ROUTED: Final = "prefix-user-agent"
REFUSED: Final = "Not allowed to access model due to tags configuration"


def _respond(request: Request) -> Reply:
    if request.method != "POST":
        if request.target.endswith("/files"):
            return Reply(status=404, body=b'{"error": {"message": "no files api"}}')
        return Reply(body=b'{"object": "list", "data": []}')
    body: Final = JSON_OBJECT.validate_json(request.body)
    backend: Final = string_value(body["model"])
    if backend.endswith("-down-backend"):
        return Reply(status=500, body=b'{"error": {"message": "upstream down"}}')
    stream: Final = body.get("stream") is True
    if request.target.endswith("/responses"):
        return responses_reply(f"resp_{uuid.uuid4().hex}", backend, "tag routed", stream=stream)
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", backend, "tag routed", stream=stream)


def _config(directory: Path, wire: Wire, prefix: str) -> Path:
    def deployment(
        group: str,
        backend: str,
        tags: Sequence[str],
        provider: str = "openai",
        fail_open: bool = False,
        tag_regex: Sequence[str] = (),
    ) -> dict[str, JsonValue]:
        return {
            "model_name": group,
            "litellm_params": {
                "model": f"{provider}/{backend}",
                "api_base": f"{wire.url}/v1",
                "api_key": "synthetic-tag-prefix-key",
                **({"tags": list(tags)} if tags else {}),
                **({"tag_regex": list(tag_regex)} if tag_regex else {}),
            },
            "model_info": {"id": backend, **({"allow_fail_open": True} if fail_open else {})},
        }

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        deployment(UNTAGGED, "untagged-backend", ()),
        deployment(REGIONAL, "eu-backend", ("region:eu",)),
        deployment(REGIONAL, "us-backend", ("region:us",)),
        deployment(BATCHED, "eu-batch-backend", ("region:eu",), "hosted_vllm"),
        deployment(BATCHED, "us-batch-backend", ("region:us",), "hosted_vllm"),
        deployment(FAIL_OPEN_BATCHED, "eu-fail-open-backend", ("default", "region:eu"), "hosted_vllm", fail_open=True),
        deployment(OUTAGE, "eu-down-backend", ("region:eu",)),
        deployment(OUTAGE, "us-up-backend", ("region:us",)),
        deployment(DOWN, "untagged-down-backend", ()),
        deployment(USER_AGENT_ROUTED, "cli-backend", (), tag_regex=("^User-Agent: claude-cli/",)),
        deployment(USER_AGENT_ROUTED, "browser-backend", ("default",)),
    ]
    config["router_settings"] = {
        **config["router_settings"],
        "enable_tag_filtering": True,
        "tag_routing_prefix": prefix,
    }
    path: Final = directory / f"tag-routing-prefix-{prefix.rstrip(':') or 'none'}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def upstream() -> Iterator[Wire]:
    with wire_server(_respond) as wire:
        yield wire


@pytest.fixture(scope="module")
def prefixed(tmp_path_factory: pytest.TempPathFactory, upstream: Wire) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("tag-routing-prefix")
    with (
        gateway_from_environment() as base,
        owned_proxy(base, directory, {}, config=_config(directory, upstream, "route:")) as proxy,
    ):
        yield proxy


@pytest.fixture(scope="module")
def unprefixed(tmp_path_factory: pytest.TempPathFactory, upstream: Wire) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("tag-routing-no-prefix")
    with (
        gateway_from_environment() as base,
        owned_proxy(base, directory, {}, config=_config(directory, upstream, "")) as proxy,
    ):
        yield proxy


def _call(
    endpoint: Endpoint, model: str, prompt: str, tags: Sequence[str], stream: bool
) -> tuple[str, Mapping[str, JsonValue], Mapping[str, str]]:
    header: Final = {"x-litellm-tags": ",".join(tags)}
    match endpoint:
        case "chat":
            messages: Final[JsonValue] = [{"role": "user", "content": prompt}]
            return (
                "/v1/chat/completions",
                {"model": model, "messages": messages, "metadata": {"tags": list(tags)}, "stream": stream},
                {},
            )
        case "messages":
            return (
                "/v1/messages",
                {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}], "stream": stream},
                header,
            )
        case "responses":
            return "/v1/responses", {"model": model, "input": prompt, "stream": stream}, header


def _send(
    gateway: Gateway,
    upstream: Wire,
    endpoint: Endpoint,
    model: str,
    tags: Sequence[str],
    *,
    key: str | None = None,
    stream: bool = False,
) -> tuple[httpx.Response, tuple[str, ...]]:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    path, body, headers = _call(endpoint, model, f"route this {marker}", tags, stream)
    response: Final = gateway.request("POST", path, body, key=key, headers=headers)
    return response, _routed_backends(upstream, marker)


def _routed_backends(upstream: Wire, marker: str) -> tuple[str, ...]:
    return tuple(
        string_value(JSON_OBJECT.validate_json(request.body)["model"])
        for request in upstream.drain()
        if request.method == "POST" and marker in request.body.decode()
    )


def _team_key(scenario: Scenario, tags: Sequence[str]) -> str:
    return scenario.key(team_id=scenario.team(tags=list(tags)))


def _policy_key(scenario: Scenario, owner: Literal["key", "team", "project"], tags: Sequence[str]) -> str:
    match owner:
        case "key":
            return scenario.key(metadata={"tags": list(tags)})
        case "team":
            return _team_key(scenario, tags)
        case "project":
            team: Final = scenario.team()
            return scenario.key(team_id=team, project_id=scenario.project(team, tags=list(tags)))


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_an_unprefixed_caller_tag_does_not_refuse_a_group_with_untagged_deployments(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint, stream: bool
) -> None:
    response, backends = _send(prefixed, upstream, endpoint, UNTAGGED, ["user_id:234"], stream=stream)

    assert response.status_code == 200, response.text
    assert backends == ("untagged-backend",)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_prefixed_caller_tag_routes_next_to_an_ignored_unprefixed_one(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint
) -> None:
    response, backends = _send(prefixed, upstream, endpoint, REGIONAL, ["user_id:234", "route:region:us"])

    assert response.status_code == 200, response.text
    assert backends == ("us-backend",)


@pytest.mark.parametrize(
    ("policy_tag", "backend"), [("&region:eu", "eu-backend"), ("!region:eu", "us-backend")], ids=["require", "exclude"]
)
@pytest.mark.parametrize("owner", ["key", "team", "project"])
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_an_unprefixed_key_team_or_project_tag_still_constrains_routing(
    prefixed: Gateway,
    upstream: Wire,
    endpoint: Endpoint,
    owner: Literal["key", "team", "project"],
    policy_tag: str,
    backend: str,
) -> None:
    with prefixed.scenario() as scenario:
        key: Final = _policy_key(scenario, owner, [policy_tag])
        routed: Final = tuple(_send(prefixed, upstream, endpoint, REGIONAL, ["user_id:234"], key=key) for _ in range(8))

    assert [response.status_code for response, _ in routed] == [200] * 8, [r.text for r, _ in routed]
    assert [backends for _, backends in routed] == [(backend,)] * 8


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_prefixed_caller_tag_cannot_route_around_a_team_requirement(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint
) -> None:
    with prefixed.scenario() as scenario:
        key: Final = _team_key(scenario, ["&region:eu"])
        response, backends = _send(prefixed, upstream, endpoint, REGIONAL, ["route:region:us"], key=key)

    assert response.status_code == 401, response.text
    assert REFUSED in response.text, response.text
    assert backends == ()


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_an_unprefixed_caller_constraint_is_ignored_once_a_prefix_is_set(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint
) -> None:
    response, backends = _send(prefixed, upstream, endpoint, REGIONAL, ["&region:nowhere"])

    assert response.status_code == 200, response.text
    assert len(backends) == 1


@pytest.mark.parametrize(
    "metadata",
    [
        {"metadata": {"tags": ["&region:nowhere"], "inherited_tags": ["&region:nowhere"]}},
        {"litellm_metadata": {"tags": ["&region:nowhere"], "inherited_tags": ["&region:nowhere"]}},
    ],
    ids=["metadata", "litellm_metadata"],
)
def test_a_caller_cannot_make_an_unprefixed_tag_route_by_sending_inherited_tags(
    prefixed: Gateway, upstream: Wire, metadata: dict[str, JsonValue]
) -> None:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    messages: Final[JsonValue] = [{"role": "user", "content": f"route this {marker}"}]
    response: Final = prefixed.request(
        "POST", "/v1/chat/completions", {"model": REGIONAL, "messages": messages, **metadata}
    )

    assert response.status_code == 200, response.text
    assert len(_routed_backends(upstream, marker)) == 1


@pytest.mark.parametrize(
    ("tags", "status", "backends"),
    [
        ([], 200, ("untagged-backend",)),
        (["user_id:234", "user_id:234"], 200, ("untagged-backend",)),
        ([f"user_id:{'x' * 5000}"], 200, ("untagged-backend",)),
        (["route:"], 401, ()),
        (["route:user_id:234"], 401, ()),
    ],
    ids=["empty", "duplicate", "5kb", "bare-prefix", "prefixed-unknown-tag"],
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_odd_caller_tag_values_on_an_untagged_group(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint, tags: list[str], status: int, backends: tuple[str, ...]
) -> None:
    response, routed = _send(prefixed, upstream, endpoint, UNTAGGED, tags)

    assert response.status_code == status, response.text
    assert routed == backends


@pytest.mark.parametrize("tags", ["user_id:234", 234, [234]], ids=["string", "int", "int-list"])
def test_malformed_body_tags_never_reach_an_upstream_error(prefixed: Gateway, upstream: Wire, tags: JsonValue) -> None:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    messages: Final[JsonValue] = [{"role": "user", "content": f"route this {marker}"}]
    response: Final = prefixed.request(
        "POST", "/v1/chat/completions", {"model": UNTAGGED, "messages": messages, "metadata": {"tags": tags}}
    )

    assert (response.status_code, _routed_backends(upstream, marker)) == (200, ("untagged-backend",)), response.text


def test_a_wrong_key_is_refused_before_routing(prefixed: Gateway, upstream: Wire) -> None:
    response, backends = _send(prefixed, upstream, "chat", REGIONAL, ["route:region:us"], key="sk-not-a-real-key")

    assert response.status_code == 401, response.text
    assert backends == ()


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_a_team_requirement_holds_while_its_only_deployment_is_down(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint
) -> None:
    with prefixed.scenario() as scenario:
        key: Final = _team_key(scenario, ["&region:eu"])
        response, backends = _send(prefixed, upstream, endpoint, OUTAGE, ["user_id:234"], key=key)

    assert response.status_code >= 500, response.text
    assert set(backends) == {"eu-down-backend"}


def test_a_fallback_group_ignores_the_unprefixed_spend_tag_too(prefixed: Gateway, upstream: Wire) -> None:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    messages: Final[JsonValue] = [{"role": "user", "content": f"route this {marker}"}]
    response: Final = prefixed.request(
        "POST",
        "/v1/chat/completions",
        {"model": DOWN, "messages": messages, "metadata": {"tags": ["user_id:234"]}, "fallbacks": [UNTAGGED]},
    )
    backends: Final = _routed_backends(upstream, marker)

    assert response.status_code == 200, response.text
    assert (set(backends), backends[-1]) == ({"untagged-down-backend", "untagged-backend"}, "untagged-backend")


@pytest.mark.parametrize(
    ("user_agent", "backend"), [("claude-cli/2.0", "cli-backend"), ("Mozilla/5.0", "browser-backend")]
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_user_agent_routing_is_unchanged_next_to_an_ignored_spend_tag(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint, user_agent: str, backend: str
) -> None:
    def send() -> tuple[int, tuple[str, ...]]:
        marker: Final = f"marker-{uuid.uuid4().hex}"
        path, body, headers = _call(endpoint, USER_AGENT_ROUTED, f"route this {marker}", ["user_id:234"], False)
        response: Final = prefixed.request("POST", path, body, headers={**headers, "User-Agent": user_agent})
        return response.status_code, _routed_backends(upstream, marker)

    assert [send() for _ in range(4)] == [(200, (backend,))] * 4


def test_a_mixed_concurrent_burst_routes_every_request_by_its_own_tags(prefixed: Gateway, upstream: Wire) -> None:
    with prefixed.scenario() as scenario:
        require_eu: Final = _team_key(scenario, ["&region:eu"])
        exclude_eu: Final = scenario.key(metadata={"tags": ["!region:eu"]})
        plans: Final = tuple(
            (endpoint, stream, model, tags, key, backend, f"marker-{uuid.uuid4().hex}")
            for round_index in range(3)
            for endpoint in ENDPOINTS
            for stream in (False, True)
            for model, tags, key, backend in (
                (UNTAGGED, ["user_id:234"], None, "untagged-backend"),
                (REGIONAL, ["user_id:234"], require_eu, "eu-backend"),
                (REGIONAL, ["user_id:234"], exclude_eu, "us-backend"),
                (REGIONAL, ["user_id:234", "route:region:us", f"round:{round_index}"], None, "us-backend"),
            )
        )

        def send(plan: tuple[Endpoint, bool, str, list[str], str | None, str, str]) -> int:
            endpoint, stream, model, tags, key, _backend, marker = plan
            path, body, headers = _call(endpoint, model, f"route this {marker}", tags, stream)
            return prefixed.request("POST", path, body, key=key, headers=headers).status_code

        with ThreadPoolExecutor(max_workers=24) as pool:
            statuses: Final = tuple(pool.map(send, plans))
        captured: Final = tuple(
            (request.body.decode(), string_value(JSON_OBJECT.validate_json(request.body)["model"]))
            for request in upstream.drain()
            if request.method == "POST"
        )

    assert statuses == (200,) * len(plans)
    assert [tuple(backend for text, backend in captured if plan[-1] in text) for plan in plans] == [
        (plan[5],) for plan in plans
    ]


@pytest.mark.parametrize(
    ("tags", "backend"),
    [(["region:us"], "us-backend"), (["!region:eu"], "us-backend"), (["&region:eu"], "eu-backend")],
    ids=["positive", "exclude", "require"],
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_without_a_prefix_unprefixed_caller_tags_still_route(
    unprefixed: Gateway, upstream: Wire, endpoint: Endpoint, tags: list[str], backend: str
) -> None:
    routed: Final = tuple(_send(unprefixed, upstream, endpoint, REGIONAL, tags) for _ in range(4))

    assert [response.status_code for response, _ in routed] == [200] * 4, [r.text for r, _ in routed]
    assert [backends for _, backends in routed] == [(backend,)] * 4


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_without_a_prefix_an_unknown_caller_tag_still_refuses_an_untagged_group(
    unprefixed: Gateway, upstream: Wire, endpoint: Endpoint
) -> None:
    response, backends = _send(unprefixed, upstream, endpoint, UNTAGGED, ["user_id:234"])

    assert response.status_code == 401, response.text
    assert REFUSED in response.text, response.text
    assert backends == ()


def _batch_rows(model: str, marker: str, count: int) -> bytes:
    rows: Final = (
        {
            "custom_id": f"row-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": f"route this {marker} {index}"}]},
        }
        for index in range(count)
    )
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def _run_batch(gateway: Gateway, key: str, model: str, marker: str, count: int) -> Mapping[str, JsonValue]:
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model},
        {"file": ("rows.jsonl", _batch_rows(model, marker, count), "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    created: Final = gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        key=key,
    )
    assert created.status_code == 200, created.text
    batch_id: Final = string_value(JSON_OBJECT.validate_json(created.content)["id"])
    return eventually(
        lambda: JSON_OBJECT.validate_json(gateway.request("GET", f"/v1/batches/{batch_id}", key=key).content),
        lambda batch: batch.get("status") in ("completed", "failed", "expired", "cancelled"),
        seconds=60,
    )


@pytest.mark.parametrize("owner", ["key", "team", "project"])
def test_a_proxy_run_batch_keeps_applying_an_unprefixed_key_team_or_project_tag(
    prefixed: Gateway, upstream: Wire, owner: Literal["key", "team", "project"]
) -> None:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    with prefixed.scenario() as scenario:
        finished: Final = _run_batch(prefixed, _policy_key(scenario, owner, ["&region:eu"]), BATCHED, marker, 8)

    assert finished["status"] == "completed", finished
    assert _routed_backends(upstream, marker) == ("eu-batch-backend",) * 8


def test_a_proxy_run_batch_cannot_fall_open_past_a_team_exclusion(unprefixed: Gateway, upstream: Wire) -> None:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    with unprefixed.scenario() as scenario:
        key: Final = _team_key(scenario, ["!region:eu"])
        finished: Final = _run_batch(unprefixed, key, FAIL_OPEN_BATCHED, marker, 2)
        batch_backends: Final = _routed_backends(upstream, marker)
        response, chat_backends = _send(unprefixed, upstream, "chat", FAIL_OPEN_BATCHED, [], key=key)

    assert (finished["status"], finished["request_counts"]) == (
        "completed",
        {"completed": 0, "failed": 2, "total": 2},
    ), finished
    assert batch_backends == ()
    assert response.status_code == 401, response.text
    assert chat_backends == ()
