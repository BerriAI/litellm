import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
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
REFUSED: Final = "Not allowed to access model due to tags configuration"


def _respond(request: Request) -> Reply:
    if request.method != "POST":
        if request.target.endswith("/files"):
            return Reply(status=404, body=b'{"error": {"message": "no files api"}}')
        return Reply(body=b'{"object": "list", "data": []}')
    body: Final = JSON_OBJECT.validate_json(request.body)
    backend: Final = string_value(body["model"])
    stream: Final = body.get("stream") is True
    if request.target.endswith("/responses"):
        return responses_reply(f"resp_{uuid.uuid4().hex}", backend, "tag routed", stream=stream)
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", backend, "tag routed", stream=stream)


def _config(directory: Path, wire: Wire, prefix: str) -> Path:
    def deployment(group: str, backend: str, tags: Sequence[str], provider: str = "openai") -> dict[str, JsonValue]:
        return {
            "model_name": group,
            "litellm_params": {
                "model": f"{provider}/{backend}",
                "api_base": f"{wire.url}/v1",
                "api_key": "synthetic-tag-prefix-key",
                **({"tags": list(tags)} if tags else {}),
            },
            "model_info": {"id": backend},
        }

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        deployment(UNTAGGED, "untagged-backend", ()),
        deployment(REGIONAL, "eu-backend", ("region:eu",)),
        deployment(REGIONAL, "us-backend", ("region:us",)),
        deployment(BATCHED, "eu-batch-backend", ("region:eu",), "hosted_vllm"),
        deployment(BATCHED, "us-batch-backend", ("region:us",), "hosted_vllm"),
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
    ("team_tag", "backend"), [("&region:eu", "eu-backend"), ("!region:eu", "us-backend")], ids=["require", "exclude"]
)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_an_unprefixed_team_tag_still_constrains_routing(
    prefixed: Gateway, upstream: Wire, endpoint: Endpoint, team_tag: str, backend: str
) -> None:
    with prefixed.scenario() as scenario:
        key: Final = _team_key(scenario, [team_tag])
        routed: Final = tuple(_send(prefixed, upstream, endpoint, REGIONAL, ["user_id:234"], key=key) for _ in range(4))

    assert [response.status_code for response, _ in routed] == [200] * 4, [r.text for r, _ in routed]
    assert [backends for _, backends in routed] == [(backend,)] * 4


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


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_without_a_prefix_an_unknown_caller_tag_still_refuses_an_untagged_group(
    unprefixed: Gateway, upstream: Wire, endpoint: Endpoint
) -> None:
    response, backends = _send(unprefixed, upstream, endpoint, UNTAGGED, ["user_id:234"])

    assert response.status_code == 401, response.text
    assert REFUSED in response.text, response.text
    assert backends == ()


def _batch_rows(marker: str, count: int) -> bytes:
    rows: Final = (
        {
            "custom_id": f"row-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": BATCHED, "messages": [{"role": "user", "content": f"route this {marker} {index}"}]},
        }
        for index in range(count)
    )
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def test_a_proxy_run_batch_keeps_applying_an_unprefixed_team_tag(prefixed: Gateway, upstream: Wire) -> None:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    with prefixed.scenario() as scenario:
        key: Final = _team_key(scenario, ["&region:eu"])
        uploaded: Final = prefixed.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": BATCHED},
            {"file": ("rows.jsonl", _batch_rows(marker, 8), "application/jsonl")},
            key=key,
        )
        assert uploaded.status_code == 200, uploaded.text
        created: Final = prefixed.request(
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
        finished: Final = eventually(
            lambda: JSON_OBJECT.validate_json(prefixed.request("GET", f"/v1/batches/{batch_id}", key=key).content),
            lambda batch: batch.get("status") in ("completed", "failed", "expired", "cancelled"),
            seconds=60,
        )

    assert finished["status"] == "completed", finished
    assert _routed_backends(upstream, marker) == ("eu-batch-backend",) * 8
