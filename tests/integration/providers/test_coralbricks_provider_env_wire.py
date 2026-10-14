from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment, string_value
from integration._support.process import owned_proxy_process
from integration._support.responses_vendor import same_response
from integration._support.wire import Request, Wire, wire_server
from integration.providers._coralbricks import (
    API_KEY,
    MODEL,
    NO_CACHE,
    PROVIDER,
    assert_billed,
    is_readiness_probe,
    marker_provider,
    not_yet_routable,
    raw_deployment,
    ready,
    spend_row,
)
from pydantic import JsonValue

ENV_KEY: Final = "cb_env_integration_key"
CONFIG_MODEL: Final = "coralbricks-env-config"


@dataclass(frozen=True, slots=True)
class EnvRig:
    gateway: Gateway
    env_wire: Wire


def env_config(env_wire: Wire, directory: Path) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **base,
        "model_list": [{"model_name": CONFIG_MODEL, "litellm_params": {"model": f"{PROVIDER}/{MODEL}"}}],
        "environment_variables": {
            **base.get("environment_variables", {}),
            "CORALBRICKS_API_KEY": ENV_KEY,
            "CORALBRICKS_API_BASE": f"{env_wire.url}/v1",
        },
    }
    path: Final = directory / "coralbricks-env.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def env_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[EnvRig]:
    directory: Final = tmp_path_factory.mktemp("coralbricks-env")
    with gateway_from_environment() as rig_gateway, wire_server(marker_provider()) as env_wire:
        config: Final = env_config(env_wire, directory)
        with owned_proxy_process(
            rig_gateway, directory, {}, remove_environment=("CORALBRICKS_API_KEY", "CORALBRICKS_API_BASE"),
            config=config, workers=2,
        ) as owned:
            ready(owned.gateway, CONFIG_MODEL, seconds=120)
            yield EnvRig(owned.gateway, env_wire)


def calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if not is_readiness_probe(request))


def body_for(target: str, model: str, marker: str) -> dict[str, JsonValue]:
    question: Final = f"Say hello marker-{marker}"
    match target:
        case "/v1/responses":
            return {"model": model, "input": question, "cache": NO_CACHE}
        case "/v1/messages":
            return {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": question}], "cache": NO_CACHE}
        case _:
            return {"model": model, "messages": [{"role": "user", "content": question}], "cache": NO_CACHE}


def only_call_with_key(wire: Wire, target: str, key: str) -> None:
    received: Final = calls(wire)
    assert [(request.method, request.target) for request in received] == [("POST", target)], [
        (request.method, request.target) for request in received
    ]
    assert received[0].headers.get("authorization") == f"Bearer {key}", received[0].headers
    assert JSON_OBJECT.validate_json(received[0].body)["model"] == MODEL, received[0].body


def row_id(target: str, marker: str, served: dict[str, JsonValue]) -> str:
    match target:
        case "/v1/responses":
            assert same_response(string_value(served["id"]), f"resp_{marker}"), served
            return string_value(served["id"])
        case "/v1/messages":
            assert served["id"] == f"msg_{marker}", served
            return f"msg_{marker}"
        case _:
            assert served["id"] == f"req_{marker}", served
            return f"req_{marker}"


@pytest.mark.parametrize(
    ("target", "call_type"),
    (("/v1/chat/completions", "acompletion"), ("/v1/responses", "aresponses"), ("/v1/messages", "anthropic_messages")),
)
def test_config_model_without_credentials_uses_the_env_key_and_base(env_rig: EnvRig, target: str, call_type: str) -> None:
    marker: Final = uuid.uuid4().hex
    env_rig.env_wire.drain()
    served: Final = env_rig.gateway.post(target, body_for(target, CONFIG_MODEL, marker))
    only_call_with_key(env_rig.env_wire, target, ENV_KEY)
    assert_billed(spend_row(row_id(target, marker, served)), CONFIG_MODEL, call_type)


def test_explicit_credentials_beat_the_env_credentials(env_rig: EnvRig) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(marker_provider()) as explicit_wire, env_rig.gateway.scenario() as scenario:
        alias: Final = raw_deployment(
            scenario,
            f"integration-{uuid.uuid4().hex}",
            {"model": f"{PROVIDER}/{MODEL}", "api_base": f"{explicit_wire.url}/v1", "api_key": API_KEY},
        )
        ready(env_rig.gateway, alias)
        env_rig.env_wire.drain()
        explicit_wire.drain()
        served: Final = env_rig.gateway.post("/v1/chat/completions", body_for("/v1/chat/completions", alias, marker))
        only_call_with_key(explicit_wire, "/v1/chat/completions", API_KEY)
        assert calls(env_rig.env_wire) == (), "the env base was called although the deployment names its own"
        assert_billed(spend_row(row_id("/v1/chat/completions", marker, served)), alias, "acompletion")


@pytest.mark.parametrize("api_key", ("missing", None, ""), ids=("missing", "null", "empty"))
def test_deployment_without_a_key_uses_the_env_key(env_rig: EnvRig, api_key: str | None) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(marker_provider()) as explicit_wire, env_rig.gateway.scenario() as scenario:
        alias: Final = raw_deployment(
            scenario,
            f"integration-{uuid.uuid4().hex}",
            {
                "model": f"{PROVIDER}/{MODEL}",
                "api_base": f"{explicit_wire.url}/v1",
                **({} if api_key == "missing" else {"api_key": api_key}),
            },
        )
        ready(env_rig.gateway, alias)
        explicit_wire.drain()
        served: Final = env_rig.gateway.post("/v1/chat/completions", body_for("/v1/chat/completions", alias, marker))
        only_call_with_key(explicit_wire, "/v1/chat/completions", ENV_KEY)
        assert_billed(spend_row(row_id("/v1/chat/completions", marker, served)), alias, "acompletion")


@pytest.mark.parametrize("api_base", ("missing", None, ""), ids=("missing", "null", "empty"))
def test_deployment_without_a_base_uses_the_env_base(env_rig: EnvRig, api_base: str | None) -> None:
    marker: Final = uuid.uuid4().hex
    with env_rig.gateway.scenario() as scenario:
        alias: Final = raw_deployment(
            scenario,
            f"integration-{uuid.uuid4().hex}",
            {
                "model": f"{PROVIDER}/{MODEL}",
                "api_key": API_KEY,
                **({} if api_base == "missing" else {"api_base": api_base}),
            },
        )
        ready(env_rig.gateway, alias)
        env_rig.env_wire.drain()
        served: Final = env_rig.gateway.post("/v1/chat/completions", body_for("/v1/chat/completions", alias, marker))
        only_call_with_key(env_rig.env_wire, "/v1/chat/completions", API_KEY)
        assert_billed(spend_row(row_id("/v1/chat/completions", marker, served)), alias, "acompletion")


def test_env_credentials_are_not_echoed_by_model_info(env_rig: EnvRig) -> None:
    entries: Final = env_rig.gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    mine: Final = [JSON_OBJECT.validate_python(entry) for entry in entries]
    config_rows: Final = [entry for entry in mine if entry["model_name"] == CONFIG_MODEL]
    assert len(config_rows) == 1, [entry["model_name"] for entry in mine]
    assert ENV_KEY not in str(config_rows[0]), config_rows[0]
    response: Final = eventually(
        lambda: env_rig.gateway.request("GET", "/health", params={"model": CONFIG_MODEL}),
        lambda candidate: not not_yet_routable(candidate),
        seconds=30,
    )
    assert response.status_code == 200, response.text
    assert ENV_KEY not in response.text, response.text
    health: Final = JSON_OBJECT.validate_json(response.content)
    assert health["healthy_count"] == 1 and health["unhealthy_count"] == 0, health
