from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    held,
    object_value,
)
from tests.integration._support.openai_wire import answering_model_discovery, chat_reply, openai_error
from tests.integration._support.process import graceful_stop_seconds, owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

SUFFIX: Final = uuid.uuid4().hex[:10]
HEALTHY: Final = f"openai/hub-healthy-{SUFFIX}"
FAILING: Final = f"openai/hub-failing-{SUFFIX}"
PUBLISHED_LATER: Final = f"openai/hub-later-{SUFFIX}"
UNPUBLISHED: Final = f"openai/hub-unpublished-{SUFFIX}"
ROUTED: Final = f"openai/hub-routed-{SUFFIX}"
EXPLICIT: Final = f"hub-explicit-{SUFFIX}"
EXPLICIT_PROVIDER_MODEL: Final = f"openai/hub-explicit-wire-{SUFFIX}"
CHECKS_OFF: Final = f"openai/hub-checks-off-{SUFFIX}"
CONFIG_PUBLISHED: Final = (HEALTHY, FAILING, EXPLICIT, CHECKS_OFF)
MARKER: Final = f"caller-traffic-{SUFFIX}"
INTERVAL_SECONDS: Final = 2
HEALTH_DEADLINE_SECONDS: Final = 90
V1_HUB: Final = "/public/v1/model_hub"
LEGACY_HUB: Final = "/public/model_hub"
HUB_PATHS: Final = (V1_HUB, LEGACY_HUB)
HUB_IDS: Final = ("v1", "legacy")
ROW_LIST: Final = TypeAdapter(list[JsonValue])
UNCACHED: Final = frozenset({"cache", "cache_params"})


def _wire_model(model: str) -> str:
    return model.removeprefix("openai/")


def _reply(request: Request) -> Reply:
    model: Final = JSON_OBJECT.validate_json(request.body).get("model")
    if model == _wire_model(FAILING):
        return openai_error(500)
    return chat_reply(f"chatcmpl-{SUFFIX}", str(model), "ok", stream=False)


class UpstreamLog:
    def __init__(self, wire: Wire) -> None:
        self.wire = wire
        self.seen: tuple[Request, ...] = ()

    def bodies(self) -> tuple[dict[str, JsonValue], ...]:
        self.seen = (*self.seen, *self.wire.drain())
        return tuple(JSON_OBJECT.validate_json(request.body) for request in self.seen if request.method == "POST")


@pytest.fixture(scope="module")
def checks_on(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory.mktemp("wildcard-hub-health-on"), background=True) as rig:
        yield rig


@pytest.fixture(scope="module")
def checks_on_nothing_published(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory.mktemp("wildcard-hub-health-unpublished"), background=True, published=()) as rig:
        yield rig


@pytest.fixture(scope="module")
def checks_off(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory.mktemp("wildcard-hub-health-off"), background=False) as rig:
        yield rig


class Rig:
    def __init__(self, gateway: Gateway, log: UpstreamLog) -> None:
        self.gateway = gateway
        self.log = log


@contextmanager
def _rig(directory: Path, *, background: bool, published: tuple[str, ...] = CONFIG_PUBLISHED) -> Iterator[Rig]:
    with (
        wire_server(answering_model_discovery(_reply)) as wire,
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway, directory, {}, config=_config(directory, wire.url, background=background, published=published)
        ) as candidate,
    ):
        yield Rig(candidate, UpstreamLog(wire))


def _config(directory: Path, upstream_url: str, *, background: bool, published: tuple[str, ...]) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    background_settings: Final = (
        {"background_health_checks": True, "enable_health_check_routing": True} if background else {}
    )
    config: Final = {
        **base,
        "general_settings": {
            **base["general_settings"],
            "health_check_interval": INTERVAL_SECONDS,
            **background_settings,
        },
        "litellm_settings": {
            **{name: value for name, value in base["litellm_settings"].items() if name not in UNCACHED},
            **({"public_model_groups": list(published)} if published else {}),
        },
        "router_settings": {},
        "model_list": [
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_base": f"{upstream_url}/v1", "api_key": "hub-wildcard"},
            },
            {
                "model_name": EXPLICIT,
                "litellm_params": {
                    "model": EXPLICIT_PROVIDER_MODEL,
                    "api_base": f"{upstream_url}/v1",
                    "api_key": "hub-explicit",
                },
            },
        ],
    }
    path: Final = directory / "wildcard_hub_health.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _from_caller(body: dict[str, JsonValue]) -> bool:
    return MARKER in json.dumps(body.get("messages"))


def _requests_for(log: UpstreamLog, model: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(body for body in log.bodies() if body.get("model") == _wire_model(model))


def _probes_for(log: UpstreamLog, model: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(body for body in _requests_for(log, model) if not _from_caller(body))


def _hub_rows(gateway: Gateway, path: str) -> tuple[dict[str, JsonValue], ...]:
    listed: Final = gateway.request("GET", path, params={"page_size": "100"} if path == V1_HUB else None)
    assert listed.status_code == 200, listed.text
    payload: Final = listed.json()
    rows: Final = ROW_LIST.validate_python(object_value(payload)["data"] if path == V1_HUB else payload)
    return tuple(object_value(row) for row in rows)


def _hub_row(gateway: Gateway, path: str, model_group: str) -> dict[str, JsonValue]:
    matches: Final = tuple(row for row in _hub_rows(gateway, path) if row.get("model_group") == model_group)
    assert len(matches) == 1, f"{model_group} appears {len(matches)} times on {path}"
    return matches[0]


def _row_once_health_is_set(gateway: Gateway, path: str, model_group: str) -> dict[str, JsonValue]:
    return eventually(
        lambda: _hub_row(gateway, path, model_group),
        lambda row: row.get("health_status") is not None,
        seconds=HEALTH_DEADLINE_SECONDS,
        return_last_on_timeout=True,
    )


def _chat(gateway: Gateway, model: str) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": MARKER}]}
    )


def _wait_two_more_background_cycles(log: UpstreamLog) -> None:
    before: Final = len(_probes_for(log, EXPLICIT_PROVIDER_MODEL))
    eventually(
        lambda: len(_probes_for(log, EXPLICIT_PROVIDER_MODEL)),
        lambda count: count >= before + 2,
        seconds=HEALTH_DEADLINE_SECONDS,
    )


@pytest.mark.parametrize("path", HUB_PATHS, ids=HUB_IDS)
def test_published_wildcard_child_is_healthy_on_the_hub_regression(checks_on: Rig, path: str) -> None:
    row: Final = _row_once_health_is_set(checks_on.gateway, path, HEALTHY)
    assert row.get("health_status") == "healthy", row
    assert isinstance(row.get("health_checked_at"), str), row


@pytest.mark.parametrize("path", HUB_PATHS, ids=HUB_IDS)
def test_published_wildcard_child_is_unhealthy_when_its_upstream_fails_regression(checks_on: Rig, path: str) -> None:
    row: Final = _row_once_health_is_set(checks_on.gateway, path, FAILING)
    assert row.get("health_status") == "unhealthy", row
    assert isinstance(row.get("health_checked_at"), str), row


@pytest.mark.parametrize("model", (HEALTHY, FAILING), ids=("healthy", "failing"))
def test_health_probe_asks_the_upstream_for_the_exact_published_child_regression(checks_on: Rig, model: str) -> None:
    probes: Final = eventually(
        lambda: _probes_for(checks_on.log, model),
        bool,
        seconds=HEALTH_DEADLINE_SECONDS,
        return_last_on_timeout=True,
    )
    seen: Final = sorted({str(body.get("model")) for body in checks_on.log.bodies()})
    assert probes, f"upstream never received a probe for {_wire_model(model)}, only for {seen}"


def test_wildcard_child_published_through_make_public_gets_hub_health_regression(
    checks_on_nothing_published: Rig,
) -> None:
    rig: Final = checks_on_nothing_published
    published: Final = rig.gateway.post("/model_group/make_public", {"model_groups": [PUBLISHED_LATER]})
    assert published.get("public_model_groups") == [PUBLISHED_LATER], published
    row: Final = _row_once_health_is_set(rig.gateway, V1_HUB, PUBLISHED_LATER)
    assert row.get("health_status") == "healthy", row
    assert _probes_for(rig.log, PUBLISHED_LATER), "upstream never received a probe for the later-published child"


def test_failing_child_probe_does_not_break_routing_through_the_wildcard_guard(checks_on: Rig) -> None:
    _wait_two_more_background_cycles(checks_on.log)
    routed: Final = _chat(checks_on.gateway, ROUTED)
    assert routed.status_code == 200, routed.text
    assert any(_from_caller(body) for body in _requests_for(checks_on.log, ROUTED)), "chat never reached the upstream"


def test_unpublished_wildcard_model_gets_no_probe_guard(checks_on: Rig) -> None:
    used: Final = _chat(checks_on.gateway, UNPUBLISHED)
    assert used.status_code == 200, used.text
    _wait_two_more_background_cycles(checks_on.log)
    requests: Final = _requests_for(checks_on.log, UNPUBLISHED)
    assert any(_from_caller(body) for body in requests), requests
    assert all(_from_caller(body) for body in requests), requests


@pytest.mark.parametrize("path", HUB_PATHS, ids=HUB_IDS)
def test_explicit_published_deployment_keeps_its_hub_health_guard(checks_on: Rig, path: str) -> None:
    row: Final = _row_once_health_is_set(checks_on.gateway, path, EXPLICIT)
    assert row.get("health_status") == "healthy", row
    assert isinstance(row.get("health_checked_at"), str), row


def test_background_checks_off_leave_the_wildcard_child_without_health_guard(checks_off: Rig) -> None:
    held(
        lambda: _hub_row(checks_off.gateway, V1_HUB, CHECKS_OFF),
        lambda row: row.get("health_status") is None,
        holding=3 * INTERVAL_SECONDS,
        seconds=HEALTH_DEADLINE_SECONDS,
    )
    assert _hub_row(checks_off.gateway, LEGACY_HUB, CHECKS_OFF).get("health_status") is None
    used: Final = _chat(checks_off.gateway, CHECKS_OFF)
    assert used.status_code == 200, used.text
    requests: Final = _requests_for(checks_off.log, CHECKS_OFF)
    assert any(_from_caller(body) for body in requests), requests
    assert all(_from_caller(body) for body in requests), requests
