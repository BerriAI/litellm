from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from integration._support.client import eventually, gateway_from_environment, string_value
from integration._support.database import scratch_database
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Wire, wire_server
from integration.routing.test_per_model_rate_limit_fallbacks import (
    SETTING,
    Groups,
    Rig,
    assert_refused,
    assert_served,
    chat,
    error_message,
    healthy_reply,
    listed_setting,
    model_entry,
    new_groups,
    new_marker,
    proxy_config,
    update_setting,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(4 * graceful_stop_seconds() + 120)

NO_OVERRIDES: Final[Mapping[str, str]] = MappingProxyType({})
LEGACY_LIMITER: Final = MappingProxyType({"LEGACY_MULTI_INSTANCE_RATE_LIMITING": "true"})
LEGACY_ATTEMPTS: Final = 10
NOT_A_BOOLEAN: Final = "is not a boolean, treating it as disabled"


def serving_groups(groups: Groups) -> tuple[str, ...]:
    return (groups.primary, groups.fallback, groups.last)


def fallback_config(
    directory: Path, groups: Groups, wires: Mapping[str, Wire], general_settings: Mapping[str, JsonValue]
) -> Path:
    return proxy_config(
        directory,
        model_list=tuple(model_entry(group, wires[group]) for group in serving_groups(groups)),
        fallbacks=({groups.primary: [groups.fallback]},),
        general_settings=general_settings,
    )


@contextmanager
def fallback_rig(
    directory: Path, general_settings: Mapping[str, JsonValue], environment: Mapping[str, str] = NO_OVERRIDES
) -> Generator[Rig]:
    groups: Final = new_groups()
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        wires: Final = MappingProxyType(
            {group: stack.enter_context(wire_server(healthy_reply)) for group in serving_groups(groups)}
        )
        config: Final = fallback_config(directory, groups, wires, general_settings)
        owned: Final = stack.enter_context(
            owned_proxy_process(rig_gateway, directory, environment, config=config, workers=2)
        )
        yield Rig(owned.gateway, owned.log, groups, wires)


@pytest.fixture(scope="module")
def lenient(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with fallback_rig(tmp_path_factory.mktemp("per-model-limits-lenient"), {SETTING: None}) as rig:
        yield rig


def through_first_refusal(responses: Iterator[httpx.Response]) -> Iterator[httpx.Response]:
    for response in responses:
        yield response
        if response.status_code != 200:
            return


def sent_until_refused(rig: Rig, key: str, markers: Sequence[str]) -> tuple[httpx.Response, ...]:
    return tuple(through_first_refusal(chat(rig.gateway, key, rig.groups.primary, marker) for marker in markers))


def test_null_setting_keeps_falling_back_on_a_per_model_429_without_a_warning(lenient: Rig) -> None:
    lenient.traffic()
    primary, fallback = lenient.groups.primary, lenient.groups.fallback
    first, second = new_marker(), new_marker()
    with lenient.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={primary: 1})
        assert_served(chat(lenient.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        assert_served(chat(lenient.gateway, key, primary, second), fallback, f"chatcmpl-{second}")
    assert lenient.traffic() == lenient.expected({primary: (first,), fallback: (second,)})
    assert NOT_A_BOOLEAN not in lenient.log.read_text(errors="replace")


def test_key_router_settings_fallbacks_still_serve_a_per_model_429_when_the_setting_is_off(lenient: Rig) -> None:
    lenient.traffic()
    primary, last = lenient.groups.primary, lenient.groups.last
    first, second = new_marker(), new_marker()
    with lenient.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={primary: 1}, router_settings={"fallbacks": [{primary: [last]}]})
        assert_served(chat(lenient.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        assert_served(chat(lenient.gateway, key, primary, second), last, f"chatcmpl-{second}")
    assert lenient.traffic() == lenient.expected({primary: (first,), last: (second,)})


def test_request_disable_fallbacks_answers_a_per_model_429_when_the_setting_is_off(lenient: Rig) -> None:
    lenient.traffic()
    primary: Final = lenient.groups.primary
    first, second = new_marker(), new_marker()
    with lenient.gateway.scenario() as scenario:
        key: Final = scenario.key(model_rpm_limit={primary: 1})
        assert_served(chat(lenient.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        assert_refused(chat(lenient.gateway, key, primary, second, disable_fallbacks=True), "model_per_key", primary)
    assert lenient.traffic() == lenient.expected({primary: (first,)})


def test_yaml_true_string_turns_the_setting_on(tmp_path: Path) -> None:
    with fallback_rig(tmp_path, {SETTING: "true"}) as rig, rig.gateway.scenario() as scenario:
        primary: Final = rig.groups.primary
        first, second = new_marker(), new_marker()
        key: Final = scenario.key(model_rpm_limit={primary: 1})
        assert_served(chat(rig.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        assert_refused(chat(rig.gateway, key, primary, second), "model_per_key", primary)
        assert rig.traffic() == rig.expected({primary: (first,)})


@pytest.mark.parametrize("raw", ("sometimes", ""), ids=("word", "empty"))
def test_yaml_non_boolean_setting_warns_and_keeps_falling_back(tmp_path: Path, raw: str) -> None:
    with fallback_rig(tmp_path, {SETTING: raw}) as rig, rig.gateway.scenario() as scenario:
        primary, fallback = rig.groups.primary, rig.groups.fallback
        first, second = new_marker(), new_marker()
        key: Final = scenario.key(model_rpm_limit={primary: 1})
        assert_served(chat(rig.gateway, key, primary, first), primary, f"chatcmpl-{first}")
        assert_served(chat(rig.gateway, key, primary, second), fallback, f"chatcmpl-{second}")
        assert rig.traffic() == rig.expected({primary: (first,), fallback: (second,)})
        eventually(
            lambda: rig.log.read_text(errors="replace"),
            lambda text: f"general_settings.{SETTING}={raw!r} {NOT_A_BOOLEAN}" in text,
            seconds=10,
        )


def test_legacy_limiter_per_model_429_skips_fallbacks(tmp_path: Path) -> None:
    with fallback_rig(tmp_path, {SETTING: True}, LEGACY_LIMITER) as rig, rig.gateway.scenario() as scenario:
        primary: Final = rig.groups.primary
        markers: Final = tuple(new_marker() for _ in range(LEGACY_ATTEMPTS))
        responses: Final = sent_until_refused(rig, scenario.key(model_rpm_limit={primary: 1}), markers)
        *served, refused = responses
        assert refused.status_code == 429, tuple(
            (response.status_code, response.headers.get("x-litellm-model-group")) for response in responses
        )
        assert "LiteLLM Rate Limit Handler for rate limit type = model_per_key." in error_message(refused.text)
        for marker, response in zip(markers, served):
            assert_served(response, primary, f"chatcmpl-{marker}")
        assert rig.traffic() == rig.expected({primary: markers[: len(served)]})


def test_database_setting_survives_a_restart_and_refuses_per_model_overflow(tmp_path: Path) -> None:
    groups: Final = new_groups()
    with ExitStack() as stack:
        rig_gateway: Final = stack.enter_context(gateway_from_environment())
        database_url: Final = stack.enter_context(scratch_database())
        wires: Final = MappingProxyType(
            {group: stack.enter_context(wire_server(healthy_reply)) for group in serving_groups(groups)}
        )
        config: Final = fallback_config(tmp_path, groups, wires, {})
        environment: Final = MappingProxyType({"DATABASE_URL": database_url})
        replica: Final = ("DATABASE_URL_READ_REPLICA",)
        with owned_proxy_process(
            rig_gateway, tmp_path, environment, config=config, remove_environment=replica, workers=2
        ) as writer:
            writes: Final = (update_setting(writer.gateway, True), update_setting(writer.gateway, True))
        assert tuple(write.status_code for write in writes) == (200, 200), tuple(write.text for write in writes)
        restarted: Final = stack.enter_context(
            owned_proxy_process(
                rig_gateway, tmp_path, environment, config=config, remove_environment=replica, workers=2
            )
        )
        listed: Final = listed_setting(restarted.gateway)
        assert [(entry["field_value"], entry["stored_in_db"]) for entry in listed] == [(True, True)], listed
        rig: Final = Rig(restarted.gateway, restarted.log, groups, wires)
        rig.traffic()
        key: Final = string_value(rig.gateway.post("/key/generate", {"model_rpm_limit": {groups.primary: 1}})["key"])
        first, second = new_marker(), new_marker()
        assert_served(chat(rig.gateway, key, groups.primary, first), groups.primary, f"chatcmpl-{first}")
        assert_refused(chat(rig.gateway, key, groups.primary, second), "model_per_key", groups.primary)
        assert rig.traffic() == rig.expected({groups.primary: (first,)})
