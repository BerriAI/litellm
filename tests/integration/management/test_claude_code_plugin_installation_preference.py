import json
import signal
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.process import owned_proxy_process

PREFERENCES: Final = ("available", "auto_install", "required")
INVALID_PREFERENCES: Final[tuple[JsonValue, ...]] = (
    "",
    0,
    ["auto_install"],
    "x" * 5000,
    "auto-install",
    "AUTO_INSTALL",
)
INVALID_IDS: Final = ("empty-string", "int", "list", "5kb-string", "hyphenated", "upper-case")
SCHEMAS_WITH_PREFERENCE: Final = ("RegisterPluginRequest", "UpdatePluginRequest", "PluginListItem")
MARKETPLACE: Final = "/claude-code/marketplace.json"
PLUGINS: Final = "/claude-code/plugins"
SKILL_HUB: Final = "/public/skill_hub"
PLUGIN_ROWS: Final = 'SELECT manifest_json, enabled, files_json FROM "LiteLLM_ClaudeCodePluginTable" WHERE name = %s'


def plugin_name() -> str:
    return f"audit-{uuid.uuid4().hex[:12]}"


def archive_source(name: str, digest: str = "a" * 64) -> dict[str, JsonValue]:
    return {"source": "archive", "url": f"https://files.example.com/{name}.zip", "sha256": digest}


def registration(name: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"name": name, "source": archive_source(name), **fields}


def replacement(name: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"source": archive_source(name), **fields}


def delete_plugin(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"{PLUGINS}/{name}")
    assert response.status_code in {200, 404}, response.text


def register(scenario: Scenario, body: Mapping[str, JsonValue]) -> httpx.Response:
    scenario.cleanups.callback(delete_plugin, scenario.gateway, string_value(body["name"]))
    return scenario.gateway.request("POST", PLUGINS, body)


def registered(scenario: Scenario, name: str, **fields: JsonValue) -> None:
    response: Final = register(scenario, registration(name, **fields))
    assert response.status_code == 200, response.text


def replace(gateway: Gateway, name: str, body: Mapping[str, JsonValue], *, key: str | None = None) -> httpx.Response:
    return gateway.request("PUT", f"{PLUGINS}/{name}", body, key=key)


def replaced(gateway: Gateway, name: str, **fields: JsonValue) -> None:
    response: Final = replace(gateway, name, replacement(name, **fields))
    assert response.status_code == 200, response.text


def entries(response: httpx.Response, name: str) -> tuple[dict[str, JsonValue], ...]:
    assert response.status_code == 200, response.text
    plugins: Final = JSON_OBJECT.validate_json(response.content)["plugins"]
    assert isinstance(plugins, list), response.text
    return tuple(object_value(plugin) for plugin in plugins if object_value(plugin)["name"] == name)


def marketplace_entries(gateway: Gateway, name: str, *, key: str | None = None) -> tuple[dict[str, JsonValue], ...]:
    return entries(gateway.client.get(MARKETPLACE, params=None if key is None else {"key": key}), name)


def marketplace_entry(gateway: Gateway, name: str, *, key: str | None = None) -> dict[str, JsonValue]:
    found: Final = marketplace_entries(gateway, name, key=key)
    assert len(found) == 1, found
    return found[0]


def listed_item(gateway: Gateway, name: str) -> dict[str, JsonValue]:
    found: Final = entries(gateway.request("GET", PLUGINS), name)
    assert len(found) == 1, found
    return found[0]


def skill_hub_entries(gateway: Gateway, name: str) -> tuple[dict[str, JsonValue], ...]:
    return entries(gateway.client.get(SKILL_HUB), name)


def detail(gateway: Gateway, name: str, *, key: str | None = None) -> httpx.Response:
    return gateway.request("GET", f"{PLUGINS}/{name}", key=key)


def detail_body(gateway: Gateway, name: str, *, key: str | None = None) -> dict[str, JsonValue]:
    response: Final = detail(gateway, name, key=key)
    assert response.status_code == 200, response.text
    return JSON_OBJECT.validate_json(response.content)


def stored_rows(name: str) -> list[dict[str, JsonValue]]:
    return read_rows(PLUGIN_ROWS, (name,))


def stored_manifest(name: str) -> dict[str, JsonValue]:
    rows: Final = stored_rows(name)
    assert len(rows) == 1, rows
    return JSON_OBJECT.validate_json(string_value(rows[0]["manifest_json"]))


def assert_preference_served(gateway: Gateway, name: str, preference: str) -> None:
    assert marketplace_entry(gateway, name)["installationPreference"] == preference
    assert listed_item(gateway, name)["installation_preference"] == preference
    assert detail_body(gateway, name)["installation_preference"] == preference
    hub: Final = skill_hub_entries(gateway, name)
    assert len(hub) == 1, hub
    assert hub[0]["installation_preference"] == preference


def assert_preference_unset(gateway: Gateway, name: str) -> None:
    assert "installationPreference" not in marketplace_entry(gateway, name)
    assert listed_item(gateway, name)["installation_preference"] is None
    assert detail_body(gateway, name)["installation_preference"] is None
    hub: Final = skill_hub_entries(gateway, name)
    assert len(hub) == 1, hub
    assert hub[0]["installation_preference"] is None


@pytest.mark.parametrize("preference", PREFERENCES)
def test_registered_preference_is_served_on_the_marketplace(gateway: Gateway, preference: str) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference=preference)
        assert marketplace_entry(gateway, name) == {
            "name": name,
            "source": archive_source(name),
            "installationPreference": preference,
            "version": "1.0.0",
        }
        assert stored_manifest(name)["installation_preference"] == preference


def test_preference_echoes_on_list_detail_and_skill_hub(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="required")
        assert_preference_served(gateway, name, "required")


def test_peer_process_serves_the_stored_preference(gateway: Gateway, peer: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="auto_install")
        assert_preference_served(peer, name, "auto_install")


def test_omitted_preference_is_unset(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name)
        assert_preference_unset(gateway, name)
        assert stored_manifest(name).get("installation_preference") is None


def test_explicit_null_preference_is_unset(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference=None)
        assert_preference_unset(gateway, name)
        assert stored_manifest(name).get("installation_preference") is None


@pytest.mark.parametrize("preference", INVALID_PREFERENCES, ids=INVALID_IDS)
def test_invalid_preference_is_refused_on_register(gateway: Gateway, preference: JsonValue) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        response: Final = register(scenario, registration(name, installation_preference=preference))
        assert response.status_code == 422, response.text
        assert "installation_preference" in response.text
        assert stored_rows(name) == []


@pytest.mark.parametrize("preference", INVALID_PREFERENCES, ids=INVALID_IDS)
def test_invalid_preference_is_refused_on_update(gateway: Gateway, preference: JsonValue) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="available")
        response: Final = replace(gateway, name, replacement(name, installation_preference=preference))
        assert response.status_code == 422, response.text
        assert "installation_preference" in response.text
        assert stored_manifest(name)["installation_preference"] == "available"
        assert marketplace_entry(gateway, name)["installationPreference"] == "available"


def test_duplicate_json_key_keeps_the_last_value(gateway: Gateway) -> None:
    name: Final = plugin_name()
    raw: Final = (
        f'{{"name": "{name}", "source": {json.dumps(archive_source(name))}, '
        '"installation_preference": "required", "installation_preference": "available"}'
    )
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(delete_plugin, gateway, name)
        response: Final = gateway.client.post(
            PLUGINS,
            content=raw.encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert response.status_code == 200, response.text
        assert marketplace_entry(gateway, name)["installationPreference"] == "available"
        assert stored_manifest(name)["installation_preference"] == "available"


def test_repeated_identical_update_is_idempotent(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="required")
        replaced(gateway, name, installation_preference="required")
        first: Final = stored_rows(name)
        replaced(gateway, name, installation_preference="required")
        second: Final = stored_rows(name)
        assert len(first) == 1, first
        assert second == first
        assert_preference_served(gateway, name, "required")


def test_duplicate_name_is_refused_and_keeps_the_first_preference(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="required")
        response: Final = register(scenario, registration(name, installation_preference="available"))
        assert response.status_code == 409, response.text
        assert len(stored_rows(name)) == 1
        assert_preference_served(gateway, name, "required")


def test_update_changes_and_clears_the_preference(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="available")
        replaced(gateway, name, installation_preference="required")
        assert_preference_served(gateway, name, "required")
        replaced(gateway, name)
        assert_preference_unset(gateway, name)
        replaced(gateway, name, installation_preference="auto_install")
        assert_preference_served(gateway, name, "auto_install")
        replaced(gateway, name, installation_preference=None)
        assert_preference_unset(gateway, name)
        assert stored_manifest(name).get("installation_preference") is None


def test_update_replaces_the_whole_manifest(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(
            scenario,
            name,
            description="first",
            keywords=["one"],
            author={"name": "audit"},
            homepage="https://example.com/first",
            category="tools",
            installation_preference="required",
        )
        before: Final = detail_body(gateway, name)
        assert before["description"] == "first"
        assert before["installation_preference"] == "required"
        replaced(gateway, name, version="2.0.0", installation_preference="auto_install")
        after: Final = detail_body(gateway, name)
        assert after["updated_at"] != before["updated_at"]
        assert {field: value for field, value in after.items() if field != "updated_at"} == {
            **{field: value for field, value in before.items() if field != "updated_at"},
            "version": "2.0.0",
            "description": None,
            "author": None,
            "homepage": None,
            "keywords": None,
            "category": None,
            "installation_preference": "auto_install",
        }
        assert marketplace_entry(gateway, name) == {
            "name": name,
            "source": archive_source(name),
            "installationPreference": "auto_install",
            "version": "2.0.0",
        }
        assert stored_rows(name)[0]["files_json"] == "{}"


def test_unauthenticated_requests_are_refused(gateway: Gateway) -> None:
    name: Final = plugin_name()
    unauthenticated: Final = gateway.client
    with gateway.scenario() as scenario:
        anonymous_registration: Final = unauthenticated.post(
            PLUGINS, json=registration(name, installation_preference="required")
        )
        assert anonymous_registration.status_code == 401, anonymous_registration.text
        assert stored_rows(name) == []
        registered(scenario, name, installation_preference="required")
        refused: Final = (
            unauthenticated.put(f"{PLUGINS}/{name}", json=replacement(name, installation_preference="available")),
            unauthenticated.post(f"{PLUGINS}/{name}/disable"),
            unauthenticated.post(f"{PLUGINS}/{name}/enable"),
            unauthenticated.delete(f"{PLUGINS}/{name}"),
            unauthenticated.get(PLUGINS),
            unauthenticated.get(f"{PLUGINS}/{name}"),
        )
        assert [response.status_code for response in refused] == [401] * len(refused), [r.text for r in refused]
        assert stored_manifest(name)["installation_preference"] == "required"
        assert stored_rows(name)[0]["enabled"] is True
        assert marketplace_entry(gateway, name)["installationPreference"] == "required"


def test_non_admin_key_cannot_modify_the_catalog(gateway: Gateway) -> None:
    name: Final = plugin_name()
    other: Final = plugin_name()
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        registered(scenario, name, installation_preference="required")
        scenario.cleanups.callback(delete_plugin, gateway, other)
        refused: Final = (
            gateway.request("POST", PLUGINS, registration(other, installation_preference="available"), key=key),
            replace(gateway, name, replacement(name, installation_preference="available"), key=key),
            gateway.request("DELETE", f"{PLUGINS}/{name}", key=key),
        )
        assert [response.status_code for response in refused] == [403] * len(refused), [r.text for r in refused]
        flips: Final = (
            gateway.request("POST", f"{PLUGINS}/{name}/disable", key=key),
            gateway.request("POST", f"{PLUGINS}/{name}/enable", key=key),
        )
        assert all(response.status_code in {401, 403} for response in flips), [r.text for r in flips]
        assert stored_rows(other) == []
        assert stored_manifest(name)["installation_preference"] == "required"
        assert stored_rows(name)[0]["enabled"] is True
        assert detail_body(gateway, name, key=key)["installation_preference"] == "required"


def test_disabled_plugin_keeps_its_preference_for_granted_keys(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="auto_install")
        granted: Final = scenario.key(object_permission={"skills": [name]})
        disabled: Final = gateway.request("POST", f"{PLUGINS}/{name}/disable")
        assert disabled.status_code == 200, disabled.text
        assert marketplace_entries(gateway, name) == ()
        assert skill_hub_entries(gateway, name) == ()
        assert marketplace_entry(gateway, name, key=granted)["installationPreference"] == "auto_install"
        assert detail_body(gateway, name, key=granted)["installation_preference"] == "auto_install"
        assert stored_rows(name)[0]["enabled"] is False
        enabled: Final = gateway.request("POST", f"{PLUGINS}/{name}/enable")
        assert enabled.status_code == 200, enabled.text
        assert_preference_served(gateway, name, "auto_install")
        assert stored_manifest(name)["installation_preference"] == "auto_install"


def test_non_granted_key_cannot_read_a_disabled_plugin(gateway: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="required")
        plain: Final = scenario.key()
        assert detail_body(gateway, name, key=plain)["installation_preference"] == "required"
        disabled: Final = gateway.request("POST", f"{PLUGINS}/{name}/disable")
        assert disabled.status_code == 200, disabled.text
        refused: Final = detail(gateway, name, key=plain)
        assert refused.status_code == 403, refused.text
        assert marketplace_entries(gateway, name, key=plain) == ()
        assert skill_hub_entries(gateway, name) == ()


def test_hand_edited_unknown_preference_is_not_served(gateway: Gateway, peer: Gateway) -> None:
    name: Final = plugin_name()
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="required")
        edited: Final = {**stored_manifest(name), "installation_preference": "someday"}
        write_rows(
            'UPDATE "LiteLLM_ClaudeCodePluginTable" SET manifest_json = %s WHERE name = %s',
            (json.dumps(edited), name),
        )
        assert stored_manifest(name)["installation_preference"] == "someday"
        assert_preference_unset(gateway, name)
        assert_preference_unset(peer, name)
        liveliness: Final = gateway.client.get("/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text


def preference_shapes(schemas: Mapping[str, JsonValue], schema: str) -> list[JsonValue]:
    properties: Final = object_value(object_value(schemas[schema])["properties"])
    assert "installation_preference" in properties, (schema, sorted(properties))
    shapes: Final = object_value(properties["installation_preference"])["anyOf"]
    assert isinstance(shapes, list), (schema, shapes)
    return shapes


@pytest.mark.parametrize("schema", SCHEMAS_WITH_PREFERENCE)
def test_openapi_declares_the_preference_enum(gateway: Gateway, schema: str) -> None:
    response: Final = gateway.client.get("/openapi.json")
    assert response.status_code == 200, response.text
    schemas: Final = object_value(object_value(JSON_OBJECT.validate_json(response.content)["components"])["schemas"])
    shapes: Final = preference_shapes(schemas, schema)
    assert {"type": "string", "enum": list(PREFERENCES)} in shapes, (schema, shapes)
    assert {"type": "null"} in shapes, (schema, shapes)


def test_burst_reads_across_processes_while_the_preference_flips(gateway: Gateway, peer: Gateway) -> None:
    name: Final = plugin_name()
    targets: Final = (gateway, peer)
    flips: Final = tuple(PREFERENCES[(index + 1) % len(PREFERENCES)] for index in range(13))
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="available")

        def job(index: int) -> tuple[str, JsonValue]:
            if index % 4 == 3:
                return "flip", replace(
                    gateway, name, replacement(name, installation_preference=flips[index // 4])
                ).status_code
            return "read", marketplace_entry(targets[index % 2], name).get("installationPreference")

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes: Final = tuple(pool.map(job, range(53)))
        assert [result for kind, result in outcomes if kind == "flip"] == [200] * len(flips), outcomes
        observed: Final = tuple(result for kind, result in outcomes if kind == "read")
        assert len(observed) == 40, outcomes
        assert all(value in PREFERENCES for value in observed), observed
        settled: Final = stored_manifest(name)["installation_preference"]
        assert settled in PREFERENCES, settled
        assert marketplace_entry(gateway, name)["installationPreference"] == settled
        assert marketplace_entry(peer, name)["installationPreference"] == settled
        assert len(stored_rows(name)) == 1
        replaced(gateway, name, installation_preference="required")
        assert_preference_served(gateway, name, "required")
        assert_preference_served(peer, name, "required")
        assert stored_manifest(name)["installation_preference"] == "required"


def test_repeated_identical_reads_are_stable(gateway: Gateway, peer: Gateway) -> None:
    name: Final = plugin_name()
    targets: Final = (gateway, peer)
    with gateway.scenario() as scenario:
        registered(scenario, name, installation_preference="required")
        before: Final = listed_item(gateway, name)
        bodies: Final = tuple(marketplace_entry(targets[index % 2], name) for index in range(10))
        assert all(body == bodies[0] for body in bodies), bodies
        assert bodies[0]["installationPreference"] == "required"
        assert listed_item(peer, name) == before
        assert len(stored_rows(name)) == 1


def test_registration_burst_survives_a_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    names: Final = tuple(plugin_name() for _ in range(20))
    preferences: Final = tuple(PREFERENCES[index % len(PREFERENCES)] for index in range(len(names)))
    with gateway.scenario() as scenario, owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned:
        for name in names:
            scenario.cleanups.callback(delete_plugin, gateway, name)
        supervisor: Final = psutil.Process(owned.process.pid)
        workers: Final = eventually(
            lambda: supervisor.children(recursive=True), lambda found: len(found) >= 2, seconds=30
        )

        def submit(index: int) -> int:
            try:
                return owned.gateway.request(
                    "POST", PLUGINS, registration(names[index], installation_preference=preferences[index])
                ).status_code
            except httpx.TransportError:
                return -1

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures: Final = tuple(pool.submit(submit, index) for index in range(len(names)))
            eventually(lambda: sum(1 for future in futures if future.done()), lambda done: done >= 3, seconds=30)
            workers[0].send_signal(signal.SIGKILL)
            statuses: Final = tuple(future.result() for future in futures)
        assert set(statuses) <= {200, -1}, statuses
        assert statuses.count(200) >= 3, statuses
        eventually(
            lambda: owned.gateway.client.get("/health/readiness").status_code, lambda code: code == 200, seconds=30
        )
        served: Final = tuple(marketplace_entries(owned.gateway, name) for name in names)
        for index, found in enumerate(served):
            assert len(found) == (1 if statuses[index] == 200 else len(found)), (names[index], statuses[index], found)
            assert len(found) <= 1, (names[index], found)
            assert all(entry["installationPreference"] == preferences[index] for entry in found), (names[index], found)
        assert [marketplace_entries(gateway, name) for name in names] == list(served)
        assert sum(len(stored_rows(name)) for name in names) == sum(len(found) for found in served)


def test_registrations_survive_a_proxy_restart(gateway: Gateway, tmp_path: Path) -> None:
    names: Final = tuple(plugin_name() for _ in range(10))
    preferences: Final = tuple(PREFERENCES[index % len(PREFERENCES)] for index in range(len(names)))
    with gateway.scenario() as scenario:
        for name in names:
            scenario.cleanups.callback(delete_plugin, gateway, name)

        def submit(target: Gateway, index: int) -> int:
            return target.request(
                "POST", PLUGINS, registration(names[index], installation_preference=preferences[index])
            ).status_code

        with owned_proxy_process(gateway, tmp_path, {}) as first, ThreadPoolExecutor(max_workers=8) as pool:
            statuses: Final = tuple(pool.map(partial(submit, first.gateway), range(len(names))))
        assert statuses == (200,) * len(names), statuses
        with owned_proxy_process(gateway, tmp_path, {}) as second:
            served: Final = tuple(marketplace_entry(second.gateway, name)["installationPreference"] for name in names)
            assert served == preferences
            listed: Final = second.gateway.request("GET", PLUGINS)
            assert listed.status_code == 200, listed.text
            plugins: Final = JSON_OBJECT.validate_json(listed.content)["plugins"]
            assert isinstance(plugins, list), listed.text
            assert sum(1 for plugin in plugins if object_value(plugin)["name"] in names) == len(names), listed.text
        assert sum(len(stored_rows(name)) for name in names) == len(names)
