"""A pass-through endpoint opted into the Model Hub is listed there under its display name.

`POST /config/pass_through_endpoint` accepts `display_name` and `show_in_model_hub`. The public hub
routes then carry one row per opted-in endpoint, named by its display name (or its path when no name
was given), in mode `passthrough`, naming the route it describes. An endpoint that did not opt in
stays off the hub. Hub rows come from the shared database, so a peer proxy lists them as well.
"""

from __future__ import annotations

import asyncio
import re
import signal
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import write_rows
from tests.integration._support.process import owned_proxy, owned_proxy_process

PASS_THROUGH_MODE: Final = "passthrough"
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
HUB_PARAMS: Final = {"filter[mode]": PASS_THROUGH_MODE, "page_size": "100"}
BURST: Final = 40


@dataclass(frozen=True, slots=True)
class PassThrough:
    endpoint_id: str
    path: str
    display_name: str | None


@contextmanager
def pass_through(
    gateway: Gateway,
    *,
    show_in_model_hub: bool,
    display_name: str | None,
    methods: tuple[str, ...] | None = None,
    path: str | None = None,
    target_path: str = "",
) -> Generator[PassThrough]:
    body: dict[str, JsonValue] = {
        "path": path or f"/integration-nlp-{uuid.uuid4().hex[:8]}",
        "target": gateway.upstream_url + target_path,
        "show_in_model_hub": show_in_model_hub,
    }
    if display_name is not None:
        body["display_name"] = display_name
    if methods is not None:
        body["methods"] = list(methods)
    created: Final = gateway.request("POST", "/config/pass_through_endpoint", body)
    assert created.status_code == 200, created.text
    endpoints: Final = object_value(created.json())["endpoints"]
    assert isinstance(endpoints, list) and len(endpoints) == 1, created.text
    endpoint_id: Final = string_value(object_value(endpoints[0])["id"])
    try:
        yield PassThrough(endpoint_id, string_value(body["path"]), display_name)
    finally:
        deleted = gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": endpoint_id})
        assert deleted.status_code == 200, deleted.text


def configured_endpoint(gateway: Gateway, endpoint_id: str) -> dict[str, JsonValue]:
    listed: Final = gateway.request("GET", "/config/pass_through_endpoint")
    assert listed.status_code == 200, listed.text
    endpoints: Final = object_value(listed.json())["endpoints"]
    assert isinstance(endpoints, list), listed.text
    matches: Final = tuple(object_value(row) for row in endpoints if object_value(row).get("id") == endpoint_id)
    assert len(matches) == 1, f"{endpoint_id} appears {len(matches)} times in {listed.text}"
    return matches[0]


def hub_rows(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    listed: Final = gateway.request("GET", "/public/v1/model_hub", params=HUB_PARAMS)
    assert listed.status_code == 200, listed.text
    return rows_of(listed)


def rows_of(listed: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = object_value(listed.json())["data"]
    assert isinstance(rows, list), listed.text
    return tuple(object_value(row) for row in rows)


def names_of(rows: tuple[dict[str, JsonValue], ...]) -> tuple[str, ...]:
    return tuple(string_value(row["model_group"]) for row in rows)


def update_pass_through(gateway: Gateway, endpoint_id: str, body: dict[str, JsonValue]) -> None:
    updated: Final = gateway.request("POST", f"/config/pass_through_endpoint/{endpoint_id}", body)
    assert updated.status_code == 200, updated.text


def relayed_contents(gateway: Gateway) -> tuple[str, ...]:
    observed: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=30, trust_env=False)
    assert observed.status_code == 200, observed.text
    requests: Final = object_value(observed.json())["requests"]
    assert isinstance(requests, list), observed.text
    return tuple(_first_message_content(object_value(request)["body"]) for request in requests)


def _first_message_content(body: JsonValue) -> str:
    messages: Final = object_value(body).get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return string_value(object_value(messages[0]).get("content", ""))


def legacy_hub_rows(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    listed: Final = gateway.request("GET", "/public/model_hub")
    assert listed.status_code == 200, listed.text
    rows: Final = TypeAdapter(list[JsonValue]).validate_python(listed.json())
    return tuple(object_value(row) for row in rows)


def hub_modes(gateway: Gateway) -> tuple[str, ...]:
    listed: Final = gateway.request("GET", "/public/v1/model_hub/modes")
    assert listed.status_code == 200, listed.text
    modes: Final = object_value(listed.json())["data"]
    assert isinstance(modes, list), listed.text
    return tuple(string_value(mode) for mode in modes)


def hub_row(rows: tuple[dict[str, JsonValue], ...], model_group: str) -> dict[str, JsonValue]:
    matches: Final = tuple(row for row in rows if row["model_group"] == model_group)
    assert len(matches) == 1, f"{model_group} appears {len(matches)} times in {rows}"
    return matches[0]


def test_opted_in_pass_through_is_listed_under_its_display_name(gateway: Gateway, peer: Gateway) -> None:
    display_name: Final = f"Clinical NER {uuid.uuid4().hex[:6]}"
    with (
        pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown,
        pass_through(gateway, show_in_model_hub=False, display_name="Kept off the hub") as hidden,
    ):
        configured: Final = configured_endpoint(gateway, shown.endpoint_id)
        assert configured["display_name"] == display_name, configured
        assert configured["show_in_model_hub"] is True, configured

        rows: Final = hub_rows(gateway)
        row: Final = hub_row(rows, display_name)
        assert row["mode"] == PASS_THROUGH_MODE, row
        assert row["pass_through_path"] == shown.path, row
        assert row["providers"] == [], row
        assert row["is_public_model_group"] is True, row
        assert not any(r["model_group"] in ("Kept off the hub", hidden.path) for r in rows), rows

        assert PASS_THROUGH_MODE in hub_modes(gateway)

        legacy: Final = hub_row(legacy_hub_rows(gateway), display_name)
        assert legacy["mode"] == PASS_THROUGH_MODE, legacy
        assert legacy["pass_through_path"] == shown.path, legacy

        peer_rows: Final = eventually(
            lambda: hub_rows(peer),
            lambda rows: any(r["model_group"] == display_name for r in rows),
            seconds=70,
        )
        assert hub_row(peer_rows, display_name)["pass_through_path"] == shown.path

    assert not any(r["model_group"] == display_name for r in hub_rows(gateway)), "deleted endpoint still on the hub"


def test_opted_in_pass_through_without_a_display_name_is_listed_by_path(gateway: Gateway) -> None:
    with pass_through(gateway, show_in_model_hub=True, display_name=None, methods=("GET",)) as shown:
        row: Final = hub_row(hub_rows(gateway), shown.path)
        assert row["mode"] == PASS_THROUGH_MODE, row
        assert row["pass_through_path"] == shown.path, row
        assert row["pass_through_methods"] == ["GET"], row


def test_a_published_pass_through_still_forwards_to_its_target(gateway: Gateway) -> None:
    marker: Final = f"relay {uuid.uuid4().hex}"
    with pass_through(
        gateway, show_in_model_hub=True, display_name="Chat relay", target_path="/v1/chat/completions"
    ) as shown:
        assert hub_row(hub_rows(gateway), "Chat relay")["pass_through_path"] == shown.path
        relayed: Final = gateway.request(
            "POST", shown.path, {"model": "relay", "messages": [{"role": "user", "content": marker}]}
        )
        assert relayed.status_code == 200, relayed.text
        assert object_value(relayed.json())["object"] == "chat.completion", relayed.text
        assert marker in relayed_contents(gateway), "the upstream never saw the relayed request"


def test_an_async_client_reads_the_same_pass_through_rows(gateway: Gateway) -> None:
    display_name: Final = f"Async NER {uuid.uuid4().hex[:6]}"

    async def read() -> httpx.Response:
        async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=30, trust_env=False) as client:
            return await client.get("/public/v1/model_hub", params=HUB_PARAMS)

    with pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown:
        listed: Final = asyncio.run(read())
        assert listed.status_code == 200, listed.text
        rows: Final = rows_of(listed)
        assert hub_row(rows, display_name)["pass_through_path"] == shown.path
        assert rows == hub_rows(gateway)


def test_pass_through_rows_paginate_one_per_page(gateway: Gateway) -> None:
    suffix: Final = uuid.uuid4().hex[:6]
    with (
        pass_through(gateway, show_in_model_hub=True, display_name=f"Pager A {suffix}"),
        pass_through(gateway, show_in_model_hub=True, display_name=f"Pager B {suffix}"),
    ):
        first: Final = gateway.request(
            "GET", "/public/v1/model_hub", params={"filter[mode]": PASS_THROUGH_MODE, "page_size": "1"}
        )
        assert first.status_code == 200, first.text
        meta: Final = object_value(object_value(first.json())["meta"])
        assert meta["page_size"] == 1, meta
        assert meta["total_count"] == meta["total_pages"], meta
        seen: list[str] = []
        page = first
        while True:
            page_rows = rows_of(page)
            assert len(page_rows) == 1, page.text
            seen.extend(names_of(page_rows))
            following = object_value(object_value(page.json())["links"])["next"]
            if following is None:
                break
            page = gateway.request("GET", string_value(following))
            assert page.status_code == 200, page.text
        assert len(seen) == meta["total_count"], seen
        assert {f"Pager A {suffix}", f"Pager B {suffix}"} <= set(seen), seen


def test_hub_fields_of_the_wrong_type_are_rejected(gateway: Gateway) -> None:
    path: Final = f"/integration-nlp-{uuid.uuid4().hex[:8]}"
    wrong: Final[tuple[tuple[str, JsonValue], ...]] = (
        ("display_name", 5),
        ("display_name", ["Clinical NER"]),
        ("show_in_model_hub", "maybe"),
    )
    for field, value in wrong:
        created = gateway.request(
            "POST", "/config/pass_through_endpoint", {"path": path, "target": gateway.upstream_url, field: value}
        )
        assert created.status_code == 422, f"{field}={value!r}: {created.status_code} {created.text}"
        assert field in created.text, created.text
    listed: Final = gateway.request("GET", "/config/pass_through_endpoint")
    assert listed.status_code == 200, listed.text
    assert path not in listed.text, listed.text
    assert path not in names_of(hub_rows(gateway))


def test_a_five_kilobyte_display_name_is_listed_verbatim(gateway: Gateway) -> None:
    display_name: Final = f"Long {uuid.uuid4().hex[:6]} " + "n" * 5000
    with pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown:
        assert configured_endpoint(gateway, shown.endpoint_id)["display_name"] == display_name
        assert hub_row(hub_rows(gateway), display_name)["pass_through_path"] == shown.path
        assert hub_row(legacy_hub_rows(gateway), display_name)["pass_through_path"] == shown.path


def test_unauthenticated_callers_read_the_hub_but_cannot_change_publication(gateway: Gateway) -> None:
    display_name: Final = f"Guarded {uuid.uuid4().hex[:6]}"
    with pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown:
        anonymous: Final = gateway.client
        body: Final[dict[str, JsonValue]] = {
            "path": shown.path,
            "target": gateway.upstream_url,
            "show_in_model_hub": False,
        }
        assert anonymous.post("/config/pass_through_endpoint", json={**body, "path": "/intruder"}).status_code == 401
        assert anonymous.post(f"/config/pass_through_endpoint/{shown.endpoint_id}", json=body).status_code == 401
        assert (
            anonymous.delete("/config/pass_through_endpoint", params={"endpoint_id": shown.endpoint_id}).status_code
            == 401
        )
        public: Final = anonymous.get("/public/v1/model_hub", params=HUB_PARAMS)
        assert public.status_code == 200, public.text
        assert hub_row(rows_of(public), display_name)["pass_through_path"] == shown.path
        legacy: Final = anonymous.get("/public/model_hub")
        assert legacy.status_code == 200, legacy.text
        assert display_name in legacy.text
        assert configured_endpoint(gateway, shown.endpoint_id)["show_in_model_hub"] is True


def test_an_empty_display_name_falls_back_to_the_path(gateway: Gateway) -> None:
    with pass_through(gateway, show_in_model_hub=True, display_name="") as shown:
        assert configured_endpoint(gateway, shown.endpoint_id)["display_name"] == ""
        assert hub_row(hub_rows(gateway), shown.path)["pass_through_path"] == shown.path
        assert "" not in names_of(hub_rows(gateway))


def test_an_update_from_an_older_client_keeps_the_hub_fields(gateway: Gateway) -> None:
    display_name: Final = f"Sticky {uuid.uuid4().hex[:6]}"
    with pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown:
        update_pass_through(gateway, shown.endpoint_id, {"path": shown.path, "target": gateway.upstream_url + "/v2"})
        configured: Final = configured_endpoint(gateway, shown.endpoint_id)
        assert configured["target"] == gateway.upstream_url + "/v2", configured
        assert configured["display_name"] == display_name, configured
        assert configured["show_in_model_hub"] is True, configured
        assert hub_row(hub_rows(gateway), display_name)["pass_through_path"] == shown.path


def test_updates_clear_the_name_and_toggle_hub_visibility(gateway: Gateway) -> None:
    display_name: Final = f"Toggled {uuid.uuid4().hex[:6]}"
    with pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown:
        base: Final[dict[str, JsonValue]] = {"path": shown.path, "target": gateway.upstream_url}
        update_pass_through(gateway, shown.endpoint_id, {**base, "display_name": ""})
        assert hub_row(hub_rows(gateway), shown.path)["pass_through_path"] == shown.path
        assert display_name not in names_of(hub_rows(gateway))

        update_pass_through(gateway, shown.endpoint_id, {**base, "show_in_model_hub": False})
        assert shown.path not in names_of(hub_rows(gateway))
        assert shown.path not in names_of(legacy_hub_rows(gateway))

        update_pass_through(
            gateway, shown.endpoint_id, {**base, "show_in_model_hub": True, "display_name": display_name}
        )
        assert hub_row(hub_rows(gateway), display_name)["pass_through_path"] == shown.path
        assert hub_row(legacy_hub_rows(gateway), display_name)["pass_through_path"] == shown.path


def test_two_pass_throughs_sharing_a_display_name_are_two_rows(gateway: Gateway) -> None:
    display_name: Final = f"Twin {uuid.uuid4().hex[:6]}"
    with (
        pass_through(gateway, show_in_model_hub=True, display_name=display_name) as first,
        pass_through(gateway, show_in_model_hub=True, display_name=display_name) as second,
    ):
        twins: Final = tuple(row for row in hub_rows(gateway) if row["model_group"] == display_name)
        assert sorted(string_value(row["pass_through_path"]) for row in twins) == sorted((first.path, second.path))


def test_concurrent_creates_each_survive(gateway: Gateway) -> None:
    display_name: Final = f"Burst {uuid.uuid4().hex[:6]}"
    paths: Final = tuple(f"/integration-nlp-{uuid.uuid4().hex[:8]}" for _index in range(8))

    def create(path: str) -> httpx.Response:
        body: dict[str, JsonValue] = {
            "path": path,
            "target": gateway.upstream_url,
            "display_name": display_name,
            "show_in_model_hub": True,
        }
        return gateway.request("POST", "/config/pass_through_endpoint", body)

    with ThreadPoolExecutor(len(paths)) as pool:
        created: Final = tuple(pool.map(create, paths))
    try:
        assert all(response.status_code == 200 for response in created), [r.text for r in created]
        listed: Final = gateway.request("GET", "/config/pass_through_endpoint")
        stored: Final = tuple(
            string_value(object_value(ep)["path"])
            for ep in listed.json()["endpoints"]
            if object_value(ep)["path"] in paths
        )
        assert sorted(stored) == sorted(paths), listed.text
        published: Final = tuple(row for row in hub_rows(gateway) if row["model_group"] == display_name)
        assert sorted(string_value(row["pass_through_path"]) for row in published) == sorted(paths)
    finally:
        for endpoint_id in endpoint_ids(gateway, paths):
            gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": endpoint_id})


def endpoint_ids(gateway: Gateway, paths: tuple[str, ...]) -> tuple[str, ...]:
    endpoints: Final = gateway.request("GET", "/config/pass_through_endpoint").json()["endpoints"]
    return tuple(string_value(object_value(ep)["id"]) for ep in endpoints if object_value(ep)["path"] in paths)


def test_a_config_file_pass_through_is_listed_once_and_owns_its_key(gateway: Gateway, tmp_path: Path) -> None:
    display_name: Final = f"Config NLP {uuid.uuid4().hex[:6]}"
    path: Final = f"/integration-config-nlp-{uuid.uuid4().hex[:8]}"
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["pass_through_endpoints"] = [
        {"path": path, "target": gateway.upstream_url, "display_name": display_name, "show_in_model_hub": True},
        {"path": f"{path}-hidden", "target": gateway.upstream_url, "display_name": "Hidden config NLP"},
    ]
    config_path: Final = tmp_path / "pass_through_hub.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with owned_proxy(gateway, tmp_path, {}, config=config_path) as owned:
        rows: Final = hub_rows(owned)
        assert names_of(rows).count(display_name) == 1, rows
        assert hub_row(rows, display_name)["pass_through_path"] == path
        assert "Hidden config NLP" not in names_of(rows), rows
        assert hub_row(legacy_hub_rows(owned), display_name)["pass_through_path"] == path
        rejected: Final = owned.request(
            "POST",
            "/config/pass_through_endpoint",
            {"path": f"{path}-db", "target": gateway.upstream_url, "show_in_model_hub": True},
        )
        assert rejected.status_code == 400, rejected.text
        assert "set in the config file" in rejected.text, rejected.text
        assert hub_rows(owned) == rows, "a rejected write changed the hub"
        listed: Final = owned.request("GET", "/config/pass_through_endpoint")
        assert listed.status_code == 200, listed.text
        assert f"{path}-db" not in listed.text, "a rejected write registered an endpoint"
    assert display_name not in names_of(hub_rows(gateway)), "the config row leaked into the shared proxy"


def test_a_health_row_under_the_pass_through_name_is_not_borrowed(gateway: Gateway) -> None:
    display_name: Final = f"gpt-shadow-{uuid.uuid4().hex[:6]}"
    health_check_id: Final = str(uuid.uuid4())
    write_rows(
        'INSERT INTO "LiteLLM_HealthCheckTable" (health_check_id, model_name, status, updated_at) '
        "VALUES (%s, %s, %s, NOW())",
        (health_check_id, display_name, "unhealthy"),
    )
    try:
        with pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown:
            row: Final = hub_row(hub_rows(gateway), display_name)
            assert (row["pass_through_path"], row["health_status"]) == (shown.path, None), row
            legacy: Final = hub_row(legacy_hub_rows(gateway), display_name)
            assert (legacy["pass_through_path"], legacy["health_status"]) == (shown.path, None), legacy
    finally:
        write_rows('DELETE FROM "LiteLLM_HealthCheckTable" WHERE health_check_id = %s', (health_check_id,))


def test_repeated_hub_reads_return_the_same_rows(gateway: Gateway) -> None:
    display_name: Final = f"Stable {uuid.uuid4().hex[:6]}"
    with pass_through(gateway, show_in_model_hub=True, display_name=display_name):
        reads: Final = tuple(hub_rows(gateway) for _ in range(3))
        assert reads[0] == reads[1] == reads[2], reads
        assert names_of(reads[0]).count(display_name) == 1, reads[0]
        assert len(legacy_hub_rows(gateway)) == len(reads[0]), "legacy and v1 disagree on the row count"


def test_hub_reads_stay_200_while_an_endpoint_is_created_and_deleted_under_load(
    gateway: Gateway, peer: Gateway
) -> None:
    display_name: Final = f"Churn {uuid.uuid4().hex[:6]}"

    def read(index: int) -> tuple[int, int]:
        listed = (gateway if index % 2 else peer).request("GET", "/public/v1/model_hub", params=HUB_PARAMS)
        return listed.status_code, names_of(rows_of(listed)).count(display_name) if listed.status_code == 200 else -1

    with ThreadPoolExecutor(8) as pool:
        before: Final = [pool.submit(read, index) for index in range(BURST // 2)]
        with pass_through(gateway, show_in_model_hub=True, display_name=display_name):
            during: Final = [pool.submit(read, index) for index in range(BURST // 2, BURST)]
            results: Final = [future.result() for future in (*before, *during)]
    assert all(status == 200 for status, _count in results), results
    assert all(count in (0, 1) for _status, count in results), results
    assert display_name not in names_of(hub_rows(gateway))
    assert display_name not in names_of(hub_rows(peer))


def test_hub_reads_survive_a_killed_proxy_worker(gateway: Gateway, tmp_path: Path) -> None:
    display_name: Final = f"Survivor {uuid.uuid4().hex[:6]}"

    def read(side: Gateway) -> object:
        try:
            return side.request("GET", "/public/v1/model_hub", params=HUB_PARAMS).status_code
        except httpx.HTTPError as error:
            return error

    with (
        pass_through(gateway, show_in_model_hub=True, display_name=display_name) as shown,
        owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned,
    ):
        workers: Final = eventually(
            lambda: tuple(int(pid) for pid in STARTED_WORKER.findall(owned.log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
        assert hub_row(hub_rows(owned.gateway), display_name)["pass_through_path"] == shown.path
        with ThreadPoolExecutor(8) as pool:
            futures: Final = [pool.submit(read, owned.gateway if index % 2 else gateway) for index in range(BURST)]
            psutil.Process(workers[0]).send_signal(signal.SIGKILL)
            results: Final = [future.result() for future in futures]
        assert all(status == 200 for status in results[0::2]), f"shared proxy answers: {results[0::2]!r}"
        answered: Final = tuple(status for status in results[1::2] if isinstance(status, int))
        assert answered and all(status == 200 for status in answered), f"owned proxy answers: {results[1::2]!r}"
        eventually(lambda: read(owned.gateway), lambda status: status == 200, seconds=60)
        assert hub_row(hub_rows(owned.gateway), display_name)["pass_through_path"] == shown.path
        assert hub_row(hub_rows(gateway), display_name)["pass_through_path"] == shown.path
