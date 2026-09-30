"""A pass-through endpoint opted into the Model Hub is listed there under its display name.

`POST /config/pass_through_endpoint` accepts `display_name` and `show_in_model_hub`. The public hub
routes then carry one row per opted-in endpoint, named by its display name (or its path when no name
was given), in mode `passthrough`, naming the route it describes. An endpoint that did not opt in
stays off the hub. Hub rows come from the shared database, so a peer proxy lists them as well.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, object_value, string_value

PASS_THROUGH_MODE: Final = "passthrough"


@dataclass(frozen=True, slots=True)
class PassThrough:
    endpoint_id: str
    path: str
    display_name: str | None


@contextmanager
def pass_through(
    gateway: Gateway, *, show_in_model_hub: bool, display_name: str | None, methods: tuple[str, ...] | None = None
) -> Generator[PassThrough]:
    path: Final = f"/integration-nlp-{uuid.uuid4().hex[:8]}"
    body: dict[str, JsonValue] = {"path": path, "target": gateway.upstream_url, "show_in_model_hub": show_in_model_hub}
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
        yield PassThrough(endpoint_id, path, display_name)
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
    listed: Final = gateway.request(
        "GET", "/public/v1/model_hub", params={"filter[mode]": PASS_THROUGH_MODE, "page_size": "100"}
    )
    assert listed.status_code == 200, listed.text
    rows: Final = object_value(listed.json())["data"]
    assert isinstance(rows, list), listed.text
    return tuple(object_value(row) for row in rows)


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
