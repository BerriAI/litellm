from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from tests.integration._support.process import graceful_stop_seconds, owned_proxy

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)

SUFFIX: Final = uuid.uuid4().hex[:10]
NAMED: Final = f"hub-named-{SUFFIX}"
OTHER: Final = f"hub-other-{SUFFIX}"
HIDDEN: Final = f"hub-hidden-{SUFFIX}"
TEAM_MODEL: Final = f"hub-team-{SUFFIX}"
WILDCARD_PREFIX: Final = f"hubwild{SUFFIX}"
WILDCARD: Final = f"{WILDCARD_PREFIX}/*"
V1_HUB: Final = "/public/v1/model_hub"
LEGACY_HUB: Final = "/public/model_hub"
HUB_PATHS: Final = (V1_HUB, LEGACY_HUB)
HUB_IDS: Final = ("v1", "legacy")
ROW_LIST: Final = TypeAdapter(list[JsonValue])
TEAM_SYNC_SECONDS: Final = 60


def _deployment(name: str, model: str, upstream_url: str, model_info: dict[str, JsonValue] | None = None) -> JsonValue:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": f"{upstream_url}/v1", "api_key": "hub-key"},
        **({"model_info": model_info} if model_info is not None else {}),
    }


def _config(directory: Path, upstream_url: str, published: list[str]) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = {
        **base,
        "litellm_settings": {**base["litellm_settings"], "public_model_groups": published},
        "router_settings": {},
        "model_list": [
            _deployment(NAMED, "openai/gpt-4o-mini", upstream_url),
            _deployment(OTHER, "openai/gpt-4o-mini", upstream_url),
            _deployment(HIDDEN, "openai/gpt-4o-mini", upstream_url, {"discoverable": False}),
            _deployment(WILDCARD, "openai/*", upstream_url),
        ],
    }
    path: Final = directory / "public_all.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _proxy(directory: Path, published: list[str]) -> Iterator[Gateway]:
    with (
        gateway_from_environment() as gateway,
        owned_proxy(gateway, directory, {}, config=_config(directory, gateway.upstream_url, published)) as candidate,
    ):
        yield candidate


@pytest.fixture(scope="module")
def all_public(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with _proxy(tmp_path_factory.mktemp("hub-public-all"), ["*"]) as gateway:
        yield gateway


@pytest.fixture(scope="module")
def named_public(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with _proxy(tmp_path_factory.mktemp("hub-public-named"), [NAMED]) as gateway:
        yield gateway


def _hub_payload(gateway: Gateway, path: str, page: int) -> JsonValue:
    listed: Final = gateway.request("GET", path, params={"page_size": "100", "page": str(page)} if path == V1_HUB else None)
    assert listed.status_code == 200, listed.text
    return listed.json()


def _row_names(rows: JsonValue) -> tuple[str, ...]:
    return tuple(str(object_value(row)["model_group"]) for row in ROW_LIST.validate_python(rows))


def _hub_names(gateway: Gateway, path: str) -> tuple[str, ...]:
    first: Final = _hub_payload(gateway, path, 1)
    if path != V1_HUB:
        return _row_names(first)
    total_pages: Final = int(str(object_value(object_value(first)["meta"])["total_pages"]))
    return tuple(
        name
        for page in range(1, total_pages + 1)
        for name in _row_names(object_value(first if page == 1 else _hub_payload(gateway, path, page))["data"])
    )


def _admin_rows(gateway: Gateway) -> dict[str, dict[str, JsonValue]]:
    rows: Final = ROW_LIST.validate_python(gateway.get("/model_group/info")["data"])
    return {str(object_value(row)["model_group"]): object_value(row) for row in rows}


@pytest.mark.parametrize("path", HUB_PATHS, ids=HUB_IDS)
def test_star_publishes_every_configured_group_and_wildcard_child(all_public: Gateway, path: str) -> None:
    names: Final = _hub_names(all_public, path)
    assert "*" not in names, names
    assert {NAMED, OTHER} <= set(names), names
    assert any(name.startswith(f"{WILDCARD_PREFIX}/") for name in names), names
    assert not any("*" in name for name in names), names


@pytest.mark.parametrize("path", HUB_PATHS, ids=HUB_IDS)
def test_star_keeps_undiscoverable_groups_off_the_hub(all_public: Gateway, path: str) -> None:
    assert HIDDEN in _admin_rows(all_public)
    names: Final = _hub_names(all_public, path)
    assert NAMED in names and HIDDEN not in names, names


@pytest.mark.parametrize("path", HUB_PATHS, ids=HUB_IDS)
def test_star_lists_what_the_admin_sees_minus_undiscoverable_groups(all_public: Gateway, path: str) -> None:
    admin_names: Final = frozenset(_admin_rows(all_public)) - {HIDDEN}
    assert frozenset(_hub_names(all_public, path)) == admin_names


def test_star_keeps_team_scoped_models_off_the_hub(all_public: Gateway) -> None:
    with all_public.scenario() as scenario:
        team: Final = scenario.team()
        created: Final = all_public.post(
            "/model/new",
            {
                "model_name": TEAM_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "hub-key",
                    "api_base": f"{all_public.upstream_url}/v1",
                },
                "model_info": {"team_id": team},
            },
        )
        identity: Final = str(object_value(created["model_info"])["id"])
        scenario.cleanups.callback(all_public.post, "/model/delete", {"id": identity})
        team_names: Final = eventually(
            lambda: tuple(
                str(object_value(row)["model_name"])
                for row in ROW_LIST.validate_python(all_public.get("/model/info")["data"])
                if object_value(object_value(row)["model_info"]).get("team_id") == team
            ),
            bool,
            seconds=TEAM_SYNC_SECONDS,
        )
        hub: Final = frozenset((*_hub_names(all_public, V1_HUB), *_hub_names(all_public, LEGACY_HUB)))
        assert TEAM_MODEL not in hub and hub.isdisjoint(team_names), (team_names, sorted(hub))


def test_star_marks_every_listed_group_public_for_the_admin(all_public: Gateway) -> None:
    rows: Final = _admin_rows(all_public)
    assert rows[NAMED].get("is_public_model_group") is True, rows[NAMED]
    assert rows[OTHER].get("is_public_model_group") is True, rows[OTHER]
    assert rows[HIDDEN].get("is_public_model_group") is False, rows[HIDDEN]


@pytest.mark.parametrize("path", HUB_PATHS, ids=HUB_IDS)
def test_named_list_still_publishes_only_the_named_groups(named_public: Gateway, path: str) -> None:
    assert _hub_names(named_public, path) == (NAMED,)
    rows: Final = _admin_rows(named_public)
    assert rows[NAMED].get("is_public_model_group") is True, rows[NAMED]
    assert rows[OTHER].get("is_public_model_group") is False, rows[OTHER]
