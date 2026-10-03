import json
import os
import signal
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.process import group_members, owned_proxy, owned_proxy_process

CLIENT_VERSION: Final = "0.159.3"
FRESH_CONNECTION: Final = {"Connection": "close"}
CONSISTENT_READS: Final = 8
CONVERGENCE_SECONDS: Final = 60
BURST: Final = 24
CATALOG_FILENAME: Final = "litellm-models.json"
SHIM_STOCK_ENV: Final = "CODEX_SHIM_STOCK"
SHIM_LAUNCH_ENV: Final = "CODEX_SHIM_LAUNCH"
SHIM_SCRIPT: Final = """\
import json
import os
import sys

argv = sys.argv[1:]
if argv[-2:] == ["debug", "models"]:
    override = next((arg for arg in argv if arg.startswith("model_catalog_json=")), None)
    if override is None:
        sys.stdout.write(open(os.environ[%(stock)r], encoding="utf-8").read())
    else:
        catalog = json.load(open(json.loads(override.partition("=")[2]), encoding="utf-8"))
        if not catalog.get("models"):
            sys.stderr.write("empty model catalog\\n")
            sys.exit(2)
        sys.stdout.write(json.dumps(catalog))
    sys.exit(0)
recorded = {name: os.environ.get(name) for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "LITELLM_PROXY_API_KEY")}
with open(os.environ[%(launch)r], "w", encoding="utf-8") as handle:
    json.dump({"argv": argv, "env": recorded}, handle)
""" % {"stock": SHIM_STOCK_ENV, "launch": SHIM_LAUNCH_ENV}


def _generic_tier(identity: str) -> dict[str, JsonValue]:
    return {"id": identity, "name": identity.capitalize(), "description": f"Sends service_tier={identity} upstream"}


def _catalog_response(proxy: Gateway, *, key: str | None = None) -> httpx.Response:
    return proxy.request(
        "GET", "/v1/models", key=key, params={"client_version": CLIENT_VERSION}, headers=FRESH_CONNECTION
    )


def _entries(response: httpx.Response) -> list[dict[str, JsonValue]]:
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert set(body) == {"models"}, response.text
    models: Final = body["models"]
    assert isinstance(models, list), response.text
    return [object_value(entry) for entry in models]


def _by_slug(entries: Sequence[Mapping[str, JsonValue]]) -> dict[str, dict[str, JsonValue]]:
    return {string_value(entry["slug"]): dict(entry) for entry in entries}


def _write_config(directory: Path, upstream_url: str, tiered: str, plain: str, alias: str) -> Path:
    config: Final = directory / f"codex_catalog_{uuid.uuid4().hex}.yaml"
    params: Final = {
        "model": "openai/gpt-4o-mini",
        "api_base": f"{upstream_url}/v1",
        "api_key": "integration-provider-key",
    }
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": tiered,
                        "litellm_params": params,
                        "model_info": {
                            "service_tiers": ["priority", {"id": "flex", "name": "Flex", "description": "cheap lane"}],
                            "display_name": "YAML Tiered",
                            "max_input_tokens": 4321,
                        },
                    },
                    {"model_name": plain, "litellm_params": params},
                ],
                "router_settings": {"model_group_alias": {alias: tiered}},
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    return config


@pytest.mark.timeout(240)
def test_yaml_tiers_and_alias_in_owned_proxy(gateway: Gateway, tmp_path: Path) -> None:
    tiered: Final = f"codex-yaml-tiered-{uuid.uuid4().hex}"
    plain: Final = f"codex-yaml-plain-{uuid.uuid4().hex}"
    alias: Final = f"codex-yaml-alias-{uuid.uuid4().hex}"
    config: Final = _write_config(tmp_path, gateway.upstream_url, tiered, plain, alias)
    with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2) as candidate:
        readings: Final = tuple(_entries(_catalog_response(candidate)) for _ in range(CONSISTENT_READS))
        assert all(reading == readings[0] for reading in readings), readings
        catalog: Final = _by_slug(readings[0])
        assert sorted(catalog) == sorted((tiered, plain, alias)), sorted(catalog)
        assert catalog[tiered]["service_tiers"] == [
            _generic_tier("priority"),
            {"id": "flex", "name": "Flex", "description": "cheap lane"},
        ], catalog[tiered]
        assert catalog[tiered]["display_name"] == "YAML Tiered", catalog[tiered]
        assert catalog[tiered]["context_window"] == 4321, catalog[tiered]
        assert catalog[plain]["service_tiers"] == [] and catalog[plain]["display_name"] == plain, catalog[plain]
        assert catalog[alias]["service_tiers"] == catalog[tiered]["service_tiers"], catalog[alias]
        assert catalog[alias]["display_name"] == alias, catalog[alias]
        assert catalog[alias]["context_window"] == 4321, catalog[alias]
        assert string_value(catalog[alias]["base_instructions"]), alias


ROOT: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
BUNDLED_STOCK_PATH: Final = ROOT / "litellm" / "proxy" / "common_utils" / "codex_bundled_models_0.159.3.json"
STOCK_UPSTREAM: Final = "gpt-5.5"


def _bundled_stock_entry(slug: str) -> dict[str, JsonValue]:
    models: Final = JSON_OBJECT.validate_json(BUNDLED_STOCK_PATH.read_bytes())["models"]
    assert isinstance(models, list), slug
    (entry,) = (object_value(model) for model in models if object_value(model)["slug"] == slug)
    return entry


def _write_team_owned_config(
    directory: Path, upstream_url: str, stock_team: str, plain_team: str, owned: str, mixed: str, alias: str
) -> Path:
    config: Final = directory / f"codex_catalog_teams_{uuid.uuid4().hex}.yaml"
    plain_params: Final = {
        "model": "openai/gpt-4o-mini",
        "api_base": f"{upstream_url}/v1",
        "api_key": "integration-provider-key",
    }
    stock_params: Final = {**plain_params, "model": f"openai/{STOCK_UPSTREAM}"}
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {"model_name": owned, "litellm_params": stock_params, "model_info": {"team_id": stock_team}},
                    {
                        "model_name": owned,
                        "litellm_params": plain_params,
                        "model_info": {"team_id": plain_team, "service_tiers": ["flex"]},
                    },
                    {"model_name": mixed, "litellm_params": stock_params, "model_info": {"team_id": stock_team}},
                    {"model_name": mixed, "litellm_params": plain_params, "model_info": {"service_tiers": ["flex"]}},
                ],
                "router_settings": {"model_group_alias": {alias: owned}},
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    return config


@pytest.mark.timeout(240)
def test_team_owned_deployments_of_one_name_serve_each_team_its_own_upstream(gateway: Gateway, tmp_path: Path) -> None:
    """Two teams own a deployment of one model name (and of a `model_group_alias` of it): Codex's stock
    entry for the first deployment's upstream goes only to the team whose requests reach it, and a
    caller outside the owning team reads the deployment no team owns."""
    owned: Final = f"codex-team-owned-{uuid.uuid4().hex}"
    mixed: Final = f"codex-team-mixed-{uuid.uuid4().hex}"
    alias: Final = f"codex-team-alias-{uuid.uuid4().hex}"
    stock: Final = _bundled_stock_entry(STOCK_UPSTREAM)
    assert stock["supported_reasoning_levels"] != [] and stock["service_tiers"] != [], stock
    with gateway.scenario() as scenario:
        stock_team: Final = scenario.team()
        plain_team: Final = scenario.team()
        stock_key: Final = scenario.key(team_id=stock_team, models=[owned, alias, mixed])
        plain_key: Final = scenario.key(team_id=plain_team, models=[owned, alias, mixed])
        teamless_key: Final = scenario.key(models=[mixed])
        config: Final = _write_team_owned_config(
            tmp_path, gateway.upstream_url, stock_team, plain_team, owned, mixed, alias
        )
        with owned_proxy(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2) as candidate:

            def stable_catalog(key: str) -> dict[str, dict[str, JsonValue]]:
                readings: Final = tuple(
                    _entries(_catalog_response(candidate, key=key)) for _ in range(CONSISTENT_READS)
                )
                assert all(reading == readings[0] for reading in readings), readings
                return _by_slug(readings[0])

            for_stock_team: Final = stable_catalog(stock_key)
            for_plain_team: Final = stable_catalog(plain_key)
            for_teamless: Final = stable_catalog(teamless_key)
        assert sorted(for_stock_team) == sorted(for_plain_team) == sorted((owned, alias, mixed)), sorted(for_stock_team)
        assert sorted(for_teamless) == [mixed], sorted(for_teamless)
        for slug in (owned, alias):
            assert for_stock_team[slug]["supported_reasoning_levels"] == stock["supported_reasoning_levels"], slug
            assert for_stock_team[slug]["model_messages"] == stock["model_messages"], slug
            assert for_stock_team[slug]["service_tiers"] == stock["service_tiers"], slug
            assert for_plain_team[slug]["supported_reasoning_levels"] == [], for_plain_team[slug]
            assert "model_messages" not in for_plain_team[slug], slug
            assert string_value(for_plain_team[slug]["base_instructions"]), slug
            assert for_plain_team[slug]["service_tiers"] == [_generic_tier("flex")], for_plain_team[slug]
        assert for_stock_team[mixed]["supported_reasoning_levels"] == stock["supported_reasoning_levels"], mixed
        assert for_stock_team[mixed]["service_tiers"] == [], for_stock_team[mixed]
        for outsider in (for_plain_team, for_teamless):
            assert outsider[mixed]["supported_reasoning_levels"] == [], outsider[mixed]
            assert "model_messages" not in outsider[mixed], mixed
            assert outsider[mixed]["service_tiers"] == [_generic_tier("flex")], outsider[mixed]


def _worker_pids(root_pid: int) -> frozenset[int]:
    def is_worker(process: psutil.Process) -> bool:
        try:
            return "spawn_main" in " ".join(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    return frozenset(process.pid for process in group_members(root_pid) if is_worker(process))


def _catalog_outcome(proxy: Gateway, slugs: Sequence[JsonValue]) -> str:
    try:
        response: Final = _catalog_response(proxy)
    except httpx.TransportError as error:
        return f"transport:{type(error).__name__}"
    assert response.status_code == 200, response.text
    assert sorted(entry["slug"] for entry in _entries(response)) == list(slugs), response.text
    return "ok"


@pytest.mark.timeout(300)
def test_catalog_survives_a_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    tiered: Final = f"codex-yaml-tiered-{uuid.uuid4().hex}"
    plain: Final = f"codex-yaml-plain-{uuid.uuid4().hex}"
    alias: Final = f"codex-yaml-alias-{uuid.uuid4().hex}"
    config: Final = _write_config(tmp_path, gateway.upstream_url, tiered, plain, alias)
    with owned_proxy_process(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2) as owned:
        candidate: Final = owned.gateway
        workers: Final = eventually(lambda: _worker_pids(owned.process.pid), lambda pids: len(pids) == 2, seconds=30)
        victim: Final = min(workers)
        slugs: Final = sorted(entry["slug"] for entry in _entries(_catalog_response(candidate)))
        assert slugs == sorted((tiered, plain, alias)), slugs

        def attempt(index: int) -> str:
            if index == 2:
                os.kill(victim, signal.SIGKILL)
            return _catalog_outcome(candidate, slugs)

        with ThreadPoolExecutor(max_workers=BURST) as pool:
            outcomes: Final = tuple(pool.map(attempt, range(BURST)))
        assert outcomes.count("ok") >= 1, outcomes
        assert all(outcome == "ok" or outcome.startswith("transport:") for outcome in outcomes), outcomes
        respawned: Final = eventually(
            lambda: _worker_pids(owned.process.pid),
            lambda pids: len(pids) == 2 and victim not in pids,
            seconds=60,
        )
        assert victim not in respawned, respawned

        def settled_burst() -> tuple[str, ...]:
            with ThreadPoolExecutor(max_workers=BURST) as pool:
                return tuple(pool.map(lambda _: _catalog_outcome(candidate, slugs), range(BURST)))

        final: Final = eventually(settled_burst, lambda values: all(value == "ok" for value in values), seconds=40)
        assert final == ("ok",) * BURST, final


def _install_shim(directory: Path, stock: Mapping[str, JsonValue]) -> tuple[Path, dict[str, str]]:
    shim_dir: Final = directory / "bin"
    shim_dir.mkdir()
    script: Final = directory / "codex_shim.py"
    script.write_text(SHIM_SCRIPT, encoding="utf-8")
    shim: Final = shim_dir / "codex"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    stock_path: Final = directory / "stock.json"
    stock_path.write_text(json.dumps(stock), encoding="utf-8")
    launch_path: Final = directory / "launch.json"
    home: Final = directory / "home"
    home.mkdir()
    codex_home: Final = directory / "codex-home"
    environment: Final = {
        **os.environ,
        "HOME": str(home),
        "CODEX_HOME": str(codex_home),
        "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        SHIM_STOCK_ENV: str(stock_path),
        SHIM_LAUNCH_ENV: str(launch_path),
    }
    return launch_path, environment


def _run_lite_codex(proxy: Gateway, key: str, environment: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", "from litellm.proxy.client.cli import cli; cli()", "codex", "--shim-marker"],
        env={**environment, "LITELLM_PROXY_URL": str(proxy.client.base_url).rstrip("/"), "LITELLM_PROXY_API_KEY": key},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def _new_model(scenario: Scenario, name: str, model_info: Mapping[str, JsonValue]) -> None:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": f"{scenario.gateway.upstream_url}/v1",
            },
            "model_info": dict(model_info),
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))


def _launch_record(path: Path) -> dict[str, JsonValue]:
    assert path.exists(), "the shim was never launched"
    return JSON_OBJECT.validate_json(path.read_bytes())


def _catalog_override(argv: Sequence[JsonValue]) -> str | None:
    overrides: Final = [
        string_value(arg) for arg in argv if isinstance(arg, str) and arg.startswith("model_catalog_json=")
    ]
    assert len(overrides) <= 1, argv
    return json.loads(overrides[0].partition("=")[2]) if overrides else None


def test_lite_codex_writes_the_shared_catalog(gateway: Gateway, tmp_path: Path) -> None:
    """`lite codex` writes the installed Codex's own entry for a proxy model whose name is a stock slug and
    the fallback entry for any other; the proxy's configured tiers never reach the file (Decision 17)."""
    stock_slug: Final = f"codex-stock-{uuid.uuid4().hex}"
    stock_tiers: Final = [{"id": "priority", "name": "Fast", "description": "1.5x speed"}]
    stock: Final = {
        "models": [
            {
                "slug": stock_slug,
                "display_name": "Stock Model",
                "priority": 7,
                "visibility": "hide",
                "supported_in_api": False,
                "service_tiers": stock_tiers,
                "default_service_tier": None,
                "supported_reasoning_levels": [{"effort": "high", "description": "thinks longer"}],
            }
        ]
    }
    launch_path, environment = _install_shim(tmp_path, stock)
    with gateway.scenario() as scenario:
        _new_model(scenario, stock_slug, {"service_tiers": ["ultrafast"]})
        fallback: Final = scenario.model(model_info={"service_tiers": ["priority"]})
        key: Final = scenario.key(models=[stock_slug, fallback])
        completed: Final = _run_lite_codex(gateway, key, environment)
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert "not syncing" not in completed.stderr, completed.stderr
        record: Final = _launch_record(launch_path)
        argv: Final = record["argv"]
        assert isinstance(argv, list) and argv[-1] == "--shim-marker", argv
        catalog_path: Final = Path(environment["CODEX_HOME"]) / CATALOG_FILENAME
        assert _catalog_override(argv) == str(catalog_path), argv
        env: Final = object_value(record["env"])
        assert env["OPENAI_BASE_URL"] == str(gateway.client.base_url).rstrip("/") + "/v1", env
        assert env["OPENAI_API_KEY"] == key, env.keys()
        written: Final = JSON_OBJECT.validate_json(catalog_path.read_bytes())
        assert set(written) == {"models"}, written.keys()
        models: Final = written["models"]
        assert isinstance(models, list), written
        file_entries: Final = _by_slug([object_value(entry) for entry in models])
        served: Final = eventually(
            lambda: _by_slug(_entries(_catalog_response(gateway, key=key))),
            lambda value: (
                stock_slug in value
                and fallback in value
                and value[stock_slug]["service_tiers"] == [_generic_tier("ultrafast")]
                and value[fallback]["service_tiers"] == [_generic_tier("priority")]
            ),
            seconds=CONVERGENCE_SECONDS,
        )
        assert list(file_entries) == list(served), (list(file_entries), list(served))
        assert file_entries[stock_slug]["service_tiers"] == stock_tiers, file_entries[stock_slug]
        assert file_entries[stock_slug]["visibility"] == "list", file_entries[stock_slug]
        assert file_entries[stock_slug]["supported_in_api"] is True, file_entries[stock_slug]
        assert file_entries[stock_slug]["priority"] == list(file_entries).index(stock_slug), file_entries[stock_slug]
        assert file_entries[stock_slug]["supported_reasoning_levels"] == [
            {"effort": "high", "description": "thinks longer"}
        ]
        assert file_entries[fallback]["service_tiers"] == [], file_entries[fallback]
        assert file_entries[fallback]["display_name"] == fallback, file_entries[fallback]
        assert string_value(file_entries[fallback]["base_instructions"]), fallback
        assert served[stock_slug]["service_tiers"] == [_generic_tier("ultrafast")], served[stock_slug]
        assert served[fallback]["service_tiers"] == [_generic_tier("priority")], served[fallback]


def test_lite_codex_skips_a_stock_dump_without_display_name(gateway: Gateway, tmp_path: Path) -> None:
    """A `codex debug models` dump that predates `display_name` is reported, not written, and Codex still
    launches without a catalog override."""
    stock: Final = {"models": [{"slug": "gpt-5.5", "priority": 0, "visibility": "list"}]}
    launch_path, environment = _install_shim(tmp_path, stock)
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model_info={"service_tiers": ["priority"]})
        key: Final = scenario.key(models=[model])
        completed: Final = _run_lite_codex(gateway, key, environment)
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert "printed no model catalog" in completed.stderr, completed.stderr
        record: Final = _launch_record(launch_path)
        argv: Final = record["argv"]
        assert isinstance(argv, list) and argv[-1] == "--shim-marker", argv
        assert _catalog_override(argv) is None, argv
        assert not (Path(environment["CODEX_HOME"]) / CATALOG_FILENAME).exists(), environment["CODEX_HOME"]
