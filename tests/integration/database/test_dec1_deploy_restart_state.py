"""A proxy restarted with the same config and the same database comes back with the same identity and state.

Contract: deployment ids for config models are derived from the model group and its `litellm_params`
(`deployment_id`, litellm/rust/crates/router-core/src/registry.rs:1178, the sha256 the Python router also uses),
so the same config gives the same ids on every boot; an explicit `model_info.id` is used verbatim; keys, teams,
models added through the API, flushed spend and budget blocks live in Postgres and survive the restart.

Rolling deploys restart every pod, and dashboards, routing rules, per-deployment rate limits and spend rows are
keyed by these ids. When the process model and the config loader move to Rust, a restart must stay invisible to
callers. Each test boots twice on one scratch database; the first proxy is stopped with SIGTERM so its shutdown
flush runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, list_value
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration.configuration.dec1_deploy_support import (
    CHAT,
    EXPECTED_SPEND,
    PROVIDER_KEY,
    SECOND_CHAT,
    await_rows,
    chat,
    deploy_config,
    deployment,
    provider,
    terminate,
    write_config,
)

pytestmark: Final = pytest.mark.timeout(900)

PINNED_ID: Final = "dec1-deploy-pinned-id"
API_MODEL: Final = "dec1-deploy-api-model"
API_MODEL_ID: Final = "dec1-deploy-api-model-id"
HEARTBEATS: Final = 'SELECT worker_id FROM "LiteLLM_ProxyWorkerHeartbeat"'


@contextmanager
def _boot(gateway: Gateway, directory: Path, database: str, config: Path) -> Iterator[OwnedProxy]:
    with owned_proxy_process(
        gateway,
        directory,
        {"DATABASE_URL": database},
        config=config,
        remove_environment=("DATABASE_URL_READ_REPLICA",),
    ) as owned:
        yield replace(owned, gateway=replace(owned.gateway, database_url=database))


def _ids(gateway: Gateway, name: str) -> list[str]:
    rows: Final = [JSON_OBJECT.validate_python(item) for item in list_value(gateway.get("/model/info")["data"])]
    return sorted(
        str(JSON_OBJECT.validate_python(row["model_info"])["id"]) for row in rows if row["model_name"] == name
    )


def _config(tmp_path: Path, api_base: str, *extra: dict[str, JsonValue], name: str = "dec1-deploy.yaml") -> Path:
    return write_config(
        tmp_path, deploy_config(api_base, general_settings={"proxy_batch_write_at": 3600}, extra_models=extra), name
    )


def test_config_deployment_ids_are_the_same_after_a_restart(gateway: Gateway, tmp_path: Path) -> None:
    """The same config is booted twice on one database. `GET /model/info` must report the same single id for
    `dec1-deploy-chat` both times: ids are a hash of the model group and its params, not a per-boot uuid."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base)
        with _boot(gateway, tmp_path / "first", database, config) as first:
            before: Final = _ids(first.gateway, CHAT)
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            after: Final = _ids(second.gateway, CHAT)
    assert len(before) == 1 and before == after, (before, after)


def test_a_pinned_model_info_id_is_served_verbatim_after_a_restart(gateway: Gateway, tmp_path: Path) -> None:
    """A second deployment sets `model_info.id: dec1-deploy-pinned-id`. After a restart `GET /model/info` must
    report exactly `["dec1-deploy-pinned-id"]` for it: operators pin ids so rules survive param edits."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        pinned: Final = {**deployment(SECOND_CHAT, upstream.api_base), "model_info": {"id": PINNED_ID}}
        config: Final = _config(tmp_path, upstream.api_base, pinned)
        with _boot(gateway, tmp_path / "first", database, config) as first:
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            assert _ids(second.gateway, SECOND_CHAT) == [PINNED_ID]


def test_changing_a_deployments_params_changes_its_derived_id(gateway: Gateway, tmp_path: Path) -> None:
    """The second boot uses a config whose only change is `timeout: 30` added to the chat deployment's params.
    `GET /model/info` must report a different id than the first boot: the id follows the params, which is why
    pinning `model_info.id` is the only stable handle across config edits.
    UNVERIFIED: that every key of `litellm_params`, including `timeout`, feeds the hash (registry.rs:1178)."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        first_config: Final = _config(tmp_path, upstream.api_base)
        edited: Final = write_config(
            tmp_path,
            {
                **deploy_config(upstream.api_base, general_settings={"proxy_batch_write_at": 3600}),
                "model_list": [deployment(CHAT, upstream.api_base, timeout=30)],
            },
            "dec1-deploy-edited.yaml",
        )
        with _boot(gateway, tmp_path / "first", database, first_config) as first:
            before: Final = _ids(first.gateway, CHAT)
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, edited) as second:
            after: Final = _ids(second.gateway, CHAT)
    assert len(after) == 1 and after != before, (before, after)


def test_a_restart_does_not_duplicate_config_models(gateway: Gateway, tmp_path: Path) -> None:
    """With `STORE_MODEL_IN_DB=True` the same config is booted twice. After the second boot `GET /model/info`
    must still list exactly one `dec1-deploy-chat` deployment: a loader that re-inserts config models into the
    database on every boot would double the traffic weight of every model."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base)
        with _boot(gateway, tmp_path / "first", database, config) as first:
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            assert len(_ids(second.gateway, CHAT)) == 1


def test_a_virtual_key_created_before_a_restart_still_authorises_calls(gateway: Gateway, tmp_path: Path) -> None:
    """A key is generated on the first boot, the proxy is stopped, and a second proxy boots on the same
    database. A chat completion with that key must answer 200 on the second proxy."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base)
        with _boot(gateway, tmp_path / "first", database, config) as first:
            key: Final = str(first.gateway.post("/key/generate", {"models": [CHAT]})["key"])
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            assert chat(second.gateway, key=key).status_code == 200


def test_a_model_added_through_the_api_is_served_after_a_restart_with_its_id(gateway: Gateway, tmp_path: Path) -> None:
    """On the first boot the master key adds `dec1-deploy-api-model` with `model_info.id:
    dec1-deploy-api-model-id` through `POST /model/new`. After a restart `GET /model/info` must report exactly
    that id for it and a chat completion on it must answer 200: database models are reloaded at boot."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base)
        with _boot(gateway, tmp_path / "first", database, config) as first:
            first.gateway.post(
                "/model/new",
                {
                    "model_name": API_MODEL,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": upstream.api_base,
                        "api_key": PROVIDER_KEY,
                    },
                    "model_info": {"id": API_MODEL_ID},
                },
            )
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            assert _ids(second.gateway, API_MODEL) == [API_MODEL_ID]
            assert chat(second.gateway, model=API_MODEL).status_code == 200


def test_key_spend_flushed_at_shutdown_is_reported_after_a_restart(gateway: Gateway, tmp_path: Path) -> None:
    """`proxy_batch_write_at: 3600`. A key makes one call costing 0.011 on the first boot, the proxy is stopped
    with SIGTERM, and a second proxy boots on the same database. `GET /key/info` for the key must report
    `spend` 0.011: the restart neither loses nor double counts the in-memory spend."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base)
        with _boot(gateway, tmp_path / "first", database, config) as first:
            key: Final = str(first.gateway.post("/key/generate", {})["key"])
            assert chat(first.gateway, key=key).status_code == 200
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            info: Final = JSON_OBJECT.validate_python(second.gateway.get("/key/info", {"key": key})["info"])
    assert float(str(info["spend"])) == pytest.approx(EXPECTED_SPEND), info


def test_a_key_over_its_budget_before_a_restart_is_still_blocked_after_it(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `max_budget: 0.005` makes one call costing 0.011 on the first boot, then the proxy is stopped
    and a second one boots on the same database. A chat completion with the key must be refused with a 4xx
    whose body mentions the budget, and must not reach the provider: a restart must not reset enforcement.
    UNVERIFIED: the exact status (400 `budget_exceeded` is expected) is not asserted."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base)
        with _boot(gateway, tmp_path / "first", database, config) as first:
            key: Final = str(first.gateway.post("/key/generate", {"max_budget": 0.005})["key"])
            assert chat(first.gateway, key=key).status_code == 200
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            upstream.posted()
            refused: Final = chat(second.gateway, key=key)
            assert 400 <= refused.status_code < 500, refused.text
            assert "budget" in refused.text.lower(), refused.text
            assert [request for request in upstream.posted() if request.method == "POST"] == []


def test_a_team_created_before_a_restart_still_scopes_its_keys(gateway: Gateway, tmp_path: Path) -> None:
    """A team limited to `dec1-deploy-chat` and a key in it are created on the first boot. After a restart the
    key must still be served on `dec1-deploy-chat` (200) and refused on `dec1-deploy-second` (401 or 403):
    team scoping is read from the database again, not lost with the process."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base, deployment(SECOND_CHAT, upstream.api_base))
        with _boot(gateway, tmp_path / "first", database, config) as first:
            team: Final = str(first.gateway.post("/team/new", {"models": [CHAT]})["team_id"])
            key: Final = str(first.gateway.post("/key/generate", {"team_id": team})["key"])
            assert terminate(first) == 0
        with _boot(gateway, tmp_path / "second", database, config) as second:
            assert chat(second.gateway, key=key).status_code == 200
            assert chat(second.gateway, key=key, model=SECOND_CHAT).status_code in (401, 403)


def test_a_restarted_worker_registers_a_new_heartbeat_row_and_the_old_one_is_gone(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A single worker proxy boots, is stopped with SIGTERM, and a second proxy boots on the same database.
    `LiteLLM_ProxyWorkerHeartbeat` must then hold exactly one row, with a `worker_id` different from the first
    boot's: each process takes a fresh uuid (proxy_worker_heartbeat.py:73) and the old row was deregistered."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = _config(tmp_path, upstream.api_base)
        with _boot(gateway, tmp_path / "first", database, config):
            first_rows: Final = await_rows(lambda: read_rows(HEARTBEATS, (), database_url=database), 1)
        with _boot(gateway, tmp_path / "second", database, config):
            second_rows: Final = await_rows(lambda: read_rows(HEARTBEATS, (), database_url=database), 1)
    assert first_rows[0]["worker_id"] != second_rows[0]["worker_id"], (first_rows, second_rows)
