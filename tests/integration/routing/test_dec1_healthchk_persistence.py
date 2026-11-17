"""Health check results stored in Postgres, and the `/health/latest` and `/health/history` reads of them.

Every targeted on-demand `GET /health` writes one `LiteLLM_HealthCheckTable` row in the background
(`_save_health_check_to_db`, litellm/proxy/health_endpoints/_health_endpoints.py:773): `model_name` is the
targeted group, `model_id` the targeted id if any, `status` is `healthy` when at least one endpoint passed,
`error_message` is the first unhealthy endpoint's error cut to 500 characters, and `checked_by` is the
caller's `user_id`. An untargeted `GET /health` writes nothing.

The background loop writes one row per deployment (`save_background_health_checks_to_db`) with
`checked_by: background_health_check`, but only when the status changed since the newest stored row or that
row is an hour old, so a stable fleet writes about once an hour and an outage transition is written at once.

`/health/latest` returns the newest row per model keyed by `model_id` (else `model_name`), and
`/health/history` pages rows newest first with `model` and `status_filter` filters. The database access
layer is being ported to Rust; the UI's model health page reads exactly these rows and fields.

Two owned proxies on the harness database: `ondemand` without background checks and `background` with a
2 second interval. Group names carry a per-run suffix so rows from other runs never match.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, held, list_value, object_value, string_value
from tests.integration.routing.dec1_healthchk_support import (
    Rig,
    deployment,
    health,
    health_rig,
    health_rows,
    report,
)

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:6]
ROW_SECONDS: Final = 30
CONVERGENCE_SECONDS: Final = 90
OK_GROUP: Final = f"db-ok-{RUN}"
BAD_GROUP: Final = f"db-bad-{RUN}"
PAIR_GROUP: Final = f"db-pair-{RUN}"
FLIP_GROUP: Final = f"db-flip-{RUN}"
STABLE_GROUP: Final = f"db-stable-{RUN}"
OK_MODEL: Final = f"ok-db-{RUN}"
BAD_MODEL: Final = f"status-503-db-{RUN}"
PAIR_MODELS: Final = (f"ok-db-pair-a-{RUN}", f"ok-db-pair-b-{RUN}")
FLIP_MODEL: Final = f"ok-db-flip-{RUN}"
STABLE_MODEL: Final = f"ok-db-stable-{RUN}"
LATEST_FIELDS: Final = frozenset(
    {
        "health_check_id",
        "model_name",
        "model_id",
        "status",
        "healthy_count",
        "unhealthy_count",
        "error_message",
        "response_time_ms",
        "details",
        "checked_by",
        "checked_at",
        "created_at",
    }
)


def _ondemand_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment(OK_GROUP, OK_MODEL, api_base),
        deployment(BAD_GROUP, BAD_MODEL, api_base),
        *(deployment(PAIR_GROUP, model, api_base) for model in PAIR_MODELS),
    )


def _background_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment(FLIP_GROUP, FLIP_MODEL, api_base),
        deployment(STABLE_GROUP, STABLE_MODEL, api_base),
        *(deployment(PAIR_GROUP, model, api_base) for model in PAIR_MODELS),
    )


@pytest.fixture(scope="module")
def ondemand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(tmp_path_factory.mktemp("dec1-healthchk-db-ondemand"), _ondemand_models) as value:
        yield value


@pytest.fixture(scope="module")
def background(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-db-background"),
        _background_models,
        general_settings={"background_health_checks": True, "health_check_interval": 2},
    ) as value:
        yield value


def _rows_by(rig: Rig, checked_by: str) -> list[dict[str, JsonValue]]:
    return rig.gateway.rows(
        'SELECT model_name, model_id, status FROM "LiteLLM_HealthCheckTable" WHERE checked_by = %s', (checked_by,)
    )


def _user_key(rig: Rig, scenario_models: Sequence[str]) -> tuple[str, str]:
    """A fresh internal user and a key of theirs limited to `scenario_models`; returns (user_id, key)."""
    user_id: Final = f"dec1-healthchk-user-{uuid.uuid4().hex[:8]}"
    rig.gateway.post("/user/new", {"user_id": user_id})
    created: Final = rig.gateway.post("/key/generate", {"user_id": user_id, "models": list(scenario_models)})
    return user_id, string_value(created["key"])


def test_a_targeted_check_stores_a_healthy_row_attributed_to_the_caller(ondemand: Rig) -> None:
    """A user's key, allowed `db-ok-<run>`, calls `GET /health?model=db-ok-<run>`.

    Within 30 seconds one `LiteLLM_HealthCheckTable` row must exist with `model_name` `db-ok-<run>`,
    `model_id` null, `status` `healthy`, `healthy_count` 1, `unhealthy_count` 0, `error_message` null and
    `checked_by` equal to that user's id.
    """
    user_id, key = _user_key(ondemand, [OK_GROUP])
    assert health(ondemand.gateway, key=key, model=OK_GROUP).status_code == 200
    rows: Final = eventually(lambda: _rows_by(ondemand, user_id), lambda found: len(found) == 1, seconds=ROW_SECONDS)
    assert rows[0]["model_name"] == OK_GROUP
    assert rows[0]["model_id"] is None
    assert rows[0]["status"] == "healthy"
    stored: Final = next(row for row in health_rows(ondemand.gateway, OK_GROUP) if row["checked_by"] == user_id)
    assert (stored["healthy_count"], stored["unhealthy_count"], stored["error_message"]) == (1, 0, None)


def test_a_failed_targeted_check_stores_an_unhealthy_row_with_the_provider_error(ondemand: Rig) -> None:
    """The master key calls `GET /health?model=db-bad-<run>`; its one deployment answers 503 with
    `scripted 503 from status-503-db-<run>`.

    Within 30 seconds the newest row for `db-bad-<run>` must have `status` `unhealthy`, `healthy_count` 0,
    `unhealthy_count` 1, and an `error_message` that contains the provider message and is at most 500
    characters long.
    """
    assert health(ondemand.gateway, model=BAD_GROUP).status_code == 503
    rows: Final = eventually(
        lambda: health_rows(ondemand.gateway, BAD_GROUP), lambda found: len(found) >= 1, seconds=ROW_SECONDS
    )
    newest: Final = rows[0]
    assert (newest["status"], newest["healthy_count"], newest["unhealthy_count"]) == ("unhealthy", 0, 1)
    message: Final = string_value(newest["error_message"])
    assert f"scripted 503 from {BAD_MODEL}" in message
    assert len(message) <= 500


def test_a_check_by_model_id_stores_the_id_and_the_deployments_group_name(ondemand: Rig) -> None:
    """The master key calls `GET /health?model_id=hc-ok-db-pair-a-<run>`, one of two deployments of
    `db-pair-<run>`.

    Within 30 seconds a row must exist with `model_id` `hc-ok-db-pair-a-<run>`, `model_name` `db-pair-<run>`
    (resolved from the id), `status` `healthy` and `healthy_count` 1.
    """
    target: Final = f"hc-{PAIR_MODELS[0]}"
    assert health(ondemand.gateway, model_id=target).status_code == 200
    rows: Final = eventually(
        lambda: [row for row in health_rows(ondemand.gateway, PAIR_GROUP) if row["model_id"] == target],
        lambda found: len(found) >= 1,
        seconds=ROW_SECONDS,
    )
    assert (rows[0]["status"], rows[0]["healthy_count"]) == ("healthy", 1)


def test_an_untargeted_check_stores_no_row(ondemand: Rig) -> None:
    """A user's key with access to `db-ok-<run>` calls `GET /health` with no model.

    For another 6 seconds after the answer, no row may have `checked_by` equal to that user: only targeted
    checks are recorded (`model_name_for_db` is `None` otherwise).
    """
    user_id, key = _user_key(ondemand, [OK_GROUP])
    assert health(ondemand.gateway, key=key).status_code == 200
    held(lambda: _rows_by(ondemand, user_id), lambda found: found == [], holding=6, seconds=20)


def test_a_stored_row_records_a_non_negative_response_time(ondemand: Rig) -> None:
    """After a targeted check of `db-ok-<run>`, the newest row's `response_time_ms` must be a number at least 0
    (no ceiling, a loaded runner can be slow): it is the wall time of the check, which the UI shows as latency.
    """
    assert health(ondemand.gateway, model=OK_GROUP).status_code == 200
    rows: Final = eventually(
        lambda: health_rows(ondemand.gateway, OK_GROUP), lambda found: len(found) >= 1, seconds=ROW_SECONDS
    )
    elapsed: Final = rows[0]["response_time_ms"]
    assert isinstance(elapsed, (int, float))
    assert elapsed >= 0


def test_latest_returns_the_newest_row_per_model_keyed_by_model_name_when_there_is_no_id(ondemand: Rig) -> None:
    """After a targeted check of `db-bad-<run>` has been stored, the master key calls `GET /health/latest`.

    `latest_health_checks["db-bad-<run>"]` must exist with `model_name` `db-bad-<run>`, `status`
    `unhealthy`, and exactly the twelve fields `health_check_id, model_name, model_id, status, healthy_count,
    unhealthy_count, error_message, response_time_ms, details, checked_by, checked_at, created_at`, and
    `total_models` must equal the number of entries.
    """
    assert health(ondemand.gateway, model=BAD_GROUP).status_code == 503
    eventually(lambda: health_rows(ondemand.gateway, BAD_GROUP), lambda found: len(found) >= 1, seconds=ROW_SECONDS)
    body: Final = ondemand.gateway.get("/health/latest")
    latest: Final = object_value(body["latest_health_checks"])
    entry: Final = object_value(latest[BAD_GROUP])
    assert set(entry) == LATEST_FIELDS
    assert (entry["model_name"], entry["status"]) == (BAD_GROUP, "unhealthy")
    assert body["total_models"] == len(latest)


def test_latest_keys_a_row_with_a_model_id_by_that_id(ondemand: Rig) -> None:
    """After `GET /health?model_id=hc-ok-db-pair-b-<run>` has been stored, `GET /health/latest` must have an
    entry under the key `hc-ok-db-pair-b-<run>` whose `model_name` is `db-pair-<run>`.
    """
    target: Final = f"hc-{PAIR_MODELS[1]}"
    assert health(ondemand.gateway, model_id=target).status_code == 200
    eventually(
        lambda: [row for row in health_rows(ondemand.gateway, PAIR_GROUP) if row["model_id"] == target],
        lambda found: len(found) >= 1,
        seconds=ROW_SECONDS,
    )
    latest: Final = object_value(ondemand.gateway.get("/health/latest")["latest_health_checks"])
    assert object_value(latest[target])["model_name"] == PAIR_GROUP


def test_history_filters_by_model_and_lists_newest_first(ondemand: Rig) -> None:
    """`db-ok-<run>` is checked twice more and both rows are stored. The master key calls
    `GET /health/history?model=db-ok-<run>`.

    Every returned row must have `model_name` `db-ok-<run>`, `checked_at` values must be in descending order,
    `total_records` must equal the number of rows, and `limit` and `offset` must echo `100` and `0`.
    """
    before: Final = len(health_rows(ondemand.gateway, OK_GROUP))
    for _ in range(2):
        assert health(ondemand.gateway, model=OK_GROUP).status_code == 200
    eventually(
        lambda: len(health_rows(ondemand.gateway, OK_GROUP)), lambda count: count >= before + 2, seconds=ROW_SECONDS
    )
    body: Final = ondemand.gateway.get("/health/history", params={"model": OK_GROUP})
    rows: Final = [object_value(row) for row in list_value(body["health_checks"])]
    assert len(rows) >= 2 and {row["model_name"] for row in rows} == {OK_GROUP}
    stamps: Final = [string_value(row["checked_at"]) for row in rows]
    assert stamps == sorted(stamps, reverse=True)
    assert (body["total_records"], body["limit"], body["offset"]) == (len(rows), 100, 0)


def test_history_status_filter_and_limit_narrow_the_rows(ondemand: Rig) -> None:
    """With rows for `db-ok-<run>` (healthy) stored, the master key asks
    `GET /health/history?model=db-ok-<run>&status_filter=unhealthy` and then `...&limit=1`.

    The first must return no rows and `total_records: 0`; the second exactly one row.
    """
    assert health(ondemand.gateway, model=OK_GROUP).status_code == 200
    eventually(lambda: health_rows(ondemand.gateway, OK_GROUP), lambda found: len(found) >= 1, seconds=ROW_SECONDS)
    unhealthy: Final = ondemand.gateway.get("/health/history", params={"model": OK_GROUP, "status_filter": "unhealthy"})
    assert unhealthy["health_checks"] == [] and unhealthy["total_records"] == 0
    limited: Final = ondemand.gateway.get("/health/history", params={"model": OK_GROUP, "limit": "1"})
    assert len(list_value(limited["health_checks"])) == 1


def test_the_background_loop_stores_one_row_per_deployment_as_background_health_check(background: Rig) -> None:
    """The `background` proxy probes every 2 seconds. Group `db-pair-<run>` has two deployments.

    Within 90 seconds there must be rows for `db-pair-<run>` with `model_id` `hc-ok-db-pair-a-<run>` and
    `hc-ok-db-pair-b-<run>`, each `healthy` with `healthy_count` 1 and `checked_by` `background_health_check`
    (no shared health manager, so no pod id).
    """
    wanted: Final = {f"hc-{model}" for model in PAIR_MODELS}
    rows: Final = eventually(
        lambda: [
            row for row in health_rows(background.gateway, PAIR_GROUP) if row["checked_by"] == "background_health_check"
        ],
        lambda found: wanted <= {row["model_id"] for row in found},
        seconds=CONVERGENCE_SECONDS,
    )
    assert all((row["status"], row["healthy_count"]) == ("healthy", 1) for row in rows if row["model_id"] in wanted)


def test_a_stable_status_is_not_written_again_every_cycle(background: Rig) -> None:
    """`db-stable-<run>` stays healthy while the loop keeps probing it every 2 seconds.

    Once its first background row exists, the number of rows for it must stay exactly 1 for another 10 seconds
    (five more cycles): rows are written only on a status change or after an hour
    (`_should_persist_health_check_result`).
    """
    eventually(
        lambda: health_rows(background.gateway, STABLE_GROUP),
        lambda found: len(found) == 1,
        seconds=CONVERGENCE_SECONDS,
    )
    probes_before: Final = len(background.upstream.probes(STABLE_MODEL))
    held(lambda: len(health_rows(background.gateway, STABLE_GROUP)), lambda count: count == 1, holding=10, seconds=30)
    assert len(background.upstream.probes(STABLE_MODEL)) >= probes_before + 3


def test_an_outage_and_its_recovery_are_each_written_once_as_they_happen(background: Rig) -> None:
    """`db-flip-<run>` is healthy and has one stored row. The upstream is then made to answer 500 for it, and
    later restored.

    Within 30 seconds of the failure a newer row must say `unhealthy` with `unhealthy_count` 1 and an
    `error_message`; within 30 seconds of the restore a newer row must say `healthy`. The stored statuses,
    newest first, must then be exactly `healthy, unhealthy, healthy`: each transition is written immediately
    and nothing in between.
    """
    eventually(
        lambda: health_rows(background.gateway, FLIP_GROUP), lambda found: len(found) == 1, seconds=CONVERGENCE_SECONDS
    )
    background.upstream.force(FLIP_MODEL, 500)
    try:
        down: Final = eventually(
            lambda: health_rows(background.gateway, FLIP_GROUP),
            lambda found: found[0]["status"] == "unhealthy",
            seconds=ROW_SECONDS,
        )
        assert down[0]["unhealthy_count"] == 1 and down[0]["error_message"]
    finally:
        background.upstream.force(FLIP_MODEL, None)
    eventually(
        lambda: health_rows(background.gateway, FLIP_GROUP),
        lambda found: found[0]["status"] == "healthy",
        seconds=ROW_SECONDS,
    )
    statuses: Final = [row["status"] for row in health_rows(background.gateway, FLIP_GROUP)]
    assert statuses == ["healthy", "unhealthy", "healthy"]


def test_latest_reflects_a_background_outage_for_the_deployment_id(background: Rig) -> None:
    """While `db-flip-<run>`'s upstream answers 500 to probes, the master key polls `GET /health/latest`.

    Within 30 seconds the entry keyed `hc-ok-db-flip-<run>` must show `status` `unhealthy` and
    `checked_by` `background_health_check`; after the upstream is restored it must return to `healthy`.
    """
    deployment_id: Final = f"hc-{FLIP_MODEL}"

    def latest_status() -> tuple[JsonValue, JsonValue]:
        latest = object_value(background.gateway.get("/health/latest")["latest_health_checks"])
        entry = object_value(latest[deployment_id]) if deployment_id in latest else {}
        return entry.get("status"), entry.get("checked_by")

    background.upstream.force(FLIP_MODEL, 500)
    try:
        assert eventually(latest_status, lambda seen: seen[0] == "unhealthy", seconds=ROW_SECONDS) == (
            "unhealthy",
            "background_health_check",
        )
    finally:
        background.upstream.force(FLIP_MODEL, None)
    eventually(latest_status, lambda seen: seen[0] == "healthy", seconds=ROW_SECONDS)


def test_history_and_latest_answer_for_a_proxy_admin_after_rows_exist(background: Rig) -> None:
    """Sanity of the read routes on the background proxy: the master key's `GET /health/history` and
    `GET /health/latest` must both answer 200 with the documented top level keys (`health_checks`,
    `total_records`, `limit`, `offset`) and (`latest_health_checks`, `total_models`).
    """
    eventually(
        lambda: health_rows(background.gateway, STABLE_GROUP),
        lambda found: len(found) >= 1,
        seconds=CONVERGENCE_SECONDS,
    )
    history: Final = background.gateway.request("GET", "/health/history")
    latest: Final = background.gateway.request("GET", "/health/latest")
    assert history.status_code == 200 and set(report(history)) == {"health_checks", "total_records", "limit", "offset"}
    assert latest.status_code == 200 and set(report(latest)) == {"latest_health_checks", "total_models"}
