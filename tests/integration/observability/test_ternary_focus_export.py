"""Ternary FOCUS export: one whole UTC day per signed-URL PUT, whatever the database session timezone."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from _ternary_support import (
    SIGNED_HEADERS,
    TERNARY_API_KEY,
    UPLOAD_URL_PATH,
    SpendRow,
    TernaryPeers,
    callbacks_config,
    refreshing_updated_at,
    scrubbed_environment,
    seed_daily_spend,
    set_database_timezone,
    signed_target,
    ternary_peers,
    utc_today,
)
from integration._support.client import Gateway, eventually
from integration._support.database import scratch_database
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

_INTERVAL_EXPORT: Final = MappingProxyType(
    {"TERNARY_EXPORT_FREQUENCY": "interval", "TERNARY_EXPORT_INTERVAL_SECONDS": "1"}
)
_PROVIDER_KEY: Final = "synthetic-provider-key"
_FOCUS_REFUSAL: Final = "Provider 'ternary' not supported for Focus export"


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(), tzinfo=timezone.utc)


def _upstream(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


@contextmanager
def _ternary_proxy(
    gateway: Gateway,
    directory: Path,
    peers: TernaryPeers,
    *,
    callbacks: tuple[str, ...],
    database_url: str | None = None,
    extra: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[OwnedProxy]:
    overrides: Final = {
        **peers.environment,
        **({"DATABASE_URL": database_url} if database_url else {}),
        **extra,
    }
    with owned_proxy_process(
        gateway,
        directory,
        overrides,
        config=callbacks_config(directory, callbacks),
        remove_environment=scrubbed_environment(os.environ),
    ) as owned:
        yield owned


def test_ternary_export_uploads_one_csv_per_utc_day_with_the_signed_headers(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex[:10]
    today: Final = utc_today()
    midnight: Final = _midnight(today)
    seeds: Final = (
        SpendRow(today - timedelta(days=3), f"before-window-{marker}", 0.5, midnight - timedelta(days=3, hours=-12)),
        SpendRow(today - timedelta(days=2), f"oldest-{marker}", 0.0123, midnight - timedelta(days=2, hours=-12)),
        SpendRow(today - timedelta(days=2), f"late-flush-{marker}", 0.0045, midnight - timedelta(hours=23, minutes=30)),
        SpendRow(today - timedelta(days=1), f"yesterday-{marker}", 0.0067, midnight - timedelta(hours=12)),
        SpendRow(today, f"today-{marker}", 0.0089, datetime.now(timezone.utc) - timedelta(minutes=1)),
    )

    with (
        scratch_database("ternary_happy") as database_url,
        ternary_peers(tmp_path) as peers,
        wire_server(_upstream) as provider,
        _ternary_proxy(
            gateway, tmp_path, peers, callbacks=("ternary",), database_url=database_url, extra=_INTERVAL_EXPORT
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            api_base=provider.url + "/v1",
            api_key=_PROVIDER_KEY,
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        key: Final = scenario.key(models=[model])
        response: Final = owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
            key=key,
        )
        assert response.status_code == 200, response.text
        seed_daily_spend(database_url, seeds)
        window: Final = tuple(str(today - timedelta(days=offset)) for offset in (2, 1, 0))
        by_day: Final = eventually(
            peers.latest_by_day,
            lambda latest: set(window) <= set(latest) and model in latest[str(today)].cost_by_service(),
            seconds=90,
        )
        requests: Final = peers.upload_url_requests()
        uploads: Final = peers.uploads()

    assert {upload.day for upload in uploads} == set(window), "a day outside the two-day window was uploaded"
    assert dict(by_day[window[0]].cost_by_service()) == pytest.approx(
        {f"oldest-{marker}": 0.0123, f"late-flush-{marker}": 0.0045}
    )
    assert dict(by_day[window[1]].cost_by_service()) == pytest.approx({f"yesterday-{marker}": 0.0067})
    assert dict(by_day[window[2]].cost_by_service()) == pytest.approx({f"today-{marker}": 0.0089, model: 0.019})
    for day, upload in by_day.items():
        assert {row["ChargePeriodStart"][:10] for row in upload.rows()} == {day}
    oldest: Final = next(row for row in by_day[window[0]].rows() if row["ServiceName"] == f"oldest-{marker}")
    assert json.loads(oldest["Tags"])["prompt_tokens"] == "7"
    assert json.loads(oldest["Tags"])["completion_tokens"] == "3"
    for request in requests:
        assert (request.method, request.target) == ("POST", UPLOAD_URL_PATH)
        assert request.headers["authorization"] == f"Bearer {TERNARY_API_KEY}"
        assert json.loads(request.body)["day"] in window
    for put in peers.puts():
        assert put.method == "PUT"
        assert "authorization" not in put.headers
        assert {name.lower(): value for name, value in SIGNED_HEADERS.items()}.items() <= put.headers.items()
        assert put.target == signed_target(put.target.rsplit("/", 1)[-1].split(".csv", 1)[0])
        assert TERNARY_API_KEY.encode() not in put.body


@pytest.mark.parametrize(
    ("zone", "label"),
    [("Asia/Tokyo", "east"), ("Pacific/Kiritimati", "east-most")],
)
def test_ternary_export_keeps_the_whole_oldest_day_when_the_database_session_is_east_of_utc(
    gateway: Gateway, tmp_path: Path, zone: str, label: str
) -> None:
    marker: Final = uuid.uuid4().hex[:10]
    today: Final = utc_today()
    oldest: Final = today - timedelta(days=2)
    seeds: Final = (
        SpendRow(oldest - timedelta(days=1), f"before-window-{marker}", 0.5, _midnight(oldest) - timedelta(hours=12)),
        SpendRow(oldest, f"early-{label}-{marker}", 0.0021, _midnight(oldest) + timedelta(hours=3)),
        SpendRow(oldest, f"midday-{label}-{marker}", 0.0034, _midnight(oldest) + timedelta(hours=15)),
    )

    with scratch_database("ternary_east") as database_url:
        set_database_timezone(database_url, zone)
        with (
            ternary_peers(tmp_path) as peers,
            _ternary_proxy(
                gateway, tmp_path, peers, callbacks=("ternary",), database_url=database_url, extra=_INTERVAL_EXPORT
            ),
        ):
            seed_daily_spend(database_url, seeds)
            by_day: Final = eventually(peers.latest_by_day, lambda latest: str(oldest) in latest, seconds=90)
            uploads: Final = peers.uploads()

    assert str(oldest - timedelta(days=1)) not in {upload.day for upload in uploads}
    assert dict(by_day[str(oldest)].cost_by_service()) == pytest.approx(
        {f"early-{label}-{marker}": 0.0021, f"midday-{label}-{marker}": 0.0034}
    ), f"the {zone} session dropped spend updated in the first hours of the oldest UTC day"


def test_ternary_export_includes_spend_updated_in_the_last_hours_when_the_database_session_is_west_of_utc(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = uuid.uuid4().hex[:10]
    today: Final = utc_today()
    yesterday: Final = today - timedelta(days=1)
    seeds: Final = (
        SpendRow(yesterday, f"midday-{marker}", 0.0056, _midnight(yesterday) + timedelta(hours=12)),
        SpendRow(yesterday, f"just-flushed-{marker}", 0.0078, datetime.now(timezone.utc) - timedelta(minutes=30)),
    )

    with scratch_database("ternary_west") as database_url:
        set_database_timezone(database_url, "America/Los_Angeles")
        with (
            ternary_peers(tmp_path) as peers,
            _ternary_proxy(
                gateway, tmp_path, peers, callbacks=("ternary",), database_url=database_url, extra=_INTERVAL_EXPORT
            ),
        ):
            seed_daily_spend(database_url, seeds)
            by_day: Final = eventually(peers.latest_by_day, lambda latest: str(yesterday) in latest, seconds=90)

    assert dict(by_day[str(yesterday)].cost_by_service()) == pytest.approx(
        {f"midday-{marker}": 0.0056, f"just-flushed-{marker}": 0.0078}
    ), "the America/Los_Angeles session dropped spend flushed in the last hours"


@pytest.mark.parametrize("export_format", ["parquet", "csv"])
def test_the_generic_focus_logger_does_not_upload_to_ternary(gateway: Gateway, tmp_path: Path, export_format: str) -> None:
    service: Final = f"generic-focus-{uuid.uuid4().hex[:10]}"
    today: Final = utc_today()

    with (
        scratch_database("ternary_generic") as database_url,
        ternary_peers(tmp_path) as peers,
        _ternary_proxy(
            gateway,
            tmp_path,
            peers,
            callbacks=("focus",),
            database_url=database_url,
            extra={
                "FOCUS_PROVIDER": "ternary",
                "FOCUS_FORMAT": export_format,
                "FOCUS_FREQUENCY": "interval",
                "FOCUS_INTERVAL_SECONDS": "1",
            },
        ) as owned,
    ):
        seed_daily_spend(database_url, (SpendRow(today, service, 0.0042, datetime.now(timezone.utc)),))
        with refreshing_updated_at(database_url, service):
            outbound, refused = eventually(
                lambda: (peers.outbound(), _FOCUS_REFUSAL in owned.log.read_text(errors="replace")),
                lambda observed: observed[0] > 0 or observed[1],
                seconds=60,
            )

    assert outbound == 0, f"callbacks: ['focus'] with FOCUS_PROVIDER=ternary uploaded {export_format} to Ternary"
    assert refused


@pytest.mark.parametrize("callbacks", [(), ("ternary",)], ids=["no-focus-callback", "ternary-callback"])
def test_a_proxy_installed_without_polars_boots_and_serves(
    gateway: Gateway, tmp_path: Path, callbacks: tuple[str, ...]
) -> None:
    shadow: Final = tmp_path / "without-polars"
    (shadow / "polars").mkdir(parents=True)
    (shadow / "polars" / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'polars'\", name='polars')\n"
    )
    python_path: Final = os.pathsep.join(filter(None, (str(shadow), os.environ.get("PYTHONPATH", ""))))
    probe: Final = subprocess.run(  # noqa: S603  # fixed argv: proves the shadow hides polars from a child
        (sys.executable, "-P", "-c", "import polars"),
        env={**os.environ, "PYTHONPATH": python_path},
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode != 0 and "No module named 'polars'" in probe.stderr, probe.stderr

    with (
        ternary_peers(tmp_path) as peers,
        wire_server(_upstream) as provider,
        _ternary_proxy(
            gateway, tmp_path, peers, callbacks=callbacks, extra={"PYTHONPATH": python_path, **_INTERVAL_EXPORT}
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=provider.url + "/v1", api_key=_PROVIDER_KEY)
        key: Final = scenario.key(models=[model])
        response: Final = owned.gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}]}, key=key
        )
        if callbacks:
            eventually(
                lambda: "No module named 'polars'" in owned.log.read_text(errors="replace"),
                lambda failed: failed,
                seconds=60,
            )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"
