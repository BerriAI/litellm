from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Final

import pytest
from e2e_config import MASTER_KEY, UI_BASE_URL, UI_PASSWORD, UI_USERNAME
from e2e_http import send_text_external, without_retries
from playwright.sync_api import Page
from setup.ui_journeys import (
    Dashboard,
    Journey,
    Sample,
    assert_within_ceiling,
    dashboard_page,
    database_failures,
    error_code,
    journeys,
    measure,
    summarize,
    ui_junit_properties,
)
from setup.ui_query_fixture import UiFixture, prepare_fixture
from setup.ui_query_isolation import isolated_database, settled_counter
from transport import HttpTransport

pytestmark = [
    pytest.mark.perf,
    pytest.mark.quiet_stack,
    pytest.mark.skipif(os.environ.get("E2E_UI_PERF_DISPOSABLE") != "1", reason="Dedicated disposable database only"),
]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "journey_name" in metafunc.fixturenames:
        metafunc.parametrize("journey_name", [journey.name for journey in journeys("")])


@pytest.fixture(scope="module")
def ui_fixture() -> UiFixture:
    return prepare_fixture()


def _metrics(transport: HttpTransport) -> float:
    urls: Final = tuple(url.strip() for url in os.environ["E2E_DATABASE_METRICS_URLS"].split(",") if url.strip())
    try:
        assert urls, "Provide a metrics URL for every proxy worker/pod"
        readings: Final = tuple(send_text_external("GET", url, headers=transport.master, timeout=10) for url in urls)
        assert all(reading.status_code == 200 for reading in readings), "Cannot read database failure metrics"
        return sum(database_failures(reading.body) for reading in readings)
    except Exception as error:
        pytest.exit(f"Database metrics unavailable ({type(error).__name__}). Stopping UI measurements", returncode=2)


def _samples(dashboard: Dashboard, journey: Journey, timeout: float, repeats: int) -> tuple[Sample, ...]:
    with dashboard_page(dashboard, timeout) as page:
        return tuple(_within_budget(page, journey, timeout, repeats))


def _within_budget(page: Page, journey: Journey, timeout: float, repeats: int) -> Generator[Sample]:
    """One hanging browser call must not eat the lane: SIGALRM ends the journey at timeout x (repeats + 2),
    the budget every sample needs (start plus act plus settle) with one spare sample of slack."""
    budget: Final = timeout * (repeats + 2)

    def expire(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"{journey.name} exceeded its {budget:.0f}s wall clock")

    previous: Final = signal.signal(signal.SIGALRM, expire)
    started: Final = time.monotonic()
    try:
        _ = signal.setitimer(signal.ITIMER_REAL, budget)
        for _ in range(repeats + 1):
            yield measure(page, journey, timeout)
    except Exception as error:
        if time.monotonic() - started < budget:
            raise
        yield Sample(seconds=time.monotonic() - started, error=f"wall_clock {type(error).__name__}: {error}"[:300])
    finally:
        _ = signal.setitimer(signal.ITIMER_REAL, 0)
        _ = signal.signal(signal.SIGALRM, previous)


def test_ui_journey_latency(journey_name: str, ui_fixture: UiFixture, request: pytest.FixtureRequest) -> None:
    journey: Final = next(journey for journey in journeys(ui_fixture.api_key) if journey.name == journey_name)
    transport: Final = HttpTransport(base_url=UI_BASE_URL, master_key=MASTER_KEY)
    timeout: Final = float(os.environ.get("E2E_UI_JOURNEY_TIMEOUT_SECONDS", "120"))
    repeats: Final = int(os.environ.get("E2E_UI_JOURNEY_REPEATS", "9"))
    assert 0 < timeout <= 600 and 1 <= repeats <= 1000, "Invalid UI performance sample settings"
    target: Final = Path(os.environ["E2E_ARTIFACT_DIR"])
    assert target.is_dir(), "E2E_ARTIFACT_DIR must exist"
    evidence: Final = target / f"ui-isolation-{journey.name}.jsonl"
    dashboard: Final = Dashboard(
        base_url=UI_BASE_URL, username=UI_USERNAME, password=UI_PASSWORD, anchor=ui_fixture.anchor
    )
    with without_retries():
        with isolated_database(journey.name):
            before: Final = settled_counter(lambda: _metrics(transport), evidence)
            samples: Final = _samples(dashboard, journey, timeout, repeats)
            with evidence.open("a", encoding="utf-8") as sample_output:
                _ = sample_output.write(
                    json.dumps({"samples": tuple(sample.model_dump() for sample in samples)}) + "\n"
                )
            after_requests: Final = _metrics(transport)
            item: Final = request.node  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # pytest: untyped
            assert isinstance(item, pytest.Item)
            item.user_properties.extend(ui_junit_properties(samples))
            item.user_properties.extend(ui_fixture.volume)
        after: Final = settled_counter(lambda: _metrics(transport), evidence)
        item.user_properties.extend(
            (
                ("perf_ui_database_failures", after_requests - before),
                ("perf_ui_cleanup_database_failures", after - after_requests),
            )
        )
        artifact: Final = {
            "name": journey.name,
            "surface": journey.surface,
            "measured_at": datetime.now(UTC).isoformat(),
            "samples": [sample.model_dump() for sample in samples],
            "error_codes": [error_code(sample) for sample in samples],
            "timings": summarize(samples).model_dump(),
            "timeouts": sum(sample.error == "timeout" for sample in samples),
            "failures": sum(bool(sample.error) for sample in samples),
            "database_failures_before": before,
            "database_failures_after_requests": after_requests,
            "database_failures_after_cleanup": after,
        }
        output: Final = target / f"ui-journey-{journey.name}.json"
        _ = output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        assert_within_ceiling(journey.name, samples)
