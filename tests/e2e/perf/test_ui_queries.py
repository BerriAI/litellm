from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pytest
from e2e_config import CONTROL_PLANE_BASE_URL, MASTER_KEY
from e2e_http import send_text_external, without_retries
from setup.ui_queries import (
    Sample,
    UiQueryCase,
    database_failures,
    error_code,
    issue_query,
    measure,
    summarize,
    ui_junit_properties,
)
from setup.ui_query_cases import Params, cases, next_page
from setup.ui_query_fixture import UiFixture, prepare_fixture
from setup.ui_query_isolation import isolated_database, settled_counter
from transport import HttpTransport

pytestmark = [
    pytest.mark.perf,
    pytest.mark.quiet_stack,
    pytest.mark.skipif(os.environ.get("E2E_UI_PERF_DISPOSABLE") != "1", reason="Dedicated disposable database only"),
]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "ui_case_name" not in metafunc.fixturenames:
        return
    if os.environ.get("E2E_UI_PERF_DISPOSABLE") == "1" and os.environ.get("E2E_RESET_SPEND_LOGS") == "1":
        raise pytest.UsageError("UI performance tests cannot run with E2E_RESET_SPEND_LOGS=1")
    names: Final = tuple(case.name for case in cases(datetime(2000, 1, 1, tzinfo=UTC), "collection-only"))
    metafunc.parametrize("ui_case_name", names, ids=names)


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


@dataclass(frozen=True, slots=True)
class Prepared:
    case: UiQueryCase
    prerequisite: Sample | None = None


def _page_two(transport: HttpTransport, case: UiQueryCase, timeout: float) -> Prepared:
    if case.cursor == "none":
        return Prepared(case)
    params: Final = Params.model_validate(case.params.model_dump()).model_copy(
        update={"page": 1 if case.cursor == "session" else None}
    )
    first: Final = replace(case, params=params)
    sample, document = measure(lambda: issue_query(transport, first, timeout=timeout), case.contract)
    if sample.error or document is None:
        return Prepared(case, sample)
    try:
        return Prepared(next_page(case, document), sample)
    except AssertionError as error:
        return Prepared(case, sample.model_copy(update={"error": str(error)}))


def _samples(transport: HttpTransport, case: UiQueryCase, timeout: float, repeats: int) -> tuple[Sample, ...]:
    measured, _ = measure(lambda: issue_query(transport, case, timeout=timeout), case.contract)
    if measured.error:
        return (measured,)
    repeated: Final = tuple(
        measure(lambda: issue_query(transport, case, timeout=timeout), case.contract)[0] for _ in range(repeats)
    )
    return (measured, *repeated)


def test_ui_query_performance(ui_case_name: str, ui_fixture: UiFixture, request: pytest.FixtureRequest) -> None:
    case: Final = next(case for case in cases(ui_fixture.anchor, ui_fixture.api_key) if case.name == ui_case_name)
    transport: Final = HttpTransport(base_url=CONTROL_PLANE_BASE_URL, master_key=MASTER_KEY)
    timeout: Final = float(os.environ.get("E2E_UI_QUERY_TIMEOUT_SECONDS", "120"))
    repeats: Final = int(os.environ.get("E2E_UI_QUERY_REPEATS", "5"))
    assert 0 < timeout <= 600 and 1 <= repeats <= 1000, "Invalid UI performance sample settings"
    target: Final = Path(os.environ["E2E_ARTIFACT_DIR"])
    assert target.is_dir(), "E2E_ARTIFACT_DIR must exist"
    evidence: Final = target / f"ui-isolation-{case.name}.jsonl"
    with without_retries():
        with isolated_database(case.name):
            before: Final = settled_counter(lambda: _metrics(transport), evidence)
            paginated: Final = _page_two(transport, case, timeout)
            samples: Final = (
                (paginated.prerequisite,)
                if paginated.prerequisite is not None and paginated.prerequisite.error
                else _samples(transport, paginated.case, timeout, repeats)
            )
            with evidence.open("a", encoding="utf-8") as sample_output:
                _ = sample_output.write(
                    json.dumps(
                        {
                            "samples": tuple(sample.model_dump() for sample in samples),
                            "prerequisite": paginated.prerequisite.model_dump()
                            if paginated.prerequisite is not None
                            else None,
                        }
                    )
                    + "\n"
                )
            after_requests: Final = _metrics(transport)
            item: Final = request.node  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # pytest: untyped
            assert isinstance(item, pytest.Item)
            item.user_properties.extend(ui_junit_properties(samples, paginated.prerequisite))
        after: Final = settled_counter(lambda: _metrics(transport), evidence)
        item.user_properties.extend(
            (
                ("perf_ui_database_failures", after_requests - before),
                ("perf_ui_cleanup_database_failures", after - after_requests),
            )
        )
        timings: Final = summarize(samples)
        artifact: Final = {
            "name": case.name,
            "surface": case.surface,
            "path": case.path,
            "method": "POST" if case.body is not None else "GET",
            "params": paginated.case.params.model_dump(mode="json", by_alias=True, exclude_none=True),
            "prerequisite": paginated.prerequisite.model_dump() if paginated.prerequisite is not None else None,
            "prerequisite_failed": paginated.prerequisite is not None and bool(paginated.prerequisite.error),
            "body": case.body.model_dump(mode="json", exclude_none=True) if case.body is not None else None,
            "samples": [sample.model_dump() for sample in samples],
            "error_codes": [error_code(sample) for sample in samples],
            "timings": timings.model_dump(),
            "timeouts": sum(sample.error == "timeout" for sample in samples),
            "failures": sum(bool(sample.error) for sample in samples),
            "status_counts": {
                str(status): sum(sample.status == status for sample in samples)
                for status in frozenset(sample.status for sample in samples)
            },
            "validation_failures": sum(
                bool(sample.error) and (sample.status == 200 or sample.error == "Response is not valid JSON")
                for sample in samples
            ),
            "database_failures_before": before,
            "database_failures_after_requests": after_requests,
            "database_failures_after_cleanup": after,
        }
        output: Final = target / f"ui-query-{case.name}.json"
        _ = output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
