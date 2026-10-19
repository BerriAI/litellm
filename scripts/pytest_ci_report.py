from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import pytest


class CIMetric(Protocol):
    def __call__(self, name: str, value: str | float, unit: str = "") -> None: ...


@dataclass(frozen=True, slots=True)
class Outcome:
    nodeid: str
    when: str
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class Metric:
    nodeid: str
    name: str
    value: str | float
    unit: str


@dataclass(frozen=True, slots=True)
class Report:
    outcomes: tuple[Outcome, ...] = ()
    metrics: tuple[Metric, ...] = ()
    version: int = 1


REPORT: Final = pytest.StashKey[Report]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--ci-report", default=None, help="Write non-gating CI outcomes and metrics as JSON")


def pytest_configure(config: pytest.Config) -> None:
    path: Final[object] = config.getoption("ci_report")
    if not isinstance(path, str) or hasattr(config, "workerinput"):
        return
    config.stash[REPORT] = Report()
    config.pluginmanager.register(CIReporter(config, Path(path)), "ci-report-writer")


def _reason(report: pytest.TestReport) -> str:
    xfail: Final[object] = getattr(report, "wasxfail", None)
    if isinstance(xfail, str):
        return xfail
    if report.skipped and isinstance(report.longrepr, tuple):
        return str(report.longrepr[2])
    return ""


def _metric(nodeid: str, prop: object) -> Metric | None:
    if not isinstance(prop, (tuple, list)) or len(prop) != 2 or prop[0] != "ci_metric":
        return None
    value: Final[object] = prop[1]
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    name, amount, unit = cast(tuple[object, object, object], value)
    if isinstance(name, str) and isinstance(amount, (str, int, float)) and isinstance(unit, str):
        return Metric(nodeid, name, amount, unit)
    return None


@dataclass(frozen=True, slots=True)
class CIReporter:
    config: pytest.Config
    path: Path

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        status: Final = cast(
            tuple[str, str, object],
            self.config.hook.pytest_report_teststatus(
                report=report,
                config=self.config,
            ),
        )
        category: Final = status[0]
        previous: Final = self.config.stash[REPORT]
        outcomes: Final = previous.outcomes + (
            (Outcome(report.nodeid, report.when, category, _reason(report)),) if category not in ("", "rerun") else ()
        )
        metrics: Final = (
            tuple(metric for metric in previous.metrics if metric.nodeid != report.nodeid)
            + tuple(metric for prop in report.user_properties if (metric := _metric(report.nodeid, prop)) is not None)
            if report.when == "teardown"
            else previous.metrics
        )
        self.config.stash[REPORT] = Report(outcomes, metrics)

    def pytest_sessionfinish(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _ = self.path.write_text(json.dumps(asdict(self.config.stash[REPORT])), encoding="utf-8")
        except (OSError, ValueError) as error:
            sys.stderr.write(f"Could not write CI report: {error}\n")


@pytest.fixture
def ci_metric(request: pytest.FixtureRequest) -> CIMetric:
    def record(name: str, value: str | float, unit: str = "") -> None:
        if request.config.getoption("ci_report") is not None:
            request.node.user_properties.append(("ci_metric", (name, value, unit)))

    return record
