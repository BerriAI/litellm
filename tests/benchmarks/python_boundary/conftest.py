from pathlib import Path
from typing import Final, cast

import pytest
from budgets import CodSpeedResults, from_codspeed
from pytest_codspeed.plugin import get_plugin


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--boundary-report", type=Path, default=None)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    destination: Final = cast(Path | None, session.config.getoption("--boundary-report"))
    if destination is None:
        return
    plugin: Final = get_plugin(session.config)
    if not plugin.is_codspeed_enabled:
        raise ValueError("Boundary reports require CodSpeed measurements")
    results: Final = CodSpeedResults.model_validate(plugin.instrument.get_result_dict())
    report: Final = from_codspeed(results, Path(__file__).resolve().parents[3])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.model_dump_json(indent=2) + "\n")
