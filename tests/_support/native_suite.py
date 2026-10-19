from collections.abc import Callable, Generator
from importlib import import_module
from pathlib import Path
from typing import Final

import pytest
from _pytest.terminal import TerminalReporter

NATIVE_PASSED: Final = "litellm_native_suite_passed"


def load_native() -> object:
    return import_module("litellm.rust_bridge._native")


class NativeSuite:
    def __init__(self, root: Path, native: Callable[[], object] = load_native) -> None:
        self.root: Final = root
        self.native: Final = native
        self.passed: bool = False

    def pytest_configure(self, config: pytest.Config) -> None:
        try:
            self.native()
        except ImportError as error:
            raise pytest.UsageError("Native tests require a compiled litellm.rust_bridge._native extension") from error

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item, call: pytest.CallInfo[None]
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        report: Final = yield
        if (
            report.when == "call"
            and report.passed
            and item.path.is_relative_to(self.root)
            and item.get_closest_marker("requires_rust_extension") is not None
        ):
            report.user_properties.append((NATIVE_PASSED, True))
        return report

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if (
            report.when == "call"
            and report.passed
            and any(name == NATIVE_PASSED and value is True for name, value in report.user_properties)
        ):
            self.passed = True

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if (
            hasattr(session.config, "workerinput")
            or session.config.getoption("collectonly", default=False) is True
            or session.exitstatus != pytest.ExitCode.OK
        ):
            return
        if not self.passed:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            reporter: Final = session.config.pluginmanager.get_plugin("terminalreporter")
            if isinstance(reporter, TerminalReporter):
                reporter.write_line("Native suite did not execute any required-native test successfully", red=True)
