from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Final

import pytest
from _pytest.terminal import TerminalReporter

SELECTED: Final = pytest.StashKey[frozenset[str]]()
PASSED: Final = pytest.StashKey[frozenset[str]]()


def load_native() -> object:
    return import_module("litellm.rust_bridge._native")


@dataclass(frozen=True, slots=True)
class NativeSuite:
    root: Path
    native: Callable[[], object] = load_native

    def pytest_configure(self, config: pytest.Config) -> None:
        try:
            self.native()
        except ImportError as error:
            raise pytest.UsageError("Native tests require a compiled litellm.rust_bridge._native extension") from error
        config.stash[SELECTED] = frozenset()
        config.stash[PASSED] = frozenset()

    @pytest.hookimpl(wrapper=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[pytest.Item]) -> Iterator[None]:
        yield
        config.stash[SELECTED] = frozenset(
            item.nodeid
            for item in items
            if item.path.is_relative_to(self.root) and item.get_closest_marker("requires_rust_extension") is not None
        )

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item, call: pytest.CallInfo[None]
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        report: Final = yield
        if report.when == "call" and report.passed and item.nodeid in item.config.stash[SELECTED]:
            item.config.stash[PASSED] = item.config.stash[PASSED] | {item.nodeid}
        return report

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if session.config.getoption("collectonly", default=False) is True or session.exitstatus != pytest.ExitCode.OK:
            return
        if not session.config.stash[SELECTED] or not session.config.stash[PASSED]:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            reporter: Final = session.config.pluginmanager.get_plugin("terminalreporter")
            if isinstance(reporter, TerminalReporter):
                reporter.write_line("Native suite did not execute any required-native test successfully", red=True)
