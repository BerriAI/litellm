import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import Final

import pytest

import litellm.llms as llms_package
from litellm._logging import verbose_logger
from litellm.proxy.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.types.utils import CallTypes

OPENAI_CHAT_TRANSLATION_MODULE: Final = "litellm.llms.openai.chat.guardrail_translation"
MCP_TRANSLATION_MODULE: Final = "litellm.proxy.mcp_server.guardrail_translation"
RETRY_SECONDS: Final = llms_package.GUARDRAIL_TRANSLATION_RETRY_SECONDS


def at(seconds: float) -> Callable[[], float]:
    return lambda: seconds


class RaisingLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, module_path: str, error: BaseException) -> None:
        self.module_path = module_path
        self.error = error
        self.attempts = 0

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.module_path:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.attempts += 1
        raise self.error


@contextmanager
def raising_on_import(module_path: str, error: BaseException) -> Iterator[RaisingLoader]:
    loader: Final = RaisingLoader(module_path, error)
    with pytest.MonkeyPatch.context() as mp:
        mp.delitem(sys.modules, module_path, raising=False)
        mp.setattr(sys, "meta_path", [loader, *sys.meta_path])
        yield loader


class DiscoveringDuringItsOwnImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, module_path: str) -> None:
        self.module_path = module_path
        self.seen_mid_import: llms_package.GuardrailTranslationDiscovery | None = None

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.module_path:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.seen_mid_import = llms_package._discover((self.module_path,), already_found={})
        module.guardrail_translation_mappings = {CallTypes.call_mcp_tool: _StandInMCPHandler}


@contextmanager
def unimportable(module_path: str) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, module_path, None)
        yield


@contextmanager
def capturing(caplog: pytest.LogCaptureFixture, level: int) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(verbose_logger, "propagate", True)
        caplog.set_level(level, logger=verbose_logger.name)
        yield


def errors_in(caplog: pytest.LogCaptureFixture) -> tuple[str, ...]:
    return tuple(record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR)


@pytest.fixture(autouse=True)
def reset_guardrail_translation_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llms_package, "guardrail_translation_discovery", None)


def test_discovery_reports_the_handler_package_it_could_not_import() -> None:
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        discovery: Final = llms_package.discover_guardrail_translations()

    assert tuple(discovery.unavailable) == (OPENAI_CHAT_TRANSLATION_MODULE,)
    assert "None in sys.modules" in discovery.unavailable[OPENAI_CHAT_TRANSLATION_MODULE]
    assert CallTypes.acompletion not in discovery.mappings
    assert CallTypes.completion not in discovery.mappings
    assert CallTypes.aembedding in discovery.mappings


def test_complete_discovery_reports_nothing_unavailable() -> None:
    discovery: Final = llms_package.discover_guardrail_translations()

    assert not discovery.unavailable
    assert CallTypes.acompletion in discovery.mappings


def test_a_lookup_after_the_retry_interval_retries_a_package_that_failed_to_import() -> None:
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        partial: Final = llms_package.load_guardrail_translation_mappings(now=at(0.0))

    assert CallTypes.acompletion not in partial
    assert CallTypes.aembedding in partial

    recovered: Final = llms_package.load_guardrail_translation_mappings(now=at(RETRY_SECONDS))

    assert CallTypes.acompletion in recovered
    assert CallTypes.completion in recovered
    assert CallTypes.aembedding in recovered


def test_a_package_that_fails_to_import_is_retried_at_most_once_per_interval() -> None:
    with raising_on_import(OPENAI_CHAT_TRANSLATION_MODULE, ImportError("simulated broken install")) as loader:
        for seconds in (0.0, RETRY_SECONDS / 2, RETRY_SECONDS * 0.99):
            assert CallTypes.acompletion not in llms_package.load_guardrail_translation_mappings(now=at(seconds))
        assert loader.attempts == 1

        llms_package.load_guardrail_translation_mappings(now=at(RETRY_SECONDS))
        assert loader.attempts == 2

        llms_package.load_guardrail_translation_mappings(now=at(RETRY_SECONDS * 1.5))
        assert loader.attempts == 2

        llms_package.load_guardrail_translation_mappings(now=at(RETRY_SECONDS * 2))
        assert loader.attempts == 3


def test_a_complete_discovery_is_not_scanned_again() -> None:
    healthy: Final = llms_package.load_guardrail_translation_mappings(now=at(0.0))

    assert CallTypes.acompletion in healthy

    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        after_the_package_breaks: Final = llms_package.load_guardrail_translation_mappings(
            now=at(RETRY_SECONDS * 10)
        )

    assert CallTypes.acompletion in after_the_package_breaks


def test_a_package_that_keeps_failing_is_reported_once_and_its_recovery_announced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with capturing(caplog, logging.INFO), unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        for attempt in range(3):
            llms_package.load_guardrail_translation_mappings(now=at(attempt * RETRY_SECONDS))

    errors: Final = errors_in(caplog)
    assert len(errors) == 1, errors
    assert OPENAI_CHAT_TRANSLATION_MODULE in errors[0]
    assert "None in sys.modules" in errors[0]

    caplog.clear()
    with capturing(caplog, logging.INFO):
        llms_package.load_guardrail_translation_mappings(now=at(3 * RETRY_SECONDS))
        llms_package.load_guardrail_translation_mappings(now=at(4 * RETRY_SECONDS))

    recoveries: Final = [record for record in caplog.records if "available again" in record.getMessage()]
    assert len(recoveries) == 1
    assert OPENAI_CHAT_TRANSLATION_MODULE in recoveries[0].getMessage()
    assert recoveries[0].levelno >= logging.WARNING
    assert not errors_in(caplog)


def test_lookup_recovers_after_a_failed_discovery() -> None:
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        with pytest.raises(ValueError, match="acompletion"):
            llms_package.get_guardrail_translation_mapping(CallTypes.acompletion, now=at(0.0))

    assert llms_package.get_guardrail_translation_mapping(CallTypes.acompletion, now=at(RETRY_SECONDS)) is not None


def test_an_mcp_package_that_fails_to_import_is_reported_and_retried(caplog: pytest.LogCaptureFixture) -> None:
    broken: Final = AttributeError("module 'mcp.types' has no attribute 'ToolCall'")

    with capturing(caplog, logging.DEBUG), raising_on_import(MCP_TRANSLATION_MODULE, broken):
        partial: Final = llms_package.load_guardrail_translation_mappings(now=at(0.0))

    assert CallTypes.call_mcp_tool not in partial
    assert CallTypes.acompletion in partial
    errors: Final = errors_in(caplog)
    assert len(errors) == 1, errors
    assert MCP_TRANSLATION_MODULE in errors[0]
    assert "AttributeError" in errors[0]

    recovered: Final = llms_package.load_guardrail_translation_mappings(now=at(RETRY_SECONDS))

    assert CallTypes.call_mcp_tool in recovered
    assert CallTypes.acompletion in recovered


def test_an_install_without_the_mcp_server_is_discovered_once_and_quietly(caplog: pytest.LogCaptureFixture) -> None:
    absent: Final = ModuleNotFoundError("No module named 'mcp'", name="mcp")

    with capturing(caplog, logging.DEBUG), unimportable("mcp"), raising_on_import(MCP_TRANSLATION_MODULE, absent):
        first: Final = llms_package.load_guardrail_translation_mappings(now=at(0.0))
        much_later: Final = llms_package.load_guardrail_translation_mappings(now=at(RETRY_SECONDS * 10))

    assert first is much_later
    assert CallTypes.call_mcp_tool not in first
    assert CallTypes.acompletion in first
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_a_broken_mcp_dependency_is_reported_and_retried(caplog: pytest.LogCaptureFixture) -> None:
    broken: Final = ModuleNotFoundError("No module named 'mcp.types'", name="mcp.types")

    with capturing(caplog, logging.DEBUG), raising_on_import(MCP_TRANSLATION_MODULE, broken):
        partial: Final = llms_package.load_guardrail_translation_mappings(now=at(0.0))

    assert CallTypes.call_mcp_tool not in partial
    assert CallTypes.acompletion in partial
    errors: Final = errors_in(caplog)
    assert len(errors) == 1, errors
    assert "mcp.types" in errors[0]

    recovered: Final = llms_package.load_guardrail_translation_mappings(now=at(RETRY_SECONDS))

    assert CallTypes.call_mcp_tool in recovered


class _StandInMCPHandler:
    pass


def test_a_package_caught_mid_import_is_retried_instead_of_cached_as_complete() -> None:
    module_path: Final = "mid_import_guardrail_translation_stub"
    loader: Final = DiscoveringDuringItsOwnImport(module_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "meta_path", [loader, *sys.meta_path])
        importlib.import_module(module_path)
        mid_import: Final = loader.seen_mid_import
        assert mid_import is not None
        retried: Final = llms_package._discover(mid_import.unavailable, already_found=mid_import.mappings)
        mp.delitem(sys.modules, module_path)

    assert tuple(mid_import.unavailable) == (module_path,)
    assert CallTypes.call_mcp_tool not in mid_import.mappings
    assert not retried.unavailable
    assert retried.mappings[CallTypes.call_mcp_tool] is _StandInMCPHandler


def test_the_mcp_package_wins_a_handler_collision_with_a_bundled_package(monkeypatch: pytest.MonkeyPatch) -> None:
    colliding: Final = ModuleType("litellm.llms.colliding_stub.guardrail_translation")
    colliding.guardrail_translation_mappings = {CallTypes.call_mcp_tool: _StandInMCPHandler}
    monkeypatch.setitem(sys.modules, colliding.__name__, colliding)

    discovery: Final = llms_package._discover((colliding.__name__, MCP_TRANSLATION_MODULE), already_found={})

    assert discovery.mappings[CallTypes.call_mcp_tool] is MCPGuardrailTranslationHandler
