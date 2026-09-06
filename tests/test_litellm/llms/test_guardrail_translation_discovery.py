import importlib.abc
import importlib.util
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import pytest

import litellm.llms as llms_package
from litellm._logging import verbose_logger
from litellm.types.utils import CallTypes

OPENAI_CHAT_TRANSLATION_MODULE = "litellm.llms.openai.chat.guardrail_translation"
MCP_TRANSLATION_MODULE = "litellm.proxy._experimental.mcp_server.guardrail_translation"


class RaisingLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves one module path, and raises the given error when Python executes it."""

    def __init__(self, module_path: str, error: BaseException) -> None:
        self.module_path = module_path
        self.error = error

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != self.module_path:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        raise self.error


@contextmanager
def raising_on_import(module_path: str, error: BaseException) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.delitem(sys.modules, module_path, raising=False)
        mp.setattr(sys, "meta_path", [RaisingLoader(module_path, error), *sys.meta_path])
        yield


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


@pytest.fixture(autouse=True)
def reset_guardrail_translation_discovery():
    saved = llms_package.guardrail_translation_discovery
    llms_package.guardrail_translation_discovery = None
    yield
    llms_package.guardrail_translation_discovery = saved


def test_discovery_reports_the_handler_package_it_could_not_import():
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        discovery = llms_package.discover_guardrail_translations()

    assert tuple(discovery.unavailable) == (OPENAI_CHAT_TRANSLATION_MODULE,)
    assert "None in sys.modules" in discovery.unavailable[OPENAI_CHAT_TRANSLATION_MODULE]
    assert CallTypes.acompletion not in discovery.mappings
    assert CallTypes.completion not in discovery.mappings
    assert CallTypes.aembedding in discovery.mappings


def test_complete_discovery_reports_nothing_unavailable():
    discovery = llms_package.discover_guardrail_translations()

    assert not discovery.unavailable
    assert CallTypes.acompletion in discovery.mappings


def test_the_next_lookup_retries_a_package_that_failed_to_import():
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        partial = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.acompletion not in partial
    assert CallTypes.aembedding in partial

    recovered = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.acompletion in recovered
    assert CallTypes.completion in recovered
    assert CallTypes.aembedding in recovered
    assert not llms_package.guardrail_translation_discovery.unavailable


def test_a_complete_discovery_is_cached():
    first = llms_package.load_guardrail_translation_mappings()
    second = llms_package.load_guardrail_translation_mappings()

    assert first is second


def test_a_package_that_keeps_failing_is_reported_once_and_its_recovery_announced(caplog):
    with capturing(caplog, logging.INFO), unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        for _ in range(3):
            llms_package.load_guardrail_translation_mappings()

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1, [record.getMessage() for record in errors]
    assert OPENAI_CHAT_TRANSLATION_MODULE in errors[0].getMessage()
    assert "None in sys.modules" in errors[0].getMessage()

    caplog.clear()
    with capturing(caplog, logging.INFO):
        llms_package.load_guardrail_translation_mappings()
        llms_package.load_guardrail_translation_mappings()

    recoveries = [record for record in caplog.records if "available again" in record.getMessage()]
    assert len(recoveries) == 1
    assert OPENAI_CHAT_TRANSLATION_MODULE in recoveries[0].getMessage()
    assert recoveries[0].levelno >= logging.WARNING
    assert not [record for record in caplog.records if record.levelno == logging.ERROR]


def test_lookup_recovers_after_a_failed_discovery():
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        with pytest.raises(ValueError, match="acompletion"):
            llms_package.get_guardrail_translation_mapping(CallTypes.acompletion)

    assert llms_package.get_guardrail_translation_mapping(CallTypes.acompletion) is not None


def test_an_mcp_package_that_fails_to_import_is_reported_and_retried():
    with raising_on_import(MCP_TRANSLATION_MODULE, AttributeError("module 'mcp.types' has no attribute 'ToolCall'")):
        partial = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.call_mcp_tool not in partial
    assert CallTypes.acompletion in partial
    assert tuple(llms_package.guardrail_translation_discovery.unavailable) == (MCP_TRANSLATION_MODULE,)
    assert "AttributeError" in llms_package.guardrail_translation_discovery.unavailable[MCP_TRANSLATION_MODULE]

    recovered = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.call_mcp_tool in recovered
    assert CallTypes.acompletion in recovered
    assert not llms_package.guardrail_translation_discovery.unavailable


def test_an_install_without_the_mcp_server_is_discovered_once_and_quietly(caplog):
    absent = ModuleNotFoundError("No module named 'mcp'", name="mcp")

    with capturing(caplog, logging.DEBUG), unimportable("mcp"), raising_on_import(MCP_TRANSLATION_MODULE, absent):
        first = llms_package.load_guardrail_translation_mappings()
        second = llms_package.load_guardrail_translation_mappings()

    assert first is second
    assert CallTypes.call_mcp_tool not in first
    assert CallTypes.acompletion in first
    assert not llms_package.guardrail_translation_discovery.unavailable
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_a_broken_mcp_dependency_is_reported_and_retried(caplog):
    """An mcp the install has but cannot import is a broken install, not a lean one, so it must be loud."""
    broken = ModuleNotFoundError("No module named 'mcp.types'", name="mcp.types")

    with capturing(caplog, logging.DEBUG), raising_on_import(MCP_TRANSLATION_MODULE, broken):
        partial = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.call_mcp_tool not in partial
    assert CallTypes.acompletion in partial
    assert tuple(llms_package.guardrail_translation_discovery.unavailable) == (MCP_TRANSLATION_MODULE,)
    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 1, [record.getMessage() for record in caplog.records]
    assert "mcp.types" in errors[0].getMessage()

    recovered = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.call_mcp_tool in recovered
    assert not llms_package.guardrail_translation_discovery.unavailable
