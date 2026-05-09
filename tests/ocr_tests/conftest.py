# conftest.py
#
# Wires OCR tests into the Redis-backed VCR cache so live provider
# calls (Mistral OCR, Azure AI OCR, Azure Document Intelligence,
# Vertex AI OCR) are replayed for 24h. See tests/llm_translation/Readme.md
# for the design overview.

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from tests._vcr_conftest_common import (  # noqa: E402
    VerboseReporterState,
    apply_vcr_auto_marker_to_items,
    record_vcr_outcome,
    register_persister_if_enabled,
    vcr_config_dict,
)

_verbose_state = VerboseReporterState()


@pytest.fixture(scope="module")
def vcr_config():
    return vcr_config_dict()


def pytest_recording_configure(config, vcr):
    register_persister_if_enabled(vcr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    yield
    record_vcr_outcome(request, vcr)


def pytest_configure(config):
    _verbose_state.remember_pluginmanager(config)


def pytest_runtest_logreport(report):
    _verbose_state.maybe_emit_verdict(report)


def pytest_collection_modifyitems(config, items):
    apply_vcr_auto_marker_to_items(items)
