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

from tests._vcr_conftest_common import (  # noqa: E402,F401
    VerboseReporterState,
    _pin_multipart_boundary,
    apply_vcr_auto_marker_to_items,
    emit_cassette_cache_session_banner,
    emit_vcr_classification_summary,
    emit_vcr_diagnostic_log,
    install_live_call_probe,
    record_vcr_outcome,
    register_persister_if_enabled,
    reset_vcr_diag_dir,
    vcr_config_dict,
)

# Vertex AI MaaS Mistral OCR tests that cannot be VCR-cached in CI.
#
# ``vertex_ai/mistral-ocr-2505`` is a Model-as-a-Service partner model that
# must be explicitly enabled in the GCP project's Model Garden. It is not
# provisioned in the CI project (``litellm-ci-cd``), so the live
# ``:rawPredict`` call fails on every run and ``BaseOCRTest`` catches the
# provider error and skips. Because the doomed live call is recorded but the
# test then skips, the persister refuses to save it (skipped tests don't
# persist) and the cassette is never seeded — so the test re-records live and
# is classified MISS:NOT_PERSISTED on every single run, forever. No cassette
# can be recorded until the model is provisioned. Mark the tests VCR-
# incompatible so they are honestly accounted as live calls (UNMARKED:LIVE_CALL)
# rather than phantom cache misses; behaviour is unchanged (they still run and
# still skip on the provider error). The sibling direct-Mistral and Azure OCR
# tests replay from cache normally and are unaffected. Remove these entries if
# the MaaS model is enabled in the CI project.
_VCR_INCOMPATIBLE_NODEID_SUFFIXES: tuple[str, ...] = (
    "test_ocr_vertex_ai.py::TestVertexAIMistralOCR::test_ocr_response_structure",
    "test_ocr_vertex_ai.py::TestVertexAIMistralOCR::test_basic_ocr_with_url[True]",
    "test_ocr_vertex_ai.py::TestVertexAIMistralOCR::test_basic_ocr_with_url[False]",
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
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


def pytest_configure(config):
    _verbose_state.remember_pluginmanager(config)
    reset_vcr_diag_dir()


def pytest_runtest_logreport(report):
    _verbose_state.maybe_emit_verdict(report)


def pytest_collection_modifyitems(config, items):
    apply_vcr_auto_marker_to_items(
        items,
        skip_nodeid_suffixes=_VCR_INCOMPATIBLE_NODEID_SUFFIXES,
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    emit_cassette_cache_session_banner(terminalreporter)
    emit_vcr_classification_summary(terminalreporter)
    emit_vcr_diagnostic_log(terminalreporter)
