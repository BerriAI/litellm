import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from tests._vcr_conftest_common import (  # noqa: E402
    VerboseReporterState,
    apply_vcr_auto_marker_to_items,
    emit_cassette_cache_session_banner,
    emit_vcr_classification_summary,
    install_live_call_probe,
    record_vcr_outcome,
    register_persister_if_enabled,
    vcr_config_dict,
)

# Tests that observe live cross-call provider state — typically a
# warm-up call followed by an assertion that the *second* call sees the
# upstream's prompt-cache (Anthropic / Bedrock prompt-caching). VCR's
# deterministic replay can't model this: both calls match the same
# cassette episode, so the second call returns the first call's
# pre-warmup response. Opt these out so they run live (no caching).
_VCR_INCOMPATIBLE_NODEID_SUFFIXES = (
    "::test_prompt_caching_returns_cache_read_tokens_on_second_call",
    "::test_prompt_caching_streaming_second_call_returns_cache_read",
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


def pytest_runtest_logreport(report):
    _verbose_state.maybe_emit_verdict(report)


def pytest_collection_modifyitems(config, items):
    apply_vcr_auto_marker_to_items(
        items, skip_nodeid_suffixes=_VCR_INCOMPATIBLE_NODEID_SUFFIXES
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    emit_cassette_cache_session_banner(terminalreporter)
    emit_vcr_classification_summary(terminalreporter)
