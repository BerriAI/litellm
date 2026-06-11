# conftest.py

import asyncio
import importlib
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm  # noqa: E402,F401

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

_verbose_state = VerboseReporterState()


# Files where VCR replay breaks the test:
# - ``test_litellm_overhead.py``: asserts overhead/total < 40%, which
#   inverts when cached replay collapses the upstream time to microseconds.
_VCR_INCOMPATIBLE_FILES = frozenset(
    {
        "test_litellm_overhead.py",
    }
)

# AWS Secrets Manager resource-lifecycle tests. Each run creates a secret
# under a per-run unique name (``litellm_test_<uuid>``) and either asserts the
# API response echoes that exact unique name or reads it straight back. The
# name *must* be unique per run because AWS enforces a >=7-day deletion
# recovery window — a fixed name can't be re-created on the daily VCR
# re-record. Deterministic replay returns the previously-recorded (different)
# name, so the unique-name round-trip cannot be reproduced offline. The
# config-parsing tests in the same file (settings / STS endpoint) make no such
# unique-resource calls and stay VCR-cached.
_VCR_INCOMPATIBLE_NODEID_SUFFIXES: tuple[str, ...] = (
    "::test_write_and_read_simple_secret",
    "::test_write_and_read_json_secret",
    "::test_read_nonexistent_secret",
    "::test_primary_secret_functionality",
    "::test_write_secret_with_description_and_tags",
)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    sys.path.insert(
        0, os.path.abspath("../..")
    )  # Adds the project directory to the system path

    import litellm

    importlib.reload(litellm)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    yield

    # Teardown code (executes after the yield point)
    loop.close()  # Close the loop created earlier
    asyncio.set_event_loop(None)  # Remove the reference to the loop


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
        skip_files=_VCR_INCOMPATIBLE_FILES,
        skip_nodeid_suffixes=_VCR_INCOMPATIBLE_NODEID_SUFFIXES,
    )

    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    emit_cassette_cache_session_banner(terminalreporter)
    emit_vcr_classification_summary(terminalreporter)
    emit_vcr_diagnostic_log(terminalreporter)
