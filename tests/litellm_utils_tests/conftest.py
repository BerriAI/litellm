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

from tests._vcr_conftest_common import (  # noqa: E402
    VerboseReporterState,
    apply_vcr_auto_marker_to_items,
    record_vcr_outcome,
    register_persister_if_enabled,
    vcr_config_dict,
)

_verbose_state = VerboseReporterState()


# Files where VCR replay actively breaks the test:
# - ``test_litellm_overhead.py`` measures ``litellm_overhead_time_ms`` as a
#   percentage of total wall-clock time. With cached responses the upstream
#   "network" time collapses to microseconds, so the overhead percentage
#   blows past the 40% threshold the test asserts on.
_VCR_INCOMPATIBLE_FILES = frozenset(
    {
        "test_litellm_overhead.py",
    }
)

# No node-id suffix skips at the moment. Tests that deliberately use a
# bad API key (e.g. ``test_get_valid_models_from_dynamic_api_key`` with
# ``api_key="123"``) are handled transparently by the ``key_fingerprint``
# matcher in ``tests/_vcr_conftest_common.py``.
_VCR_INCOMPATIBLE_NODEID_SUFFIXES: tuple[str, ...] = ()


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
    yield
    record_vcr_outcome(request, vcr)


def pytest_configure(config):
    _verbose_state.remember_pluginmanager(config)


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
