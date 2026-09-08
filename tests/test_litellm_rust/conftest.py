import os
from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.rust_bridge.configuration import reset_rust_configuration
from tests.test_litellm_rust.recording_server import recording_server  # noqa: F401  # pytest fixture export


@pytest.fixture(autouse=True)
def isolate_rust_state() -> Iterator[None]:
    callback_attributes: Final = (
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_input_callback",
        "_async_success_callback",
        "_async_failure_callback",
    )
    original_callbacks: Final = {attribute: list(getattr(litellm, attribute)) for attribute in callback_attributes}
    original_cache: Final = litellm.cache
    for attribute in callback_attributes:
        getattr(litellm, attribute).clear()
    litellm.cache = None  # test-quality-ok: isolate the process-global cache from native extension tests
    reset_rust_configuration()
    litellm.rust(True)
    yield
    for attribute, callbacks in original_callbacks.items():
        target = getattr(litellm, attribute)
        target.clear()
        target.extend(callbacks)
    litellm.cache = original_cache  # test-quality-ok: restore the process-global cache after native extension tests
    reset_rust_configuration()


def pytest_collection_modifyitems(items):
    rust_enabled = os.environ.get("LITELLM_RUST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not rust_enabled:
        skip = pytest.mark.skip(reason="requires LITELLM_RUST=1 and a compiled Rust extension")
        for item in items:
            item.add_marker(skip)
        return

    try:
        from litellm.rust_bridge import _native  # noqa: F401  # validates the installed extension
    except ImportError as error:
        raise pytest.UsageError("LITELLM_RUST=1 requires a compiled litellm.rust_bridge._native extension") from error
