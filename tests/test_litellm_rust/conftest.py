import os

import pytest


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
        raise pytest.UsageError(
            "LITELLM_RUST=1 requires a compiled litellm.rust_bridge._native extension"
        ) from error
