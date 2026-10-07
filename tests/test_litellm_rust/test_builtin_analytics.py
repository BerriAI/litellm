import json
from typing import Final

import pytest

from litellm.rust_bridge import _native

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.parametrize(
    ("dnt", "explicit", "license_configured", "invalid"),
    (
        ("1", "true", False, False),
        ("true", "true", True, False),
        (None, None, True, False),
        (None, "garbage", True, True),
        (None, "garbage", False, True),
        (None, "", False, True),
        (" ", "true", False, True),
        ("y", "t", False, False),
    ),
)
def test_disabled_policy_never_starts_native_workers(
    dnt: str | None, explicit: str | None, license_configured: bool, invalid: bool
) -> None:
    native: Final = _native.NativeDiagnosticLogger()
    native.shutdown_analytics()
    before: Final = _native.process_state_started()
    configuration: Final = json.dumps(
        {
            "inputs": {"do_not_track": dnt, "explicit": explicit, "license_configured": license_configured},
            "surface": "python_sdk",
            "version": "1.0",
        }
    )
    try:
        assert native.initialize_analytics(configuration) == (False, invalid)
        native.emit_analytics("completion")
        assert _native.process_state_started() is before
    finally:
        native.shutdown_analytics()
