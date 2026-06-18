"""Runtime helpers for the external batch e2e suite."""

import json
import os
import tempfile
import time
from typing import Callable

from .config import ExpectedError

# Terminal batch states. Once a batch reaches any of these it will not change.
TERMINAL_STATES = {"completed", "failed", "cancelled", "expired"}


def build_input_jsonl(request_model: str, endpoint: str) -> str:
    """Write a minimal two-request batch input file and return its path.

    Generated per case so the model name always matches the deployment under
    test instead of relying on stale committed fixtures.
    """
    rows = [
        {
            "custom_id": "e2e-request-1",
            "method": "POST",
            "url": endpoint,
            "body": {
                "model": request_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello world!"},
                ],
                "max_tokens": 10,
            },
        },
        {
            "custom_id": "e2e-request-2",
            "method": "POST",
            "url": endpoint,
            "body": {
                "model": request_model,
                "messages": [
                    {"role": "user", "content": "Say hi in one word."},
                ],
                "max_tokens": 10,
            },
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="batch_e2e_")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _error_text(err: Exception) -> str:
    return str(getattr(err, "message", None) or err)


def assert_expected_error(call: Callable[[], object], spec: ExpectedError) -> None:
    """Assert that ``call`` raises an error matching the declared expectation.

    A success, or an error that does not match the spec, is a failure -- this is
    how we distinguish a sanctioned, documented contract gap from a regression.
    """
    try:
        call()
    except Exception as err:  # noqa: BLE001 - we re-assert on the caught error
        if spec.status is not None:
            actual = getattr(err, "status_code", None)
            assert (
                actual == spec.status
            ), f"expected status {spec.status} but got {actual!r}: {_error_text(err)}"
        if spec.match is not None:
            assert (
                spec.match.lower() in _error_text(err).lower()
            ), f"expected error to contain {spec.match!r} but got: {_error_text(err)}"
        return
    raise AssertionError(f"expected an error matching {spec} but the call succeeded.")


def poll_until_terminal(retrieve: Callable[[], object], timeout_s: int):
    """Poll ``retrieve`` until the batch reaches a terminal state or times out."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = retrieve()
        if getattr(last, "status", None) in TERMINAL_STATES:
            return last
        time.sleep(15)
    raise AssertionError(
        f"batch did not reach a terminal state within {timeout_s}s; "
        f"last status={getattr(last, 'status', None)!r}"
    )
