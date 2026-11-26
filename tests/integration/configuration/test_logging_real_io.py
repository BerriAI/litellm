"""JSON logs keep an exception a destructor raises while the interpreter shuts down.

Python runs the destructors of objects still alive at exit after it has started tearing down imports, so a
JSON hook that imports anything then fails, and Python reports the hook's own failure instead of the original
exception. This starts a real interpreter with JSON logs on that exits holding such an object.
"""

import os
from typing import Final

from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

EXIT_TIMEOUT_SECONDS: Final = 120
FAILS_AT_EXIT_SOURCE: Final = """
class _FailsAtExit:
    def __del__(self):
        raise RuntimeError("cleanup at exit failed")


kept_until_exit = _FailsAtExit()
"""


def test_json_logs_keep_the_exception_a_destructor_raises_at_interpreter_exit() -> None:
    finished: Final = run_child_interpreter(
        FAILS_AT_EXIT_SOURCE, env={**os.environ, "JSON_LOGS": "true"}, timeout=EXIT_TIMEOUT_SECONDS
    )

    assert "RuntimeError: cleanup at exit failed" in finished.stderr, finished.stderr
    assert "Exception ignored in sys.unraisablehook" not in finished.stderr, finished.stderr
