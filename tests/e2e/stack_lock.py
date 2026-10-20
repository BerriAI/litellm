"""Cross-process reader/writer lock over the proxy stack every xdist worker shares.
Every collected test holds it shared, marker or not, since the Claude Code cells and
other unmarked suites drive the same stack; a `quiet_stack` test holds it exclusive,
and the `gate` file makes a waiting exclusive holder win over readers that arrive
after it."""

from __future__ import annotations

import fcntl
import hashlib
import tempfile
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Final

from e2e_config import PROXY_BASE_URL

STACK_DIGEST: Final = hashlib.sha256(PROXY_BASE_URL.encode()).hexdigest()[:12]
LOCK_DIR: Final = Path(tempfile.gettempdir()) / f"litellm-e2e-stack-{STACK_DIGEST}"
GATE_FILE: Final = LOCK_DIR / "gate"
STACK_FILE: Final = LOCK_DIR / "stack"


@contextmanager
def _flock(path: Path, operation: int) -> Generator[None]:
    with path.open("a") as handle:
        fcntl.flock(handle, operation)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def stack_lock(exclusive: bool) -> Generator[None]:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with _flock(GATE_FILE, fcntl.LOCK_EX), _flock(STACK_FILE, fcntl.LOCK_EX):
            yield
        return
    with ExitStack() as held:
        with _flock(GATE_FILE, fcntl.LOCK_SH):
            held.enter_context(_flock(STACK_FILE, fcntl.LOCK_SH))
        yield
