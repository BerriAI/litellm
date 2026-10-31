"""What a coding-agent CLI under test may see of the host.

The CI job holds the proxy's upstream credentials (provider keys, AWS, Azure, the database URL, GitHub tokens) in its
environment, and the cron VM runs as a real user with a populated home. A compromised CLI release, or a model told to
read a file, must reach neither, so the CLI gets only the allowlisted runtime variables plus what its suite passes in,
and a fresh empty HOME that is deleted when it exits.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final

CLI_ENV_ALLOWLIST: Final = (
    "PATH",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NODE_PATH",
    "NVM_DIR",
    "NVM_BIN",
)


@contextmanager
def isolated_home(prefix: str) -> Generator[Path]:
    home: Final = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


def locked_down_env(
    *, home: Path, overrides: Mapping[str, str], source: Mapping[str, str] = os.environ
) -> Mapping[str, str]:
    allowed: Final = {name: source[name] for name in CLI_ENV_ALLOWLIST if name in source}
    return MappingProxyType({**allowed, "HOME": str(home), **overrides})
