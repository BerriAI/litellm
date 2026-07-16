"""Local conftest for the driver unit tests.

Installs a hermetic, no-op rate limiter for every test in this
subdirectory. Without this, importing `cli_driver` and calling
`run_claude(..., runner=fake)` would silently consume tokens from the
shared default limiter (which writes to `$TMPDIR/...`), polluting the
on-disk state another test run might rely on and adding flakiness if
the env vars say "rate=0.1/s".

A no-op limiter (rate=0 for every provider) returns immediately from
`acquire(...)`, so unit tests behave exactly as they did before the
limiter was added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

from claude_code.rate_limiter import (
    ALL_PROVIDERS,
    ProviderConfig,
    RateLimiter,
    use_limiter,
)


@pytest.fixture(autouse=True)
def _hermetic_rate_limiter(  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    tmp_path: Path,
) -> Generator[None]:
    config = {p: ProviderConfig(rate_per_sec=0.0, burst=0.0) for p in ALL_PROVIDERS}
    limiter = RateLimiter(config=config, state_dir=tmp_path)
    with use_limiter(limiter):
        yield
