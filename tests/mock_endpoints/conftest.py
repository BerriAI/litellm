"""Pytest fixtures for the vendored mock OpenAI endpoint.

Any test suite can opt into a per-session local mock server like so::

    pytest_plugins = ("tests.mock_endpoints.conftest",)

    def test_something(mock_openai_endpoint_server):
        url = mock_openai_endpoint_server  # e.g. "http://127.0.0.1:53892"
        ...

This avoids hitting the Railway-hosted deployment of
``BerriAI/example_openai_endpoint``, which has historically been a source of
flaky CI when Railway has incidents.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from . import start_mock_server


@pytest.fixture(scope="session")
def mock_openai_endpoint_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """Session-scoped fixture that boots the vendored mock server.

    Yields the base URL (e.g. ``http://127.0.0.1:53892``). The server is
    killed at session teardown. Server logs are written to
    ``<pytest tmp>/mock_openai_endpoint.log`` for debugging.
    """
    log_dir: Path = tmp_path_factory.mktemp("mock_openai_endpoint")
    log_file = log_dir / "mock_openai_endpoint.log"

    handle = start_mock_server(log_file=log_file)

    # Expose the URL to any code that reads the env var (this is what the
    # ``MOCK_OPENAI_BASE_URL`` helper checks). Restore the previous value on
    # teardown so we don't leak it across test sessions.
    previous = os.environ.get("LITELLM_MOCK_OPENAI_BASE_URL")
    os.environ["LITELLM_MOCK_OPENAI_BASE_URL"] = handle.base_url
    try:
        yield handle.base_url
    finally:
        if previous is None:
            os.environ.pop("LITELLM_MOCK_OPENAI_BASE_URL", None)
        else:
            os.environ["LITELLM_MOCK_OPENAI_BASE_URL"] = previous
        handle.stop()
