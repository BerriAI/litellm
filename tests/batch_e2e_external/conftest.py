"""
Pytest wiring for the external batch e2e suite.

This suite is opt-in. It is never collected by CI (no workflow references this
folder) and is gated behind the ``batch_e2e`` marker. It runs against a live,
published LiteLLM proxy image using the real OpenAI SDK -- no mocks, no VCR.

Hard-fail contract: configuration is loaded once at session start. If anything
required is missing or malformed the whole session errors out, rather than
silently skipping coverage.
"""

import pytest
from openai import OpenAI

from .config import Settings, load_settings

_SETTINGS: Settings = None  # type: ignore[assignment]


def _get_settings() -> Settings:
    """Load + cache the config once. Raises ConfigError (fatal) if misconfigured."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "batch_e2e: external end-to-end batch lifecycle tests (live providers).",
    )


def pytest_generate_tests(metafunc):
    """Expand the test matrix from the configured cases at collection time."""
    if "case" in metafunc.fixturenames:
        cases = _get_settings().cases
        metafunc.parametrize("case", cases, ids=[c.id for c in cases])


@pytest.fixture(scope="session")
def settings() -> Settings:
    return _get_settings()


@pytest.fixture(scope="session")
def client(settings: Settings) -> OpenAI:
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key)
