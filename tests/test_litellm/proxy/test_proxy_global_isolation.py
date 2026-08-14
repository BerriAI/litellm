"""Regression coverage for the proxy_server global isolation hooks in conftest.py."""

from pathlib import Path
from typing import Final

import pytest

pytest_plugins: Final = ("pytester",)

_PARENT_CONFTEST_SHAPE: Final = '''

@pytest.fixture(autouse=True)
def _early_monkeypatch_user(monkeypatch):
    """Mirrors tests/test_litellm/conftest.py, whose autouse env isolation pulls in monkeypatch
    before anything else, so monkeypatch undo lands after every other finalizer."""
    yield
'''

_CONFTEST_SOURCE: Final = (Path(__file__).parent / "conftest.py").read_text() + _PARENT_CONFTEST_SHAPE

_LEAKY_MODULE: Final = '''
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _autouse_patched_prisma_client():
    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        yield


def test_monkeypatches_the_already_patched_global(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", MagicMock())
'''

_WITNESS_MODULE: Final = '''
def test_the_real_global_is_back():
    from litellm.proxy import proxy_server

    assert proxy_server.prisma_client is None
'''


def test_a_monkeypatched_prisma_client_cannot_outlive_its_test(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(_CONFTEST_SOURCE)
    pytester.makepyfile(test_a_leaks=_LEAKY_MODULE, test_b_witness=_WITNESS_MODULE)

    pytester.runpytest("-p", "no:randomly").assert_outcomes(passed=2)
