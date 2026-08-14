"""Regression coverage for the proxy_server global isolation hooks in conftest.py."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

_PROXY_TESTS_DIR: Final = Path(__file__).parent
_REPO_ROOT: Final = _PROXY_TESTS_DIR.parents[2]

_PARENT_CONFTEST_SHAPE: Final = '''

@pytest.fixture(autouse=True)
def _early_monkeypatch_user(monkeypatch):
    """Mirrors tests/test_litellm/conftest.py, whose autouse env isolation pulls in monkeypatch
    before anything else, so monkeypatch undo lands after every other finalizer."""
    yield
'''

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


def test_a_monkeypatched_prisma_client_cannot_outlive_its_test(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text((_PROXY_TESTS_DIR / "conftest.py").read_text() + _PARENT_CONFTEST_SHAPE)
    (tmp_path / "test_a_leaks.py").write_text(_LEAKY_MODULE)
    (tmp_path / "test_b_witness.py").write_text(_WITNESS_MODULE)

    completed: Final = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "test_a_leaks.py",
            "test_b_witness.py",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTEST_ADDOPTS": "", "PYTHONPATH": str(_REPO_ROOT)},
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
