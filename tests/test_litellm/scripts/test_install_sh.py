"""
Regression tests for scripts/install.sh (syntax and critical install-path strings).

These are lightweight static checks so CI catches accidental rewrites of the
pipx inject / fallback messaging without running a full network install.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTALL_SH = _REPO_ROOT / "scripts" / "install.sh"


@pytest.fixture
def install_sh_text() -> str:
    assert _INSTALL_SH.is_file(), f"missing {_INSTALL_SH}"
    return _INSTALL_SH.read_text(encoding="utf-8")


def test_install_sh_exists():
    assert _INSTALL_SH.is_file()


def test_install_sh_passes_bash_syntax_check():
    result = subprocess.run(
        ["bash", "-n", str(_INSTALL_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_install_sh_uses_pipx_inject_for_proxy_extras_after_upgrade(install_sh_text: str):
    """Prefer pipx inject over runpip: inject is older / wider pipx support."""
    assert "pipx inject litellm" in install_sh_text
    assert "pipx runpip litellm" not in install_sh_text


def test_install_sh_warns_on_pipx_install_failure_fallback(install_sh_text: str):
    assert 'warn "pipx install failed (see above), falling back to venv"' in install_sh_text


def test_install_sh_warn_message_for_failed_proxy_inject(install_sh_text: str):
    assert (
        'warn "could not inject proxy extras (proxy features may fail until manually fixed)."'
        in install_sh_text
    )
