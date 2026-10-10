from itertools import chain
import re
from pathlib import Path
from typing import Final

import pytest

from tests.fake_openai_endpoint import _LOCAL_DEFAULT, _resolve_base

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_LIVE_HOSTED_MOCK: Final = re.compile(r"railway\.app(?:/|\")")


def _hosted_mock_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    python_files: Final = sorted(chain.from_iterable(root.rglob("*.py") for root in roots))
    return tuple(path for path in python_files if _LIVE_HOSTED_MOCK.search(path.read_text()))


@pytest.mark.parametrize("host", ("127.0.0.1", "localhost", "[::1]"))
def test_loopback_env_base_is_honored(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    base: Final = f"http://{host}:9191"
    monkeypatch.setenv("FAKE_OPENAI_API_BASE", base)

    assert _resolve_base() == base


def test_remote_env_base_resolves_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "FAKE_OPENAI_API_BASE",
        "https://remote.example.invalid",
    )

    assert _resolve_base() == _LOCAL_DEFAULT


def test_migrated_files_have_no_live_hosted_mock() -> None:
    roots: Final = (_REPO_ROOT / "tests/unit", _REPO_ROOT / "tests/integration")
    offenders: Final = _hosted_mock_files(roots)

    assert not offenders, "\n".join(str(path.relative_to(_REPO_ROOT)) for path in offenders)


def test_hosted_mock_scan_finds_a_matching_file(tmp_path: Path) -> None:
    unit_root: Final = tmp_path / "tests/unit"
    integration_root: Final = tmp_path / "tests/integration"
    unit_root.mkdir(parents=True)
    integration_root.mkdir(parents=True)
    match: Final = unit_root / "hosted_mock.py"
    host: Final = ".".join(("railway", "app"))
    match.write_text(f'api_base = "https://mock.{host}"\n')

    assert _hosted_mock_files((unit_root, integration_root)) == (match,)
