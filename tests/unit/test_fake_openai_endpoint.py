import re
from pathlib import Path
from typing import Final

import pytest

from tests.fake_openai_endpoint import _LOCAL_DEFAULT, _resolve_base

_TESTS_ROOT: Final = Path(__file__).resolve().parents[1]
_LIVE_HOSTED_MOCK: Final = re.compile(r"railway\.app(?:/|\")")
_LOCAL_FAKE_IMPORT: Final = "fake_openai_endpoint"


def _local_fake_users_with_hosted_mock(root: Path) -> tuple[Path, ...]:
    sources: Final = ((path, path.read_text()) for path in sorted(root.rglob("*.py")))
    return tuple(
        path for path, source in sources if _LOCAL_FAKE_IMPORT in source and _LIVE_HOSTED_MOCK.search(source)
    )


@pytest.mark.parametrize("host", ("127.0.0.1", "localhost", "[::1]"))
def test_loopback_env_base_is_honored(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    base: Final = f"http://{host}:9191"
    monkeypatch.setenv("FAKE_OPENAI_API_BASE", base)

    assert _resolve_base() == base


def test_remote_env_base_resolves_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_OPENAI_API_BASE", "https://remote.example.invalid")

    assert _resolve_base() == _LOCAL_DEFAULT


def test_migrated_files_have_no_live_hosted_mock() -> None:
    offenders: Final = _local_fake_users_with_hosted_mock(_TESTS_ROOT)

    assert offenders == (), [str(path.relative_to(_TESTS_ROOT)) for path in offenders]


def test_hosted_mock_scan_flags_only_local_fake_users(tmp_path: Path) -> None:
    host: Final = ".".join(("railway", "app"))
    both: Final = tmp_path / "integration/uses_fake_and_hosted.py"
    both.parent.mkdir(parents=True)
    both.write_text(f'from tests.{_LOCAL_FAKE_IMPORT} import FAKE_OPENAI_API_BASE\napi_base = "https://mock.{host}"\n')
    (tmp_path / "integration/uses_fake_only.py").write_text(
        f"from tests.{_LOCAL_FAKE_IMPORT} import FAKE_OPENAI_API_BASE\n"
    )
    (tmp_path / "integration/hosted_string_only.py").write_text(f'api_base = "https://mock.{host}/"\n')

    assert _local_fake_users_with_hosted_mock(tmp_path) == (both,)
