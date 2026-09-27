import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]


def _backend_env(**overrides: str) -> dict[str, str]:
    env: Final = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'scripts'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.pop("LITELLM_SKIP_RUST_BRIDGE", None)
    env.update(overrides)
    return env


def test_skip_rust_bridge_builds_a_pure_python_wheel(tmp_path: Path) -> None:
    result: Final = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "import sys, litellm_build_backend as b; print(b.build_wheel(sys.argv[1]))",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=_backend_env(LITELLM_SKIP_RUST_BRIDGE="1"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"returncode={result.returncode} stderr={result.stderr}"
    wheel: Final = tmp_path / result.stdout.strip()
    assert wheel.is_file(), f"wheel not produced: stdout={result.stdout} stderr={result.stderr}"
    names: Final = tuple(zipfile.ZipFile(wheel).namelist())
    assert not any(
        name.startswith("litellm/rust_bridge/_native") and (name.endswith(".so") or name.endswith(".pyd"))
        for name in names
    ), names
    for expected in (
        "litellm/proxy/proxy_server.py",
        "litellm/rust_bridge/loader.py",
        "litellm/rust_bridge/_native.pyi",
        "litellm/proxy/_experimental/out/index.html",
        "litellm/router_strategy/complexity_router/fuse_presets.json",
        "litellm/model_prices_and_context_window_backup.json",
        "litellm/proxy/client/cli/commands/codex_base_instructions.md",
    ):
        assert expected in names, f"{expected} missing from wheel"
    assert not any(name.startswith("litellm/proxy/enterprise/") or "__pycache__" in name for name in names), names


def test_without_the_switch_the_hooks_are_maturins() -> None:
    result: Final = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "import litellm_build_backend as b, maturin; "
            "print(b.build_wheel is maturin.build_wheel and b.build_sdist is maturin.build_sdist "
            "and b.build_editable is maturin.build_editable)",
        ],
        cwd=REPO_ROOT,
        env=_backend_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"returncode={result.returncode} stderr={result.stderr}"
    assert result.stdout.strip() == "True", result.stdout


def test_sdist_carries_the_build_backend(tmp_path: Path) -> None:
    result: Final = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "import sys, litellm_build_backend as b; print(b.build_sdist(sys.argv[1]))",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=_backend_env(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"returncode={result.returncode} stderr={result.stderr}"
    sdist: Final = tmp_path / result.stdout.strip().splitlines()[-1]
    assert sdist.is_file(), f"sdist not produced: stdout={result.stdout} stderr={result.stderr}"
    names: Final = tuple(tarfile.open(sdist).getnames())
    assert any(name.endswith("scripts/litellm_build_backend.py") for name in names), names
    assert any(name.endswith("/pyproject.toml") for name in names), names
