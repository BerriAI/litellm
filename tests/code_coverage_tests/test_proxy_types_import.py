import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]


def _run_import_check(
    block_enterprise: bool, repo_root: Path
) -> subprocess.CompletedProcess[str]:
    program: Final = f"""
import importlib.abc
import pathlib
import sys
import traceback

repo_root = sys.argv[1]

class ImportTracer(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if {block_enterprise!r} and name.startswith("litellm_enterprise"):
            raise ImportError("blocked litellm_enterprise imports")
        if name == "litellm.proxy._types":
            repository_frames = [
                frame for frame in traceback.extract_stack()
                if repo_root in frame.filename
            ]
            print(
                "FIRST _types importer chain:",
                " | ".join(
                    f"{{frame.filename}}:{{frame.lineno}}"
                    for frame in repository_frames[-8:]
                ),
            )
        return None

sys.meta_path.insert(0, ImportTracer())
import litellm
if not pathlib.Path(litellm.__file__).resolve().is_relative_to(pathlib.Path(repo_root)):
    raise SystemExit(f"imported litellm from {{litellm.__file__}}, expected under {{repo_root}}")
print("litellm.__file__:", litellm.__file__)
print("_types loaded:", "litellm.proxy._types" in sys.modules)
    """
    return subprocess.run(
        [sys.executable, "-I", "-c", program, str(repo_root.resolve())],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_proxy_types_attribute_access(repo_root: Path) -> subprocess.CompletedProcess[str]:
    program: Final = """
import pathlib
import sys
import litellm

repo_root = sys.argv[1]
if not pathlib.Path(litellm.__file__).resolve().is_relative_to(pathlib.Path(repo_root)):
    raise SystemExit(f"imported litellm from {litellm.__file__}, expected under {repo_root}")

assert "litellm.proxy._types" not in sys.modules
print("_types loaded before attribute access: False")
user_api_key_auth = litellm.proxy._types.UserAPIKeyAuth
assert "litellm.proxy._types" in sys.modules
proxy_types = sys.modules["litellm.proxy._types"]
assert user_api_key_auth is proxy_types.UserAPIKeyAuth
print("_types loaded after attribute access: True")
print("UserAPIKeyAuth identity: True")
    """
    return subprocess.run(
        [sys.executable, "-I", "-c", program, str(repo_root.resolve())],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("block_enterprise", (False, True))
def test_import_litellm_does_not_load_proxy_types(block_enterprise: bool) -> None:
    result: Final = _run_import_check(block_enterprise, _REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "_types loaded: False" in result.stdout, result.stdout + result.stderr


def test_proxy_types_attribute_access_still_works() -> None:
    result: Final = _run_proxy_types_attribute_access(_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "_types loaded before attribute access: False" in result.stdout
    assert "_types loaded after attribute access: True" in result.stdout
    assert "UserAPIKeyAuth identity: True" in result.stdout


def main(repo_root: Path = _REPO_ROOT) -> None:
    results: Final = (
        ("enterprise blocked: False", _run_import_check(False, repo_root), "_types loaded: False"),
        ("enterprise blocked: True", _run_import_check(True, repo_root), "_types loaded: False"),
        (
            "proxy._types attribute access",
            _run_proxy_types_attribute_access(repo_root),
            "UserAPIKeyAuth identity: True",
        ),
    )
    for label, result, expected_output in results:
        print(label)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")

    if any(
        result.returncode != 0 or expected_output not in result.stdout
        for _, result, expected_output in results
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
