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
import sys
import traceback

repo_root = {str(repo_root.resolve())!r}

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
print("litellm.__file__:", litellm.__file__)
print("_types loaded:", "litellm.proxy._types" in sys.modules)
"""
    return subprocess.run(
        [sys.executable, "-c", program],
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


def main(repo_root: Path = _REPO_ROOT) -> None:
    results: Final = (
        (False, _run_import_check(False, repo_root)),
        (True, _run_import_check(True, repo_root)),
    )
    has_failures = False
    for block_enterprise, result in results:
        print(f"enterprise blocked: {block_enterprise}")
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0 or "_types loaded: False" not in result.stdout:
            has_failures = True

    if has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
