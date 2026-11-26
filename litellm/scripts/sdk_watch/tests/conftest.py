import subprocess
from pathlib import Path

import pytest


def git(repo: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    checkout = tmp_path / "checkout"
    subprocess.run(("git", "init", "--quiet", "--bare", "--initial-branch=main", str(origin)), check=True)
    subprocess.run(("git", "clone", "--quiet", str(origin), str(checkout)), check=True, capture_output=True)
    (checkout / ".gitignore").write_text("__pycache__/\n")
    (checkout / "litellm").mkdir()
    (checkout / "litellm" / "shout.py").write_text("def shout(text):\n    return text\n")
    (checkout / "tests" / "unit").mkdir(parents=True)
    (checkout / "tests" / "unit" / "test_existing.py").write_text("def test_existing():\n    assert True\n")
    git(checkout, "add", ".")
    git(checkout, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "--quiet", "-m", "base")
    git(checkout, "push", "--quiet", "origin", "HEAD:main")
    return checkout
