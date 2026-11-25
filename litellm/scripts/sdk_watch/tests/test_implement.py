import sys
from pathlib import Path

import pytest
from conftest import git
from sdk_watch.implement import collect_patch, fetch_sdk_diff_tool, is_test_path, run_tests_tool
from sdk_watch.sources import Sdk

SDK_DIFF = (
    "diff --git a/src/anthropic/types/message_param.py b/src/anthropic/types/message_param.py\n"
    "+    role: Literal['user', 'assistant', 'system']\n"
    "diff --git a/README.md b/README.md\n"
    "+docs\n"
)


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        ("tests/unit/test_x.py", True),
        ("tests/unit/llms/anthropic/test_chat.py", True),
        ("tests/conftest.py", False),
        ("tests/unit/../../litellm/x.py", False),
        ("litellm/x.py", False),
        ("tests/unit/data.json", False),
        ("tests", False),
    ],
)
def test_only_python_files_under_tests_can_be_run(path: str, allowed: bool) -> None:
    assert is_test_path(path) is allowed


def test_run_tests_refuses_a_path_outside_tests_without_running_anything(repo: Path) -> None:
    output = run_tests_tool(repo, Path(sys.executable)).invoke({"test_files": ["litellm/shout.py"]})

    assert output.startswith("Refused")


def test_run_tests_reports_the_pytest_exit_code_and_output(repo: Path) -> None:
    (repo / "tests" / "unit" / "test_shout.py").write_text(
        "from litellm.shout import shout\n\ndef test_shout():\n    assert shout('hi') == 'HI'\n"
    )

    output = run_tests_tool(repo, Path(sys.executable)).invoke({"test_files": ["tests/unit/test_shout.py"]})

    assert output.startswith("exit code 1")
    assert "assert 'hi' == 'HI'" in output


def test_collect_patch_captures_edits_and_new_files_then_restores_the_checkout(repo: Path, tmp_path: Path) -> None:
    (repo / "litellm" / "shout.py").write_text("def shout(text):\n    return text.upper()\n")
    (repo / "tests" / "unit" / "test_shout.py").write_text("def test_shout():\n    assert True\n")

    patch = collect_patch(repo)

    assert git(repo, "status", "--porcelain") == ""
    (tmp_path / "change.patch").write_text(patch)
    git(repo, "apply", str(tmp_path / "change.patch"))
    assert (repo / "litellm" / "shout.py").read_text().endswith("return text.upper()\n")
    assert (repo / "tests" / "unit" / "test_shout.py").exists()


def test_fetch_sdk_diff_lists_every_file_and_keeps_only_the_matching_ones() -> None:
    urls = []
    tool = fetch_sdk_diff_tool(
        Sdk(name="anthropic", repo="anthropics/anthropic-sdk-python"), lambda url: urls.append(url) or SDK_DIFF
    )

    output = tool.invoke({"commit_sha": "f18b014", "path_contains": "types/"})

    assert urls == ["https://github.com/anthropics/anthropic-sdk-python/commit/f18b014.diff"]
    assert output.startswith("2 files changed:")
    assert "'system'" in output
    assert "+docs" not in output


def test_fetch_sdk_diff_refuses_anything_but_a_commit_sha() -> None:
    tool = fetch_sdk_diff_tool(Sdk(name="openai", repo="openai/openai-python"), lambda url: SDK_DIFF)

    assert tool.invoke({"commit_sha": "../../evil"}).startswith("Refused")
