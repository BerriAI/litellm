import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_root_file_count.py"
_spec = importlib.util.spec_from_file_location("check_root_file_count", _MODULE_PATH)
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


def test_root_files_of_keeps_only_top_level_paths():
    tracked = (
        "README.md",
        ".gitignore",
        "litellm/main.py",
        "scripts/check_root_file_count.py",
        "docs/my-website/sidebars.js",
    )
    assert checker.root_files_of(tracked) == ("README.md", ".gitignore")


def test_count_equal_to_limit_is_within_limit():
    check = checker.RootFileCheck(root_files=("a", "b", "c"), max_root_files=3)
    assert check.count == 3
    assert check.within_limit is True


def test_count_over_limit_fails_and_lists_offenders():
    check = checker.RootFileCheck(root_files=("zzz.sh", "aaa.py", "bbb.json"), max_root_files=2)
    assert check.within_limit is False
    message = checker.report(check)
    assert "::error::" in message
    assert message.index("aaa.py") < message.index("bbb.json") < message.index("zzz.sh")
    assert "MAX_ROOT_FILES" in message


def test_report_ok_message_when_within_limit():
    check = checker.RootFileCheck(root_files=("a", "b"), max_root_files=44)
    message = checker.report(check)
    assert "OK" in message
    assert "::error::" not in message
