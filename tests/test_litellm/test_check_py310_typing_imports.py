import importlib.util
import sys
from pathlib import Path


_CHECKER_PATH = Path(__file__).resolve().parents[1] / "code_coverage_tests" / "check_py310_typing_imports.py"
_SPEC = importlib.util.spec_from_file_location("check_py310_typing_imports", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)


def _scan(tmp_path: Path, source: str) -> tuple:
    file_path = tmp_path / "fixture.py"
    file_path.write_text(source, encoding="utf-8")
    return checker.scan_file(file_path)


def test_typing_import_flags_python_311_name(tmp_path):
    violations = _scan(tmp_path, "from typing import NotRequired, TypedDict\n")
    assert tuple(violation.name for violation in violations) == ("NotRequired",)


def test_typing_extensions_import_passes(tmp_path):
    assert _scan(tmp_path, "from typing_extensions import NotRequired\n") == ()


def test_typing_attribute_flags_python_311_name(tmp_path):
    violations = _scan(tmp_path, "import typing\nx: typing.Self\n")
    assert tuple(violation.name for violation in violations) == ("Self",)


def test_version_guarded_typing_import_passes(tmp_path):
    source = (
        "import sys\n"
        "if sys.version_info >= (3, 11):\n"
        "    from typing import NotRequired\n"
        "else:\n"
        "    from typing_extensions import NotRequired\n"
    )
    assert _scan(tmp_path, source) == ()


def test_python_310_typing_name_passes(tmp_path):
    assert _scan(tmp_path, "from typing import Optional\n") == ()
