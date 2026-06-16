import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_any_discipline.py"
)
_spec = importlib.util.spec_from_file_location("check_any_discipline", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

Violation = mod.Violation


def _v(path="litellm/x.py", line=10, code="LIT009"):
    return Violation(Path(path), line, 0, code, "Any-typed value")


def test_violation_on_a_changed_line_is_in_scope():
    assert mod._in_scope(_v(line=10), {"litellm/x.py": {10, 11}}) is True


def test_violation_on_an_unchanged_line_of_a_changed_file_is_out_of_scope():
    assert mod._in_scope(_v(line=99), {"litellm/x.py": {10, 11}}) is False


def test_whole_new_file_puts_every_line_in_scope():
    assert mod._in_scope(_v(line=99999), {"litellm/x.py": mod.ALL_LINES}) is True


def test_file_absent_from_line_map_is_out_of_scope():
    # Regression: ALL_LINES is a distinct sentinel, so a path missing from the map
    # (line_map.get -> None) is NOT mistaken for "whole file in scope".
    assert mod._in_scope(_v(path="litellm/other.py"), {"litellm/x.py": {1}}) is False


def test_no_line_map_means_no_line_filtering():
    assert mod._in_scope(_v(line=12345), None) is True


def test_build_error_is_always_in_scope():
    assert mod._in_scope(_v(code="LIT000", line=1), {"litellm/x.py": {2}}) is True
