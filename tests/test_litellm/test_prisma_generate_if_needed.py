import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "prisma_generate_if_needed.py"
)
_spec = importlib.util.spec_from_file_location("prisma_generate_if_needed", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_stamp_changes_with_schema_and_with_prisma_version():
    stamp = mod.stamp_value(b"model A {}", "0.11.0")
    assert mod.stamp_value(b"model A {}", "0.11.0") == stamp
    assert mod.stamp_value(b"model B {}", "0.11.0") != stamp
    assert mod.stamp_value(b"model A {}", "0.12.0") != stamp


def test_skip_requires_a_matching_stamp(tmp_path):
    stamp = tmp_path / "stamp"
    expected = mod.stamp_value(b"schema", "0.11.0")
    assert mod.should_skip(stamp, expected, client_generated=True) is False
    stamp.write_text(expected)
    assert mod.should_skip(stamp, expected, client_generated=True) is True
    assert (
        mod.should_skip(stamp, mod.stamp_value(b"other", "0.11.0"), client_generated=True)
        is False
    )


def test_skip_requires_a_generated_client_even_with_a_matching_stamp(tmp_path):
    stamp = tmp_path / "stamp"
    expected = mod.stamp_value(b"schema", "0.11.0")
    stamp.write_text(expected)
    assert mod.should_skip(stamp, expected, client_generated=False) is False
