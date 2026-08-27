import importlib.util
import os
import sys
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


def test_env_puts_this_interpreters_bin_dir_first_on_path():
    env = mod.env_with_own_bin_first({"PATH": "/usr/bin", "HOME": "/home"})
    bin_dir = str(Path(sys.executable).parent)
    assert env["PATH"].split(os.pathsep) == [bin_dir, "/usr/bin"]
    assert env["HOME"] == "/home"


def test_env_without_an_inherited_path_is_just_the_bin_dir():
    env = mod.env_with_own_bin_first({})
    assert env["PATH"] == str(Path(sys.executable).parent)


def test_generate_runs_prisma_with_its_own_bin_dir_leading_the_childs_path():
    seen = {}

    def recorder(cmd, cwd, env):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        seen["env"] = env
        return 0

    assert mod.run_generate(run=recorder) == 0
    assert seen["cmd"][:4] == [sys.executable, "-m", "prisma", "generate"]
    assert seen["cwd"] == mod.REPO_ROOT
    assert seen["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
