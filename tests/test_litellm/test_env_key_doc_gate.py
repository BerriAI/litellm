"""Tests for the env-var extraction used by tests/documentation_tests/test_env_keys.py.

That script is the CI gate that fails when a user-facing environment variable read
under litellm/ has no row in the docs reference table. It only sees a key if one of its
patterns matches the call, so a call shape the patterns miss silently bypasses the gate.
Each supported shape is asserted here, along with the shapes that must not be treated as
env var reads, so narrowing a pattern makes a test fail instead of quietly reopening the
hole.
"""

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "tests" / "documentation_tests" / "test_env_keys.py"
_spec = importlib.util.spec_from_file_location("documentation_test_env_keys", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)


def test_bare_get_secret_bool_is_captured() -> None:
    assert gate.extract_env_keys('flag = get_secret_bool("QSTASH_FLUSH_ON_BOOT")') == {"QSTASH_FLUSH_ON_BOOT"}


def test_get_secret_bool_with_default_is_captured() -> None:
    assert gate.extract_env_keys('if get_secret_bool("QSTASH_FLUSH_ON_BOOT", False) is not True:') == {
        "QSTASH_FLUSH_ON_BOOT"
    }


def test_get_secret_bool_with_keyword_default_is_captured() -> None:
    assert gate.extract_env_keys('get_secret_bool("QSTASH_FLUSH_ON_BOOT", default_value=False)') == {
        "QSTASH_FLUSH_ON_BOOT"
    }


def test_litellm_prefixed_get_secret_bool_is_captured() -> None:
    assert gate.extract_env_keys('litellm.get_secret_bool("QSTASH_FLUSH_ON_BOOT")') == {"QSTASH_FLUSH_ON_BOOT"}


def test_previously_supported_call_shapes_are_still_captured() -> None:
    source = "\n".join(
        (
            'os.getenv("QSTASH_ALPHA")',
            'os.getenv("QSTASH_BRAVO", "fallback")',
            'litellm.get_secret("QSTASH_CHARLIE")',
            'litellm.get_secret_str("QSTASH_DELTA", default_value=None)',
        )
    )
    assert gate.extract_env_keys(source) == {"QSTASH_ALPHA", "QSTASH_BRAVO", "QSTASH_CHARLIE", "QSTASH_DELTA"}


def test_get_secret_calls_on_unrelated_objects_are_not_env_reads() -> None:
    source = "\n".join(
        (
            'vault_client.get_secret("QSTASH_ALPHA")',
            'self.get_secret_str("QSTASH_BRAVO")',
            'provider.get_secret_bool("QSTASH_CHARLIE")',
        )
    )
    assert gate.extract_env_keys(source) == frozenset()


def test_similarly_named_helpers_are_not_env_reads() -> None:
    assert gate.extract_env_keys('get_secret_bundle("QSTASH_ALPHA")') == frozenset()


def test_non_literal_arguments_are_not_env_reads() -> None:
    assert gate.extract_env_keys("get_secret_bool(flag_name)") == frozenset()


def test_excluded_keys_are_filtered_for_every_call_shape() -> None:
    source = "\n".join(
        (
            'os.getenv("TERM_PROGRAM")',
            'get_secret_bool("LITELLM_RUST")',
            'litellm.get_secret_str("MAVVRIK_FOCUS_FREQUENCY")',
        )
    )
    assert gate.extract_env_keys(source) == frozenset()


def test_documented_keys_are_read_from_the_reference_table_only() -> None:
    docs = "\n".join(
        (
            "### general_settings - Reference",
            "| BEFORE_THE_TABLE | not the env var table",
            "",
            "### environment variables - Reference",
            "",
            "| Name | Description |",
            "|------|-------------|",
            "| QSTASH_ALPHA | first key",
            "| QSTASH_BRAVO | second key",
            "",
            "### another section - Reference",
            "| AFTER_THE_TABLE | also not the env var table",
        )
    )
    assert gate.extract_documented_keys(docs) == {"QSTASH_ALPHA", "QSTASH_BRAVO"}
