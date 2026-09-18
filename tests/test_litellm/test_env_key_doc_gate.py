"""Tests for the env-var extraction used by tests/documentation_tests/test_env_keys.py.

That script is the CI gate that fails when a user-facing environment variable read
under litellm/ is mentioned nowhere on the docs site. It only sees a key if one of its
patterns matches the call, so a call shape the patterns miss silently bypasses the gate.
Each supported shape is asserted here, along with the shapes that must not be treated as
env var reads, so narrowing a pattern makes a test fail instead of quietly reopening the
hole. The docs side is asserted too, since a key documented on a provider page rather
than in the central reference table still counts as documented.
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


def test_bare_get_secret_is_captured() -> None:
    assert gate.extract_env_keys('key = get_secret("QSTASH_ALPHA")') == {"QSTASH_ALPHA"}


def test_bare_get_secret_with_default_is_captured() -> None:
    assert gate.extract_env_keys('key = get_secret("QSTASH_ALPHA", "fallback")') == {"QSTASH_ALPHA"}


def test_bare_get_secret_str_is_captured() -> None:
    assert gate.extract_env_keys('key = get_secret_str("QSTASH_BRAVO")') == {"QSTASH_BRAVO"}


def test_bare_get_secret_str_with_keyword_default_is_captured() -> None:
    assert gate.extract_env_keys('get_secret_str("QSTASH_BRAVO", default_value=None)') == {"QSTASH_BRAVO"}


def test_get_secret_reached_through_the_utils_module_is_captured() -> None:
    assert gate.extract_env_keys('litellm.utils.get_secret("QSTASH_ALPHA")') == {"QSTASH_ALPHA"}


def test_get_secret_on_an_unrelated_utils_attribute_is_not_an_env_read() -> None:
    source = "\n".join(
        (
            'vault.utils.get_secret("QSTASH_ALPHA")',
            'self.utils.get_secret_str("QSTASH_BRAVO")',
        )
    )
    assert gate.extract_env_keys(source) == frozenset()


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


def test_a_key_mentioned_outside_the_reference_table_counts_as_documented() -> None:
    docs = "\n".join(
        (
            "# Qstash",
            "",
            "Set `QSTASH_ALPHA` to your endpoint before calling the provider.",
            "",
            "```bash",
            'export QSTASH_BRAVO="sk-..."',
            "```",
        )
    )
    documented = gate.extract_documented_keys(docs)
    assert "QSTASH_ALPHA" in documented
    assert "QSTASH_BRAVO" in documented


def test_a_name_glued_to_surrounding_text_is_not_a_mention() -> None:
    documented = gate.extract_documented_keys("the useQSTASH_ALPHA helper reads it")
    assert "QSTASH_ALPHA" not in documented


def test_a_longer_name_does_not_document_the_key_it_ends_with() -> None:
    documented = gate.extract_documented_keys("Set AZURE_QSTASH_ALPHA in your environment")
    assert "AZURE_QSTASH_ALPHA" in documented
    assert "QSTASH_ALPHA" not in documented


def test_lowercase_mentions_are_not_treated_as_env_var_names() -> None:
    assert gate.extract_documented_keys("pass qstash_alpha as a config key") == frozenset()


def test_documented_keys_are_collected_from_every_page_of_the_docs_site(tmp_path: Path) -> None:
    (tmp_path / "providers").mkdir()
    (tmp_path / "providers" / "qstash.md").write_text("Set `QSTASH_ALPHA` to your endpoint.\n", encoding="utf-8")
    (tmp_path / "providers" / "qstash_batches.mdx").write_text("| QSTASH_BRAVO | second key |\n", encoding="utf-8")
    documented = gate.collect_documented_keys(str(tmp_path))
    assert "QSTASH_ALPHA" in documented
    assert "QSTASH_BRAVO" in documented
    assert "QSTASH_CHARLIE" not in documented


def _write_tree(tmp_path: Path, source: str, docs_page: str) -> tuple[str, str]:
    source_dir = tmp_path / "litellm"
    docs_dir = tmp_path / "docs" / "providers"
    source_dir.mkdir()
    docs_dir.mkdir(parents=True)
    (source_dir / "qstash.py").write_text(source, encoding="utf-8")
    (docs_dir / "qstash.md").write_text(docs_page, encoding="utf-8")
    return str(source_dir), str(tmp_path / "docs")


def test_a_key_documented_only_on_a_provider_page_satisfies_the_gate(tmp_path: Path) -> None:
    source_dir, docs_dir = _write_tree(
        tmp_path,
        'api_key = get_secret_str("QSTASH_ALPHA")\n',
        "# Qstash\n\nSet `QSTASH_ALPHA` to your API key.\n",
    )
    assert gate.undocumented_env_keys(source_dir, docs_dir) == frozenset()


def test_a_key_documented_on_no_page_at_all_fails_the_gate(tmp_path: Path) -> None:
    source_dir, docs_dir = _write_tree(
        tmp_path,
        'api_key = get_secret_str("QSTASH_ALPHA")\n',
        "# Qstash\n\nThis provider needs an API key.\n",
    )
    assert gate.undocumented_env_keys(source_dir, docs_dir) == {"QSTASH_ALPHA"}


def test_only_documentation_pages_are_scanned_for_mentions(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("QSTASH_ALPHA\n", encoding="utf-8")
    (tmp_path / "example.py").write_text('get_secret("QSTASH_BRAVO")\n', encoding="utf-8")
    assert gate.collect_documented_keys(str(tmp_path)) == frozenset()
