"""Tests for scripts/dict_usage_gate.py.

The breach/ratchet logic is imported from scripts/type_discipline_gate.py and
pinned by its own tests, so what is pinned here is the glue this gate adds:
parsing the checker's output format, the checker-content cache fingerprint
(a rule-logic change must re-key cached base counts), and the cache-prefix
isolation that keeps this gate's entries and the basedpyright gate's entries
from evicting each other in the shared lint-cache directory.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

_gate_spec = importlib.util.spec_from_file_location("dict_usage_gate", _SCRIPTS / "dict_usage_gate.py")
gate = importlib.util.module_from_spec(_gate_spec)
sys.modules[_gate_spec.name] = gate
_gate_spec.loader.exec_module(gate)

typecheck_env = gate.typecheck_env


def test_parse_violations_reads_absolute_and_relative_paths(tmp_path):
    root = tmp_path.resolve()
    output = "\n".join(
        (
            f"{root}/litellm/a.py:12: DICT001 expression is typed as mutable builtins.dict",
            "litellm/b.py:3: DICT001 expression is typed as mutable collections.defaultdict",
            "litellm/c.py:9: LIT002 mutable collection constructed",
            "",
            "2 DICT001 violation(s)",
        )
    )
    parsed = gate.parse_violations(output, root)
    assert parsed == [
        gate.Violation("litellm/a.py", 12, "DICT001"),
        gate.Violation("litellm/b.py", 3, "DICT001"),
    ]
    assert gate.count_by_rule(parsed) == {"DICT001": 2}


def test_cache_fingerprints_rekey_when_the_checker_changes(tmp_path):
    first = tmp_path / "checker_v1.py"
    second = tmp_path / "checker_v2.py"
    first.write_text("RULES = 1\n")
    second.write_text("RULES = 2\n")
    prints_v1 = gate.cache_fingerprints(first)
    prints_v2 = gate.cache_fingerprints(second)
    assert prints_v1 != prints_v2
    assert prints_v1[:-1] == prints_v2[:-1]
    assert prints_v1[-1].startswith("checker:")


def test_cache_entries_use_the_dict_usage_prefix(tmp_path):
    path = typecheck_env.cache_path(tmp_path, "abc123", ("f",), gate.CACHE_PREFIX)
    assert path.name.startswith("dict-usage-base-")
    assert not path.name.startswith(typecheck_env.CACHE_FILE_PREFIX)


def test_cache_eviction_never_crosses_prefixes(tmp_path):
    foreign = typecheck_env.cache_path(tmp_path, "bp-base", ("f",))
    typecheck_env.store_counts(tmp_path, foreign, "bp-base", {"reportAny": 1})
    for index in range(typecheck_env.CACHE_KEEP_ENTRIES + 3):
        mine = typecheck_env.cache_path(tmp_path, f"base-{index}", ("f",), gate.CACHE_PREFIX)
        typecheck_env.store_counts(tmp_path, mine, f"base-{index}", {"DICT001": index + 1}, gate.CACHE_PREFIX)
    assert foreign.exists()
    dict_entries = list(tmp_path.glob(f"{gate.CACHE_PREFIX}*.json"))
    assert len(dict_entries) == typecheck_env.CACHE_KEEP_ENTRIES


def test_base_counts_cached_round_trips_through_the_disk_cache(tmp_path, monkeypatch):
    calls = []

    def fake_base_counts(ref):
        calls.append(ref)
        return {"DICT001": 7}

    monkeypatch.setattr(gate, "base_counts", fake_base_counts)
    first = gate.base_counts_cached("deadbeef", cache_dir=tmp_path)
    second = gate.base_counts_cached("deadbeef", cache_dir=tmp_path)
    assert first == second == {"DICT001": 7}
    assert calls == ["deadbeef"]
