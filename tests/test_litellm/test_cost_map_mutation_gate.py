"""Unit tests for scripts/cost_map_mutation_gate.py."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[2]
GATE_PATH: Final = ROOT / "scripts" / "cost_map_mutation_gate.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cost_map_mutation_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cost_map_mutation_gate"] = module
    spec.loader.exec_module(module)
    return module


gate: Final = _load()


def _entry() -> dict[str, object]:
    return {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "max_tokens": 4096,
        "max_input_tokens": 3000,
        "max_output_tokens": 1000,
        "search_context_cost_per_query": {"search_context_size_low": 0.01},
        "tiered": [{"input_cost_per_token": 5e-06}],
        "supports_vision": True,
    }


BASE_MAP: Final = {
    "sample_spec": {"input_cost_per_token": "USD per prompt token"},
    "fallback_generalizations": {"rules": [{"name": "r", "pattern": "^x"}]},
    "openrouter/a": _entry(),
}


def test_mutation_scales_cost_fields_including_nested() -> None:
    mutated: Final = gate.mutate_cost_map(BASE_MAP)
    entry: Final = mutated["openrouter/a"]
    assert entry["input_cost_per_token"] == pytest.approx(1e-06 * 1.37)
    assert entry["output_cost_per_token"] == pytest.approx(2e-06 * 1.37)
    assert entry["search_context_cost_per_query"]["search_context_size_low"] == pytest.approx(0.01 * 1.37)
    assert entry["tiered"][0]["input_cost_per_token"] == pytest.approx(5e-06 * 1.37)


def test_mutation_adds_deprecation_date_and_bumps_limits() -> None:
    mutated: Final = gate.mutate_cost_map(BASE_MAP)
    entry: Final = mutated["openrouter/a"]
    assert entry["deprecation_date"] == "2030-01-01"
    assert entry["max_tokens"] == 4096 + 1000
    assert entry["max_input_tokens"] == 3000 + 1000
    assert entry["max_output_tokens"] == 1000 + 1000
    assert entry["supports_vision"] is True
    assert entry["mode"] == "chat"


def test_mutation_leaves_non_model_root_keys_untouched() -> None:
    mutated: Final = gate.mutate_cost_map(BASE_MAP)
    assert mutated["sample_spec"] == BASE_MAP["sample_spec"]
    assert mutated["fallback_generalizations"] == BASE_MAP["fallback_generalizations"]


def test_mutation_preserves_key_order() -> None:
    assert tuple(gate.mutate_cost_map(BASE_MAP)) == tuple(BASE_MAP)


def test_changed_test_files_filters_conftest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate,
        "_run",
        lambda cmd, cwd=gate.REPO_ROOT: (
            "tests/test_litellm/test_a.py\n"
            "tests/test_litellm/conftest.py\n"
            "tests/test_litellm/llms/conftest.py\n"
            "tests/test_litellm/llms/test_b.py\n"
            "litellm/utils.py\n"
        ),
    )
    assert gate._changed_test_files("BASE") == (
        "tests/test_litellm/test_a.py",
        "tests/test_litellm/llms/test_b.py",
    )


def test_dirty_cost_map_refuses(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(gate, "_run", lambda cmd, cwd=gate.REPO_ROOT: " M model_prices_and_context_window.json\n")
    assert gate.main(["tests/test_litellm/test_a.py"]) == 2
    assert "Refusing to run" in capsys.readouterr().err


def test_no_files_selected_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert gate.main([]) == 0
    assert "nothing to gate" in capsys.readouterr().out


def test_pytest_command_adds_workers_only_when_absent() -> None:
    without_n: Final = gate._pytest_command(("a.py",), ())
    assert "-n" in without_n and without_n[without_n.index("-n") + 1] == "4"
    with_n: Final = gate._pytest_command(("a.py",), ("-n", "8"))
    assert list(with_n).count("-n") == 1 and with_n[with_n.index("-n") + 1] == "8"


def test_serialized_mutation_round_trips() -> None:
    text: Final = gate._serialize(gate.mutate_cost_map(BASE_MAP))
    parsed: Final = json.loads(text)
    assert parsed["openrouter/a"]["deprecation_date"] == "2030-01-01"
    assert parsed["sample_spec"] == BASE_MAP["sample_spec"]
    assert text.endswith("\n")
