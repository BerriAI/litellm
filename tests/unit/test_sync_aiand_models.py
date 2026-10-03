import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "sync_aiand_models.py"

_spec = importlib.util.spec_from_file_location("sync_aiand_models", SCRIPT)
assert _spec is not None and _spec.loader is not None
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def _model(**overrides: object) -> dict[str, object]:
    model: dict[str, object] = {
        "id": "acme/chat-1",
        "name": "Chat 1",
        "family": "chat",
        "reasoning": False,
        "tool_call": True,
        "structured_output": True,
        "temperature": True,
        "attachment": False,
        "open_weights": False,
        "cost": {"input": 1.0, "output": 2.0, "cache_read": 0.5},
        "limit": {"context": 8192, "output": 1024},
        "modalities": {"input": ["text"]},
    }
    model.update(overrides)
    return model


def _spec_json(*models: dict[str, object]) -> bytes:
    payload: dict[str, object] = {"aiand": {"models": {model["id"]: model for model in models}}}
    return json.dumps(payload).encode()


@pytest.mark.parametrize(
    ("per_million", "expected"),
    [
        (3, 3e-06),
        (15, 1.5e-05),
        (1.4, 1.4e-06),
        (0.25999999999999995, 2.6e-07),
        (0.060000000000000005, 6e-08),
        (1.0399999999999998, 1.04e-06),
        (0, 0.0),
    ],
)
def test_per_token_normalizes_float_artifacts(per_million: float, expected: float) -> None:
    assert sync.per_token(per_million) == expected


def test_load_spec_raises_on_shape_change() -> None:
    with pytest.raises(sync.SyncError):
        sync.load_spec(b'{"aiand": {"models": [{"id": "x"}]}}')


def test_load_spec_raises_when_no_models_remain() -> None:
    with pytest.raises(sync.SyncError):
        sync.load_spec(b'{"aiand": {"models": {}}}')


def test_load_spec_raises_when_id_mismatches_key() -> None:
    raw = json.dumps({"aiand": {"models": {"acme/chat-1": _model(id="acme/other")}}}).encode()
    with pytest.raises(sync.SyncError):
        sync.load_spec(raw)


def test_added_model_lands_in_cost_map_with_expected_fields() -> None:
    spec = sync.load_spec(_spec_json(_model()))
    outcome = sync.compute_sync({}, spec)
    entry = outcome.cost_map["aiand/acme/chat-1"]
    assert entry["litellm_provider"] == "aiand"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == 1e-06
    assert entry["output_cost_per_token"] == 2e-06
    assert entry["cache_read_input_token_cost"] == 5e-07
    assert entry["max_input_tokens"] == 8192
    assert entry["max_output_tokens"] == 1024
    assert entry["max_tokens"] == 1024
    assert entry["supports_function_calling"] is True
    assert entry["supports_parallel_function_calling"] is True
    assert entry["supports_tool_choice"] is True
    assert entry["supports_response_schema"] is True
    assert entry["supports_reasoning"] is False
    assert entry["supports_vision"] is False
    assert entry["source"] == "https://api.aiand.com/v1/api.json"
    assert entry["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
    assert outcome.added == ("aiand/acme/chat-1",)
    assert outcome.has_changes is True


def test_updated_price_is_detected_and_rendered() -> None:
    baseline = sync.compute_sync({}, sync.load_spec(_spec_json(_model())))
    changed = _model(cost={"input": 3.0, "output": 2.0, "cache_read": 0.5})
    outcome = sync.compute_sync(baseline.cost_map, sync.load_spec(_spec_json(changed)))
    assert outcome.added == ()
    assert len(outcome.updated) == 1
    assert outcome.updated[0].startswith("aiand/acme/chat-1:")
    assert "input_cost_per_token" in outcome.updated[0]
    assert outcome.cost_map["aiand/acme/chat-1"]["input_cost_per_token"] == 3e-06
    assert outcome.has_changes is True


def test_removed_model_is_stamped_and_counted_as_updated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sync, "_today", lambda: "2026-10-03")
    baseline = sync.compute_sync({}, sync.load_spec(_spec_json(_model(), _model(id="acme/chat-2", name="Chat 2"))))
    remaining = sync.load_spec(_spec_json(_model()))
    outcome = sync.compute_sync(baseline.cost_map, remaining)
    assert outcome.added == ()
    assert outcome.removed == ("aiand/acme/chat-2",)
    assert len(outcome.updated) == 1
    assert outcome.updated[0].startswith("aiand/acme/chat-2:")
    assert "absent_from_spec_since" in outcome.updated[0]
    assert outcome.cost_map["aiand/acme/chat-2"]["metadata"] == {"absent_from_spec_since": "2026-10-03"}
    assert any("aiand/acme/chat-2" in warning and "human" in warning for warning in outcome.warnings)
    assert outcome.has_changes is True


def test_already_stamped_absent_model_keeps_the_earliest_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sync, "_today", lambda: "2026-10-03")
    baseline = sync.compute_sync({}, sync.load_spec(_spec_json(_model(), _model(id="acme/chat-2", name="Chat 2"))))
    stamped_entry = dict(baseline.cost_map["aiand/acme/chat-2"])
    stamped_entry["metadata"] = {"absent_from_spec_since": "2026-09-01"}
    registry = {**baseline.cost_map, "aiand/acme/chat-2": stamped_entry}
    outcome = sync.compute_sync(registry, sync.load_spec(_spec_json(_model())))
    assert outcome.removed == ("aiand/acme/chat-2",)
    assert outcome.updated == ()
    assert outcome.cost_map["aiand/acme/chat-2"]["metadata"] == {"absent_from_spec_since": "2026-09-01"}
    assert outcome.has_changes is True


def test_reappeared_model_clears_the_stamp_and_counts_as_updated() -> None:
    spec = sync.load_spec(_spec_json(_model()))
    baseline = sync.compute_sync({}, spec).cost_map
    stamped_entry = dict(baseline["aiand/acme/chat-1"])
    stamped_entry["metadata"] = {"absent_from_spec_since": "2026-09-01"}
    registry = {**baseline, "aiand/acme/chat-1": stamped_entry}
    outcome = sync.compute_sync(registry, spec)
    assert outcome.added == ()
    assert outcome.removed == ()
    assert len(outcome.updated) == 1
    assert outcome.updated[0].startswith("aiand/acme/chat-1:")
    assert "absent_from_spec_since" in outcome.updated[0]
    assert "metadata" not in outcome.cost_map["aiand/acme/chat-1"]
    assert outcome.has_changes is True


def test_updated_entry_preserves_the_absence_marker_as_an_extra() -> None:
    model = sync.load_spec(_spec_json(_model()))["acme/chat-1"]
    entry = {**sync._new_entry(model), "metadata": {"absent_from_spec_since": "2026-09-01"}}
    new_entry, changes = sync._updated_entry(entry, model)
    assert new_entry["metadata"] == {"absent_from_spec_since": "2026-09-01"}
    assert changes == ()


def test_parallel_function_calling_follows_tool_call() -> None:
    spec = sync.load_spec(_spec_json(_model(tool_call=False)))
    outcome = sync.compute_sync({}, spec)
    entry = outcome.cost_map["aiand/acme/chat-1"]
    assert entry["supports_function_calling"] is False
    assert entry["supports_parallel_function_calling"] is False
    assert entry["supports_tool_choice"] is False


def test_new_keys_land_at_the_end_of_the_provider_block() -> None:
    registry = {
        "aaa": {},
        "aiand/acme/chat-1": {},
        "zzz": {},
    }
    spec = sync.load_spec(_spec_json(_model(), _model(id="acme/new", name="New")))
    outcome = sync.compute_sync(registry, spec)
    assert list(outcome.cost_map) == ["aaa", "aiand/acme/chat-1", "aiand/acme/new", "zzz"]


def test_pr_body_renders_none_placeholders_for_empty_sections() -> None:
    outcome = sync.compute_sync({}, sync.load_spec(_spec_json(_model())))
    body = sync.render_pr_body(outcome)
    assert "### Added (1)" in body
    assert "### Updated (0)\n- none" in body
    assert "### Removed from the spec (0)\n- none" in body
    assert sync.render_summary(outcome) == "added=1 updated=0 removed=0 warnings=0"


def test_write_updates_root_and_backup_maps_identically(tmp_path: Path) -> None:
    repo_root = tmp_path
    for relpath in sync.COST_MAP_RELPATHS:
        target = repo_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
    spec_path = repo_root / "spec.json"
    spec_path.write_bytes(_spec_json(_model()))
    pr_body = repo_root / "pr_body.md"
    exit_code = sync.main(
        [
            "--write",
            "--spec-json",
            str(spec_path),
            "--pr-body-file",
            str(pr_body),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert exit_code == 0
    root_map = json.loads((repo_root / sync.COST_MAP_RELPATHS[0]).read_text())
    backup_map = json.loads((repo_root / sync.COST_MAP_RELPATHS[1]).read_text())
    assert root_map == backup_map
    assert "aiand/acme/chat-1" in root_map


def test_removal_only_sync_stamps_and_writes_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync, "_today", lambda: "2026-10-03")
    repo_root = tmp_path
    spec = sync.load_spec(_spec_json(_model(), _model(id="acme/chat-2", name="Chat 2")))
    registry = sync.compute_sync({}, spec).cost_map
    for relpath in sync.COST_MAP_RELPATHS:
        target = repo_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(registry, indent=4) + "\n")
    remaining = repo_root / "remaining.json"
    remaining.write_bytes(_spec_json(_model()))
    pr_body = repo_root / "pr_body.md"
    exit_code = sync.main(
        [
            "--write",
            "--spec-json",
            str(remaining),
            "--pr-body-file",
            str(pr_body),
            "--repo-root",
            str(repo_root),
        ]
    )
    assert exit_code == 0
    assert "updated=1 removed=1 warnings=1" in capsys.readouterr().out
    body = pr_body.read_text()
    assert "### Added (0)" in body
    assert "### Updated (1)" in body
    assert "### Removed from the spec (1)" in body
    assert "`aiand/acme/chat-2`" in body
    assert "### Warnings needing a human call" not in body
    root_map = json.loads((repo_root / sync.COST_MAP_RELPATHS[0]).read_text())
    backup_map = json.loads((repo_root / sync.COST_MAP_RELPATHS[1]).read_text())
    assert root_map == backup_map
    assert root_map["aiand/acme/chat-2"]["metadata"] == {"absent_from_spec_since": "2026-10-03"}
