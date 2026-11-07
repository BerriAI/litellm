import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[2]
CI_CD: Final = ROOT / "ci_cd"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, CI_CD / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard: Final = _load("cost_map_guard")

MAP_FILES: Final = (guard.COST_MAP_PATH,)
BOT_REF: Final = "litellm_cost_map_sync_2026-09-04T12-00Z"


def _entry(price: float = 1e-06, **extra: object) -> dict[str, object]:
    return {
        "input_cost_per_token": price,
        "output_cost_per_token": price * 2,
        "litellm_provider": "openrouter",
        "mode": "chat",
        "max_tokens": 4096,
        **extra,
    }


BASE_MAP: Final = {
    "sample_spec": {"input_cost_per_token": "USD per prompt token"},
    "fallback_generalizations": {"rules": [{"name": "r", "pattern": "^x"}]},
    "openrouter/a": _entry(supports_vision=True),
    "openrouter/b": _entry(2e-06),
}


def _serialize(cost_map: dict[str, object]) -> str:
    return json.dumps(cost_map, indent=4, ensure_ascii=False) + "\n"


NO_PINS: Final = "{}\n"


def _snapshot(cost_map: dict[str, object], backup: str | None = None, pins: str = NO_PINS) -> object:
    text = _serialize(cost_map)
    return guard.Snapshot(cost_map=text, backup=text if backup is None else backup, pins=pins)


BASE: Final = _snapshot(BASE_MAP)
UP_TO_DATE: Final = object()


def _failures(
    head: object,
    changed_files: tuple[str, ...] = MAP_FILES,
    bot: bool = True,
    evaluation_output_price_sources: Mapping[str, str] = guard.EVALUATION_OUTPUT_PRICE_SOURCES,
    base: object = BASE,
    merged: object = UP_TO_DATE,
) -> tuple[str, ...]:
    return guard.guard_failures(
        base,
        head,
        head if merged is UP_TO_DATE else merged,
        changed_files,
        bot,
        evaluation_output_price_sources=evaluation_output_price_sources,
    )


def test_in_sync_files_pass_for_humans_and_bots() -> None:
    assert _failures(BASE, bot=False) == ()
    assert _failures(BASE, bot=True) == ()


EVALUATION_MODEL: Final = "azure_ai/Microsoft-Decision-1"
DECISIONS_MODEL: Final = "azure_ai/decisions/Microsoft-Decision-1"
INPUT_ONLY_OUTPUT_PRICE_FAILURE: Final = (
    "mode evaluation and decisions rows bill input only; output_cost_per_token must be 0 unless the vendor prices "
    "output tokens for this model (add the source to EVALUATION_OUTPUT_PRICE_SOURCES)"
)


@pytest.mark.parametrize(("model", "mode"), ((EVALUATION_MODEL, "evaluation"), (DECISIONS_MODEL, "decisions")))
@pytest.mark.parametrize("bot", (True, False))
def test_evaluation_and_decisions_rows_with_output_price_fail_for_humans_and_bots(
    model: str, mode: str, bot: bool
) -> None:
    head: Final = _snapshot({**BASE_MAP, model: _entry(mode=mode, output_cost_per_token=4.2e-08)})
    assert _failures(head, bot=bot) == (f"{model}: {INPUT_ONLY_OUTPUT_PRICE_FAILURE}",)


@pytest.mark.parametrize(("model", "mode"), ((EVALUATION_MODEL, "evaluation"), (DECISIONS_MODEL, "decisions")))
def test_evaluation_and_decisions_rows_with_zero_or_missing_output_price_pass(model: str, mode: str) -> None:
    zero_output: Final = _snapshot({**BASE_MAP, model: _entry(mode=mode, output_cost_per_token=0.0)})
    missing_output_entry: Final = {
        key: value for key, value in _entry(mode=mode).items() if key != "output_cost_per_token"
    }
    missing_output: Final = _snapshot({**BASE_MAP, model: missing_output_entry})

    assert _failures(zero_output) == ()
    assert _failures(missing_output) == ()


def test_chat_rows_with_output_price_pass() -> None:
    head: Final = _snapshot(
        {**BASE_MAP, "openrouter/chat-with-output": _entry(output_cost_per_token=4.2e-08)}
    )
    assert _failures(head) == ()


@pytest.mark.parametrize(("model", "mode"), ((EVALUATION_MODEL, "evaluation"), (DECISIONS_MODEL, "decisions")))
def test_evaluation_and_decisions_output_price_source_allowlist_passes(model: str, mode: str) -> None:
    head: Final = _snapshot({**BASE_MAP, model: _entry(mode=mode, output_cost_per_token=4.2e-08)})
    source: Final = 'https://vendor.example/pricing "Output tokens are charged per token."'

    assert _failures(head, evaluation_output_price_sources={model: source}) == ()


def test_bot_may_add_and_reprice_models() -> None:
    head = _snapshot({**BASE_MAP, "openrouter/a": _entry(9e-06, supports_vision=True), "openrouter/c": _entry()})
    assert _failures(head) == ()


def test_broken_json_is_reported() -> None:
    head = guard.Snapshot(cost_map="{not json", backup="{not json", pins=NO_PINS)
    assert _failures(head, bot=False) == (
        f"{guard.COST_MAP_PATH} is not valid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
    )


def test_non_object_root_is_reported() -> None:
    head = guard.Snapshot(cost_map="[]", backup="[]", pins=NO_PINS)
    assert _failures(head, bot=False) == (f"{guard.COST_MAP_PATH} must be a JSON object at the root",)


def test_backup_drift_is_reported() -> None:
    head = _snapshot(BASE_MAP, backup=_serialize({**BASE_MAP, "openrouter/b": _entry(3e-06)}))
    assert [failure for failure in _failures(head, bot=False) if failure.startswith(guard.BACKUP_PATH)]


STALE_HEAD: Final = _snapshot(BASE_MAP, backup=_serialize({**BASE_MAP, "openrouter/b": _entry(3e-06)}))
CODE_ONLY: Final = (
    "litellm/utils.py",
    "tests/test_litellm/test_utils.py",
    "docs/model_prices_and_context_window.json",
)


def test_human_pr_that_leaves_the_cost_map_alone_skips_the_file_checks() -> None:
    unparseable: Final = guard.Snapshot(cost_map="{not json", backup="", pins=NO_PINS)
    assert _failures(STALE_HEAD, changed_files=CODE_ONLY, bot=False) == ()
    assert _failures(STALE_HEAD, changed_files=(), bot=False) == ()
    assert _failures(unparseable, changed_files=CODE_ONLY, bot=False) == ()


@pytest.mark.parametrize("checked_path", guard.CHECKED_PATHS)
def test_touching_any_cost_map_file_keeps_the_file_checks(checked_path: str) -> None:
    failures: Final = _failures(STALE_HEAD, changed_files=(*CODE_ONLY, checked_path), bot=False)
    assert [failure for failure in failures if failure.startswith(guard.BACKUP_PATH)]


def test_bot_pr_always_gets_the_file_checks() -> None:
    failures: Final = _failures(STALE_HEAD, changed_files=CODE_ONLY, bot=True)
    assert [failure for failure in failures if failure.startswith(guard.BACKUP_PATH)]


def test_contract_names_the_skip() -> None:
    assert guard.contract_for(False, CODE_ONLY) == "human PR, cost map untouched"
    assert guard.contract_for(False, (*CODE_ONLY, guard.COST_MAP_PATH)) == "human PR, file checks only"
    assert guard.contract_for(True, CODE_ONLY) == "bot contract enforced"


def test_bot_may_only_touch_the_cost_map_files() -> None:
    changed = (*guard.GUARDED_PATHS, "litellm/utils.py", ".github/workflows/cost-map-guard.yml")
    assert _failures(BASE, changed_files=changed, bot=False) == ()
    assert _failures(BASE, changed_files=changed) == (
        "bot PRs may only change the cost map files, not litellm/utils.py",
        "bot PRs may only change the cost map files, not .github/workflows/cost-map-guard.yml",
    )


def test_bot_may_not_remove_models() -> None:
    head = _snapshot({key: value for key, value in BASE_MAP.items() if key != "openrouter/b"})
    assert _failures(head, bot=False) == ()
    assert _failures(head) == ("bot PRs may not remove models: openrouter/b",)


def test_bot_may_not_remove_fields() -> None:
    head = _snapshot({**BASE_MAP, "openrouter/a": _entry()})
    assert _failures(head, bot=False) == ()
    assert _failures(head) == ("bot PRs may not remove fields: openrouter/a.supports_vision",)


def test_bot_may_not_change_special_root_keys() -> None:
    head = _snapshot({**BASE_MAP, "fallback_generalizations": {"rules": []}})
    assert _failures(head, bot=False) == ()
    assert _failures(head) == ("bot PRs may not change fallback_generalizations",)


PINNED_SOURCE: Final = "a 512x512 probe image was read back over the provider's chat endpoint"
PINS: Final = _serialize(
    {"openrouter/a": {"supports_vision": {"value": True, "checked": "2026-09-26", "source": PINNED_SOURCE}}}
)


def test_a_pinned_capability_that_holds_passes_for_humans_and_bots() -> None:
    head = _snapshot(BASE_MAP, pins=PINS)
    assert _failures(head, bot=False) == ()
    assert _failures(head, bot=True) == ()


LOWERED: Final = (
    f"{guard.COST_MAP_PATH}: openrouter/a.supports_vision must stay true (verified 2026-09-26: {PINNED_SOURCE}); "
    f"change {guard.PINS_PATH} with new evidence first"
)
NO_VISION_MAP: Final = {**BASE_MAP, "openrouter/a": _entry(supports_vision=False)}


@pytest.mark.parametrize("bot", [False, True])
def test_lowering_a_pinned_capability_fails_with_its_evidence(bot: bool) -> None:
    assert _failures(_snapshot(NO_VISION_MAP, pins=PINS), bot=bot) == (LOWERED,)


def test_dropping_a_pinned_field_or_model_is_reported() -> None:
    no_field = _snapshot({**BASE_MAP, "openrouter/a": _entry()}, pins=PINS)
    assert [failure for failure in _failures(no_field, bot=False) if "must stay true" in failure]
    no_model = _snapshot({key: value for key, value in BASE_MAP.items() if key != "openrouter/a"}, pins=PINS)
    assert _failures(no_model, bot=False) == (
        f"{guard.PINS_PATH} pins openrouter/a.supports_vision but {guard.COST_MAP_PATH} has no openrouter/a entry",
    )


@pytest.mark.parametrize(
    ("pins", "expected"),
    [
        (
            "{not json",
            f"{guard.PINS_PATH} is not valid JSON: Expecting property name enclosed in double quotes: "
            "line 1 column 2 (char 1)",
        ),
        (
            _serialize({"openrouter/a": {"supports_vision": {"value": True}}}),
            f"{guard.PINS_PATH}: openrouter/a.supports_vision needs value, checked, and source",
        ),
        (_serialize({"openrouter/a": True}), f"{guard.PINS_PATH}: openrouter/a must map field names to pins"),
    ],
)
def test_malformed_pins_are_reported(pins: str, expected: str) -> None:
    assert _failures(_snapshot(BASE_MAP, pins=pins), bot=False) == (expected,)


MISSING_PINS: Final = f"{guard.PINS_PATH} is missing; restore it, its pins were verified against live provider calls"


@pytest.mark.parametrize("bot", [False, True])
def test_a_pr_that_deletes_the_pins_file_fails(bot: bool) -> None:
    head: Final = _snapshot(BASE_MAP, pins="")
    assert _failures(head, bot=bot, base=_snapshot(BASE_MAP, pins=PINS)) == (MISSING_PINS,)


@pytest.mark.parametrize("bot", [False, True])
def test_pin_checks_are_skipped_when_no_side_of_the_merge_has_the_pins_file(bot: bool) -> None:
    assert _failures(_snapshot(NO_VISION_MAP, pins=""), bot=bot, base=_snapshot(NO_VISION_MAP, pins="")) == ()


@pytest.mark.parametrize("bot", [False, True])
def test_a_branch_cut_before_a_pin_was_added_cannot_lower_it(bot: bool) -> None:
    head: Final = _snapshot(NO_VISION_MAP, pins="")
    merged: Final = _snapshot(NO_VISION_MAP, pins=PINS)
    assert _failures(head, bot=bot, base=_snapshot(BASE_MAP, pins=""), merged=merged) == (LOWERED,)


@pytest.mark.parametrize("bot", [False, True])
def test_a_branch_cut_before_a_pin_was_added_passes_when_merging_keeps_the_pinned_value(bot: bool) -> None:
    repriced: Final = {"openrouter/b": _entry(3e-06)}
    head: Final = _snapshot({**NO_VISION_MAP, **repriced}, pins="")
    merged: Final = _snapshot({**BASE_MAP, **repriced}, pins=PINS)
    assert _failures(head, bot=bot, base=_snapshot(NO_VISION_MAP, pins=""), merged=merged) == ()


def test_a_pr_that_does_not_merge_cleanly_is_told_to_merge_the_base_branch_in() -> None:
    assert _failures(BASE, bot=False, merged=None) == (
        "the pull request does not merge cleanly into its base branch; merge the base branch in first",
    )


def test_a_merge_result_that_is_not_valid_json_is_reported() -> None:
    merged: Final = guard.Snapshot(cost_map="{broken", backup="{broken", pins=PINS)
    assert _failures(BASE, bot=False, merged=merged) == (
        f"after merging into the base branch, {guard.COST_MAP_PATH} is not valid JSON: "
        "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
    )


def test_a_pr_that_adds_the_pins_file_gets_its_pins_checked() -> None:
    head: Final = _snapshot({**BASE_MAP, "openrouter/a": _entry(supports_vision=False)}, pins=PINS)
    changed: Final = (*MAP_FILES, guard.PINS_PATH)
    failures: Final = _failures(head, changed_files=changed, bot=False, base=_snapshot(BASE_MAP, pins=""))
    assert [failure for failure in failures if "openrouter/a.supports_vision must stay true" in failure]


def test_bot_may_not_touch_the_pins_file() -> None:
    head = _snapshot(BASE_MAP, pins=PINS)
    changed = (*guard.GUARDED_PATHS, guard.PINS_PATH)
    assert _failures(head, changed_files=changed, bot=False) == ()
    assert _failures(head, changed_files=changed) == (
        f"bot PRs may only change the cost map files, not {guard.PINS_PATH}",
    )


PRICE_SYNC_PIN_FAILURE: Final = (
    f"price sync PRs may not change {guard.PINS_PATH}; a pin moves only in a human PR with new evidence"
)
BOT_AUTHOR_PIN_FAILURE: Final = (
    f"PRs opened by a bot may not change {guard.PINS_PATH}; a pin moves only in a human PR with new evidence"
)


@pytest.mark.parametrize(
    ("head_ref", "author_type", "changed", "expected"),
    [
        ("litellm-providers/price-sync-fireworks-ai", "User", (*MAP_FILES, guard.PINS_PATH), (PRICE_SYNC_PIN_FAILURE,)),
        ("litellm-providers/price-sync-fireworks-ai", "Bot", MAP_FILES, ()),
        ("litellm_azure_ai_mai_image_2_5_pro", "Bot", (*MAP_FILES, guard.PINS_PATH), (BOT_AUTHOR_PIN_FAILURE,)),
        ("litellm_azure_ai_mai_image_2_5_pro", "Bot", MAP_FILES, ()),
        ("fix-openrouter-price-sync", "User", (*MAP_FILES, guard.PINS_PATH), ()),
        ("fix-openrouter-price-sync", "", (*MAP_FILES, guard.PINS_PATH), ()),
    ],
)
def test_only_human_opened_non_price_sync_prs_may_touch_the_pins_file(
    head_ref: str, author_type: str, changed: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert guard.pin_author_failures(head_ref, author_type, changed) == expected


def test_checked_in_pins_hold_in_the_checked_in_cost_map() -> None:
    cost_map = json.loads((ROOT / guard.COST_MAP_PATH).read_text())
    pins = (ROOT / guard.PINS_PATH).read_text()
    assert json.loads(pins), "the pins file must pin at least one capability"
    assert guard.pin_failures(pins, cost_map) == ()


def _commit(repo: Path, cost_map: dict[str, object], message: str, pins: str | None = None) -> str:
    text = _serialize(cost_map)
    pins_file = repo / guard.PINS_PATH
    if pins is not None or not pins_file.exists():
        pins_file.parent.mkdir(exist_ok=True)
        pins_file.write_text(NO_PINS if pins is None else pins)
    (repo / guard.COST_MAP_PATH).write_text(text)
    (repo / guard.BACKUP_PATH).parent.mkdir(exist_ok=True)
    (repo / guard.BACKUP_PATH).write_text(text)
    return _git_commit(repo, message)


def _commit_code_only(repo: Path, message: str) -> str:
    (repo / "litellm").mkdir(exist_ok=True)
    (repo / "litellm" / "utils.py").write_text(f"print('{message}')\n")
    return _git_commit(repo, message)


def _git_commit(repo: Path, message: str) -> str:
    subprocess.run(("git", "add", "-A"), cwd=repo, check=True)
    subprocess.run(
        ("git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", message),
        cwd=repo,
        check=True,
    )
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _run_guard(
    repo: Path,
    base: str,
    head: str,
    head_ref: str,
    *,
    target: str | None = None,
    target_env: str | None = None,
    author_type: str | None = None,
) -> subprocess.CompletedProcess[str]:
    target_args: Final = () if target_env is not None else ("--target", target or base)
    env_overrides: Final = {
        **({} if target_env is None else {guard.TARGET_ENV: target_env}),
        **({} if author_type is None else {guard.AUTHOR_TYPE_ENV: author_type}),
    }
    return subprocess.run(
        (
            sys.executable,
            str(CI_CD / "cost_map_guard.py"),
            "--base",
            base,
            "--head",
            head,
            *target_args,
            "--head-ref",
            head_ref,
        ),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env_overrides},
    )


@pytest.mark.parametrize(
    ("head_ref", "expected_code", "expected_line"),
    [
        (BOT_REF, 1, "- bot PRs may not remove models: openrouter/b"),
        ("litellm_fix_pricing", 0, "cost map guard passed (human PR, file checks only)"),
    ],
)
def test_main_reads_both_revisions_from_git(
    tmp_path: Path, head_ref: str, expected_code: int, expected_line: str
) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base = _commit(tmp_path, BASE_MAP, "base")
    head = _commit(tmp_path, {key: value for key, value in BASE_MAP.items() if key != "openrouter/b"}, "head")
    result = _run_guard(tmp_path, base, head, head_ref)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert expected_line in result.stdout.splitlines()


def test_main_skips_the_file_checks_on_a_stale_base_the_pr_never_touched(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    _commit(tmp_path, BASE_MAP, "base")
    (tmp_path / guard.BACKUP_PATH).write_text(_serialize({**BASE_MAP, "openrouter/b": _entry(3e-06)}))
    stale_base: Final = _commit_code_only(tmp_path, "stale base with drifted backup")
    head: Final = _commit_code_only(tmp_path, "code change on the stale base")
    human: Final = _run_guard(tmp_path, stale_base, head, "litellm_fix_pricing")
    assert human.returncode == 0, human.stdout + human.stderr
    assert "cost map guard passed (human PR, cost map untouched)" in human.stdout.splitlines()
    bot: Final = _run_guard(tmp_path, stale_base, head, BOT_REF)
    assert bot.returncode == 1
    backup_failure: Final = f"- {guard.BACKUP_PATH} differs from {guard.COST_MAP_PATH}; copy the root file over it"
    assert backup_failure in bot.stdout.splitlines()


def test_main_keeps_the_file_checks_when_a_cost_map_file_is_renamed(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base: Final = _commit(tmp_path, BASE_MAP, "base")
    subprocess.run(("git", "mv", guard.COST_MAP_PATH, "renamed.json"), cwd=tmp_path, check=True)
    head: Final = _git_commit(tmp_path, "rename the cost map")
    result: Final = _run_guard(tmp_path, base, head, "litellm_fix_pricing")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cost map guard failed (human PR, file checks only):" in result.stdout.splitlines()


def test_main_fails_when_the_changed_files_cannot_be_read(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    head: Final = _commit(tmp_path, BASE_MAP, "head")
    result: Final = _run_guard(tmp_path, "0" * 40, head, "litellm_fix_pricing")
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith("cost map guard failed: git diff ")


def test_main_rejects_a_bot_pr_that_edits_code(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base = _commit(tmp_path, BASE_MAP, "base")
    (tmp_path / "litellm" / "utils.py").write_text("print('hi')\n")
    head = _commit(tmp_path, {**BASE_MAP, "openrouter/c": _entry()}, "head")
    assert _run_guard(tmp_path, base, head, BOT_REF).returncode == 1
    assert _run_guard(tmp_path, base, head, "litellm_fix_pricing").returncode == 0


@pytest.mark.parametrize("head_ref", [BOT_REF, "litellm_fix_pricing"])
def test_main_reads_the_pins_from_the_head_revision(tmp_path: Path, head_ref: str) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base: Final = _commit(tmp_path, BASE_MAP, "base", pins=PINS)
    head: Final = _commit(tmp_path, {**BASE_MAP, "openrouter/a": _entry(supports_vision=False)}, "sync lowers it")
    result: Final = _run_guard(tmp_path, base, head, head_ref)
    assert result.returncode == 1, result.stdout + result.stderr
    pin_line: Final = f"- {guard.COST_MAP_PATH}: openrouter/a.supports_vision must stay true (verified 2026-09-26: "
    assert [line for line in result.stdout.splitlines() if line.startswith(pin_line)]


@pytest.mark.parametrize("head_ref", [BOT_REF, "litellm_fix_pricing"])
def test_main_holds_a_pin_the_base_branch_added_after_the_pr_branched(tmp_path: Path, head_ref: str) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    merge_base: Final = _commit(tmp_path, BASE_MAP, "merge base")
    target: Final = _commit(tmp_path, BASE_MAP, "pin on the base branch", pins=PINS)
    subprocess.run(("git", "checkout", "-q", merge_base), cwd=tmp_path, check=True)
    head: Final = _commit(tmp_path, NO_VISION_MAP, "branch cut before the pin lowers it", pins=NO_PINS)
    result: Final = _run_guard(tmp_path, merge_base, head, head_ref, target_env=target)
    assert result.returncode == 1, result.stdout + result.stderr
    pin_line: Final = f"- {guard.COST_MAP_PATH}: openrouter/a.supports_vision must stay true (verified 2026-09-26: "
    assert [line for line in result.stdout.splitlines() if line.startswith(pin_line)]


def test_main_reports_a_failed_merge_apart_from_a_conflict(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base: Final = _commit(tmp_path, BASE_MAP, "base")
    head: Final = _commit(tmp_path, {**BASE_MAP, "openrouter/c": _entry()}, "head")
    result: Final = _run_guard(tmp_path, base, head, "litellm_fix_pricing", target="0" * 40)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith(f"cost map guard failed: git merge-tree into {'0' * 40} failed: ")


def test_main_tells_a_conflicting_pr_to_merge_the_base_branch_in(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    merge_base: Final = _commit(tmp_path, BASE_MAP, "merge base")
    target: Final = _commit(tmp_path, {**BASE_MAP, "openrouter/b": _entry(3e-06)}, "base branch reprices b")
    subprocess.run(("git", "checkout", "-q", merge_base), cwd=tmp_path, check=True)
    head: Final = _commit(tmp_path, {**BASE_MAP, "openrouter/b": _entry(4e-06)}, "pr reprices b differently")
    result: Final = _run_guard(tmp_path, merge_base, head, "litellm_fix_pricing", target=target)
    assert result.returncode == 1, result.stdout + result.stderr
    conflict_line: Final = "- the pull request does not merge cleanly into its base branch; merge the base branch in first"
    assert conflict_line in result.stdout.splitlines()


def test_main_skips_the_merge_for_a_pr_that_leaves_the_cost_map_alone(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base: Final = _commit(tmp_path, BASE_MAP, "base")
    head: Final = _commit_code_only(tmp_path, "code change")
    result: Final = _run_guard(tmp_path, base, head, "litellm_fix_pricing", target="0" * 40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["cost map guard passed (human PR, cost map untouched)"]


@pytest.mark.parametrize(("author_type", "expected_code"), [("Bot", 1), ("User", 0)])
def test_main_lets_only_a_human_opened_pr_change_the_pins(tmp_path: Path, author_type: str, expected_code: int) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base: Final = _commit(tmp_path, BASE_MAP, "base", pins=PINS)
    head: Final = _commit(tmp_path, NO_VISION_MAP, "lower the flag and drop its pin", pins=NO_PINS)
    result: Final = _run_guard(
        tmp_path, base, head, "litellm_azure_ai_mai_image_2_5_pro", author_type=author_type
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert (f"- {BOT_AUTHOR_PIN_FAILURE}" in result.stdout.splitlines()) == (author_type == "Bot")
