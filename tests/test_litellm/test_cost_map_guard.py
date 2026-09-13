import importlib.util
import json
import subprocess
import sys
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


schema_module: Final = _load("generate_model_prices_schema")
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


def _snapshot(cost_map: dict[str, object], backup: str | None = None, schema: str | None = None) -> object:
    text = _serialize(cost_map)
    rendered = schema_module.render(schema_module.build_schema(cost_map))
    return guard.Snapshot(
        cost_map=text, backup=text if backup is None else backup, schema=rendered if schema is None else schema
    )


BASE: Final = _snapshot(BASE_MAP)


def _failures(head: object, changed_files: tuple[str, ...] = MAP_FILES, bot: bool = True) -> tuple[str, ...]:
    return guard.guard_failures(BASE, head, changed_files, bot)


def test_in_sync_files_pass_for_humans_and_bots() -> None:
    assert _failures(BASE, bot=False) == ()
    assert _failures(BASE, bot=True) == ()


def test_bot_may_add_and_reprice_models() -> None:
    head = _snapshot({**BASE_MAP, "openrouter/a": _entry(9e-06, supports_vision=True), "openrouter/c": _entry()})
    assert _failures(head) == ()


def test_broken_json_is_reported() -> None:
    head = guard.Snapshot(cost_map="{not json", backup="{not json", schema="{}")
    assert _failures(head, bot=False) == (
        f"{guard.COST_MAP_PATH} is not valid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
    )


def test_non_object_root_is_reported() -> None:
    head = guard.Snapshot(cost_map="[]", backup="[]", schema="{}")
    assert _failures(head, bot=False) == (f"{guard.COST_MAP_PATH} must be a JSON object at the root",)


def test_backup_drift_is_reported() -> None:
    head = _snapshot(BASE_MAP, backup=_serialize({**BASE_MAP, "openrouter/b": _entry(3e-06)}))
    assert [failure for failure in _failures(head, bot=False) if failure.startswith(guard.BACKUP_PATH)]


def test_schema_out_of_sync_is_reported() -> None:
    head = _snapshot({**BASE_MAP, "openrouter/c": _entry(supports_audio_input=True)}, schema=BASE.schema)
    assert [failure for failure in _failures(head, bot=False) if failure.startswith(guard.SCHEMA_PATH)]


def test_schema_validation_errors_are_reported() -> None:
    head = _snapshot({**BASE_MAP, "openrouter/c": _entry(-1e-06)})
    prefix = f"{guard.COST_MAP_PATH} does not validate against its schema: openrouter/c."
    assert [failure.removeprefix(prefix).split(":")[0] for failure in _failures(head, bot=False)] == [
        "input_cost_per_token",
        "output_cost_per_token",
    ]


def test_unclassified_entry_key_is_reported() -> None:
    text = _serialize({**BASE_MAP, "openrouter/c": _entry(weird_thing=1)})
    head = guard.Snapshot(cost_map=text, backup=text, schema=BASE.schema)
    (failure,) = _failures(head, bot=False)
    assert "Unclassified keys" in failure and "weird_thing" in failure


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


def _commit(repo: Path, cost_map: dict[str, object], message: str) -> str:
    text = _serialize(cost_map)
    (repo / guard.COST_MAP_PATH).write_text(text)
    (repo / guard.BACKUP_PATH).parent.mkdir(exist_ok=True)
    (repo / guard.BACKUP_PATH).write_text(text)
    (repo / guard.SCHEMA_PATH).write_text(schema_module.render(schema_module.build_schema(cost_map)))
    subprocess.run(("git", "add", "-A"), cwd=repo, check=True)
    subprocess.run(
        ("git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", message),
        cwd=repo,
        check=True,
    )
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _run_guard(repo: Path, base: str, head: str, head_ref: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(CI_CD / "cost_map_guard.py"), "--base", base, "--head", head, "--head-ref", head_ref),
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
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


def test_main_rejects_a_bot_pr_that_edits_code(tmp_path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    base = _commit(tmp_path, BASE_MAP, "base")
    (tmp_path / "litellm" / "utils.py").write_text("print('hi')\n")
    head = _commit(tmp_path, {**BASE_MAP, "openrouter/c": _entry()}, "head")
    assert _run_guard(tmp_path, base, head, BOT_REF).returncode == 1
    assert _run_guard(tmp_path, base, head, "litellm_fix_pricing").returncode == 0
