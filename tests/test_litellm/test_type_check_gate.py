import hashlib
import json
from pathlib import Path

import pytest

import lint_base_counts
import type_check_gate as gate

ROOT = gate.REPO_ROOT


def _bpr(file, severity, rule):
    diag = {"file": str(file), "severity": severity, "message": "msg"}
    if rule is not None:
        diag["rule"] = rule
    return diag


def test_basedpyright_counts_per_rule_from_json_not_warnings():
    # basedpyright wraps long messages across lines, so the (reportRule) lands on
    # a continuation line away from the `- error:` marker; --outputjson avoids it.
    payload = json.dumps(
        {
            "generalDiagnostics": [
                _bpr(f"{ROOT}/litellm/utils.py", "error", "reportUnknownVariableType"),
                _bpr(f"{ROOT}/litellm/utils.py", "error", "reportUnknownVariableType"),
                _bpr(f"{ROOT}/litellm/main.py", "error", "reportArgumentType"),
                _bpr(f"{ROOT}/litellm/main.py", "warning", "reportUnusedImport"),
            ]
        }
    )
    assert gate.count_basedpyright(payload) == {
        "reportUnknownVariableType": 2,
        "reportArgumentType": 1,
    }


def test_basedpyright_error_without_a_rule_is_bucketed():
    payload = json.dumps(
        {"generalDiagnostics": [_bpr(f"{ROOT}/litellm/x.py", "error", None)]}
    )
    assert gate.count_basedpyright(payload) == {gate.UNCODED: 1}


def test_paths_outside_repo_are_skipped():
    payload = json.dumps(
        {
            "generalDiagnostics": [
                _bpr("/tmp/elsewhere.py", "error", "reportArgumentType")
            ]
        }
    )
    assert gate.count_basedpyright(payload) == {}


def test_symlinked_root_keeps_diagnostics_in_tree(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    payload = json.dumps(
        {
            "generalDiagnostics": [
                _bpr(link / "litellm" / "x.py", "error", "reportArgumentType")
            ]
        }
    )
    assert gate.count_basedpyright(payload, root=link) == {"reportArgumentType": 1}


def test_node_options_with_heap_sets_the_flag_in_a_bare_env():
    assert gate.node_options_with_heap({}) == gate.NODE_HEAP_OPTION


def test_node_options_with_heap_appends_after_caller_flags_so_it_wins():
    # node resolves a repeated --max-old-space-size last-wins, so ours must come
    # after any caller-set value while keeping their other flags.
    merged = gate.node_options_with_heap(
        {"NODE_OPTIONS": "--max-old-space-size=4096 --no-warnings"}
    )
    assert merged == f"--max-old-space-size=4096 --no-warnings {gate.NODE_HEAP_OPTION}"


def _stub_env(tmp_path, script_body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "basedpyright"
    stub.write_text(f"#!/bin/sh\n{script_body}\n")
    stub.chmod(0o755)
    return tmp_path


def test_run_basedpyright_exports_the_raised_heap_to_the_child(tmp_path, monkeypatch):
    captured = tmp_path / "node_options.txt"
    env_dir = _stub_env(
        tmp_path,
        f'echo "$NODE_OPTIONS" > "{captured}"\necho \'{{"generalDiagnostics": []}}\'',
    )
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    assert json.loads(gate.run_basedpyright(cwd=tmp_path, env_dir=env_dir)) == {
        "generalDiagnostics": []
    }
    assert captured.read_text().strip() == gate.NODE_HEAP_OPTION


def test_run_basedpyright_pins_import_resolution_to_the_owned_env(tmp_path):
    # basedpyright auto-detects a `.venv` in the project root, and that beats
    # PATH order and VIRTUAL_ENV; only an explicit --pythonpath keeps the
    # caller's fatter venv (whose extra typed packages flip diagnostics vs CI)
    # out of the measurement.
    captured = tmp_path / "argv.txt"
    env_dir = _stub_env(
        tmp_path,
        f'echo "$@" > "{captured}"\necho \'{{"generalDiagnostics": []}}\'',
    )
    gate.run_basedpyright(cwd=tmp_path, env_dir=env_dir)
    argv = captured.read_text().split()
    assert argv[argv.index("--pythonpath") + 1] == str(env_dir / "bin" / "python")


def test_run_basedpyright_fails_loudly_on_a_crash_exit_code(tmp_path):
    import pytest

    # 134 is SIGABRT, what node dies with on a heap OOM; it must never read as a
    # clean zero-error run.
    env_dir = _stub_env(tmp_path, "exit 134")
    with pytest.raises(SystemExit):
        gate.run_basedpyright(cwd=tmp_path, env_dir=env_dir)


def test_malformed_basedpyright_json_exits_loudly_not_as_zero_errors():
    import pytest

    with pytest.raises(SystemExit):
        gate.count_basedpyright("startup warning\n{not json")


def test_empty_basedpyright_payload_counts_zero():
    # Empty (not malformed) output parses to zero; the vacuous-run guard, not the
    # parser, is what rejects an empty run.
    assert gate.count_basedpyright("") == {}


def test_fingerprints_carry_the_dependency_group_set():
    # Counts measured under one group set must never be compared against
    # another's: the fingerprint difference re-keys every cache entry and
    # artifact name, so a changed canonical set falls back to recompute.
    assert gate.environment_fingerprints() == gate.environment_fingerprints()
    assert gate.environment_fingerprints(
        dep_groups=("proxy-dev",)
    ) != gate.environment_fingerprints(dep_groups=("proxy-dev", "e2e-dev"))
    assert gate.environment_fingerprints()[-1] == "groups:" + ",".join(
        gate.TYPECHECK_DEP_GROUPS
    )


def test_fingerprints_cover_the_prisma_schema():
    schema_hash = hashlib.sha256(gate.PRISMA_SCHEMA.read_bytes()).hexdigest()
    assert schema_hash in gate.environment_fingerprints()


def test_env_commands_sync_the_canonical_groups_then_generate_prisma():
    sync, generate = gate.typecheck_env_commands(Path("/envdir"))
    assert sync[:3] == ("uv", "sync", "--frozen")
    adjacent = list(zip(sync, sync[1:]))
    for group in gate.TYPECHECK_DEP_GROUPS:
        assert ("--group", group) in adjacent
    assert generate == (
        str(Path("/envdir") / "bin" / "python"),
        str(gate.PRISMA_GENERATE_SCRIPT),
    )


def test_env_interpreter_pin_tracks_pyrightconfigs_python_version():
    configured = json.loads((ROOT / "pyrightconfig.json").read_text())[
        "pythonVersion"
    ]
    assert gate.typecheck_python_version() == configured
    sync = gate.typecheck_env_commands()[0]
    assert sync[sync.index("--python") + 1] == configured


def test_ensure_env_targets_the_owned_dir_and_runs_sync_then_generate(tmp_path):
    calls = []

    def runner(cmd, env):
        calls.append((cmd[:2], env["UV_PROJECT_ENVIRONMENT"]))
        return 0

    assert gate.ensure_typecheck_env(env_dir=tmp_path, run=runner) == tmp_path
    assert calls == [
        (("uv", "sync"), str(tmp_path)),
        ((str(tmp_path / "bin" / "python"), str(gate.PRISMA_GENERATE_SCRIPT)), str(tmp_path)),
    ]


def test_ensure_env_fails_loudly_and_stops_at_the_first_failed_step(tmp_path):
    import pytest

    calls = []

    def failing(cmd, env):
        calls.append(cmd)
        return 2

    with pytest.raises(SystemExit):
        gate.ensure_typecheck_env(env_dir=tmp_path, run=failing)
    assert len(calls) == 1


def test_ensure_env_announces_a_cold_provision(tmp_path, capsys):
    def runner(cmd, env):
        return 0

    gate.ensure_typecheck_env(env_dir=tmp_path / "fresh", run=runner)
    assert "provisioning" in capsys.readouterr().err


def test_ensure_env_is_silent_when_the_env_already_exists(tmp_path, capsys):
    def runner(cmd, env):
        return 0

    gate.ensure_typecheck_env(env_dir=tmp_path, run=runner)
    assert capsys.readouterr().err == ""


def test_the_checker_identity_is_keyed_on_the_environment_fingerprints():
    assert gate.checker_identity() == lint_base_counts.Checker("basedpyright", gate.environment_fingerprints())


def test_headroom_is_kept_only_for_the_any_discipline_rules():
    assert set(gate.HEADROOM) == {"reportAny", "reportExplicitAny"}
    assert all(isinstance(value, int) and value > 0 for value in gate.HEADROOM.values())


def test_no_head_output_is_refused_as_vacuous_before_any_base_lookup(capsys):
    with pytest.raises(SystemExit) as exit_info:
        gate.cmd_check({}, "irrelevant-base-ref")
    assert exit_info.value.code == 1
    assert "vacuous" in capsys.readouterr().out
