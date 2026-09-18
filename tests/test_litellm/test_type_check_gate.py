import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "type_check_gate.py"
_spec = importlib.util.spec_from_file_location("type_check_gate", _MODULE_PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

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


def test_at_or_under_ceiling_passes():
    budget = {"no-any-return": {"limit": 5}}
    assert gate.evaluate({"no-any-return": 5}, {}, budget) == []


def test_one_more_error_than_ceiling_fails():
    budget = {"no-any-return": {"limit": 5}}
    assert gate.evaluate({"no-any-return": 6}, {}, budget) == [
        gate.Breach("no-any-return", 6, 5, 6)
    ]


def test_limit_absorbs_increase_up_to_it_then_fails_past_it():
    budget = {"arg-type": {"limit": 10}}
    assert gate.evaluate({"arg-type": 10}, {}, budget) == []
    assert gate.evaluate({"arg-type": 11}, {}, budget) == [
        gate.Breach("arg-type", 11, 10, 11)
    ]


def test_unbudgeted_new_code_uses_default_limit():
    assert gate.evaluate({"brand-new": gate.DEFAULT_LIMIT}, {}, {}) == []
    assert gate.evaluate({"brand-new": gate.DEFAULT_LIMIT + 1}, {}, {}) == [
        gate.Breach(
            "brand-new",
            gate.DEFAULT_LIMIT + 1,
            gate.DEFAULT_LIMIT,
            gate.DEFAULT_LIMIT + 1,
        )
    ]


def test_drift_already_over_cap_in_base_is_not_blamed_on_a_flat_change():
    # The bystander case: a rule sits over its limit because two earlier PRs
    # summed past it. A PR that branches off that base and adds nothing must pass
    # -- total > limit but total == base, so the `> base` guard spares it.
    budget = {"arg-type": {"limit": 10}}
    assert gate.evaluate({"arg-type": 12}, {"arg-type": 12}, budget) == []


def test_change_that_grows_an_over_cap_rule_is_blamed_for_only_what_it_added():
    # Over limit AND above base: blamed, and `added` is the delta vs base, not the
    # whole overage, so the message points at this change's contribution.
    budget = {"arg-type": {"limit": 10}}
    assert gate.evaluate({"arg-type": 14}, {"arg-type": 12}, budget) == [
        gate.Breach("arg-type", 14, 10, 2)
    ]


def test_reducing_an_over_cap_rule_below_base_passes():
    budget = {"arg-type": {"limit": 10}}
    assert gate.evaluate({"arg-type": 11}, {"arg-type": 12}, budget) == []


def test_no_output_against_a_nonempty_budget_is_a_vacuous_run():
    # A crashed type checker emits nothing; the gate must not certify it as clean.
    budget = {"no-untyped-def": {"limit": 4898}}
    assert gate.is_vacuous_run({}, budget) is True


def test_genuine_zero_and_empty_budget_are_not_vacuous():
    assert gate.is_vacuous_run({}, {}) is False
    assert gate.is_vacuous_run({}, {"no-untyped-def": {"limit": 0}}) is False
    assert (
        gate.is_vacuous_run({"arg-type": 1}, {"arg-type": {"limit": 10}}) is False
    )


def test_update_ratchets_a_limit_down_by_what_the_branch_fixed():
    # A rule that dropped from 40 (branch point) to 30 (current) fixed 10, so its
    # limit of 100 falls to 90 -- the granted headroom (60) is preserved, not the
    # raw count.
    budget = {"reportAny": {"limit": 100}}
    assert gate.ratcheted_budget(budget, {"reportAny": 30}, {"reportAny": 40}) == {
        "reportAny": {"limit": 90}
    }


def test_update_never_raises_a_limit_when_a_rule_grows():
    # Adding violations must not loosen the ceiling; the limit holds flat.
    budget = {"reportAny": {"limit": 100}}
    assert gate.ratcheted_budget(budget, {"reportAny": 55}, {"reportAny": 40}) == {
        "reportAny": {"limit": 100}
    }


def test_update_clamps_a_limit_at_zero_never_negative():
    budget = {"reportAny": {"limit": 5}}
    assert gate.ratcheted_budget(budget, {"reportAny": 0}, {"reportAny": 40}) == {
        "reportAny": {"limit": 0}
    }


def test_malformed_basedpyright_json_exits_loudly_not_as_zero_errors():
    import pytest

    with pytest.raises(SystemExit):
        gate.count_basedpyright("startup warning\n{not json")


def test_empty_basedpyright_payload_counts_zero():
    # Empty (not malformed) output parses to zero; the vacuous-run guard, not the
    # parser, is what rejects an empty run.
    assert gate.count_basedpyright("") == {}


def test_over_ceiling_flags_only_rules_above_their_limit():
    budget = {"reportAny": {"limit": 10}}
    assert gate.over_ceiling({"reportAny": 10}, budget) == frozenset()
    assert gate.over_ceiling({"reportAny": 11}, budget) == frozenset({"reportAny"})
    assert gate.over_ceiling({}, budget) == frozenset()


def test_over_ceiling_holds_unbudgeted_rules_to_the_default_limit():
    assert gate.over_ceiling({"brand-new": gate.DEFAULT_LIMIT}, {}) == frozenset()
    assert gate.over_ceiling({"brand-new": gate.DEFAULT_LIMIT + 1}, {}) == frozenset(
        {"brand-new"}
    )


def test_over_ceiling_is_independent_across_rules():
    budget = {"reportAny": {"limit": 10}, "reportArgumentType": {"limit": 5}}
    assert gate.over_ceiling(
        {"reportAny": 9, "reportArgumentType": 6}, budget
    ) == frozenset({"reportArgumentType"})


def test_cache_key_changes_with_base_point_and_each_fingerprint():
    key = gate.cache_key("abc", ("cfg", "lock"))
    assert gate.cache_key("abc", ("cfg", "lock")) == key
    assert gate.cache_key("def", ("cfg", "lock")) != key
    assert gate.cache_key("abc", ("cfg2", "lock")) != key
    assert gate.cache_key("abc", ("cfg", "lock2")) != key


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


def test_cached_counts_round_trip(tmp_path):
    path = gate.cache_path(tmp_path, "abc123", ("f1", "f2"))
    gate.store_counts(tmp_path, path, "abc123", {"reportAny": 3, "reportCall": 1})
    assert gate.load_cached_counts(path) == {"reportAny": 3, "reportCall": 1}


def test_missing_corrupt_or_misshapen_cache_reads_as_none(tmp_path):
    path = tmp_path / "cache.json"
    assert gate.load_cached_counts(path) is None
    path.write_text("{not json")
    assert gate.load_cached_counts(path) is None
    path.write_text(json.dumps(["counts"]))
    assert gate.load_cached_counts(path) is None
    path.write_text(json.dumps({"base_point": "abc"}))
    assert gate.load_cached_counts(path) is None
    path.write_text(json.dumps({"counts": {"reportAny": "three"}}))
    assert gate.load_cached_counts(path) is None
    path.write_text(json.dumps({"counts": {"reportAny": True}}))
    assert gate.load_cached_counts(path) is None


def test_scratch_is_invisible_to_the_prune_glob():
    import fnmatch

    scratch = gate.scratch_path(gate.cache_path(Path("/c"), "abc", ("f",)))
    assert not fnmatch.fnmatch(scratch.name, f"{gate.CACHE_FILE_PREFIX}*")


def test_store_prune_spares_a_concurrent_runs_in_flight_scratch(tmp_path):
    foreign = gate.scratch_path(gate.cache_path(tmp_path, "other", ("f",)))
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text("{}")
    mine = gate.cache_path(tmp_path, "mine", ("f",))
    gate.store_counts(tmp_path, mine, "mine", {"reportAny": 1})
    assert foreign.exists()
    assert gate.load_cached_counts(mine) == {"reportAny": 1}


def test_store_keeps_a_concurrent_worktrees_entry_for_another_branch_point(tmp_path):
    old = gate.cache_path(tmp_path, "old", ("f",))
    gate.store_counts(tmp_path, old, "old", {"reportAny": 1})
    new = gate.cache_path(tmp_path, "new", ("f",))
    gate.store_counts(tmp_path, new, "new", {"reportAny": 2})
    assert gate.load_cached_counts(old) == {"reportAny": 1}
    assert gate.load_cached_counts(new) == {"reportAny": 2}


def test_store_evicts_only_the_oldest_entries_beyond_the_cap(tmp_path):
    aged = [
        gate.cache_path(tmp_path, f"base{i}", ("f",))
        for i in range(gate.CACHE_KEEP_ENTRIES)
    ]
    for age, path in enumerate(aged):
        gate.store_counts(tmp_path, path, f"base{age}", {"reportAny": age})
        os.utime(path, (age, age))
    newest = gate.cache_path(tmp_path, "newest", ("f",))
    gate.store_counts(tmp_path, newest, "newest", {"reportAny": 99})
    assert not aged[0].exists()
    assert all(path.exists() for path in aged[1:])
    assert gate.load_cached_counts(newest) == {"reportAny": 99}


def test_store_never_evicts_the_entry_it_just_wrote_even_on_mtime_ties(tmp_path):
    others = [
        gate.cache_path(tmp_path, f"base{i}", ("f",))
        for i in range(gate.CACHE_KEEP_ENTRIES + 2)
    ]
    for path in others:
        gate.store_counts(tmp_path, path, path.name, {"reportAny": 1})
        os.utime(path, (9_999_999_999, 9_999_999_999))
    mine = gate.cache_path(tmp_path, "mine", ("f",))
    gate.store_counts(tmp_path, mine, "mine", {"reportAny": 2})
    assert gate.load_cached_counts(mine) == {"reportAny": 2}
    survivors = list(tmp_path.glob(f"{gate.CACHE_FILE_PREFIX}*.json"))
    assert len(survivors) == gate.CACHE_KEEP_ENTRIES


def _no_fetch(ref):
    return None


def _never(reason):
    def callback(ref):
        raise AssertionError(reason)

    return callback


def test_base_counts_cached_returns_the_hit_without_recomputing(tmp_path):
    path = gate.cache_path(tmp_path, "abc123", gate.environment_fingerprints())
    gate.store_counts(tmp_path, path, "abc123", {"reportAny": 7})

    assert gate.base_counts_cached(
        "abc123",
        cache_dir=tmp_path,
        compute=_never("a cache hit must not re-run the base pass"),
        fetch=_never("a cache hit must not reach for CI"),
    ) == {"reportAny": 7}


def test_base_counts_cached_computes_once_then_hits(tmp_path):
    calls = []

    def fake(ref):
        calls.append(ref)
        return {"reportAny": 4}

    first = gate.base_counts_cached(
        "abc123", cache_dir=tmp_path, compute=fake, fetch=_no_fetch
    )
    second = gate.base_counts_cached(
        "abc123", cache_dir=tmp_path, compute=fake, fetch=_no_fetch
    )
    assert first == second == {"reportAny": 4}
    assert calls == ["abc123"]


def test_an_empty_base_pass_is_never_cached(tmp_path):
    calls = []

    def crashed(ref):
        calls.append(ref)
        return {}

    assert (
        gate.base_counts_cached(
            "abc123", cache_dir=tmp_path, compute=crashed, fetch=_no_fetch
        )
        == {}
    )
    assert (
        gate.base_counts_cached(
            "abc123", cache_dir=tmp_path, compute=crashed, fetch=_no_fetch
        )
        == {}
    )
    assert calls == ["abc123", "abc123"]
    assert list(tmp_path.iterdir()) == []


def test_base_counts_cached_uses_fetched_counts_and_persists_them(tmp_path):
    counts = gate.base_counts_cached(
        "abc123",
        cache_dir=tmp_path,
        compute=_never("fetched counts must skip the local base pass"),
        fetch=lambda ref: {"reportAny": 9},
    )
    assert counts == {"reportAny": 9}
    path = gate.cache_path(tmp_path, "abc123", gate.environment_fingerprints())
    assert gate.load_cached_counts(path) == {"reportAny": 9}
    assert gate.base_counts_cached(
        "abc123",
        cache_dir=tmp_path,
        compute=_never("the persisted fetch must satisfy later runs"),
        fetch=_never("the persisted fetch must satisfy later runs"),
    ) == {"reportAny": 9}


def test_base_counts_cached_falls_back_to_compute_on_a_fetch_miss(tmp_path):
    calls = []

    def local(ref):
        calls.append(ref)
        return {"reportAny": 4}

    assert gate.base_counts_cached(
        "abc123", cache_dir=tmp_path, compute=local, fetch=_no_fetch
    ) == {"reportAny": 4}
    assert calls == ["abc123"]


def test_base_counts_cached_treats_empty_fetched_counts_as_a_miss(tmp_path):
    assert gate.base_counts_cached(
        "abc123",
        cache_dir=tmp_path,
        compute=lambda ref: {"reportAny": 2},
        fetch=lambda ref: {},
    ) == {"reportAny": 2}
    path = gate.cache_path(tmp_path, "abc123", gate.environment_fingerprints())
    assert gate.load_cached_counts(path) == {"reportAny": 2}


def test_origin_slug_parsing_supports_ssh_and_https_github_forms():
    assert gate.parse_origin_slug("git@github.com:BerriAI/litellm.git") == "BerriAI/litellm"
    assert gate.parse_origin_slug("git@github.com:BerriAI/litellm") == "BerriAI/litellm"
    assert gate.parse_origin_slug("https://github.com/BerriAI/litellm.git") == "BerriAI/litellm"
    assert gate.parse_origin_slug("https://github.com/BerriAI/litellm") == "BerriAI/litellm"
    assert gate.parse_origin_slug("https://github.com/BerriAI/litellm/") == "BerriAI/litellm"


def test_origin_slug_parsing_rejects_non_github_urls():
    assert gate.parse_origin_slug("https://gitlab.com/BerriAI/litellm.git") is None
    assert gate.parse_origin_slug("git@bitbucket.org:BerriAI/litellm.git") is None
    assert gate.parse_origin_slug("not a url") is None
    assert gate.parse_origin_slug("") is None


def _artifact_zip(payload):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("basedpyright-counts.json", json.dumps(payload))
    return buffer.getvalue()


def _gh_stub(listing, zip_bytes):
    def gh_output(args):
        if args[-1].startswith("repos/"):
            return json.dumps(listing).encode()
        return zip_bytes

    return gh_output


def _live_listing():
    return {
        "artifacts": [
            {"expired": False, "archive_download_url": "https://api.github.com/x/zip"}
        ]
    }


def test_fetcher_returns_counts_from_a_matching_artifact(capsys):
    payload = {"base_point": "abc123", "counts": {"reportAny": 3}}
    fetched = gate.fetch_ci_base_counts(
        "abc123", gh_output=_gh_stub(_live_listing(), _artifact_zip(payload))
    )
    assert fetched == {"reportAny": 3}
    assert "fetched from CI artifact" in capsys.readouterr().err


def test_fetcher_rejects_an_artifact_for_a_different_base_point():
    payload = {"base_point": "someothersha", "counts": {"reportAny": 3}}
    assert (
        gate.fetch_ci_base_counts(
            "abc123", gh_output=_gh_stub(_live_listing(), _artifact_zip(payload))
        )
        is None
    )


def test_fetcher_rejects_empty_or_misshapen_artifact_counts():
    for counts in ({}, {"reportAny": "three"}, {"reportAny": True}):
        payload = {"base_point": "abc123", "counts": counts}
        assert (
            gate.fetch_ci_base_counts(
                "abc123", gh_output=_gh_stub(_live_listing(), _artifact_zip(payload))
            )
            is None
        )


def test_fetcher_rejects_an_expired_artifact():
    listing = {
        "artifacts": [
            {"expired": True, "archive_download_url": "https://api.github.com/x/zip"}
        ]
    }
    payload = {"base_point": "abc123", "counts": {"reportAny": 3}}
    assert (
        gate.fetch_ci_base_counts(
            "abc123", gh_output=_gh_stub(listing, _artifact_zip(payload))
        )
        is None
    )


def test_fetcher_misses_when_no_artifact_is_published():
    assert (
        gate.fetch_ci_base_counts(
            "abc123", gh_output=_gh_stub({"artifacts": []}, b"")
        )
        is None
    )


def test_fetcher_misses_when_gh_is_unusable(capsys):
    assert gate.fetch_ci_base_counts("abc123", gh_output=lambda args: None) is None
    assert "computing base counts locally" in capsys.readouterr().err


def test_fetcher_misses_on_a_corrupt_artifact_archive():
    assert (
        gate.fetch_ci_base_counts(
            "abc123", gh_output=_gh_stub(_live_listing(), b"not a zip")
        )
        is None
    )


def test_emit_writes_the_artifact_json_named_by_the_head_key(tmp_path, capsys):
    gate.cmd_emit_counts({"reportAny": 3, "aRule": 1}, tmp_path, "deadbeef")
    key = gate.cache_key("deadbeef", gate.environment_fingerprints())
    path = tmp_path / f"basedpyright-counts-{key}.json"
    assert json.loads(path.read_text()) == {
        "base_point": "deadbeef",
        "counts": {"aRule": 1, "reportAny": 3},
    }
    summary = capsys.readouterr().out
    assert "deadbeef" in summary
    assert key in summary
    assert "4" in summary


def test_emit_refuses_to_publish_empty_counts(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        gate.cmd_emit_counts({}, tmp_path, "deadbeef")
    assert list(tmp_path.iterdir()) == []


def test_emitted_file_round_trips_through_the_fetch_validation(tmp_path):
    gate.cmd_emit_counts({"reportAny": 3}, tmp_path, "deadbeef")
    key = gate.cache_key("deadbeef", gate.environment_fingerprints())
    payload = json.loads((tmp_path / f"basedpyright-counts-{key}.json").read_text())
    assert gate.counts_for_base(payload, "deadbeef") == {"reportAny": 3}
    assert gate.counts_for_base(payload, "someothersha") is None


def _git(cwd, *args):
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(cwd, name):
    (cwd / name).write_text(name)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "-m", name)
    return _git(cwd, "rev-parse", "HEAD")


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "gate@example.com")
    _git(repo, "config", "user.name", "gate")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _branched_repo(tmp_path):
    repo = _init_repo(tmp_path)
    branch_point = _commit(repo, "shared.txt")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.txt")
    _git(repo, "checkout", "-q", "main")
    base_tip = _commit(repo, "drift.txt")
    _git(repo, "checkout", "-q", "feature")
    return repo, branch_point, base_tip


def test_base_point_is_the_branch_point_when_no_merge_is_in_progress(tmp_path):
    repo, branch_point, _ = _branched_repo(tmp_path)
    assert gate.resolve_base_point("main", cwd=repo) == branch_point


def test_base_point_mid_merge_advances_to_the_merged_in_base_tip(tmp_path):
    repo, _, base_tip = _branched_repo(tmp_path)
    _git(repo, "merge", "--no-commit", "--no-ff", "main")
    assert gate.resolve_base_point("main", cwd=repo) == base_tip


def test_base_point_mid_merge_of_an_older_side_branch_keeps_the_newer_branch_point(tmp_path):
    repo = _init_repo(tmp_path)
    _commit(repo, "shared.txt")
    _git(repo, "checkout", "-q", "-b", "old-side")
    _commit(repo, "old.txt")
    _git(repo, "checkout", "-q", "main")
    newer_point = _commit(repo, "drift.txt")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.txt")
    _git(repo, "merge", "--no-commit", "--no-ff", "old-side")
    assert gate.resolve_base_point("main", cwd=repo) == newer_point
