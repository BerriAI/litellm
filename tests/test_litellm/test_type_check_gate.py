import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

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


def test_store_prunes_entries_for_other_branch_points(tmp_path):
    old = gate.cache_path(tmp_path, "old", ("f",))
    gate.store_counts(tmp_path, old, "old", {"reportAny": 1})
    new = gate.cache_path(tmp_path, "new", ("f",))
    gate.store_counts(tmp_path, new, "new", {"reportAny": 2})
    assert not old.exists()
    assert gate.load_cached_counts(new) == {"reportAny": 2}


def test_base_counts_cached_returns_the_hit_without_recomputing(tmp_path):
    path = gate.cache_path(tmp_path, "abc123", gate.environment_fingerprints())
    gate.store_counts(tmp_path, path, "abc123", {"reportAny": 7})

    def explode(ref):
        raise AssertionError("a cache hit must not re-run the base pass")

    assert gate.base_counts_cached("abc123", cache_dir=tmp_path, compute=explode) == {
        "reportAny": 7
    }


def test_base_counts_cached_computes_once_then_hits(tmp_path):
    calls = []

    def fake(ref):
        calls.append(ref)
        return {"reportAny": 4}

    first = gate.base_counts_cached("abc123", cache_dir=tmp_path, compute=fake)
    second = gate.base_counts_cached("abc123", cache_dir=tmp_path, compute=fake)
    assert first == second == {"reportAny": 4}
    assert calls == ["abc123"]


def test_an_empty_base_pass_is_never_cached(tmp_path):
    calls = []

    def crashed(ref):
        calls.append(ref)
        return {}

    assert gate.base_counts_cached("abc123", cache_dir=tmp_path, compute=crashed) == {}
    assert gate.base_counts_cached("abc123", cache_dir=tmp_path, compute=crashed) == {}
    assert calls == ["abc123", "abc123"]
    assert list(tmp_path.iterdir()) == []


def _proc(returncode, stdout="", stderr=""):
    return subprocess.CompletedProcess(["basedpyright"], returncode, stdout, stderr)


@pytest.mark.parametrize("code", [0, 1])
def test_base_pass_accepts_clean_and_errorful_exit_codes_without_retry(code):
    calls = []

    def run():
        calls.append(code)
        return _proc(code, stdout="{}")

    assert gate.run_base_pass(run).returncode == code
    assert calls == [code]


@pytest.mark.parametrize("code", [2, 137, -9])
def test_base_pass_retries_after_a_crash_and_reports_the_evidence(code, capsys):
    procs = iter([_proc(code, stderr="node blew up"), _proc(0, stdout="{}")])
    assert gate.run_base_pass(lambda: next(procs)).returncode == 0
    err = capsys.readouterr().err
    assert f"exited {code}" in err
    assert "node blew up" in err


def test_base_pass_that_keeps_crashing_exits_loudly_not_as_zero_counts(capsys):
    attempts = []

    def crash():
        attempts.append(1)
        return _proc(134, stderr="JavaScript heap out of memory")

    with pytest.raises(SystemExit):
        gate.run_base_pass(crash)
    assert len(attempts) == gate.BASE_PASS_ATTEMPTS
    err = capsys.readouterr().err
    assert err.count("exited 134") == gate.BASE_PASS_ATTEMPTS
    assert "JavaScript heap out of memory" in err


def _payload(rule, count):
    return json.dumps(
        {
            "generalDiagnostics": [
                _bpr(f"{ROOT}/litellm/x.py", "error", rule) for _ in range(count)
            ]
        }
    )


def _raise_if_called(*args):
    raise AssertionError("must not be called")


def _budget_file(tmp_path, limit):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"reportAny": {"limit": limit}}))
    return path


def test_check_rejects_a_vacuous_head_run_before_touching_the_base(tmp_path, capsys):
    with pytest.raises(SystemExit):
        gate.cmd_check(
            "origin/main",
            head_payload=lambda: "",
            base_counts_for=_raise_if_called,
            merge_base=_raise_if_called,
            budget_path=_budget_file(tmp_path, 5),
        )
    assert "vacuous" in capsys.readouterr().out


def test_check_within_limits_passes_without_a_base_pass(tmp_path, capsys):
    gate.cmd_check(
        "origin/main",
        head_payload=lambda: _payload("reportAny", 3),
        base_counts_for=_raise_if_called,
        merge_base=_raise_if_called,
        budget_path=_budget_file(tmp_path, 5),
    )
    assert capsys.readouterr().out.startswith("OK")


def test_check_refuses_to_blame_the_change_for_a_vacuous_base_pass(tmp_path, capsys):
    with pytest.raises(SystemExit):
        gate.cmd_check(
            "origin/main",
            head_payload=lambda: _payload("reportAny", 6),
            base_counts_for=lambda ref: {},
            merge_base=lambda base: "a" * 40,
            budget_path=_budget_file(tmp_path, 5),
        )
    assert "refusing to blame" in capsys.readouterr().out


def test_check_spares_a_bystander_whose_base_matches_head(tmp_path, capsys):
    gate.cmd_check(
        "origin/main",
        head_payload=lambda: _payload("reportAny", 6),
        base_counts_for=lambda ref: {"reportAny": 6},
        merge_base=lambda base: "a" * 40,
        budget_path=_budget_file(tmp_path, 5),
    )
    assert capsys.readouterr().out.startswith("OK")


def test_check_fails_a_change_that_grew_a_rule_past_its_limit(tmp_path, capsys):
    with pytest.raises(SystemExit):
        gate.cmd_check(
            "origin/main",
            head_payload=lambda: _payload("reportAny", 6),
            base_counts_for=lambda ref: {"reportAny": 4},
            merge_base=lambda base: "a" * 40,
            budget_path=_budget_file(tmp_path, 5),
        )
    assert "BREACHED RULES" in capsys.readouterr().out
