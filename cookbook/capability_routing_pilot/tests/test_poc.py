import csv
import json
import math
import os
from pathlib import Path

import pytest
from minisweagent.exceptions import LimitsExceeded

from poc.analyze import analyze
from poc.guarded_model import (
    GuardedLitellmModel,
    GuardedModelConfig,
    budget_allows,
    estimate_input_tokens,
    gateway_cost,
    reserve_usd,
)
from poc.label import parse_jev_response, parse_label_response
from poc.policy import (
    PolicyConfig,
    build_capability_map,
    choose_model,
    paired_comparison,
    paired_lower_bound,
    parse_category,
    task_outcomes,
)
from poc.run_pilot import (
    _utility_env,
    classify_error,
    global_budget_allows,
    sum_call_costs,
    sum_call_tokens,
)


def _record(
    task_id: str,
    model: str,
    *,
    error_class: str = "solver",
    success: bool = False,
    cost_usd: float = 0.0,
    exit_status: str = "Submitted",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "model": model,
        "error_class": error_class,
        "success": success,
        "cost_usd": cost_usd,
        "exit_status": exit_status,
        "solve_latency_s": 1.0,
        "eval_latency_s": 2.0,
    }


def test_category_parsing_handles_aliases_garbage_and_non_strings() -> None:
    assert parse_category("Repository Debugging") == "repository_debugging"
    assert parse_category("Shell/Build Work") == "shell_build"
    assert parse_category("Algorithms & Data Processing") == "algorithms_data"
    assert parse_category("garbage") == "unknown"
    assert parse_category(None) == "unknown"
    assert parse_category(4) == "unknown"


def test_label_response_parser_accepts_fences_and_rejects_garbage() -> None:
    assert parse_label_response(
        'Answer: ```json\n{"label":"Feature Implementation","rationale":"Adds an API."}\n```'
    ) == ("feature_implementation", "Adds an API.", True)
    assert parse_label_response('{"label":"nonsense","rationale":"x"}') == ("unknown", None, False)
    assert parse_label_response("no json") == ("unknown", None, False)
    assert parse_label_response('{"label":"unknown","rationale":"unclear"}') == ("unknown", "unclear", True)


def test_jev_response_parser_reads_structured_answer_and_fails_closed() -> None:
    response = {
        "answers": {
            "capability": {
                "choice": "shell_build",
                "confidence": 0.82,
                "probabilities": {
                    "shell_build": 0.82,
                    "repository_debugging": 0.18,
                    "not_a_category": 0.7,
                },
            }
        }
    }
    assert parse_jev_response(response) == (
        "shell_build",
        0.82,
        {"shell_build": 0.82, "repository_debugging": 0.18},
    )
    assert parse_jev_response({"answers": {"capability": {"choice": "unknown"}}}) == (
        "unknown",
        None,
        {},
    )
    assert parse_jev_response({"answers": {"capability": {"confidence": 0.5}}}) == (None, None, {})
    assert parse_jev_response({"answers": {"capability": {"choice": "unrecognized"}}}) == (None, None, {})
    assert parse_jev_response("malformed") == (None, None, {})


def test_task_outcomes_cluster_retries_and_keep_all_attempt_costs() -> None:
    records = [
        _record("task-a", "haiku-4.5", error_class="infra", cost_usd=0.1),
        _record("task-a", "haiku-4.5", error_class="infra", cost_usd=0.2),
        _record("task-a", "haiku-4.5", success=True, cost_usd=0.4),
        _record("task-b", "haiku-4.5", error_class="infra", cost_usd=0.5),
    ]
    outcomes = task_outcomes(records)
    task_a_outcome = outcomes[("task-a", "haiku-4.5")]
    assert {
        "graded": task_a_outcome["graded"],
        "success": task_a_outcome["success"],
        "attempts": task_a_outcome["attempts"],
    } == {
        "graded": True,
        "success": True,
        "attempts": 3,
    }
    assert task_a_outcome["cost_usd"] == pytest.approx(0.7)
    assert outcomes[("task-b", "haiku-4.5")] == {
        "graded": False,
        "success": None,
        "cost_usd": 0.5,
        "attempts": 1,
    }
    capability_map = build_capability_map(
        records,
        {"task-a": "repository_debugging", "task-b": "repository_debugging"},
        ["haiku-4.5", "opus-5"],
        PolicyConfig(),
    )
    assert capability_map["repository_debugging"]["haiku-4.5"]["n"] == 1
    assert capability_map["repository_debugging"]["haiku-4.5"]["s"] == 1
    assert capability_map["repository_debugging"]["haiku-4.5"]["posterior_mean"] == pytest.approx(2 / 3)
    assert capability_map["repository_debugging"]["haiku-4.5"]["mean_cost_per_task_usd"] == pytest.approx(0.6)
    assert capability_map["repository_debugging"]["haiku-4.5"]["graded_tasks"] == ["task-a"]
    assert capability_map["repository_debugging"]["haiku-4.5"]["task_success"] == {"task-a": True}
    task_cost = capability_map["repository_debugging"]["haiku-4.5"]["task_cost"]
    assert task_cost["task-a"] == pytest.approx(0.7)
    assert task_cost["task-b"] == pytest.approx(0.5)


def test_task_outcomes_reject_multiple_graded_attempts() -> None:
    records = [
        _record("task-a", "haiku-4.5", success=False),
        _record("task-a", "haiku-4.5", success=True),
    ]
    with pytest.raises(ValueError, match=r"duplicate graded attempts for \('task-a', 'haiku-4.5'\): single-attempt pilot"):
        task_outcomes(records)


def test_paired_lower_bound_matches_one_sided_hoeffding_formula() -> None:
    expected = -math.sqrt(2 * math.log(20) / 1000)
    assert PolicyConfig().alpha == 0.05
    assert paired_lower_bound([0] * 1000, 0.05) == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError, match="cannot be empty"):
        paired_lower_bound([], 0.05)


def test_model_selection_falls_back_for_failures_unknown_and_sparse_evidence() -> None:
    cfg = PolicyConfig(min_paired_tasks=10)
    assert choose_model(None, {}, cfg).reason == "fallback: classifier failure"
    assert choose_model("unknown", {}, cfg).model == "opus-5"
    assert choose_model("repository_debugging", {}, cfg).reason == (
        "fallback: no profile for category repository_debugging"
    )
    sparse_map = build_capability_map(
        [
            _record(f"task-{index}", model, success=index < successes, cost_usd=cost)
            for index in range(9)
            for model, successes, cost in (("opus-5", 9, 0.1), ("haiku-4.5", 9, 0.01))
        ],
        {f"task-{index}": "repository_debugging" for index in range(9)},
        ["haiku-4.5", "opus-5"],
        cfg,
    )
    assert choose_model("repository_debugging", sparse_map, cfg).model == "opus-5"


def test_model_selection_requires_paired_evidence() -> None:
    cfg = PolicyConfig(min_paired_tasks=10, tolerance=0.05)
    disjoint_records = [
        _record(f"reference-{index}", "opus-5", success=False, cost_usd=0.1)
        for index in range(12)
    ] + [
        _record(f"candidate-{index}", "haiku-4.5", success=True, cost_usd=0.01)
        for index in range(12)
    ]
    disjoint_categories = {
        **{f"reference-{index}": "repository_debugging" for index in range(12)},
        **{f"candidate-{index}": "repository_debugging" for index in range(12)},
    }
    disjoint_map = build_capability_map(
        disjoint_records,
        disjoint_categories,
        ["haiku-4.5", "opus-5"],
        cfg,
    )
    disjoint_decision = choose_model("repository_debugging", disjoint_map, cfg)
    assert disjoint_decision.model == "opus-5"
    assert disjoint_decision.reason == (
        "fallback: no cheaper model with >= 10 paired tasks and paired lower bound >= -0.05 (alpha=0.05)"
    )

    both_success_records = [
        _record(f"task-{index}", model, success=True, cost_usd=cost)
        for index in range(12)
        for model, cost in (("opus-5", 0.1), ("haiku-4.5", 0.01))
    ]
    both_success_categories = {f"task-{index}": "repository_debugging" for index in range(12)}
    both_success_map = build_capability_map(
        both_success_records,
        both_success_categories,
        ["haiku-4.5", "opus-5"],
        cfg,
    )
    comparison = paired_comparison(
        both_success_map["repository_debugging"]["haiku-4.5"],
        both_success_map["repository_debugging"]["opus-5"],
        cfg.alpha,
    )
    assert comparison["n"] == 12
    assert comparison["mean_diff"] == 0
    assert comparison["lower_bound"] == pytest.approx(-math.sqrt(2 * math.log(20) / 12))
    assert choose_model("repository_debugging", both_success_map, cfg).model == "opus-5"


def test_model_selection_uses_paired_cost_and_chooses_cheapest_eligible() -> None:
    cfg = PolicyConfig(min_paired_tasks=10, tolerance=0.1)
    paired_records = [
        _record(f"task-{index}", model, success=True, cost_usd=cost)
        for index in range(1000)
        for model, cost in (("opus-5", 0.1), ("haiku-4.5", 0.05))
    ]
    paired_categories = {f"task-{index}": "repository_debugging" for index in range(1000)}
    paired_map = build_capability_map(
        paired_records,
        paired_categories,
        ["haiku-4.5", "opus-5"],
        cfg,
    )
    decision = choose_model("repository_debugging", paired_map, cfg)
    assert decision.model == "haiku-4.5"
    assert decision.reason == "cheapest model with paired lower bound within tolerance of reference"

    multiple_candidate_records = [
        _record(f"task-{index}", model, success=True, cost_usd=cost)
        for index in range(1000)
        for model, cost in (("opus-5", 0.1), ("haiku-4.5", 0.05), ("sonnet-5", 0.03))
    ]
    multiple_candidate_map = build_capability_map(
        multiple_candidate_records,
        paired_categories,
        ["haiku-4.5", "sonnet-5", "opus-5"],
        cfg,
    )
    assert choose_model("repository_debugging", multiple_candidate_map, cfg).model == "sonnet-5"

    paired_cost_records = [
        _record(f"paired-{index}", "opus-5", success=True, cost_usd=0.1)
        for index in range(1000)
    ] + [
        _record(f"paired-{index}", "haiku-4.5", success=True, cost_usd=0.11)
        for index in range(1000)
    ] + [
        _record(f"candidate-only-{index}", "haiku-4.5", success=True, cost_usd=0.0001)
        for index in range(1000)
    ]
    paired_cost_categories = {
        **{f"paired-{index}": "repository_debugging" for index in range(1000)},
        **{f"candidate-only-{index}": "repository_debugging" for index in range(1000)},
    }
    paired_cost_map = build_capability_map(
        paired_cost_records,
        paired_cost_categories,
        ["haiku-4.5", "opus-5"],
        cfg,
    )
    assert paired_cost_map["repository_debugging"]["haiku-4.5"]["mean_cost_per_task_usd"] < (
        paired_cost_map["repository_debugging"]["opus-5"]["mean_cost_per_task_usd"]
    )
    assert choose_model("repository_debugging", paired_cost_map, cfg).model == "opus-5"


def test_reservations_and_spend_guards_include_boundary(tmp_path: Path) -> None:
    config = GuardedModelConfig(
        model_name="litellm_proxy/test-model",
        run_budget_usd=1.0,
        worst_input_usd_per_token=0.000002,
        output_usd_per_token=0.00001,
        max_output_tokens=100,
        call_log_path="/unused/calls.jsonl",
    )
    assert reserve_usd(100, config) == 0.00125
    assert budget_allows(0.99875, 0.00125, 1.0)
    assert not budget_allows(0.998751, 0.00125, 1.0)
    assert global_budget_allows(13.0, 1, 15.0)
    assert not global_budget_allows(13.0001, 1, 15.0)
    assert _utility_env(
        {
            "LITELLM_PROXY_API_KEY": "proxy-test-value",
            "LITELLM_GATEWAY_ADMIN_KEY": "admin-test-value",
            "PATH": "/bin",
        }
    ) == {"PATH": "/bin"}
    guarded_model = GuardedLitellmModel(
        model_name="litellm_proxy/anthropic/claude-haiku-4-5-20251001",
        model_kwargs={"api_base": "https://gateway.litellm-sandbox.ai"},
        run_budget_usd=0.0001,
        worst_input_usd_per_token=1.25e-6,
        output_usd_per_token=5e-6,
        max_output_tokens=64,
        call_log_path=str(tmp_path / "guard_refusal.jsonl"),
    )
    with pytest.raises(LimitsExceeded):
        guarded_model.query([{"role": "user", "content": "Reply OK."}])
    assert os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "1"
    refusal = json.loads((tmp_path / "guard_refusal.jsonl").read_text())
    assert refusal["event"] == "budget_guard_refused"


def test_gateway_and_run_cost_accounting_handles_missing_costs() -> None:
    from poc.guarded_model import COST_HEADER

    assert gateway_cost({"additional_headers": {COST_HEADER: "0.0042"}}) == 0.0042
    assert gateway_cost({}) is None
    calls = [
        {
            "event": "call",
            "cost_usd": 0.1,
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 1,
            "reasoning_tokens": 1,
        },
        {"event": "call", "cost_usd": None, "prompt_tokens": 5},
        {"event": "budget_guard_refused", "cost_usd": 50},
    ]
    assert sum_call_costs(calls) == (0.1, True)
    assert sum_call_tokens(calls) == {
        "prompt": 15,
        "completion": 3,
        "cache_read": 2,
        "cache_creation": 1,
        "reasoning": 1,
    }


def test_error_classification_distinguishes_infra_and_solver_failures() -> None:
    assert classify_error("Submitted", 0, "patch", "resolved", True) == "none"
    assert classify_error("Submitted", 0, "patch", "unresolved", True) == "solver"
    assert classify_error("BudgetGuard", 1, "", "skipped_empty_patch", True) == "solver"
    assert classify_error("RuntimeError", 1, "", "skipped_empty_patch", True, "Docker startup command failed") == "infra"
    assert classify_error(None, 1, "", "skipped_empty_patch", False) == "infra"
    assert classify_error("Submitted", 124, "patch", "unresolved", True) == "solver"
    assert classify_error("Submitted", 1, "patch", "error", True) == "infra"
    assert classify_error("ModuleNotFoundError", 1, "", "skipped_empty_patch", True) == "infra"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_analysis_reports_fixed_model_costs_oracle_and_case_table(tmp_path: Path) -> None:
    task_ids = ["task-a", "task-b"]
    _write_json(tmp_path / "tasks.json", {"tasks": [{"instance_id": task_id} for task_id in task_ids]})
    _write_json(
        tmp_path / "poc" / "pilot_models.json",
        {
            "solvers": {
                "haiku-4.5": {},
                "sonnet-5": {},
                "opus-5": {},
            }
        },
    )
    _write_json(
        tmp_path / "labels.json",
        [
            {
                "task_id": "task-a",
                "offline": {"label": "repository_debugging", "latency_s": 0.1},
                "jev": {"label": "repository_debugging", "latency_s": 0.2},
            },
            {
                "task_id": "task-b",
                "offline": {"label": "feature_implementation", "latency_s": 0.1},
                "jev": {"label": "shell_build", "latency_s": 0.2},
            },
        ],
    )
    _write_jsonl(
        tmp_path / "runs.jsonl",
        [
            _record("task-a", "haiku-4.5", success=True, cost_usd=0.1),
            _record("task-a", "haiku-4.5", error_class="infra", cost_usd=0.05),
            _record("task-a", "sonnet-5", success=False, cost_usd=0.2),
            _record("task-a", "opus-5", success=True, cost_usd=0.5),
            _record("task-b", "haiku-4.5", error_class="infra", cost_usd=0.03),
            _record("task-b", "sonnet-5", success=True, cost_usd=0.25),
            _record("task-b", "opus-5", success=False, cost_usd=0.4),
        ],
    )
    _write_jsonl(
        tmp_path / "costs_aux.jsonl",
        [
            {"kind": "offline_label", "cost_usd": 0.01},
            {"kind": "jev_classifier", "cost_usd": 0.02},
            {"kind": "smoke", "cost_usd": 0.003},
        ],
    )
    _write_json(
        tmp_path / "preflight_probes.json",
        [{"cost": "0.0006", "latency": 0.05}],
    )

    result = analyze(tmp_path)
    haiku = result["fixed_models"]["haiku-4.5"]
    assert haiku["solved"] == 1
    assert haiku["total_tasks"] == 2
    assert haiku["total_cost_usd"] == pytest.approx(0.18)
    assert haiku["mean_cost_per_attempted_task_usd"] == pytest.approx(0.09)
    assert haiku["cost_per_success_usd"] == pytest.approx(0.18)
    assert [row["retrospective_oracle"] for row in result["cases"]] == ["haiku-4.5", "sonnet-5"]
    assert all(row["jev_decision"] == "opus-5" for row in result["cases"])
    assert "RETROSPECTIVE UPPER BOUND, not achieved routing" in result["report"]
    assert "EXPLORATORY leave-one-task-out" in result["report"]
    assert "task_success and task_cost keys" in result["report"]
    assert "Paired comparisons by category:" in result["report"]
    assert "| repository_debugging | haiku-4.5 | 1 | 0.000 | -2.448 | $0.150000 | $0.500000 |" in result["report"]
    assert "No threshold was tuned" in result["report"]
    assert "Spend: solver" in result["report"]
    assert "total $1.563600" in result["report"]
    assert "Offline label coverage: 2/2. Offline/Jev agreement: 1/2 comparable labels" in result["report"]
    assert "Total recorded runtime: 21.65s" in result["report"]
    assert (tmp_path / "capability_map.json").exists()
    assert (tmp_path / "report.md").exists()
    with (tmp_path / "case_table.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["retrospective_oracle"] == "haiku-4.5"
    assert rows[1]["sonnet-5_success"] == "True"


def test_input_estimate_covers_tool_overhead_and_tracks_observed_prompt_growth():
    config = GuardedModelConfig(
        model_name="litellm_proxy/test-model", run_budget_usd=1.0, worst_input_usd_per_token=0.000002,
        output_usd_per_token=0.00001, max_output_tokens=64, call_log_path="unused.jsonl",
    )
    assert estimate_input_tokens(59, None, None, config) == 59 + config.fixed_overhead_tokens
    assert estimate_input_tokens(9000, 5000, 7000, config) == 11000
    assert estimate_input_tokens(5100, 5000, 5200, config) == 6124
    assert estimate_input_tokens(4000, 5000, 7000, config) == 7000
