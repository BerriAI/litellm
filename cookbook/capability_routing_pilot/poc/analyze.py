import csv
import json
import math
from pathlib import Path

from poc.policy import (
    PolicyConfig,
    beta_cell,
    build_capability_map,
    choose_model,
    load_jsonl,
    paired_comparison,
    parse_category,
    task_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("haiku-4.5", "sonnet-5", "opus-5")
PREFLIGHT_PATH = "preflight_probes.json"


def _json(path: Path, fallback: object) -> object:
    return json.loads(path.read_text()) if path.exists() else fallback


def _cost(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else 0.0
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return 0.0
        return number if math.isfinite(number) else 0.0
    return 0.0


def _labels(path: Path) -> dict[str, dict[str, object]]:
    raw = _json(path / "labels.json", [])
    return {
        row["task_id"]: row
        for row in raw
        if isinstance(row, dict) and isinstance(row.get("task_id"), str)
    }


def _task_ids(path: Path) -> list[str]:
    payload = _json(path / "tasks.json", {"tasks": []})
    return [task["instance_id"] for task in payload["tasks"]]


def _preflight_cost(path: Path) -> float:
    probes = _json(path / PREFLIGHT_PATH, [])
    return sum(_cost(probe.get("cost")) for probe in probes if isinstance(probe, dict))


def _jev_cost_by_task(aux: list[dict[str, object]]) -> dict[str, float]:
    jev_rows = [
        row
        for row in aux
        if row.get("kind") in {"jev", "jev_classifier"} and isinstance(row.get("task_id"), str)
    ]
    return {
        task_id: sum(_cost(row.get("cost_usd")) for row in jev_rows if row.get("task_id") == task_id)
        for task_id in {str(row["task_id"]) for row in jev_rows}
    }


def _aux_cost_for_kinds(aux: list[dict[str, object]], kinds: set[str]) -> float:
    return sum(_cost(row.get("cost_usd")) for row in aux if row.get("kind") in kinds)


def _aux_costs(path: Path) -> list[dict[str, object]]:
    cost_path = path / "costs_aux.jsonl"
    return load_jsonl(str(cost_path)) if cost_path.exists() else []


def _model_outcome(outcomes: dict[tuple[str, str], dict[str, object]], task_id: str, model: str) -> dict[str, object]:
    return outcomes.get((task_id, model), {"graded": False, "success": None, "cost_usd": 0.0, "attempts": 0})


def _last_exit_status(
    records: list[dict[str, object]], task_id: str, model: str
) -> str:
    attempts = [row for row in records if row["task_id"] == task_id and row["model"] == model]
    return " | ".join(str(row["exit_status"]) for row in attempts if row.get("exit_status")) or "not_run"


def _oracle(outcomes: dict[tuple[str, str], dict[str, object]], task_id: str) -> str:
    successful = [
        model for model in MODEL_ORDER if _model_outcome(outcomes, task_id, model)["success"] is True
    ]
    return min(successful, key=lambda model: (_model_outcome(outcomes, task_id, model)["cost_usd"], MODEL_ORDER.index(model))) if successful else "none"


def _fixed_model_summaries(
    task_ids: list[str],
    outcomes: dict[tuple[str, str], dict[str, object]],
    records: list[dict[str, object]],
    cfg: PolicyConfig,
) -> dict[str, dict[str, object]]:
    return {
        model: _fixed_model_summary(task_ids, model, outcomes, records, cfg)
        for model in MODEL_ORDER
    }


def _fixed_model_summary(
    task_ids: list[str],
    model: str,
    outcomes: dict[tuple[str, str], dict[str, object]],
    records: list[dict[str, object]],
    cfg: PolicyConfig,
) -> dict[str, object]:
    model_outcomes = [_model_outcome(outcomes, task_id, model) for task_id in task_ids]
    graded_outcomes = [outcome for outcome in model_outcomes if outcome["graded"]]
    solved = sum(outcome["success"] is True for outcome in model_outcomes)
    total_cost = sum(float(outcome["cost_usd"]) for outcome in model_outcomes)
    attempted_tasks = sum(outcome["attempts"] > 0 for outcome in model_outcomes)
    n_graded = len(graded_outcomes)
    n_missing = sum(outcome["attempts"] == 0 for outcome in model_outcomes)
    n_infra_only = sum(outcome["attempts"] > 0 and not outcome["graded"] for outcome in model_outcomes)
    cell = beta_cell(solved, n_graded, cfg.credible_mass, total_cost / attempted_tasks if attempted_tasks else None)
    return {
        "solved": solved,
        "n_graded": n_graded,
        "n_missing": n_missing,
        "n_infra_only": n_infra_only,
        "total_tasks": len(task_ids),
        "raw_rate": solved / n_graded if n_graded else None,
        "posterior_mean": cell.posterior_mean,
        "ci_low": cell.ci_low,
        "ci_high": cell.ci_high,
        "total_cost_usd": total_cost,
        "attempted_tasks": attempted_tasks,
        "mean_cost_per_attempted_task_usd": total_cost / attempted_tasks if attempted_tasks else None,
        "cost_per_success_usd": total_cost / solved if solved else None,
        "attempt_records": sum(row["model"] == model for row in records),
    }


def _case_rows(
    task_ids: list[str],
    labels: dict[str, dict[str, object]],
    records: list[dict[str, object]],
    outcomes: dict[tuple[str, str], dict[str, object]],
    capability_map: dict[str, dict[str, object]],
    cfg: PolicyConfig,
    jev_cost_by_task: dict[str, float],
) -> list[dict[str, object]]:
    rows = []
    for task_id in task_ids:
        task_labels = labels.get(task_id, {})
        offline_data = task_labels.get("offline", {})
        jev_data = task_labels.get("jev", {})
        offline_label = offline_data.get("label") if isinstance(offline_data, dict) else None
        jev_label = jev_data.get("label") if isinstance(jev_data, dict) else None
        decision = choose_model(jev_label, capability_map, cfg)
        row: dict[str, object] = {
            "task_id": task_id,
            "offline_label": offline_label,
            "jev_label": jev_label,
        }
        for model in MODEL_ORDER:
            outcome = _model_outcome(outcomes, task_id, model)
            row[f"{model}_success"] = outcome["success"]
            row[f"{model}_cost_usd"] = outcome["cost_usd"]
            row[f"{model}_exit_status"] = _last_exit_status(records, task_id, model)
            row[f"{model}_attempts"] = outcome["attempts"]
        row["jev_decision"] = decision.model
        row["jev_decision_reason"] = decision.reason
        row["retrospective_oracle"] = _oracle(outcomes, task_id)
        replay_outcome = _model_outcome(outcomes, task_id, decision.model)
        jev_cost = jev_cost_by_task.get(task_id, 0.0)
        row["replay_model"] = decision.model
        row["replay_success"] = replay_outcome["success"]
        row["replay_model_cost_usd"] = replay_outcome["cost_usd"]
        row["jev_classification_cost_usd"] = jev_cost
        row["replay_total_cost_usd"] = float(replay_outcome["cost_usd"]) + jev_cost
        rows.append(row)
    return rows


def _leave_one_out(
    task_ids: list[str],
    labels: dict[str, dict[str, object]],
    records: list[dict[str, object]],
    categories: dict[str, str],
    cfg: PolicyConfig,
) -> list[tuple[str, str, str]]:
    return [
        _leave_one_out_row(task_id, task_ids, labels, records, categories, cfg)
        for task_id in task_ids
    ]


def _leave_one_out_row(
    task_id: str,
    task_ids: list[str],
    labels: dict[str, dict[str, object]],
    records: list[dict[str, object]],
    categories: dict[str, str],
    cfg: PolicyConfig,
) -> tuple[str, str, str]:
    task_labels = labels.get(task_id, {})
    jev_data = task_labels.get("jev", {})
    jev_label = jev_data.get("label") if isinstance(jev_data, dict) else None
    profile = build_capability_map(records, categories, list(MODEL_ORDER), cfg, exclude_task=task_id)
    decision = choose_model(jev_label, profile, cfg)
    return task_id, decision.model, decision.reason


def _render_report(
    task_ids: list[str],
    labels: dict[str, dict[str, object]],
    records: list[dict[str, object]],
    outcomes: dict[tuple[str, str], dict[str, object]],
    case_rows: list[dict[str, object]],
    fixed: dict[str, dict[str, object]],
    loo_rows: list[tuple[str, str, str]],
    aux: list[dict[str, object]],
    preflight_cost: float,
    total_runtime_s: float,
    capability_map: dict[str, dict[str, object]],
    cfg: PolicyConfig,
) -> str:
    offline_count = sum(
        isinstance(labels.get(task_id, {}).get("offline"), dict)
        and labels[task_id]["offline"].get("label") is not None
        for task_id in task_ids
    )
    agreement_rows = [
        (labels[task_id]["offline"].get("label"), labels[task_id]["jev"].get("label"))
        for task_id in task_ids
        if isinstance(labels.get(task_id, {}).get("offline"), dict)
        and isinstance(labels.get(task_id, {}).get("jev"), dict)
        and labels[task_id]["offline"].get("label") is not None
        and labels[task_id]["jev"].get("label") is not None
    ]
    agreement = sum(offline == jev for offline, jev in agreement_rows)
    replay_graded = sum(row["replay_success"] is not None for row in case_rows)
    replay_successes = sum(row["replay_success"] is True for row in case_rows)
    replay_model_cost = sum(float(row["replay_model_cost_usd"]) for row in case_rows)
    replay_jev_cost = sum(float(row["jev_classification_cost_usd"]) for row in case_rows)
    replay_total_cost = sum(float(row["replay_total_cost_usd"]) for row in case_rows)
    label_cost = _aux_cost_for_kinds(aux, {"offline_label", "labeler"})
    classifier_cost = _aux_cost_for_kinds(aux, {"jev", "jev_classifier"})
    smoke_cost = _aux_cost_for_kinds(aux, {"smoke"})
    other_aux_cost = sum(
        _cost(row.get("cost_usd"))
        for row in aux
        if row.get("kind") not in {"offline_label", "labeler", "jev", "jev_classifier", "smoke"}
    )
    solver_cost = sum(_cost(row.get("cost_usd")) for row in records)
    infra_failures = sum(row.get("error_class") == "infra" for row in records)
    solver_failures = sum(row.get("error_class") == "solver" for row in records)
    report = [
        "# Capability routing pilot analysis",
        "",
        "Five frozen tasks are too few to establish general routing superiority. No threshold was tuned, no validation set exists, and these results are not evidence of routing improvement",
        "",
        "## Per-task cases",
        "",
        "| Task | Offline | Jev | Haiku success / cost / exit / attempts | Sonnet success / cost / exit / attempts | Opus success / cost / exit / attempts | Jev decision | Decision reason | Retrospective oracle |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in case_rows:
        cells = [
            str(row["task_id"]),
            str(row["offline_label"] or "missing"),
            str(row["jev_label"] or "classifier failure"),
        ]
        for model in MODEL_ORDER:
            success = row[f"{model}_success"]
            cells.append(
                f"{success if success is not None else 'not graded'} / ${float(row[f'{model}_cost_usd']):.6f} / "
                f"{row[f'{model}_exit_status']} / {row[f'{model}_attempts']}"
            )
        cells.extend([str(row["jev_decision"]), str(row["jev_decision_reason"]), str(row["retrospective_oracle"])])
        report.append("| " + " | ".join(value.replace("|", "\\|") for value in cells) + " |")
    report.extend(["", "## Fixed-model results", "", "| Model | Solved / n_graded | n_missing | n_infra_only | Raw rate | Beta(1,1) mean | 90% CI | Total cost | Mean cost per attempted task | Cost per success |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"])
    for model in MODEL_ORDER:
        result = fixed[model]
        ci = f"{result['ci_low']:.3f}-{result['ci_high']:.3f}"
        mean_cost = result["mean_cost_per_attempted_task_usd"]
        success_cost = result["cost_per_success_usd"]
        raw_rate = result["raw_rate"]
        report.append(
            f"| {model} | {result['solved']}/{result['n_graded']} | {result['n_missing']} | "
            f"{result['n_infra_only']} | {f'{raw_rate:.3f}' if raw_rate is not None else 'n/a'} | "
            f"{result['posterior_mean']:.3f} | {ci} | ${result['total_cost_usd']:.6f} | "
            f"{f'${mean_cost:.6f}' if mean_cost is not None else 'n/a'} | "
            f"{f'${success_cost:.6f}' if success_cost is not None else 'n/a'} |"
        )
    report.extend(
        [
            "",
            "## Retrospective router replay (not an observed routed run)",
            "",
            f"The retrospective replay selected {replay_successes}/{replay_graded} graded outcomes and measured model "
            f"costs of ${replay_model_cost:.6f}. Recorded Jev classification cost was ${replay_jev_cost:.6f}, "
            f"for a retrospective total of ${replay_total_cost:.6f}. It fell back to the reference model for "
            f"{sum(row['jev_decision'] == cfg.reference for row in case_rows)}/{len(task_ids)} tasks. "
            f"For each candidate, T is the sorted intersection of reference and candidate task_success and task_cost keys. "
            f"For each task in T, d is candidate success minus reference success, and the one-sided Hoeffding lower "
            f"bound is mean(d) - sqrt(2*ln(1/alpha)/n), with alpha={cfg.alpha}. A candidate is eligible iff "
            f"n >= {cfg.min_paired_tasks}, the lower bound is >= -{cfg.tolerance}, and its paired mean cost is lower "
            "than the reference paired mean cost. The router picks the eligible candidate with the lowest paired "
            "mean cost, breaking ties by model name, and otherwise falls back to the reference model",
            "",
            "| Task | Jev label | Selected model | Selected success | Selected model cost | Jev classification cost | Retrospective total cost |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in case_rows:
        jev_label = row["jev_label"] or "classifier failure"
        replay_success = row["replay_success"]
        report.append(
            f"| {row['task_id']} | {jev_label} | {row['replay_model']} | "
            f"{replay_success if replay_success is not None else 'not graded'} | "
            f"${float(row['replay_model_cost_usd']):.6f} | "
            f"${float(row['jev_classification_cost_usd']):.6f} | "
            f"${float(row['replay_total_cost_usd']):.6f} |"
        )
    report.extend(
        [
            "",
            "Paired comparisons by category:",
            "",
            "| Category | Candidate | Paired n | Mean diff | Lower bound | Candidate paired mean cost | Reference paired mean cost |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, cells in sorted(capability_map.items()):
        reference = cells.get(cfg.reference)
        candidates = [(model, cell) for model, cell in cells.items() if model != cfg.reference]
        if not candidates:
            report.append(f"| {category} | n/a | 0 | n/a | n/a | n/a | n/a |")
            continue
        if reference is None:
            report.extend(f"| {category} | {model} | 0 | n/a | n/a | n/a | n/a |" for model, _ in candidates)
            continue
        for model, candidate in sorted(candidates, key=lambda item: item[0]):
            comparison = paired_comparison(candidate, reference, cfg.alpha)
            mean_diff = comparison["mean_diff"]
            lower_bound = comparison["lower_bound"]
            candidate_cost = comparison["cand_cost"]
            reference_cost = comparison["ref_cost"]
            report.append(
                f"| {category} | {model} | {comparison['n']} | "
                f"{f'{mean_diff:.3f}' if mean_diff is not None else 'n/a'} | "
                f"{f'{lower_bound:.3f}' if lower_bound is not None else 'n/a'} | "
                f"{f'${candidate_cost:.6f}' if candidate_cost is not None else 'n/a'} | "
                f"{f'${reference_cost:.6f}' if reference_cost is not None else 'n/a'} |"
            )
    report.extend(
        [
            "",
            "## RETROSPECTIVE UPPER BOUND, not achieved routing",
            "",
            "For each task, the oracle selects the cheapest model that succeeded on that task. This uses observed outcomes and is not an executable routing result",
            "",
            "## EXPLORATORY leave-one-task-out",
            "",
            "| Excluded task | Jev label decision | Reason |",
            "| --- | --- | --- |",
        ]
    )
    report.extend(f"| {task_id} | {model} | {reason} |" for task_id, model, reason in loo_rows)
    report.extend(
        [
            "",
            "## Label agreement and resource use",
            "",
            f"Offline label coverage: {offline_count}/{len(task_ids)}. Offline/Jev agreement: {agreement}/{len(agreement_rows)} comparable labels",
            f"Spend totals: solver ${solver_cost:.6f}, labeler ${label_cost:.6f}, Jev ${classifier_cost:.6f}, "
            f"smoke ${smoke_cost:.6f}, preflight ${preflight_cost:.6f}, other auxiliary calls ${other_aux_cost:.6f}, "
            f"total ${solver_cost + label_cost + classifier_cost + smoke_cost + preflight_cost + other_aux_cost:.6f}",
            f"Total recorded runtime: {total_runtime_s:.2f}s. Infrastructure failure attempts: {infra_failures}. "
            f"Solver failure attempts: {solver_failures}",
            f"Capability map categories: {', '.join(sorted(capability_map)) or 'none'}",
        ]
    )
    return "\n".join(report) + "\n"


def analyze(data_dir: Path = ROOT) -> dict[str, object]:
    records = load_jsonl(str(data_dir / "runs.jsonl")) if (data_dir / "runs.jsonl").exists() else []
    labels = _labels(data_dir)
    task_ids = _task_ids(data_dir)
    model_payload = _json(data_dir / "poc" / "pilot_models.json", {"solvers": {}})
    models = [model for model in MODEL_ORDER if model in model_payload["solvers"]]
    cfg = PolicyConfig()
    categories = {
        task_id: parse_category(
            str(labels.get(task_id, {}).get("offline", {}).get("label", "unknown"))
            if isinstance(labels.get(task_id, {}).get("offline"), dict)
            else "unknown"
        )
        for task_id in task_ids
    }
    capability_map = build_capability_map(records, categories, models, cfg)
    outcomes = task_outcomes(records)
    fixed = _fixed_model_summaries(task_ids, outcomes, records, cfg)
    aux = _aux_costs(data_dir)
    jev_cost_by_task = _jev_cost_by_task(aux)
    case_rows = _case_rows(task_ids, labels, records, outcomes, capability_map, cfg, jev_cost_by_task)
    loo_rows = _leave_one_out(task_ids, labels, records, categories, cfg)
    preflight_cost = _preflight_cost(data_dir)
    label_runtime = sum(
        float(value.get("latency_s", 0.0))
        for task in labels.values()
        for key in ("offline", "jev")
        if isinstance((value := task.get(key)), dict)
        and isinstance(value.get("latency_s"), (int, float))
    )
    preflight = _json(data_dir / PREFLIGHT_PATH, [])
    preflight_runtime = sum(
        float(row.get("latency", 0.0))
        for row in preflight
        if isinstance(row, dict) and isinstance(row.get("latency"), (int, float))
    )
    total_runtime_s = (
        sum(_cost(row.get("solve_latency_s")) + _cost(row.get("eval_latency_s")) for row in records)
        + label_runtime
        + preflight_runtime
    )
    report = _render_report(
        task_ids,
        labels,
        records,
        outcomes,
        case_rows,
        fixed,
        loo_rows,
        aux,
        preflight_cost,
        total_runtime_s,
        capability_map,
        cfg,
    )
    (data_dir / "capability_map.json").write_text(json.dumps(capability_map, indent=2) + "\n")
    (data_dir / "report.md").write_text(report)
    if case_rows:
        with (data_dir / "case_table.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(case_rows[0]))
            writer.writeheader()
            writer.writerows(case_rows)
    return {
        "capability_map": capability_map,
        "fixed_models": fixed,
        "cases": case_rows,
        "leave_one_out": loo_rows,
        "report": report,
    }


def main() -> None:
    result = analyze()
    print(
        json.dumps(
            {
                "task_count": len(result["cases"]),
                "model_summaries": result["fixed_models"],
                "report": str(ROOT / "report.md"),
                "case_table": str(ROOT / "case_table.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
