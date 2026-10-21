import json
import math
import re
from collections.abc import Mapping, Sequence
from collections import defaultdict
from dataclasses import dataclass
from typing import Final

from scipy.stats import beta

CATEGORIES = ("repository_debugging", "feature_implementation", "shell_build", "algorithms_data", "unknown")


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    reference: str = "opus-5"
    min_paired_tasks: int = 10
    tolerance: float = 0.05
    alpha: float = 0.05
    credible_mass: float = 0.90


@dataclass(frozen=True, slots=True)
class Cell:
    n: int
    s: int
    raw_rate: float | None
    posterior_mean: float
    ci_low: float
    ci_high: float
    mean_cost_per_task_usd: float | None


@dataclass(frozen=True, slots=True)
class Decision:
    model: str
    reason: str


@dataclass(frozen=True, slots=True)
class SpendSuccessCell:
    total_spend_usd: float
    successes: int
    graded_tasks: int
    score_usd_per_success: float | None
    sparse: bool


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    model: str | None
    category: str
    reason: str


def build_spend_success_map(
    records: Sequence[Mapping[str, object]],
    models: Sequence[str],
    sparse_threshold: int = 3,
) -> dict[str, dict[str, SpendSuccessCell]]:
    task_models: Final[dict[tuple[str, str], list[Mapping[str, object]]]] = defaultdict(list)
    for record in records:
        task_id: Final = record.get("task_id")
        model: Final = record.get("model")
        if not isinstance(task_id, str) or not isinstance(model, str):
            raise ValueError("training records require string task_id and model")
        task_models[(task_id, model)].append(record)
    task_outcomes_by_category: Final[dict[str, dict[str, list[tuple[float, bool, bool]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    global_outcomes: Final[dict[str, list[tuple[float, bool, bool]]]] = defaultdict(list)
    for (task_id, model), attempts in task_models.items():
        categories: Final = {
            parse_category(record.get("category_jev"))
            for record in attempts
        }
        if len(categories) > 1:
            raise ValueError(f"training category changed between attempts for {(task_id, model)}")
        costs: Final = tuple(
            float(cost)
            for record in attempts
            if isinstance((cost := record.get("cost_usd")), (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost >= 0
        )
        if len(costs) != len(attempts):
            raise ValueError(f"training record has invalid cost for {(task_id, model)}")
        graded: Final = tuple(record for record in attempts if record.get("error_class") != "infra")
        if len(graded) > 1:
            raise ValueError(f"duplicate graded attempts for {(task_id, model)}")
        success: Final = bool(graded and graded[0].get("success") is True)
        category: Final = next(iter(categories), "unknown")
        outcome: Final = (sum(costs), bool(graded), success)
        task_outcomes_by_category[category][model].append(outcome)
        global_outcomes[model].append(outcome)
    buckets: Final = {**task_outcomes_by_category, "global": global_outcomes}
    return {
        category: {
            model: _spend_success_cell(
                outcomes_by_model.get(model, ()),
                sparse_threshold,
            )
            for model in models
        }
        for category, outcomes_by_model in buckets.items()
    }


def _spend_success_cell(
    outcomes: Sequence[tuple[float, bool, bool]],
    sparse_threshold: int,
) -> SpendSuccessCell:
    total_spend: Final = sum(outcome[0] for outcome in outcomes)
    graded_tasks: Final = sum(outcome[1] for outcome in outcomes)
    successes: Final = sum(outcome[2] for outcome in outcomes)
    return SpendSuccessCell(
        total_spend_usd=total_spend,
        successes=successes,
        graded_tasks=graded_tasks,
        score_usd_per_success=total_spend / successes if successes else None,
        sparse=graded_tasks < sparse_threshold,
    )


def choose_capability_model(
    category: str | None,
    score_map: Mapping[str, Mapping[str, SpendSuccessCell]],
    models: Sequence[str],
    classifier_failed: bool = False,
) -> CapabilityDecision:
    category_name: Final = parse_category(category)
    fallback_reason: Final = (
        "classifier_failure"
        if classifier_failed or category is None
        else "unknown_category"
        if category_name == "unknown"
        else "category_all_zero_success"
    )
    category_cells: Final = score_map.get(category_name, {})
    category_has_success: Final = any(
        (cell := category_cells.get(model)) is not None and cell.score_usd_per_success is not None
        for model in models
    )
    selected_category: Final = (
        category_name
        if category is not None
        and not classifier_failed
        and category_name != "unknown"
        and category_has_success
        else "global"
    )
    reason: Final = (
        "category_score"
        if selected_category != "global"
        else f"global_fallback:{fallback_reason}"
    )
    global_cells: Final = score_map.get(selected_category, {})
    ranked: Final = tuple(
        (cell.score_usd_per_success, model)
        for model in models
        if (cell := global_cells.get(model)) is not None and cell.score_usd_per_success is not None
    )
    if not ranked:
        return CapabilityDecision(
            model=None,
            category=selected_category,
            reason="unproven",
        )
    score, model = min(ranked)
    return CapabilityDecision(model=model, category=selected_category, reason=reason)


def parse_category(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    norm = re.sub(r"[^a-z]+", "_", raw.strip().lower()).strip("_")
    aliases = {"shell_build_work": "shell_build", "algorithms_data_processing": "algorithms_data"}
    norm = aliases.get(norm, norm)
    return norm if norm in CATEGORIES else "unknown"


def beta_cell(s: int, n: int, mass: float, mean_cost: float | None) -> Cell:
    a, b = s + 1, n - s + 1
    tail = (1 - mass) / 2
    return Cell(n=n, s=s, raw_rate=(s / n) if n else None, posterior_mean=a / (a + b),
                ci_low=float(beta.ppf(tail, a, b)), ci_high=float(beta.ppf(1 - tail, a, b)),
                mean_cost_per_task_usd=mean_cost)


def task_outcomes(records: list[dict]) -> dict[tuple[str, str], dict]:
    """One outcome per (task, model). Infra-failed attempts add cost but never success or n."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["task_id"], r["model"])].append(r)
    out = {}
    for key, attempts in grouped.items():
        graded = [a for a in attempts if a["error_class"] != "infra"]
        if len(graded) > 1:
            raise ValueError(f"duplicate graded attempts for {key}: single-attempt pilot")
        out[key] = {
            "graded": bool(graded),
            "success": graded[0]["success"] if graded else None,
            "cost_usd": sum(a["cost_usd"] for a in attempts),
            "attempts": len(attempts),
        }
    return out


def build_capability_map(records: list[dict], task_category: dict[str, str], models: list[str],
                         cfg: PolicyConfig, exclude_task: str | None = None) -> dict:
    outcomes = {k: v for k, v in task_outcomes(records).items() if k[0] != exclude_task}
    cmap: dict = {}
    for cat in CATEGORIES:
        tasks = sorted({t for (t, _m) in outcomes if task_category.get(t, "unknown") == cat})
        if not tasks:
            continue
        cmap[cat] = {}
        for m in models:
            graded = [outcomes[(t, m)] for t in tasks if (t, m) in outcomes and outcomes[(t, m)]["graded"]]
            attempted = [outcomes[(t, m)] for t in tasks if (t, m) in outcomes]
            mean_cost = sum(o["cost_usd"] for o in attempted) / len(attempted) if attempted else None
            cell = beta_cell(sum(o["success"] for o in graded), len(graded), cfg.credible_mass, mean_cost)
            cmap[cat][m] = {
                f: getattr(cell, f) for f in Cell.__slots__} | {"graded_tasks": sorted(
                    t for t in tasks if (t, m) in outcomes and outcomes[(t, m)]["graded"])} | {
                        "task_success": {
                            t: outcomes[(t, m)]["success"]
                            for t in tasks
                            if (t, m) in outcomes and outcomes[(t, m)]["graded"]
                        },
                        "task_cost": {
                            t: outcomes[(t, m)]["cost_usd"]
                            for t in tasks
                            if (t, m) in outcomes
                        },
                    }
    return cmap


def paired_lower_bound(diffs: list[int], alpha: float) -> float:
    """One-sided Hoeffding lower bound for paired differences in [-1, 1]."""
    if not diffs:
        raise ValueError("paired differences cannot be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return sum(diffs) / len(diffs) - math.sqrt(2 * math.log(1 / alpha) / len(diffs))


def paired_comparison(cand_cell: dict, ref_cell: dict, alpha: float) -> dict[str, int | float | None]:
    cand_success = cand_cell["task_success"]
    ref_success = ref_cell["task_success"]
    cand_cost = cand_cell["task_cost"]
    ref_cost = ref_cell["task_cost"]
    paired_tasks = sorted(set(cand_success) & set(ref_success) & set(cand_cost) & set(ref_cost))
    if not paired_tasks:
        return {"n": 0, "mean_diff": None, "lower_bound": None, "cand_cost": None, "ref_cost": None}
    diffs = [int(cand_success[task]) - int(ref_success[task]) for task in paired_tasks]
    return {
        "n": len(paired_tasks),
        "mean_diff": sum(diffs) / len(diffs),
        "lower_bound": paired_lower_bound(diffs, alpha),
        "cand_cost": sum(cand_cost[task] for task in paired_tasks) / len(paired_tasks),
        "ref_cost": sum(ref_cost[task] for task in paired_tasks) / len(paired_tasks),
    }


def choose_model(category: str | None, cmap: dict, cfg: PolicyConfig) -> Decision:
    cat = parse_category(category)
    if category is None:
        return Decision(cfg.reference, "fallback: classifier failure")
    if cat == "unknown" or cat not in cmap or cfg.reference not in cmap[cat]:
        return Decision(cfg.reference, f"fallback: no profile for category {cat}")
    cells = cmap[cat]
    ref = cells[cfg.reference]
    eligible = []
    for m, c in cells.items():
        if m == cfg.reference:
            continue
        comparison = paired_comparison(c, ref, cfg.alpha)
        if comparison["n"] < cfg.min_paired_tasks:
            continue
        if (
            comparison["lower_bound"] is not None
            and comparison["lower_bound"] >= -cfg.tolerance
            and comparison["cand_cost"] is not None
            and comparison["ref_cost"] is not None
            and comparison["cand_cost"] < comparison["ref_cost"]
        ):
            eligible.append((comparison["cand_cost"], m))
    if not eligible:
        return Decision(cfg.reference, f"fallback: no cheaper model with >= {cfg.min_paired_tasks} paired tasks "
                                       f"and paired lower bound >= -{cfg.tolerance} (alpha={cfg.alpha})")
    return Decision(min(eligible)[1], "cheapest model with paired lower bound within tolerance of reference")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
