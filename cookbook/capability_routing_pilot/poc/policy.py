import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass

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
