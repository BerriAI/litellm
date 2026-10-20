import argparse
import fcntl
import json
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import yaml

from poc.guarded_model import valid_cost

ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "runs.jsonl"
AUX_COSTS_PATH = ROOT / "costs_aux.jsonl"
RUNS_ROOT = ROOT / "runs_raw"
GLOBAL_BUDGET_USD = 15.0
RESERVATION_USD = 1.0
MIN_FREE_BYTES = 50 * 1024**3
EMERGENCY_FREE_BYTES = 20 * 1024**3
MAX_LOG_BYTES = 200 * 1024
INFRA_EXIT_STATUSES = {
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "RateLimitError",
    "BadGatewayError",
    "Timeout",
    "ImportError",
    "ModuleNotFoundError",
    "DockerException",
    "ImageNotFound",
    "ContainerError",
    "EnvironmentError",
    "ConnectionError",
}
SOLVER_EXIT_STATUSES = {"Submitted", "LimitsExceeded", "BudgetGuard", "TimeExceeded", "FormatError"}
DATASET_METADATA = {
    "agent": "princeton-nlp/SWE-bench@e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6 dev (sanitized)",
    "eval": "SWE-bench/SWE-bench@c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2 dev",
}
TOKEN_FIELDS = {
    "prompt": "prompt_tokens",
    "completion": "completion_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_creation": "cache_creation_input_tokens",
    "reasoning": "reasoning_tokens",
}


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(json.dumps(row, separators=(",", ":")) + "\n")
        file.flush()
        fcntl.flock(file, fcntl.LOCK_UN)


def read_jsonl_locked(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open() as file:
        fcntl.flock(file, fcntl.LOCK_SH)
        rows = [json.loads(line) for line in file if line.strip()]
        fcntl.flock(file, fcntl.LOCK_UN)
    return [row for row in rows if isinstance(row, dict)]


def committed_spend_usd(runs_path: Path = RUNS_PATH, aux_path: Path = AUX_COSTS_PATH) -> float:
    run_cost = sum(_cost(row.get("cost_usd")) for row in read_jsonl_locked(runs_path))
    aux_cost = sum(_cost(row.get("cost_usd")) for row in read_jsonl_locked(aux_path))
    return run_cost + aux_cost


def _cost(value: object) -> float:
    return float(value) if valid_cost(value) else 0.0


def _utility_env(env: dict[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in env.items()
        if name not in {"LITELLM_PROXY_API_KEY", "LITELLM_GATEWAY_ADMIN_KEY"}
    }


def global_budget_allows(committed_spend: float, in_flight: int, budget: float = GLOBAL_BUDGET_USD) -> bool:
    return committed_spend + RESERVATION_USD * (in_flight + 1) <= budget


def sum_call_costs(calls: list[dict[str, object]]) -> tuple[float, bool]:
    present_costs = [call.get("cost_usd") for call in calls if call.get("event") == "call"]
    return sum(_cost(cost) for cost in present_costs), any(not valid_cost(cost) for cost in present_costs)


def sum_call_tokens(calls: list[dict[str, object]]) -> dict[str, int]:
    totals = {name: 0 for name in TOKEN_FIELDS}
    for call in calls:
        if call.get("event") != "call":
            continue
        for name, source in TOKEN_FIELDS.items():
            value = call.get(source)
            totals[name] += int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
    return totals


def classify_error(
    exit_status: str | None,
    rc: int,
    submission: str,
    eval_status: str | None,
    trajectory_found: bool,
    exception_text: str = "",
    cost_incomplete: bool = False,
    emergency_disk_stop: bool = False,
) -> str:
    if exit_status == "CostAccountingError" or cost_incomplete:
        return "accounting"
    if emergency_disk_stop:
        return "infra"
    if eval_status == "resolved":
        return "none"
    if eval_status == "error":
        return "infra"
    if rc == 124 or exit_status in SOLVER_EXIT_STATUSES:
        return "solver"
    if not trajectory_found and rc != 124:
        return "infra"
    if exit_status in INFRA_EXIT_STATUSES:
        return "infra"
    if exit_status == "RuntimeError" and any(
        marker in exception_text.lower() for marker in ("environment", "docker", "startup command")
    ):
        return "infra"
    return "solver" if submission or eval_status == "unresolved" or trajectory_found else "infra"


def retry_eligible(
    error_class: str,
    attempt: int,
    n_calls: int,
    cost_usd: float,
    emergency_disk_stop: bool = False,
) -> bool:
    return (
        attempt == 1
        and error_class == "infra"
        and n_calls == 0
        and cost_usd == 0
        and not emergency_disk_stop
    )


def emergency_disk_stop_required(free_bytes: int) -> bool:
    return free_bytes < EMERGENCY_FREE_BYTES


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_pilot_key() -> str:
    path = ROOT / ".private" / "pilot_key"
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError("pilot key must have mode 0600")
    key = path.read_text().strip()
    if not key:
        raise ValueError("pilot key is empty")
    return key


def _read_json(path: Path) -> object:
    return json.loads(path.read_text())


def _labels_by_task() -> dict[str, dict[str, object]]:
    path = ROOT / "labels.json"
    if not path.exists():
        return {}
    entries = _read_json(path)
    return {row["task_id"]: row for row in entries if isinstance(row, dict) and isinstance(row.get("task_id"), str)}


def _model_config(model: str, run_id: str, model_data: dict[str, object], base_url: str) -> dict[str, object]:
    return {
        "model": {
            "model_name": f"litellm_proxy/{model_data['gateway_model']}",
            "model_kwargs": {"api_base": base_url},
            "worst_input_usd_per_token": max(
                float(model_data["input"]),
                float(model_data.get("cache_write", model_data["input"])),
            ),
            "output_usd_per_token": float(model_data["output"]),
            "call_log_path": str(RUNS_ROOT / run_id / "calls.jsonl"),
        },
        "environment": {
            "run_args": ["--rm", "--label", "routing_poc=1", "--label", f"routing_poc_run={run_id}"],
        },
    }


def _bounded_log(path: Path) -> None:
    size = path.stat().st_size
    if size <= MAX_LOG_BYTES:
        return
    with path.open("rb") as file:
        file.seek(-MAX_LOG_BYTES, os.SEEK_END)
        tail = file.read()
    path.write_bytes(tail)


def _calls(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trajectory(path: Path) -> tuple[dict[str, object], bool]:
    if not path.exists():
        return {}, False
    payload = _read_json(path)
    info = payload.get("info") if isinstance(payload, dict) else None
    return (info if isinstance(info, dict) else {}), True


def _tail_exception_text(info: dict[str, object]) -> str:
    return str(info.get("exception_str", ""))


def _report_status(report_dir: Path, task_id: str) -> str | None:
    for path in report_dir.rglob("*.json"):
        try:
            report = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        if task_id in report.get("resolved_ids", []):
            return "resolved"
        if task_id in report.get("unresolved_ids", []):
            return "unresolved"
        if task_id in report.get("error_ids", []):
            return "error"
    return None


def _grade(task_id: str, model: str, submission: str, run_id: str, run_dir: Path, env: dict[str, str]) -> tuple[str, float]:
    if not submission.strip():
        return "skipped_empty_patch", 0.0
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": task_id,
                    "model_name_or_path": model,
                    "model_patch": submission,
                }
            ],
            indent=2,
        )
        + "\n"
    )
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        "SWE-bench/SWE-bench",
        "-s",
        "dev",
        "-i",
        task_id,
        "-p",
        str(predictions_path),
        "-id",
        run_id,
        "--max_workers",
        "1",
        "-t",
        "900",
        "--report_dir",
        str(eval_dir),
    ]
    started = time.monotonic()
    with (eval_dir / "eval.log").open("wb") as log:
        try:
            subprocess.run(
                command,
                cwd=eval_dir,
                env=_utility_env(env),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=930,
            )
        except subprocess.TimeoutExpired:
            pass
    _bounded_log(eval_dir / "eval.log")
    status = _report_status(eval_dir, task_id)
    if status is None:
        return "error", time.monotonic() - started
    return status, time.monotonic() - started


def _cleanup_containers(run_id: str, env: dict[str, str]) -> int:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label=routing_poc_run={run_id}"],
        cwd=ROOT,
        env=_utility_env(env),
        capture_output=True,
        text=True,
        check=False,
    )
    container_ids = result.stdout.split() if result.returncode == 0 else []
    if not container_ids:
        return 0
    removed = subprocess.run(
        ["docker", "rm", "-f", *container_ids],
        cwd=ROOT,
        env=_utility_env(env),
        capture_output=True,
    )
    return len(container_ids) if removed.returncode == 0 else 0


def _dataset_metadata(task_id: str) -> tuple[str | None, dict[str, str]]:
    rows = [json.loads(line) for line in (ROOT / "agent_dataset" / "train.jsonl").read_text().splitlines()]
    task = next((row for row in rows if row.get("instance_id") == task_id), {})
    image_lines = (ROOT / "images_pinned.txt").read_text().splitlines()
    image_pins = {
        parts[0]: parts[1]
        for line in image_lines
        if len(parts := line.split()) >= 2
    }
    return task.get("base_commit"), image_pins


def _attempt(
    task_id: str,
    model: str,
    attempt: int,
    task: dict[str, object],
    model_data: dict[str, object],
    category_offline: object,
    category_jev: object,
    env: dict[str, str],
    base_url: str,
) -> dict[str, object]:
    started_at = _utc_now()
    started = time.monotonic()
    run_id = f"{task_id}__{model}__a{attempt}__{_timestamp()}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = _model_config(model, run_id, model_data, base_url)
    model_config_path = run_dir / "model.yaml"
    model_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    solve_dir = run_dir / "solve"
    solve_log = run_dir / "solve.log"
    command = [
        "timeout",
        "600",
        str(ROOT / ".venv" / "bin" / "mini-extra"),
        "swebench",
        "--subset",
        str(ROOT / "agent_dataset"),
        "--split",
        "train",
        "--filter",
        f"^{task_id}$",
        "-o",
        str(solve_dir),
        "-w",
        "1",
        "-c",
        "swebench.yaml",
        "-c",
        "poc/pilot.yaml",
        "-c",
        str(model_config_path),
    ]
    emergency_disk_stop = False
    with solve_log.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if emergency_disk_stop_required(shutil.disk_usage("/").free):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                emergency_disk_stop = True
                break
            time.sleep(10)
        rc = process.wait()
    _bounded_log(solve_log)
    solve_latency_s = time.monotonic() - started
    containers_removed = _cleanup_containers(run_id, env)
    trajectory_path = solve_dir / task_id / f"{task_id}.traj.json"
    info, trajectory_found = _trajectory(trajectory_path)
    exit_status = info.get("exit_status")
    submission = info.get("submission") if isinstance(info.get("submission"), str) else ""
    calls_path = run_dir / "calls.jsonl"
    calls = _calls(calls_path)
    cost_usd, cost_incomplete = sum_call_costs(calls)
    tokens = sum_call_tokens(calls)
    eval_status, eval_latency_s = _grade(task_id, model, submission, run_id, run_dir, env)
    if eval_status == "skipped_empty_patch":
        success = False
    else:
        success = eval_status == "resolved"
    error_class = classify_error(
        str(exit_status) if exit_status is not None else None,
        rc,
        submission,
        eval_status,
        trajectory_found,
        _tail_exception_text(info),
        cost_incomplete,
        emergency_disk_stop,
    )
    base_commit, image_pins = _dataset_metadata(task_id)
    call_rows = [call for call in calls if call.get("event") == "call"]
    model_stats = info.get("model_stats")
    trajectory_calls = model_stats.get("api_calls") if isinstance(model_stats, dict) else None
    result = {
        "task_id": task_id,
        "run_id": run_id,
        "attempt": attempt,
        "model": model,
        "gateway_model": model_data["gateway_model"],
        "returned_models": sorted({str(call["returned_model"]) for call in call_rows if call.get("returned_model")}),
        "deployment_ids": sorted({str(call["gateway_model_id"]) for call in call_rows if call.get("gateway_model_id")}),
        "category_offline": category_offline,
        "category_jev": category_jev,
        "versions": {
            "mini_swe_agent": _package_version("mini-swe-agent"),
            "swebench": _package_version("swebench"),
            "litellm": _package_version("litellm"),
            "python": os.sys.version.split()[0],
        },
        "dataset": DATASET_METADATA,
        "image": task.get("image"),
        "image_digest": image_pins.get(str(task.get("image"))),
        "base_commit": base_commit,
        "exit_status": exit_status,
        "traj_api_calls": trajectory_calls,
        "success": success,
        "eval_status": eval_status,
        "error_class": error_class,
        "rc": rc,
        "cost_usd": cost_usd,
        "cost_incomplete": cost_incomplete,
        "emergency_disk_stop": emergency_disk_stop,
        "n_calls": len(call_rows),
        "tokens": tokens,
        "solve_latency_s": solve_latency_s,
        "eval_latency_s": eval_latency_s,
        "call_ids": [call.get("call_id") for call in call_rows if call.get("call_id")],
        "budget_guard_refused": any(call.get("event") == "budget_guard_refused" for call in calls),
        "retry_eligible": retry_eligible(
            error_class,
            attempt,
            len(call_rows),
            cost_usd,
            emergency_disk_stop,
        ),
        "containers_removed": containers_removed,
        "started_at": started_at,
        "finished_at": _utc_now(),
    }
    append_jsonl(RUNS_PATH, result)
    return result


def _package_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


class DispatchGuard:
    def __init__(
        self,
        stop_path: Path,
        budget: float = GLOBAL_BUDGET_USD,
        free_bytes: Callable[[], int] = lambda: shutil.disk_usage("/").free,
        committed_spend: Callable[[], float] = committed_spend_usd,
    ):
        self._lock = threading.Lock()
        self._in_flight = 0
        self._stopped = False
        self._stop_path = stop_path
        self._budget = budget
        self._free_bytes = free_bytes
        self._committed_spend = committed_spend

    def acquire(self, task_id: str) -> bool:
        with self._lock:
            if self._stopped:
                return False
            free_bytes = self._free_bytes()
            if free_bytes < MIN_FREE_BYTES:
                reason = "disk_below_reserve_50GiB"
            elif not global_budget_allows(self._committed_spend(), self._in_flight, self._budget):
                reason = "global_spend_reservation_exceeds_budget"
            else:
                self._in_flight += 1
                return True
            self._stopped = True
            append_jsonl(
                self._stop_path,
                {"task_id": task_id, "reason": reason, "free_bytes": free_bytes, "ts": _utc_now()},
            )
            return False

    def stop_for_accounting(self, task_id: str) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            append_jsonl(
                self._stop_path,
                {"task_id": task_id, "reason": "cost_accounting_incomplete", "ts": _utc_now()},
            )

    def release(self) -> None:
        with self._lock:
            self._in_flight -= 1


def _run_task(
    task_id: str,
    task: dict[str, object],
    models: list[str],
    solvers: dict[str, dict[str, object]],
    labels: dict[str, dict[str, object]],
    guard: DispatchGuard,
    base_url: str,
    base_env: dict[str, str],
) -> list[dict[str, object]]:
    offline = labels.get(task_id, {}).get("offline", {})
    jev = labels.get(task_id, {}).get("jev", {})
    category_offline = offline.get("label") if isinstance(offline, dict) else None
    category_jev = jev.get("label") if isinstance(jev, dict) else None
    results = []
    for model in models:
        attempt = 1
        while attempt <= 2:
            if not guard.acquire(task_id):
                return results
            try:
                row = _attempt(
                    task_id,
                    model,
                    attempt,
                    task,
                    solvers[model],
                    category_offline,
                    category_jev,
                    base_env,
                    base_url,
                )
                results.append(row)
                if row["error_class"] == "accounting":
                    guard.stop_for_accounting(task_id)
            finally:
                guard.release()
            if row["retry_eligible"]:
                attempt += 1
                continue
            break
    return results


def run_pilot(workers: int = 3, only: str | None = None, models: list[str] | None = None) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks_payload = _read_json(ROOT / "tasks.json")
    model_payload = _read_json(ROOT / "poc" / "pilot_models.json")
    if not isinstance(tasks_payload, dict) or not isinstance(model_payload, dict):
        raise ValueError("invalid task or model manifest")
    tasks = tasks_payload["tasks"]
    solvers = model_payload["solvers"]
    model_order = ["haiku-4.5", "sonnet-5", "opus-5"]
    if models is not None and (not models or any(model not in model_order for model in models)):
        raise ValueError("models must be selected from the pinned solver list")
    selected_models = [model for model in model_order if models is None or model in models]
    if not selected_models or any(model not in solvers for model in selected_models):
        raise ValueError("models must be selected from the pinned solver list")
    selected_tasks = [task for task in tasks if only is None or task["instance_id"] == only]
    if only is not None and not selected_tasks:
        raise ValueError(f"unknown task ID: {only}")
    labels = _labels_by_task()
    base_env = os.environ.copy()
    base_env.pop("LITELLM_GATEWAY_ADMIN_KEY", None)
    base_env["LITELLM_PROXY_API_KEY"] = _read_pilot_key()
    base_env["PYTHONPATH"] = str(ROOT)
    base_env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    stop_path = ROOT / "dispatch_stops.jsonl"
    guard = DispatchGuard(stop_path)
    base_url = os.environ.get("LITELLM_GATEWAY_BASE_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_task,
                task["instance_id"],
                task,
                selected_models,
                solvers,
                labels,
                guard,
                base_url,
                base_env,
            ): task["instance_id"]
            for task in selected_tasks
        }
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen capability-routing pilot matrix")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only")
    parser.add_argument("--models", nargs="+")
    arguments = parser.parse_args()
    results = run_pilot(arguments.workers, arguments.only, arguments.models)
    print(
        json.dumps(
            [
                {
                    "task_id": row["task_id"],
                    "model": row["model"],
                    "attempt": row["attempt"],
                    "error_class": row["error_class"],
                    "success": row["success"],
                    "cost_usd": row["cost_usd"],
                }
                for row in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
