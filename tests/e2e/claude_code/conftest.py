"""Pytest plumbing for the Claude Code compatibility matrix.

Three responsibilities live here:

1. The `compat_result` fixture — the only API a test author needs to learn.
   Tests call `compat_result.set({"status": "pass"})` (or fail / not_applicable)
   to report their outcome as a tagged union. Multi-model tests call
   `.add(...)` once per Claude tier so each tier lands as its own row in
   the results artifact.

2. The `pytest_runtest_makereport` hook — captures each test's reported result,
   infers (feature, provider) from the file path, and accumulates rows into
   a per-process collector. At session end we serialize them to
   `compat-results.json` (or a per-worker file under xdist) so the Matrix
   JSON Builder can consume them.

3. xdist coordination — when `pytest -n auto` is used, every worker writes
   its own results shard and the controller merges them into the canonical
   `compat-results.json` in `pytest_sessionfinish`. Without this, the
   workers race on the same path and the artifact only reflects whichever
   worker finished last. The same merge step also emits a rate-limit
   summary that the binary-search helper consumes to decide whether the
   current X/Y/Z values were too aggressive.

The (feature, provider) inference comes from the test file path: the parent
directory name is the feature_id (matching `manifest.yaml`), and the file
stem after the leading `test_` is the provider id. This avoids per-file
metadata that drifts.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import pytest
import yaml

VALID_STATUSES = {"pass", "fail", "not_applicable", "not_tested"}
RESULTS_ARTIFACT_ENV = "COMPAT_RESULTS_PATH"
DEFAULT_ARTIFACT_PATH = "compat-results.json"
RATE_LIMIT_SUMMARY_ENV = "COMPAT_RATE_LIMIT_SUMMARY_PATH"
DEFAULT_RATE_LIMIT_SUMMARY_PATH = "compat-rate-limit-summary.json"

# Heuristic: detect 429s and rate-limit-shaped errors anywhere in the
# error string. The CLI buries upstream errors in `assistant.message.content`
# text on stdout (see `failure_diagnostic`), so we don't get a structured
# status code in every code path — a regex over the joined error text is
# the most reliable signal we have.
#
# We also treat a CLI timeout (`claude CLI timed out after Ns`) as a
# rate-limit-shaped failure for binary-search purposes: in practice the
# only reason every model in a cell stalls past the timeout is the
# upstream collapsing under concurrency, which is exactly the situation
# the rate limiter is supposed to back off from. False positives on a
# genuinely slow upstream are tolerable here because the worst case is
# the binary search runs at a slightly lower rate than necessary.
_RATE_LIMIT_RE = re.compile(
    r"(?:\b429\b|rate[\s_-]?limit|too\s+many\s+requests|throttl(?:ed|ing)|"
    r"claude\s+CLI\s+timed\s+out)",
    re.IGNORECASE,
)


@dataclass
class CompatResult:
    """Per-test recorder for compatibility outcomes.

    Tests interact via `.set(...)` (single result) or `.add(...)` (one
    result per Claude tier when the test fans the three models out in
    parallel). `.value` and `.values` are read by the
    `pytest_runtest_makereport` hook after the test body finishes.

    Multi-result usage exists because every cell in the compat matrix is
    backed by three model invocations (Haiku/Sonnet/Opus) per (feature,
    provider). When a test runs them concurrently in a single pytest
    node, each model needs its own entry in the results artifact so the
    matrix builder's per-cell aggregator can apply its "all three must
    pass" rule.
    """

    value: Optional[Dict[str, Any]] = None
    values: List[Dict[str, Any]] = field(default_factory=list)

    def set(self, result: Dict[str, Any]) -> None:
        validated = self._validate(result)
        self.value = validated

    def add(self, result: Dict[str, Any]) -> None:
        """Append one model's outcome to the per-test results list.

        Use this when a single test exercises multiple Claude tiers
        concurrently and needs to report one outcome per tier. The
        conftest hook will emit one entry per appended result.
        """
        validated = self._validate(result)
        self.values.append(validated)

    @staticmethod
    def _validate(result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError("compat_result requires a dict")
        status = result.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"compat_result status must be one of {sorted(VALID_STATUSES)}, "
                f"got {status!r}"
            )
        if status == "fail" and not result.get("error"):
            raise ValueError("compat_result {'status': 'fail'} requires 'error'")
        if status == "not_applicable" and not result.get("reason"):
            raise ValueError(
                "compat_result {'status': 'not_applicable'} requires 'reason'"
            )
        return dict(result)

    def collected(self) -> List[Dict[str, Any]]:
        """Return every result reported during the test, preserving order.

        Multi-model tests use `.add(...)` per model; legacy tests use
        `.set(...)` once. We surface both shapes in a single list so
        the makereport hook only has to think about a list of results.
        """
        if self.values:
            return list(self.values)
        if self.value is not None:
            return [dict(self.value)]
        return []


@dataclass
class _CollectedResult:
    feature_id: str
    provider: str
    nodeid: str
    result: Dict[str, Any]


@dataclass
class _Collector:
    items: List[_CollectedResult] = field(default_factory=list)


_COLLECTOR = _Collector()


@pytest.fixture
def compat_result() -> CompatResult:
    """Per-test recorder for the (feature, provider) outcome.

    Tests should call `compat_result.set({"status": "pass"})` (or fail /
    not_applicable) before returning. If a test exits without calling `.set()`
    the harness records `status="fail"` with an explanatory error so that
    every collected node maps to a real cell.
    """
    return CompatResult()


@functools.lru_cache(maxsize=1)
def _manifest_feature_ids() -> FrozenSet[str]:
    """Return the set of feature_ids declared in `manifest.yaml`.

    Used as a positive filter so only directories that correspond to a
    real matrix row contribute results — utility/support directories
    (e.g. `cron_vm`, `_driver_unit_tests`) are dropped regardless of
    naming convention, and the rate-limit summary stays clean.

    Returns an empty set if the manifest is missing or malformed; the
    caller treats that as "no path is a feature path", which is the
    safe default — we'd rather drop a real result than pollute the
    artifact with a garbage cell.
    """
    manifest_path = Path(__file__).resolve().parent / "manifest.yaml"
    try:
        raw = yaml.safe_load(manifest_path.read_text())
    except (OSError, yaml.YAMLError):
        return frozenset()
    if not isinstance(raw, dict):
        return frozenset()
    features = raw.get("features")
    if not isinstance(features, list):
        return frozenset()
    return frozenset(
        entry["id"]
        for entry in features
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


def _infer_feature_and_provider(node_path: Path) -> Optional[tuple]:
    """Infer (feature_id, provider) from a test file path.

    Path shape:  tests/e2e/claude_code/<feature_id>/test_<provider>.py
    Returns None if the file is not a per-feature test (e.g. unit tests
    under `_driver_unit_tests/` or support code under `cron_vm/`), so
    those don't pollute the matrix artifact. We positively filter the
    parent directory against `manifest.yaml` rather than relying on
    naming conventions, because non-feature siblings don't all share
    an underscore prefix.
    """
    name = node_path.name
    if not name.startswith("test_") or not name.endswith(".py"):
        return None
    provider = name[len("test_") : -len(".py")]
    feature_id = node_path.parent.name
    if feature_id not in _manifest_feature_ids():
        return None
    return feature_id, provider


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture compat_result reports at end-of-test and remember them for the artifact.

    A single test may report multiple results (one per Claude tier when
    the three are run in parallel inside one node). We emit one
    `_CollectedResult` per reported entry so the matrix builder's
    per-cell aggregator sees the same shape it would have seen if the
    test were parametrized — every model lands in the artifact.
    """
    outcome = yield
    report = outcome.get_result()
    # We record on two phases:
    #   - "call": the normal end-of-test path.
    #   - "setup" but only on failure: fixture/import errors that prevent the
    #     test body from running. Without recording these, a broken setup
    #     silently becomes "not_tested" in the published matrix instead of
    #     "fail". Teardown is ignored — by then "call" already recorded the
    #     outcome, and a teardown-only failure (e.g. fixture finalizer) is
    #     not a cell-level signal.
    if report.when == "setup":
        if not report.failed:
            return
    elif report.when != "call":
        return

    # A skipped test (e.g. `pytest.skip(...)` called inside the body or
    # by a `pytest.mark.skipif` evaluated at call time) is neither a
    # pass nor a fail — it just didn't run. Recording it as anything
    # here would produce a spurious row (the not-failed/empty-collected
    # branch below would mark it as a fail with "test passed without
    # reporting via compat_result"), so bail out and let the cell stay
    # "not_tested" in the published matrix.
    if report.skipped:
        return

    inferred = _infer_feature_and_provider(Path(str(item.path)))
    if inferred is None:
        return
    feature_id, provider = inferred

    fixture = item.funcargs.get("compat_result") if hasattr(item, "funcargs") else None
    collected: List[Dict[str, Any]] = (
        fixture.collected() if isinstance(fixture, CompatResult) else []
    )

    if report.failed and not any(entry.get("status") == "fail" for entry in collected):
        # The test body (or setup) raised and the test author hasn't
        # already recorded a fail row via `.add(...)`. If the test had
        # recorded only per-model passes before crashing, those partial
        # entries would aggregate to "pass" and hide the crash from the
        # published matrix; append an explicit "fail" row so the cell
        # aggregator (which gives precedence to any fail) surfaces the
        # breakage. We skip the append when a fail row is already
        # present so that the common pattern — `.add({"status": "fail",
        # ...})` per failing model, then `pytest.fail("; ".join(...))`
        # to surface them — doesn't produce a phantom duplicate row.
        collected = collected + [
            {
                "status": "fail",
                "error": (str(report.longrepr) if report.longrepr else "test failed"),
            }
        ]
    elif not report.failed and not collected:
        collected = [
            {
                "status": "fail",
                "error": "test passed without reporting via compat_result; "
                "every compat test must report a status.",
            }
        ]

    for reported in collected:
        _COLLECTOR.items.append(
            _CollectedResult(
                feature_id=feature_id,
                provider=provider,
                nodeid=report.nodeid,
                result=reported,
            )
        )


def _is_xdist_worker(session) -> bool:
    """Return True iff the current pytest session is an xdist worker.

    The standard idiom is to look up `workerinput` on the config; the
    controller process doesn't have it, the workers do. We deliberately
    don't `import xdist` because the suite must keep running when xdist
    isn't installed at all.
    """
    return hasattr(session.config, "workerinput")


def _xdist_worker_id(session) -> Optional[str]:
    info = getattr(session.config, "workerinput", None)
    if not info:
        return None
    return info.get("workerid")


def _shard_dir(artifact_path: Path) -> Path:
    """Workers write their shards next to the canonical results path.

    Putting shards in a sibling directory (rather than inline JSON
    files in the same dir) keeps the controller's merge step simple
    — it just lists `*.json` in `<artifact>.shards/` — and avoids
    accidental shard/canonical filename collisions.
    """
    return artifact_path.with_name(artifact_path.name + ".shards")


def _serialize_items(items: List["_CollectedResult"]) -> List[Dict[str, Any]]:
    return [
        {
            "feature_id": item.feature_id,
            "provider": item.provider,
            "nodeid": item.nodeid,
            "result": item.result,
        }
        for item in items
    ]


def _build_rate_limit_summary(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-provider rate-limit signals from the result rows.

    We classify any failure whose error string matches `_RATE_LIMIT_RE`
    as a "rate-limited" failure. The binary-search helper reads this
    summary to decide whether the current X/Y/Z values were too
    aggressive: if any provider has `rate_limited > 0`, the harness
    should back off that provider's rate and retry.

    Returns a dict shaped:

        {
          "totals": {"pass": N, "fail": N, "rate_limited": N, ...},
          "per_provider": {
            "anthropic": {"pass": ..., "fail": ..., "rate_limited": ...},
            ...
          },
          "rate_limited_examples": [
            {"feature_id": ..., "provider": ..., "error": "..."}, ...
          ],
        }
    """
    totals: Counter = Counter()
    per_provider: Dict[str, Counter] = defaultdict(Counter)
    rate_limited_examples: List[Dict[str, Any]] = []

    for row in rows:
        result = row.get("result") or {}
        status = result.get("status") or "unknown"
        provider = row.get("provider") or "unknown"
        totals[status] += 1
        per_provider[provider][status] += 1

        if status == "fail":
            error = str(result.get("error") or "")
            if _RATE_LIMIT_RE.search(error):
                totals["rate_limited"] += 1
                per_provider[provider]["rate_limited"] += 1
                # Cap examples so a stuck-throttled run doesn't write
                # a multi-MB summary file the helper has to slurp.
                if len(rate_limited_examples) < 25:
                    rate_limited_examples.append(
                        {
                            "feature_id": row.get("feature_id"),
                            "provider": provider,
                            "error": error[:500],
                        }
                    )

    return {
        "totals": dict(totals),
        "per_provider": {p: dict(c) for p, c in per_provider.items()},
        "rate_limited_examples": rate_limited_examples,
    }


def _print_rate_limit_summary(summary: Dict[str, Any]) -> None:
    """Emit a human-readable per-provider table to stderr.

    Pytest only captures stderr when `-s` isn't set; we deliberately
    write here anyway because the binary-search workflow runs pytest
    with `-q` and grep-checks the structured JSON artifact, while a
    human running locally with `-s` sees the same numbers inline.
    """
    totals = summary.get("totals", {})
    per_provider = summary.get("per_provider", {})
    lines: List[str] = []
    lines.append("[compat] session totals:")
    for status in ("pass", "fail", "rate_limited", "not_applicable", "not_tested"):
        if status in totals:
            lines.append(f"  {status:<16s} {totals[status]}")
    if per_provider:
        lines.append("[compat] per-provider breakdown:")
        for provider in sorted(per_provider):
            counts = per_provider[provider]
            parts = " ".join(
                f"{k}={v}"
                for k, v in sorted(counts.items())
                if k != "not_tested" or v > 0
            )
            lines.append(f"  {provider:<20s} {parts}")
    if totals.get("rate_limited", 0):
        lines.append(
            "[compat] WARNING: at least one cell hit a rate-limit-shaped error; "
            "lower the corresponding LITELLM_COMPAT_RATE_<PROVIDER> and retry"
        )
    print("\n".join(lines), file=sys.stderr, flush=True)


def pytest_sessionstart(session):
    """Reset per-session state before tests run.

    Two responsibilities:

    1. Clear the module-level `_COLLECTOR` singleton, which survives
       across `pytest.main()` invocations within the same Python
       process. Without this reset, results from a prior session
       would leak into the next run's `compat-results.json` artifact.

    2. Remove stale per-worker shards from any prior session. Without
       this, a previous run's shard directory leaks into the next
       `pytest_sessionfinish` merge — yielding a `compat-results.json`
       that includes results from runs that aren't part of the current
       session, and a misleading rate-limit summary that re-flags
       failures the user already saw and addressed. Only the
       controller (non-xdist-worker) clears; workers must not race
       the controller while it's wiping the directory.
    """
    _COLLECTOR.items.clear()
    _manifest_feature_ids.cache_clear()

    if _is_xdist_worker(session):
        return
    artifact_path = Path(os.environ.get(RESULTS_ARTIFACT_ENV) or DEFAULT_ARTIFACT_PATH)
    shard_dir = _shard_dir(artifact_path)
    if not shard_dir.exists():
        return
    for stale in shard_dir.glob("*.json"):
        try:
            stale.unlink()
        except OSError:
            # If we can't remove a stale shard (permissions, race with
            # an unrelated process), keep going — the merge step is
            # robust to malformed shards, and a stale row landing in
            # the artifact is recoverable; aborting the session isn't.
            continue


def pytest_sessionfinish(session, exitstatus):
    """Write the per-process results shard, then merge if we're the controller.

    Worker processes (xdist `gw0`, `gw1`, ...) only write their shard
    under `<artifact>.shards/<workerid>.json`. The controller writes
    its own shard if it ran any tests itself, then walks the shards
    directory and produces the canonical `compat-results.json` plus
    the rate-limit summary. Single-process runs (no xdist) take the
    same code path with a single shard, so behavior is consistent.

    Skip when no compat results were collected — this conftest is
    loaded for every test under `tests/e2e/claude_code/`, including sibling
    unit-test trees (e.g. `_driver_unit_tests/`). Writing an empty
    artifact would silently overwrite a real artifact from a prior
    compat-test run on the same checkout.

    The xdist controller hits this hook with `_COLLECTOR.items` empty
    (it never executes tests itself) and `_is_xdist_worker` False, so
    we additionally allow the merge step to run when worker shards
    are already on disk — otherwise the canonical artifact would
    never be produced under `pytest -n auto`.
    """
    artifact_path = Path(os.environ.get(RESULTS_ARTIFACT_ENV) or DEFAULT_ARTIFACT_PATH)
    shard_dir = _shard_dir(artifact_path)
    has_worker_shards = shard_dir.is_dir() and any(shard_dir.glob("*.json"))
    if not _COLLECTOR.items and not _is_xdist_worker(session) and not has_worker_shards:
        return
    shard_dir.mkdir(parents=True, exist_ok=True)

    worker_id = _xdist_worker_id(session) or "main"
    shard_path = shard_dir / f"{worker_id}.json"
    shard_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "worker_id": worker_id,
                "results": _serialize_items(_COLLECTOR.items),
            },
            indent=2,
            sort_keys=True,
        )
    )

    # Workers stop here. The controller merges; if we're not running
    # under xdist, we are effectively the controller.
    if _is_xdist_worker(session):
        return

    merged_rows: List[Dict[str, Any]] = []
    for shard_file in sorted(shard_dir.glob("*.json")):
        try:
            shard = json.loads(shard_file.read_text())
        except (OSError, ValueError):
            continue
        rows = shard.get("results")
        if isinstance(rows, list):
            merged_rows.extend(rows)

    # Skip writing artifact + summary entirely for unit-test-only runs
    # (no per-feature compat rows). Otherwise every `pytest tests/...`
    # run — including local unit-test invocations — would silently
    # overwrite a real artifact from a prior compat-test run.
    if not merged_rows:
        return

    artifact_path.write_text(
        json.dumps(
            {"schema_version": "1", "results": merged_rows},
            indent=2,
            sort_keys=True,
        )
    )

    summary = _build_rate_limit_summary(merged_rows)
    summary_path = Path(
        os.environ.get(RATE_LIMIT_SUMMARY_ENV) or DEFAULT_RATE_LIMIT_SUMMARY_PATH
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _print_rate_limit_summary(summary)


# ---------------------------------------------------------------------------
# Session-scoped compat model registration.
#
# The compat cells probe hardcoded virtual names like ``claude-sonnet-4-6``
# and ``claude-sonnet-4-6-bedrock-invoke``. On stage those live in the
# gateway's model_list at deploy time; locally the docker-config.yaml
# under tests/e2e/ only declares one of them, so every non-haiku cell
# 400s with ``Invalid model name``. The fixture here reconciles the two:
# it reads ``test_config.yaml`` (the ground-truth compat matrix config)
# and POSTs ``/model/new`` for the subset whose provider credentials are
# actually set in the current environment, then tears them all down at
# session end.
#
# Kept below the rest of the conftest so the compat-artifact hooks stay
# grouped up top. The fixture is opt-in via autouse=True on the session
# scope, so a cell that hits the proxy sees the deployment ready without
# any per-cell wiring, and pure unit tests that never reach the proxy
# pay only one skipped-liveness check.
# ---------------------------------------------------------------------------

from claude_code._env import resolve_proxy  # noqa: E402
from claude_code._compat_models import (  # noqa: E402
    CompatDeployment,
    load_all_deployments,
)


def _build_control_gateway():
    """Local import of the shared harness so the pure-unit-test tree
    under ``_driver_unit_tests/`` etc. never has to pull it in. The
    control plane transport is what /model/new lives on; SplitTransport
    routes it correctly for both monolithic and split deployments."""
    from e2e_gateway import build_gateway

    return build_gateway()


def _register_deployment(gateway, deployment: CompatDeployment) -> str:
    """Register one deployment and return its proxy-assigned model_id
    once it is servable on the data plane."""
    return gateway.create_model(
        deployment.model_name,
        deployment.litellm_params,
    )


@pytest.fixture(scope="session", autouse=True)
def _compat_models_registered() -> Any:
    """Register every compat deployment against the running proxy, then
    tear them all down on session exit.

    Skips silently if the proxy env is not configured (no
    ``LITELLM_PROXY_URL``/``LITELLM_MASTER_KEY``) so unit-test runs
    stay hermetic.

    Design note: we always attempt to register all 15 deployments,
    regardless of what credentials are exported in the test-runner's
    shell. The credentials live in the proxy container's environment
    (via docker-compose ``env_file``), not the shell running pytest -
    so gating on shell env would filter out deployments the proxy can
    actually serve. Per-deployment ``/model/new`` failures are printed
    but do not abort the session: the cells that need that specific
    deployment will 400 with "Invalid model name" and fail loudly,
    which is the right signal (missing cred on the proxy side)."""
    if resolve_proxy() is None:
        yield
        return

    from requests import RequestException

    gateway = _build_control_gateway()
    registered_ids: list[str] = []
    failures: list[tuple[str, str]] = []
    try:
        for deployment in load_all_deployments():
            try:
                model_id = _register_deployment(gateway, deployment)
                registered_ids.append(model_id)
            except (AssertionError, RequestException) as exc:
                failures.append((deployment.model_name, str(exc)))
        if failures:
            summary = "\n".join(
                f"  - {name}: {reason}" for name, reason in failures
            )
            print(
                f"[compat fixture] {len(failures)} of "
                f"{len(failures) + len(registered_ids)} deployments "
                f"failed to register (proxy likely missing that provider's "
                f"credentials); cells that target them will fail loudly:\n"
                f"{summary}"
            )
        yield
    finally:
        for model_id in registered_ids:
            try:
                gateway.delete_model(model_id)
            except (AssertionError, RequestException):
                # Best-effort — teardown surfaces via warnings inside
                # ``delete_model`` already; swallowing here so one flaky
                # delete does not mask real test failures.
                pass
