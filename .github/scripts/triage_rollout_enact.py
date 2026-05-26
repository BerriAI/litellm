#!/usr/bin/env python3
"""One-shot day-7 enactment sweep for the Agent Shin rollout.

This runs once, on the merge commit of the enactment PR, exactly 7 days after
the heads-up sweep. It walks every open external PR/issue and lets Agent
Shin's steady-state logic decide what to do for each:

  * PR passing the rubric  -> tag `ready for review` (via `review_gate`)
  * PR failing the rubric, has the heads-up marker, past 24h since the warning
                            -> close + post the standard close comment
  * PR failing the rubric, no heads-up marker yet (created this week)
                            -> post the 24h grace warning (steady-state path)
  * Issue passing           -> no-op
  * Issue failing past grace -> close + comment
  * Issue failing in grace   -> warn (or already-warned skip)
  * Internal-author / closed -> skip

The script doesn't replicate the rubric logic — it calls into the existing
``review_gate`` and ``triage`` paths in dry-run mode, then routes their
verdicts through the ``maybe_*`` wrappers so a single ``dry_run`` boolean
toggles between logging and real GitHub mutations.

Time-travel dry-run
-------------------
``--simulate-future-hours N`` (default 24h+1s when --dry-run is set and no
explicit value is given) shifts the script's notion of "now" forward by N
hours. This lets you preview what the *next* scheduled run will do: any PR
currently in the grace window will tip into "past grace" after 24h, and the
preview shows those would-close decisions before they actually fire.

Time-travel is implemented in exactly one place: a ``current_time`` variable
is computed at the top of ``run()`` and threaded through ``review_gate(now=)``
for PRs. For issues (whose grace check goes through
``seconds_since_latest_marker_comment``), we patch ``triage_with_llm``'s
``dt.datetime.now`` under a context manager for the duration of each
``triage()`` call — one small surface, easy to audit.

CLI examples
------------

::

    # Pure preview at current time (no GitHub writes):
    python3 .github/scripts/triage_rollout_enact.py --repo BerriAI/litellm

    # Preview what the next daily cron will do (24h+1s in the future):
    python3 .github/scripts/triage_rollout_enact.py --repo BerriAI/litellm \\
        --simulate-future-hours 24

    # Real run (what the workflow does on merge):
    python3 .github/scripts/triage_rollout_enact.py --repo BerriAI/litellm --close
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import triage_with_llm  # noqa: E402
from _agent_shin_actions import (  # noqa: E402
    maybe_add_label,
    maybe_close_issue,
    maybe_close_pr,
    maybe_post_comment,
    maybe_remove_label,
)
from triage_with_llm import (  # noqa: E402
    DEFAULT_MODEL,
    READY_FOR_REVIEW_LABEL,
    fetch_issue,
    fetch_pr,
    format_grace_warning_issue_comment,
    format_issue_close_comment,
    gh,
    is_internal_contributor,
    review_gate,
    triage,
)

# Default time-travel offset: the next daily cron runs 24h after this script's
# real run, so 24h+1s gives a "what fires tomorrow" preview without any edge
# cases at the boundary itself.
DEFAULT_SIMULATE_HOURS = 24


@contextlib.contextmanager
def _fake_now(when: dt.datetime) -> Iterator[None]:
    """Patch ``dt.datetime.now`` in triage_with_llm so issue triage's grace
    check resolves against ``when`` rather than wall-clock time.

    Patches ``triage_with_llm.dt`` (the module's local alias) rather than
    the global ``datetime`` so we don't leak the override into unrelated code.
    The issue grace check reads the wall clock inside
    ``_seconds_since_latest_marker_comment`` via this module's ``dt`` alias,
    so this is the only surface that needs to be frozen. The patch is scoped
    to the ``with`` block — once it exits, the original ``dt`` module is
    restored, so a per-item call to ``triage()`` is the only code that ever
    sees the fake clock.
    """
    real_dt = triage_with_llm.dt

    # We only need to override `dt.datetime.now`. Easiest path: install a
    # tiny shim that proxies to the real `datetime` module for everything
    # except `.now()`.
    class _DtShim:
        timezone = real_dt.timezone

        class datetime(real_dt.datetime):  # noqa: N801 - mirror stdlib name
            @classmethod
            def now(cls, tz: dt.tzinfo | None = None) -> dt.datetime:
                return when if tz is None else when.astimezone(tz)

        # Pass everything else through to the real module.
        def __getattr__(self, name: str) -> Any:  # pragma: no cover - shim
            return getattr(real_dt, name)

    triage_with_llm.dt = _DtShim()
    try:
        yield
    finally:
        triage_with_llm.dt = real_dt


def _list_open_numbers(repo: str, kind: str) -> list[int]:
    cmd = "pr" if kind == "pr" else "issue"
    raw = gh(
        cmd,
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        "1000",
        "--json",
        "number",
    )
    return [item["number"] for item in json.loads(raw)]


# ---------------------------------------------------------------------------
# Per-PR / per-issue dispatch.
#
# Each helper takes the verdict-shaped result from review_gate / triage, then
# routes it to the matching maybe_* wrapper. The wrappers each carry the
# `dry_run` boolean, so a dry-run preview hits exactly the same code path as
# the real run except for the final GitHub API call.


def _apply_pr_result(*, repo: str, number: int, result: dict, dry_run: bool) -> dict:
    """Translate a ``review_gate`` result into the matching GitHub mutation
    (or a dry-run log line). Returns the augmented result with the action
    taken (``"applied"``, ``"would-apply"``, or ``"noop"``)."""
    action = result.get("action") or "unknown"
    comment = result.get("comment")
    base = {"kind": "pr", "number": number, "review_gate_action": action}

    # `review_gate(close=False)` returns `would-*` strings for every
    # transition; `review_gate(close=True)` returns the already-applied
    # counterparts. The dispatcher below treats both forms identically so the
    # enactment script can be driven in either mode (we always run it in
    # close=False mode to capture the would-* preview, then re-apply the
    # mutations through the dry-run wrappers).
    if action in ("noop-passing", "skip-not-open", "skip-internal-author"):
        return {**base, "result": "noop"}
    if action in ("skip-no-llm-key", "skip-llm-error"):
        return {
            **base,
            "result": "noop-llm-unavailable",
            "error": result.get("error"),
        }
    if action in ("would-label-ready", "labeled-ready"):
        assert comment, "review_gate must supply a comment for label-ready"
        maybe_post_comment(repo, number, comment, dry_run=dry_run)
        maybe_add_label(repo, number, READY_FOR_REVIEW_LABEL, dry_run=dry_run)
        return {**base, "result": "labeled-ready"}
    if action in ("would-remove-label", "label-removed-regressed"):
        assert comment
        maybe_remove_label(repo, number, READY_FOR_REVIEW_LABEL, dry_run=dry_run)
        maybe_post_comment(repo, number, comment, dry_run=dry_run)
        return {**base, "result": "label-removed-regressed"}
    if action in ("would-close", "closed"):
        assert comment
        maybe_post_comment(repo, number, comment, dry_run=dry_run)
        maybe_close_pr(repo, number, dry_run=dry_run)
        return {**base, "result": "closed"}
    if action in ("would-notify-within-grace", "within-grace-notified"):
        assert comment
        maybe_post_comment(repo, number, comment, dry_run=dry_run)
        return {**base, "result": "warned-within-grace"}
    if action in ("within-grace-already-notified", "regressed-already-notified"):
        return {**base, "result": "noop-already-notified"}
    # Anything else falls through as a no-op so an unexpected verdict from
    # review_gate (e.g. a future action string) doesn't cause a partial write.
    return {**base, "result": "noop-unknown-action"}


def _apply_issue_result(*, repo: str, number: int, result: dict, dry_run: bool) -> dict:
    """Translate a ``triage`` (kind='issue') result into the matching
    mutation. Mirrors `_apply_pr_result` for the issue half of the flow."""
    action = result.get("action") or "unknown"
    verdict = result.get("verdict") or {}
    base = {"kind": "issue", "number": number, "triage_action": action}

    if action in (
        "pass-llm",
        "pass-linked-issue",
        "skip-not-open",
        "skip-internal-author",
    ):
        return {**base, "result": "noop"}
    if action in ("skip-no-llm-key", "skip-llm-error"):
        return {
            **base,
            "result": "noop-llm-unavailable",
            "error": result.get("error"),
        }
    if action in ("would-warn-grace", "warned-grace"):
        body = format_grace_warning_issue_comment(verdict)
        maybe_post_comment(repo, number, body, dry_run=dry_run)
        return {**base, "result": "warned-within-grace"}
    if action in ("skip-in-grace-period",):
        return {**base, "result": "noop-already-warned"}
    if action in ("would-close", "closed"):
        body = format_issue_close_comment(verdict)
        maybe_post_comment(repo, number, body, dry_run=dry_run)
        maybe_close_issue(repo, number, dry_run=dry_run)
        return {**base, "result": "closed"}
    return {**base, "result": "noop-unknown-action"}


def _evaluate_pr(
    *,
    repo: str,
    number: int,
    model: str,
    current_time: dt.datetime,
    judge: Any = None,
) -> dict:
    """Run ``review_gate`` in preview mode against ``current_time``.

    Always uses ``close=False`` so the underlying review_gate never mutates
    GitHub directly — the enactment script is the single source of mutations
    and routes everything through the dry-run wrappers.
    """
    return review_gate(
        repo=repo,
        number=number,
        close=False,
        model=model,
        judge=judge,
        now=current_time,
    )


def _evaluate_issue(
    *,
    repo: str,
    number: int,
    model: str,
    current_time: dt.datetime,
    judge: Any = None,
) -> dict:
    """Run ``triage(kind='issue')`` in preview mode against ``current_time``.

    ``triage`` doesn't accept a ``now`` parameter, so the time-travel patch is
    applied here (the only place issues touch the wall clock is the
    grace-warning age check inside ``seconds_since_latest_marker_comment``).
    """
    with _fake_now(current_time):
        return triage(
            repo=repo,
            kind="issue",
            number=number,
            close=False,
            model=model,
            judge=judge,
        )


def _process_one(
    *,
    repo: str,
    kind: str,
    number: int,
    model: str,
    dry_run: bool,
    current_time: dt.datetime,
    judge: Any = None,
) -> dict:
    """Evaluate one PR/issue and apply the resulting mutation via the
    maybe_* wrappers. Skip-cases (not-open, internal author, no key) short-
    circuit before any LLM call."""
    fetcher = fetch_pr if kind == "pr" else fetch_issue
    item = fetcher(repo, number)

    if (item.get("state") or "") != "open":
        return {"kind": kind, "number": number, "result": "skip-not-open"}
    if is_internal_contributor(item):
        return {"kind": kind, "number": number, "result": "skip-internal-author"}

    if kind == "pr":
        result = _evaluate_pr(
            repo=repo,
            number=number,
            model=model,
            current_time=current_time,
            judge=judge,
        )
        return _apply_pr_result(
            repo=repo, number=number, result=result, dry_run=dry_run
        )
    result = _evaluate_issue(
        repo=repo,
        number=number,
        model=model,
        current_time=current_time,
        judge=judge,
    )
    return _apply_issue_result(repo=repo, number=number, result=result, dry_run=dry_run)


def _print_summary(results: list[dict], *, current_time: dt.datetime) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.get("result") or "unknown"] = (
            counts.get(r.get("result") or "unknown", 0) + 1
        )
    print(f"\n=== enactment summary (clock={current_time.isoformat()}) ===")
    for action in sorted(counts):
        print(f"  {action:35s} {counts[action]}")
    print(f"  total                                {len(results)}")


def run(
    *,
    repo: str,
    close: bool,
    model: str,
    current_time: dt.datetime,
    kinds: tuple[str, ...] = ("pr", "issue"),
    judge: Any = None,
    only_numbers: dict[str, list[int]] | None = None,
) -> list[dict]:
    """Sweep ``repo`` and apply the enactment verdicts. Returns per-item results."""
    dry_run = not close
    mode_label = "DRY RUN" if dry_run else "REAL RUN"
    print(
        f"[{mode_label}] enactment sweep over {repo} at clock={current_time.isoformat()}"
    )

    results: list[dict] = []
    for kind in kinds:
        numbers = list((only_numbers or {}).get(kind, [])) or _list_open_numbers(
            repo, kind
        )
        print(f"\n--- {kind}s: {len(numbers)} open ---")
        for n in numbers:
            try:
                result = _process_one(
                    repo=repo,
                    kind=kind,
                    number=n,
                    model=model,
                    dry_run=dry_run,
                    current_time=current_time,
                    judge=judge,
                )
            except Exception as exc:  # noqa: BLE001 - per-item errors don't abort
                result = {
                    "kind": kind,
                    "number": n,
                    "result": "error",
                    "error": str(exc),
                }
                print(f"!! {kind}#{n}: {exc}", file=sys.stderr)
            print(f"  {kind}#{n}: {result.get('result')}")
            results.append(result)
    _print_summary(results, current_time=current_time)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument(
        "--close",
        action="store_true",
        help=(
            "Actually post comments and close PRs/issues. Without this flag "
            "the script runs in dry-run mode and only logs what it would do."
        ),
    )
    parser.add_argument(
        "--simulate-future-hours",
        type=float,
        default=None,
        help=(
            "Dry-run only: pretend the wall clock is N hours in the future "
            f"(default when --close is NOT set: {DEFAULT_SIMULATE_HOURS}h+1s, "
            "so you preview exactly what the next daily run will do). Set to "
            "0 to preview at the current clock instead."
        ),
    )
    parser.add_argument(
        "--simulate-now",
        type=str,
        default=None,
        help=(
            "Dry-run only: pin the wall clock to this ISO-8601 timestamp "
            "(e.g. '2026-06-02T09:00:00Z'). Overrides --simulate-future-hours."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL,
        help=f"Model for the rubric LLM judge (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--kind",
        choices=("pr", "issue", "both"),
        default="both",
        help="Restrict the sweep to PRs or issues only (default: both).",
    )
    parser.add_argument(
        "--only-pr",
        type=int,
        action="append",
        default=[],
        help="Limit the PR sweep to these PR numbers (repeat for several).",
    )
    parser.add_argument(
        "--only-issue",
        type=int,
        action="append",
        default=[],
        help="Limit the issue sweep to these issue numbers (repeat for several).",
    )
    args = parser.parse_args()

    if args.close and (
        args.simulate_future_hours is not None or args.simulate_now is not None
    ):
        parser.error(
            "--simulate-future-hours / --simulate-now are only valid in dry-run "
            "(omit --close to preview a future clock)."
        )

    if args.close and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set for --close (real-run) mode.")

    # Resolve the script's notion of "now".
    if args.simulate_now is not None:
        current_time = dt.datetime.fromisoformat(
            args.simulate_now.replace("Z", "+00:00")
        )
        if current_time.tzinfo is None:
            parser.error(
                "--simulate-now must include a timezone offset "
                "(e.g. '2026-06-02T09:00:00Z' or '2026-06-02T09:00:00+00:00')."
            )
    else:
        offset = (
            args.simulate_future_hours
            if args.simulate_future_hours is not None
            else (0 if args.close else DEFAULT_SIMULATE_HOURS + 1 / 3600)
        )
        current_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=offset)

    kinds: tuple[str, ...]
    if args.kind == "pr":
        kinds = ("pr",)
    elif args.kind == "issue":
        kinds = ("issue",)
    else:
        kinds = ("pr", "issue")

    only: dict[str, list[int]] = {}
    if args.only_pr:
        only["pr"] = args.only_pr
    if args.only_issue:
        only["issue"] = args.only_issue

    run(
        repo=args.repo,
        close=args.close,
        model=args.model,
        current_time=current_time,
        kinds=kinds,
        only_numbers=only or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
