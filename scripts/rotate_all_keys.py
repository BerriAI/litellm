#!/usr/bin/env python3
"""
Bulk-rotate LiteLLM virtual keys to match the latest tier definition.

Context
-------
xcity-home (the Astro portal at xcity.one) is the source of truth for tier
definitions: ``free`` / ``pro_monthly`` / ``team_monthly`` each declare a
list of LiteLLM models, a ``max_budget`` USD figure, a ``budget_duration``
window and an ``rpm_limit``. xcity-home mints one LiteLLM virtual key per
authenticated user via ``POST /key/generate`` with those values stamped
inline (``metadata.plan = <plan>``).

When a tier definition is edited (e.g. Pro adds a model), already-issued
keys keep the OLD whitelist and budget. This script walks the keys for one
or more plans and rewrites them to match the new spec.

Algorithm
---------
1. Page through ``GET /key/list?return_full_object=true&size=100`` and
   filter client-side by ``metadata.plan == <target>``. The endpoint
   exposes ``user_id`` / ``team_id`` / ``key_alias`` filters but not a
   ``metadata`` filter (see
   ``litellm/proxy/management_endpoints/key_management_endpoints.py``).
2. For each matched key, diff the live ``models`` / ``max_budget`` /
   ``budget_duration`` / ``rpm_limit`` against the requested values.
3. If anything differs, ``POST /key/update`` with the deltas. The
   ``UpdateKeyRequest`` schema (``litellm/proxy/_types.py``) inherits
   ``KeyRequestBase`` and accepts all four of these fields directly, so no
   delete+regenerate fallback is required.
4. The script is idempotent: re-running with the same input yields no API
   calls when nothing has drifted.

Usage
-----
::

    export LITELLM_BASE_URL="https://tokenhub.xcity.one"
    export LITELLM_MASTER_KEY="sk-..."

    # rotate every Free-tier key to the latest Free spec
    python scripts/rotate_all_keys.py \\
        --plan free \\
        --models tencent_cloud/deepseek-v3.2,tencent_cloud/deepseek-v4-flash,tencent_cloud/glm-5 \\
        --max-budget 0.20 \\
        --budget-duration 1d \\
        --rpm-limit 5

    # preview-only
    python scripts/rotate_all_keys.py --plan pro_monthly ... --dry-run

    # rotate every plan in one shot (requires a TOML spec, see --spec)
    python scripts/rotate_all_keys.py --plan all --spec ./tier_spec.toml

Limitations & open issues
-------------------------
* LiteLLM has no native filter for ``metadata.plan``. The script paginates
  ALL keys and filters client-side; on a large deployment this is O(N).
* If a user_id has been deleted but the key is still in the table, the
  rotation succeeds; xcity-home is responsible for garbage-collecting
  orphaned keys.
* This script is the LITELLM-SIDE rotation only. xcity-home stores
  ``app_metadata.litellm.key`` per Supabase user; if you ever need to
  delete-and-regenerate (which this script does NOT do, since /key/update
  covers every field we care about), the new token must be pushed back
  into Supabase by the operator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

DEFAULT_BASE_URL = "https://tokenhub.xcity.one"
PAGE_SIZE = 100
ALLOWED_PLANS = ("free", "pro_monthly", "team_monthly")
ALLOWED_DURATIONS = ("1d", "30d")


@dataclass(frozen=True)
class TargetSpec:
    """The desired post-rotation state for keys on a given plan."""

    plan: str
    models: Tuple[str, ...]
    max_budget: float
    budget_duration: str
    rpm_limit: Optional[int]  # None == unlimited


@dataclass
class RotationCounters:
    rotated: int = 0
    no_op: int = 0
    skipped: int = 0


class LiteLLMAdminClient:
    """Thin wrapper around the LiteLLM admin HTTP API."""

    def __init__(self, base_url: str, master_key: str, timeout: float = 30.0) -> None:
        if not master_key:
            raise ValueError(
                "master key is empty; pass --master-key or set LITELLM_MASTER_KEY"
            )
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LiteLLMAdminClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def iter_all_keys(self) -> Iterable[Dict[str, Any]]:
        """Yield every key in the deployment as a full key object.

        Pages through ``/key/list?return_full_object=true``. Stops as soon
        as the API reports that the current page is the last one.
        """
        page = 1
        while True:
            resp = self._client.get(
                "/key/list",
                params={
                    "page": page,
                    "size": PAGE_SIZE,
                    "return_full_object": "true",
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            keys = payload.get("keys") or []
            for key_obj in keys:
                # When return_full_object=true the entries are dicts; when
                # false they are bare strings. Skip the latter to be safe.
                if isinstance(key_obj, dict):
                    yield key_obj
            total_pages = int(payload.get("total_pages") or 1)
            if page >= total_pages or not keys:
                return
            page += 1

    def update_key(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.post("/key/update", json=payload)
        resp.raise_for_status()
        return resp.json()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate xcity-home LiteLLM keys to a new tier spec."
    )
    parser.add_argument(
        "--plan",
        required=True,
        choices=(*ALLOWED_PLANS, "all"),
        help="Target plan (or 'all' to rotate every plan; requires --spec).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of allowed model IDs for the target plan.",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=None,
        help="Per-key max_budget in USD (e.g. 0.20, 25, 90).",
    )
    parser.add_argument(
        "--budget-duration",
        choices=ALLOWED_DURATIONS,
        default=None,
        help="Budget reset window.",
    )
    parser.add_argument(
        "--rpm-limit",
        default=None,
        help="Per-key requests-per-minute limit. Pass 'none' for unlimited.",
    )
    parser.add_argument(
        "--spec",
        default=None,
        help=(
            "Path to a JSON file with one block per plan when --plan all is used. "
            "Each block carries 'models', 'max_budget', 'budget_duration', 'rpm_limit'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diffs without calling /key/update.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LITELLM_BASE_URL", DEFAULT_BASE_URL),
        help="LiteLLM proxy base URL (default: $LITELLM_BASE_URL or %s)."
        % DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--master-key",
        default=os.environ.get("LITELLM_MASTER_KEY", ""),
        help="LiteLLM master key (default: $LITELLM_MASTER_KEY).",
    )
    return parser.parse_args(argv)


def parse_rpm_limit(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("none", "null", ""):
        return None
    return int(raw)


def build_targets(args: argparse.Namespace) -> Dict[str, TargetSpec]:
    """Return {plan_name: TargetSpec} based on CLI flags or --spec file."""
    if args.plan == "all":
        if not args.spec:
            raise SystemExit(
                "--plan all requires --spec <path-to-json> with one entry per plan"
            )
        with open(args.spec, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        targets: Dict[str, TargetSpec] = {}
        for plan in ALLOWED_PLANS:
            if plan not in blob:
                continue
            entry = blob[plan]
            targets[plan] = TargetSpec(
                plan=plan,
                models=tuple(entry["models"]),
                max_budget=float(entry["max_budget"]),
                budget_duration=str(entry["budget_duration"]),
                rpm_limit=parse_rpm_limit(entry.get("rpm_limit")),
            )
        if not targets:
            raise SystemExit(
                "--spec %s contained none of the recognised plan keys %s"
                % (args.spec, ALLOWED_PLANS)
            )
        return targets

    # Single-plan mode: every flag is required.
    missing: List[str] = []
    if not args.models:
        missing.append("--models")
    if args.max_budget is None:
        missing.append("--max-budget")
    if args.budget_duration is None:
        missing.append("--budget-duration")
    if missing:
        raise SystemExit(
            "missing required arguments for single-plan mode: " + ", ".join(missing)
        )

    return {
        args.plan: TargetSpec(
            plan=args.plan,
            models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
            max_budget=float(args.max_budget),
            budget_duration=args.budget_duration,
            rpm_limit=parse_rpm_limit(args.rpm_limit),
        )
    }


def diff_key_against_target(
    key_obj: Dict[str, Any], target: TargetSpec
) -> Dict[str, Any]:
    """Return a dict of fields that differ between the live key and target.

    Empty dict means the key is already in the desired state.
    """
    delta: Dict[str, Any] = {}

    live_models = list(key_obj.get("models") or [])
    if sorted(live_models) != sorted(target.models):
        delta["models"] = list(target.models)

    live_max_budget = key_obj.get("max_budget")
    # max_budget is stored as float; treat None != float as a diff.
    if live_max_budget is None or float(live_max_budget) != target.max_budget:
        delta["max_budget"] = target.max_budget

    if (key_obj.get("budget_duration") or None) != target.budget_duration:
        delta["budget_duration"] = target.budget_duration

    live_rpm = key_obj.get("rpm_limit")
    if live_rpm != target.rpm_limit:
        delta["rpm_limit"] = target.rpm_limit

    return delta


def keys_for_plan(
    keys: Iterable[Dict[str, Any]], plan: str
) -> Iterable[Dict[str, Any]]:
    """Filter keys whose metadata.plan matches the target plan."""
    for key_obj in keys:
        metadata = key_obj.get("metadata") or {}
        if metadata.get("plan") == plan:
            yield key_obj


def rotate_plan(
    client: LiteLLMAdminClient,
    target: TargetSpec,
    all_keys: List[Dict[str, Any]],
    dry_run: bool,
    counters: RotationCounters,
) -> None:
    print(f"\n=== plan={target.plan} ===")
    print(f"  target models     = {list(target.models)}")
    print(f"  target max_budget = ${target.max_budget}")
    print(f"  target budget_dur = {target.budget_duration}")
    print(f"  target rpm_limit  = {target.rpm_limit}")

    matched = list(keys_for_plan(all_keys, target.plan))
    print(f"  matched {len(matched)} key(s)")

    for key_obj in matched:
        token = key_obj.get("token") or key_obj.get("key_name")
        user_id = key_obj.get("user_id")
        delta = diff_key_against_target(key_obj, target)

        if not delta:
            counters.no_op += 1
            print(f"  no-op  user_id={user_id} token={token}")
            continue

        print(f"  diff   user_id={user_id} token={token} -> {delta}")

        if dry_run:
            # Diff printed; don't call the API.
            counters.rotated += 1
            continue

        try:
            client.update_key({"key": token, **delta})
            counters.rotated += 1
        except httpx.HTTPStatusError as exc:
            counters.skipped += 1
            print(
                f"  ERROR  user_id={user_id} token={token}: "
                f"{exc.response.status_code} {exc.response.text[:200]}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            counters.skipped += 1
            print(
                f"  ERROR  user_id={user_id} token={token}: {exc}",
                file=sys.stderr,
            )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    targets = build_targets(args)

    counters = RotationCounters()

    with LiteLLMAdminClient(
        base_url=args.base_url, master_key=args.master_key
    ) as client:
        # Page through once; filter per plan in-memory.
        all_keys = list(client.iter_all_keys())
        print(f"fetched {len(all_keys)} key(s) total from {args.base_url}")

        for target in targets.values():
            rotate_plan(
                client=client,
                target=target,
                all_keys=all_keys,
                dry_run=args.dry_run,
                counters=counters,
            )

    print()
    print(f"rotated: {counters.rotated}")
    print(f"no-op: {counters.no_op}")
    if counters.skipped:
        print(f"skipped: {counters.skipped}")
    if args.dry_run:
        print("(dry-run: no /key/update calls were issued)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
