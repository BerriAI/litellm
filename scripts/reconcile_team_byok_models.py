#!/usr/bin/env python3
# ruff: noqa: T201, PLR0915
"""
One-time reconciliation utility for orphaned team BYOK model names in LiteLLM_TeamTable.models.

Defaults to dry-run. Use --apply to persist changes.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from prisma import Prisma

SPECIAL_MODEL_TOKENS = {"*", "all-proxy-models", "all-team-models"}


def _safe_parse_dict(payload: object) -> Dict[str, str]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _safe_parse_model_info(payload: object) -> Dict[str, object]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass
class TeamReconcileResult:
    team_id: str
    before_count: int
    after_count: int
    removed_names: List[str]
    error: Optional[str] = None


def _is_reserved_team_access_group(model_name: str, team_id: str) -> bool:
    return model_name.startswith(f"team-{team_id}-")


async def reconcile_team_byok_models(
    team_id_filter: Optional[str],
    limit: Optional[int],
    apply_changes: bool,
) -> int:
    db = Prisma()
    await db.connect()

    try:
        deployments = await db.litellm_proxymodeltable.find_many()

        global_model_names: Set[str] = set()
        team_public_names_by_team: Dict[str, Set[str]] = {}

        for deployment in deployments:
            model_info = _safe_parse_model_info(getattr(deployment, "model_info", None))
            deployment_team_id = model_info.get("team_id")
            if isinstance(deployment_team_id, str) and deployment_team_id:
                team_public_name = model_info.get("team_public_model_name")
                if isinstance(team_public_name, str) and team_public_name:
                    team_public_names_by_team.setdefault(deployment_team_id, set()).add(
                        team_public_name
                    )
                continue

            model_name = getattr(deployment, "model_name", None)
            if isinstance(model_name, str) and model_name:
                global_model_names.add(model_name)

        team_query: Dict[str, object] = {"include": {"litellm_model_table": True}}
        if team_id_filter:
            team_query["where"] = {"team_id": team_id_filter}
        if limit is not None:
            team_query["take"] = limit

        teams = await db.litellm_teamtable.find_many(**team_query)
        if not teams:
            print("No teams matched the filter.")
            return 0

        results: List[TeamReconcileResult] = []
        failures = 0

        for team in teams:
            team_id = getattr(team, "team_id", None)
            if not isinstance(team_id, str) or not team_id:
                continue

            team_models = list(getattr(team, "models", []) or [])
            model_table = getattr(team, "litellm_model_table", None)
            alias_map = _safe_parse_dict(
                getattr(model_table, "model_aliases", None) if model_table else None
            )
            alias_public_names = set(alias_map.keys())
            active_team_public_names = team_public_names_by_team.get(team_id, set())

            removed_names: List[str] = []
            for model_name in team_models:
                if not isinstance(model_name, str):
                    continue
                if model_name in SPECIAL_MODEL_TOKENS:
                    continue
                if _is_reserved_team_access_group(model_name=model_name, team_id=team_id):
                    continue
                if model_name in alias_public_names:
                    continue
                if model_name in active_team_public_names:
                    continue
                if model_name in global_model_names:
                    continue
                removed_names.append(model_name)

            if not removed_names:
                continue

            deduped_removed = set(removed_names)
            updated_models = [m for m in team_models if m not in deduped_removed]
            result = TeamReconcileResult(
                team_id=team_id,
                before_count=len(team_models),
                after_count=len(updated_models),
                removed_names=sorted(deduped_removed),
            )

            if apply_changes:
                try:
                    await db.litellm_teamtable.update(
                        where={"team_id": team_id},
                        data={"models": updated_models},
                    )
                except Exception as exc:
                    failures += 1
                    result.error = str(exc)
            results.append(result)

        if not results:
            print("No orphaned team BYOK model names detected.")
            return 0

        mode_label = "APPLY" if apply_changes else "DRY-RUN"
        print(f"Mode: {mode_label}")
        print(f"Teams analyzed: {len(teams)}")
        print(f"Teams with candidate changes: {len(results)}")
        for result in results:
            removed = ", ".join(result.removed_names)
            print(
                f"[team_id={result.team_id}] before={result.before_count} after={result.after_count} removed=[{removed}]"
            )
            if result.error:
                print(f"  ERROR: {result.error}")

        if apply_changes and failures > 0:
            print(f"Completed with partial failures: {failures} team(s) failed.")
            return 1

        if not apply_changes:
            print("Dry-run only. Re-run with --apply to persist changes.")

        return 0
    finally:
        await db.disconnect()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile orphaned team BYOK model names in LiteLLM_TeamTable.models."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist updates. If omitted, script runs in dry-run mode.",
    )
    parser.add_argument(
        "--team-id",
        dest="team_id",
        help="Optional team_id filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of teams to scan.",
    )
    return parser


async def _main_async() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    return await reconcile_team_byok_models(
        team_id_filter=args.team_id,
        limit=args.limit,
        apply_changes=args.apply,
    )


def main() -> None:
    exit_code = asyncio.run(_main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
