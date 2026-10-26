"""Refuses deleting a team that still holds a PTU share, on every route that deletes teams."""

from collections.abc import Sequence
from typing import Final

from fastapi import HTTPException

from litellm.router import Router
from litellm.router_utils.ptu_shares import deployments_shared_with


def raise_if_teams_hold_ptu_shares(team_ids: Sequence[str], llm_router: Router | None) -> None:
    """A team still named in a deployment's ``ptu_shares`` would keep being billed for PTUs nobody can use."""
    if llm_router is None:
        return
    held: Final = tuple(
        (team_id, deployment_ids)
        for team_id in team_ids
        if (deployment_ids := deployments_shared_with(llm_router.get_model_list() or (), team_id))
    )
    if not held:
        return
    team_id, deployment_ids = held[0]
    raise HTTPException(
        status_code=400,
        detail={
            "error": f"Team {team_id} holds a PTU share on deployment {', '.join(deployment_ids)}. "
            "Move its share to another team in model_info.ptu_shares before deleting the team."
        },
    )
