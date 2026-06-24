"""Utilities for loading models from the proxy database."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("litellm.proxy.server")


def strip_env_refs_for_team_model(
    litellm_params: dict,
    model_info: Optional[dict],
) -> dict:
    """Remove os.environ/ references from team-scoped DB model litellm_params.

    Team admins can create models via /model/new.  If os.environ/ references
    were resolved for team-scoped models, a team admin could store
    api_key: os.environ/LITELLM_MASTER_KEY together with an api_base they
    control, invoke the model, and receive the real secret in the
    Authorization header -- exfiltrating any proxy environment variable
    (issue #31052).

    Only proxy-admin-created models (team_id is None) and config.yaml models
    may use os.environ/ expansion.  For team-scoped DB models the reference
    is replaced with an empty string and a warning is logged.
    """
    if model_info is None:
        return litellm_params
    team_id = model_info.get("team_id") if isinstance(model_info, dict) else None
    if team_id is None:
        return litellm_params
    stripped: dict = {}
    for k, v in litellm_params.items():
        if isinstance(v, str) and v.startswith("os.environ/"):
            logger.warning(
                "Ignoring os.environ/ reference in team-scoped DB model "
                "(key=%s, team_id=%s) -- team admins cannot reference proxy "
                "environment variables.",
                k,
                team_id,
            )
            stripped[k] = ""
        else:
            stripped[k] = v
    return stripped
