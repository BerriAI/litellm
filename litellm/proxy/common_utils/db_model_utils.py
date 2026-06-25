"""Utilities for loading models from the proxy database."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("litellm.proxy.server")


def _strip_env_refs(value: Any, key: str, team_id: str) -> Any:
    """Recursively replace os.environ/ references with empty string."""
    if isinstance(value, str) and value.startswith("os.environ/"):
        logger.warning(
            "Ignoring os.environ/ reference in team-scoped DB model "
            "(key=%s, team_id=%s) -- team admins cannot reference proxy "
            "environment variables.",
            key,
            team_id,
        )
        return ""
    if isinstance(value, dict):
        return {k: _strip_env_refs(v, f"{key}.{k}", team_id) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_env_refs(item, f"{key}[{i}]", team_id) for i, item in enumerate(value)]
    return value


def strip_env_refs_for_team_model(
    litellm_params: dict,
    model_info: Optional[dict],
) -> dict:
    """Remove os.environ/ references from team-scoped DB model litellm_params.

    Only proxy-admin-created models (team_id is None) and config.yaml models
    may use os.environ/ expansion.  For team-scoped DB models the reference
    is replaced with an empty string and a warning is logged.
    """
    if model_info is None:
        return litellm_params
    team_id = model_info.get("team_id") if isinstance(model_info, dict) else None
    if team_id is None:
        return litellm_params
    return {k: _strip_env_refs(v, k, team_id) for k, v in litellm_params.items()}
