"""Utilities for loading models from the proxy database."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, List, Optional

logger = logging.getLogger("litellm.proxy.server")


def _blank_if_env_ref(value: str, path: str, team_id: str) -> str:
    logger.warning(
        "Ignoring os.environ/ reference in team-scoped DB model "
        "(key=%s, team_id=%s) -- team admins cannot reference proxy "
        "environment variables.",
        path,
        team_id,
    )
    return ""


def _process_list(items: list, path: str, team_id: str, work: deque) -> list:
    """Build result list; push nested dicts onto work queue for deferred fill."""
    out: List[Any] = []
    for i, item in enumerate(items):
        item_path = f"{path}[{i}]"
        if isinstance(item, str) and item.startswith("os.environ/"):
            out.append(_blank_if_env_ref(item, item_path, team_id))
        elif isinstance(item, dict):
            child: dict = {}
            out.append(child)
            work.append((item, child, item_path))
        elif isinstance(item, list):
            out.append(_process_list(item, item_path, team_id, work))
        else:
            out.append(item)
    return out


def _strip_env_refs_iterative(params: dict, team_id: str) -> dict:
    result: dict = {}
    work: deque = deque()
    work.append((params, result, "root"))
    while work:
        src, dst, path = work.popleft()
        for key, value in src.items():
            key_path = f"{path}.{key}" if path != "root" else key
            if isinstance(value, str) and value.startswith("os.environ/"):
                dst[key] = _blank_if_env_ref(value, key_path, team_id)
            elif isinstance(value, dict):
                child: dict = {}
                dst[key] = child
                work.append((value, child, key_path))
            elif isinstance(value, list):
                dst[key] = _process_list(value, key_path, team_id, work)
            else:
                dst[key] = value
    return result


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
    return _strip_env_refs_iterative(litellm_params, team_id)
