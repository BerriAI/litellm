"""Shared deployment URL resolution for SAP GenAI Hub providers.

Both chat (orchestration) and embedding configs discover the same
orchestration deployment, so a single process-wide cache keyed on
(base_url, resource_group) is shared across all SAP provider types.
"""

import os
import threading
from collections.abc import Callable, Mapping
from typing import Final

import litellm
from litellm._logging import verbose_logger

from .chat.handler import GenAIHubOrchestrationError
from .credentials import AICoreDeploymentList
from litellm.litellm_core_utils.url_utils import join_api_base

_deployment_url_cache: dict[tuple[str, str], str] = {}  # mutable-ok: process-wide cache, guarded by _deployment_url_lock
_deployment_url_lock: Final = threading.Lock()


def fetch_deployment_url(
    base_url: str,
    headers: Mapping[str, str],
) -> str:
    deployments: Final[AICoreDeploymentList] = (
        litellm.module_level_client.get(
            join_api_base(base_url, "lm/deployments"),
            headers=headers,
            params={  # mutable-ok: one-shot query-params dict passed directly to httpx, not stored
                "scenarioId": "orchestration",
                "executableIds": ["orchestration"],
                "status": "RUNNING",
            },
        ).json()
    )

    candidates: Final[list[tuple[str, str, str]]] = sorted(
        (
            (resource["deploymentUrl"], resource["createdAt"], resource["configurationName"])
            for resource in deployments.get("resources", [])
            if resource.get("deploymentUrl")
        ),
        key=lambda c: c[1],
        reverse=True,
    )
    if not candidates:
        raise GenAIHubOrchestrationError(status_code=404, message="No orchestration deployment found in SAP AI Core.")
    if len(candidates) > 1:
        verbose_logger.warning(
            "SAP: %d orchestration deployments found; using newest (name=%r, url=%r). Others ignored: %s.",
            len(candidates),
            candidates[0][2],
            candidates[0][0],
            tuple(v[2] or v[0] for v in candidates[1:]),
        )
    return candidates[0][0]


def resolve_deployment_url(
    base_url: str,
    resource_group: str,
    env_var: str,
    fetcher: Callable[[], str],
) -> str:
    cache_key: Final[tuple[str, str]] = (base_url, resource_group)
    cached_url = _deployment_url_cache.get(cache_key)
    if cached_url is not None:
        return cached_url
    with _deployment_url_lock:
        rechecked_url = _deployment_url_cache.get(cache_key)
        if rechecked_url is not None:
            return rechecked_url
        fetched_url: Final[str] = os.environ.get(env_var) or fetcher()
        _deployment_url_cache[cache_key] = fetched_url
        return fetched_url
