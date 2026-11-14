"""Shared deployment URL resolution for SAP GenAI Hub providers."""

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import join_api_base

from .chat.handler import GenAIHubOrchestrationError
from .credentials import AICoreDeployment, AICoreDeploymentList

AICORE_DEPLOYMENT_URL_ENV_VAR: Final = "AICORE_ORCHESTRATION_DEPLOYMENT_URL"

_deployment_url_cache: Final[  # mutable-ok: process-wide cache, guarded by _deployment_url_lock
    dict[tuple[str, str], str]
] = {}
_deployment_url_lock: Final = threading.Lock()


def select_deployment_url(resources: Sequence[AICoreDeployment]) -> str:
    candidates: Final = tuple(
        sorted(
            ((r["deploymentUrl"], r["createdAt"], r["configurationName"]) for r in resources if r.get("deploymentUrl")),
            key=lambda c: c[1],
            reverse=True,
        )
    )
    if len(candidates) == 0:
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


def fetch_deployment_url(
    base_url: str,
    headers: Mapping[str, str],
) -> str:
    deployments: Final[AICoreDeploymentList] = litellm.module_level_client.get(
        join_api_base(base_url, "lm/deployments"),
        headers=headers,
        params={
            "scenarioId": "orchestration",
            "executableIds": ["orchestration"],
            "status": "RUNNING",
        },
    ).json()
    return select_deployment_url(deployments.get("resources", []))


def resolve_deployment_url(
    base_url: str,
    resource_group: str,
    fetcher: Callable[[], str],
) -> str:
    cache_key: Final[tuple[str, str]] = (base_url, resource_group)
    cached_url = _deployment_url_cache.get(cache_key)  # rebind-ok: fast-path read before acquiring the lock
    if cached_url is not None:
        return cached_url
    with _deployment_url_lock:
        rechecked_url = _deployment_url_cache.get(cache_key)  # rebind-ok: double-check under lock after fast-path miss
        if rechecked_url is not None:
            return rechecked_url
        fetched_url: Final[str] = os.environ.get(AICORE_DEPLOYMENT_URL_ENV_VAR) or fetcher()
        _deployment_url_cache[cache_key] = fetched_url
        return fetched_url
