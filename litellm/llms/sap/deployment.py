"""Shared deployment URL resolution for SAP GenAI Hub providers."""

import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import join_api_base

from .chat.handler import GenAIHubOrchestrationError
from .credentials import AICoreDeployment, AICoreDeploymentList

AICORE_DEPLOYMENT_URL_ENV_VAR: Final = "AICORE_ORCHESTRATION_DEPLOYMENT_URL"
AICORE_DEPLOYMENT_CACHE_TTL_ENV_VAR: Final = "AICORE_ORCHESTRATION_DEPLOYMENT_CACHE_TTL_SECONDS"

_deployment_url_cache: Final[  # mutable-ok: process-wide cache storing (url, monotonic_timestamp)
    dict[tuple[str, str], tuple[str, float]]
] = {}


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


def fetch_deployment_url_once(
    base_url: str,
    headers: Mapping[str, str],
) -> str:
    response: Final = litellm.module_level_client.get(
        join_api_base(base_url, "lm/deployments"),
        headers=headers,
        params={
            "scenarioId": "orchestration",
            "executableIds": ["orchestration"],
            "status": "RUNNING",
        },
    )
    if response.status_code != 200:
        raise GenAIHubOrchestrationError(status_code=response.status_code, message=response.text)
    deployments: Final[AICoreDeploymentList] = response.json()
    return select_deployment_url(deployments.get("resources", []))


def fetch_deployment_url(
    base_url: str,
    get_headers: Callable[[], Mapping[str, str]],
    invalidate_token: Callable[[str], None],
) -> str:
    headers: Final = get_headers()
    try:
        return fetch_deployment_url_once(base_url, headers)
    except GenAIHubOrchestrationError as err:
        if err.status_code != 401:
            raise
        invalidate_token(headers["Authorization"])
        return fetch_deployment_url_once(base_url, get_headers())


def _is_expired(cached_at: float, now: float) -> bool:
    raw: Final = os.environ.get(AICORE_DEPLOYMENT_CACHE_TTL_ENV_VAR)
    if raw is None:
        return False
    try:
        ttl: Final = float(raw)
    except ValueError:
        return False
    if ttl <= 0:
        return False
    return (now - cached_at) >= ttl


def resolve_deployment_url(
    base_url: str,
    resource_group: str,
    fetcher: Callable[[], str],
) -> str:
    cache_key: Final[tuple[str, str]] = (base_url, resource_group)
    now: Final[float] = time.monotonic()

    cached_entry = _deployment_url_cache.get(cache_key)  # rebind-ok: cache lookup before potential fetch
    if cached_entry is not None and not _is_expired(cached_entry[1], now):
        return cached_entry[0]

    fetched_url: Final[str] = os.environ.get(AICORE_DEPLOYMENT_URL_ENV_VAR) or fetcher()
    _deployment_url_cache[cache_key] = (fetched_url, now)
    return fetched_url
