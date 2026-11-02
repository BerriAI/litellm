"""
Health-check routing, allowed-fails policies and the failover flags on the host side of the Rust
core: how the constructor resolves them, the static document router-core reads them from, and the
``health()`` members that read the registry's live model list or the background health checks'
results. router-core makes every routing decision (the health-check filter, the cooldown rules,
weighted failover); this module only reads Python values into its documents and back.

The rules are the Python router core's, ported (never imported).
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.router_utils.health_state_cache import DeploymentHealthCache
from litellm.types.router import AllowedFailsPolicy

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.router_config import RouterConfig
    from litellm.types.router import Deployment, DeploymentTypedDict

HEALTH_STATE_KEY: Final = DeploymentHealthCache.CACHE_KEY
# The failover flags the Python core keeps as plain router attributes, with their defaults.
FAILOVER_FLAGS: Final = ("enable_weighted_failover", "fallback_on_unsupported_params")


def resolved_allowed_fails_policy(value: object) -> AllowedFailsPolicy | None:
    """The router's ``allowed_fails_policy`` as the constructor keeps it: a dict is read into an
    ``AllowedFailsPolicy`` (a bad field raises the model's own error), a policy is kept, anything
    else is no policy."""
    if isinstance(value, AllowedFailsPolicy):
        return value
    if isinstance(value, dict):
        return AllowedFailsPolicy.model_validate(value)
    return None


def _staleness(threshold: object) -> float:
    """``health_check_staleness_threshold or DEFAULT_HEALTH_CHECK_INTERVAL * multiplier``."""
    from litellm.constants import DEFAULT_HEALTH_CHECK_INTERVAL, DEFAULT_HEALTH_CHECK_STALENESS_MULTIPLIER

    given: Final = threshold if isinstance(threshold, int | float) and not isinstance(threshold, bool) else None
    return float(given or (DEFAULT_HEALTH_CHECK_INTERVAL * DEFAULT_HEALTH_CHECK_STALENESS_MULTIPLIER))


@dataclasses.dataclass(frozen=True, slots=True)
class HealthRouting:
    """The health-check routing settings, as the constructor resolves them."""

    enable_health_check_routing: bool = False
    health_check_staleness: float = 600.0
    health_check_ignore_transient_errors: bool = False
    background_health_check_model_groups: frozenset[str] | None = None

    @classmethod
    def of(cls, config: "RouterConfig") -> "HealthRouting":
        groups: Final = config.get("background_health_check_model_groups")
        return cls(
            enable_health_check_routing=bool(config.get("enable_health_check_routing")),
            health_check_staleness=_staleness(config.get("health_check_staleness_threshold")),
            health_check_ignore_transient_errors=bool(config.get("health_check_ignore_transient_errors")),
            background_health_check_model_groups=frozenset(groups) if isinstance(groups, Sequence | set) else None,  # pyright: ignore[reportUnknownArgumentType]  # the group names as passed
        )

    def static_document(self) -> Mapping[str, object]:
        groups: Final = self.background_health_check_model_groups
        return {
            "enable_health_check_routing": True if self.enable_health_check_routing else None,
            "health_check_staleness_threshold": self.health_check_staleness,
            "background_health_check_model_groups": sorted(groups) if groups is not None else None,
        }

    def health_checks(self, cache: "DualCache") -> DeploymentHealthCache:
        return DeploymentHealthCache(cache=cache, staleness_threshold=self.health_check_staleness)


def failover_document(plain: Mapping[str, object]) -> Mapping[str, object]:
    """The failover flags router-core reads, from the router's plain attributes (read for
    truthiness, as the Python core's checks read them)."""
    return {
        "enable_weighted_failover": True if plain.get("enable_weighted_failover") else None,
        "fallback_on_unsupported_params": False if not plain.get("fallback_on_unsupported_params", True) else None,
    }


def _model_info(deployment: Mapping[str, object]) -> Mapping[str, object]:
    return as_str_mapping(deployment.get("model_info")) or {}


def is_blocked(deployment: "Deployment") -> bool:
    """``_core_is_deployment_blocked``."""
    model_info: Final[object | None] = getattr(deployment, "model_info", None)
    if model_info is None:
        return False
    return getattr(model_info, "blocked", None) is True


def all_blocked(deployments: "Sequence[DeploymentTypedDict]") -> bool:
    """``_core_are_all_deployments_blocked``."""
    return len(deployments) > 0 and all(
        _model_info(deployment).get("blocked") is True  # pyright: ignore[reportArgumentType]  # a deployment dict
        for deployment in deployments
    )


def fully_blocked_model_names(
    model_list: Sequence[Mapping[str, object]],
) -> set[str]:  # mutable-ok: the core's type, a fresh set
    """``_core_get_fully_blocked_model_names``: groups whose every deployment is blocked."""
    blocked_by_name: Final[dict[str, bool]] = {}  # mutable-ok: built here
    for deployment in model_list:
        name = deployment.get("model_name") or ""
        if not isinstance(name, str) or not name:
            continue
        blocked = _model_info(deployment).get("blocked") is True
        blocked_by_name[name] = blocked_by_name.get(name, True) and blocked
    return {name for name, fully in blocked_by_name.items() if fully}


async def fully_unhealthy_model_names(
    model_list: Sequence[Mapping[str, object]], health_checks: DeploymentHealthCache, policy: object
) -> set[str]:  # mutable-ok: the core's type, a fresh set
    """``_core_async_get_fully_unhealthy_model_names``: groups (and team public names) whose
    every deployment failed its health check; nothing under an allowed_fails_policy."""
    if policy is not None:
        return set()
    unhealthy_ids: Final = await health_checks.async_get_unhealthy_deployment_ids()
    if not unhealthy_ids:
        return set()
    unhealthy_by_name: Final[dict[str, bool]] = {}  # mutable-ok: built here
    for deployment in model_list:
        model_info = _model_info(deployment)
        names = [deployment.get("model_name") or ""]
        team_public_model_name = model_info.get("team_public_model_name")
        if team_public_model_name:
            names.append(team_public_model_name)
        unhealthy = model_info.get("id") in unhealthy_ids
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            unhealthy_by_name[name] = unhealthy_by_name.get(name, True) and unhealthy
    return {name for name, fully in unhealthy_by_name.items() if fully}


__all__ = [
    "FAILOVER_FLAGS",
    "HEALTH_STATE_KEY",
    "HealthRouting",
    "all_blocked",
    "failover_document",
    "fully_blocked_model_names",
    "fully_unhealthy_model_names",
    "is_blocked",
    "resolved_allowed_fails_policy",
]
