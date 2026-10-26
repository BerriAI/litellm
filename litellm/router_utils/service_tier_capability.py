from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter, ValidationError

from litellm.types.utils import ServiceTierInfo, ServiceTierMetadata


class _ConfiguredServiceTier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    name: str | None = None
    description: str | None = None


_PUBLISHED_TIERS: Final = TypeAdapter(Sequence[ServiceTierInfo])
_CONFIGURED_TIERS: Final = TypeAdapter(Sequence[str | _ConfiguredServiceTier])


@dataclass(frozen=True, slots=True)
class ServiceTierCapabilities:
    tiers: tuple[ServiceTierInfo, ...] | None = None
    default: str | None = None
    declared: bool = False


def _configured_tier(item: str | _ConfiguredServiceTier) -> ServiceTierInfo:
    if isinstance(item, str):
        return {"id": item.strip()}
    if item.name is None:
        return {"id": item.id} if item.description is None else {"id": item.id, "description": item.description}
    return (
        {"id": item.id, "name": item.name}
        if item.description is None
        else {"id": item.id, "name": item.name, "description": item.description}
    )


def service_tier_capabilities(model_info: Mapping[str, object], *, configured: bool = False) -> ServiceTierCapabilities:
    raw: Final = model_info.get("service_tiers")
    if raw is None:
        return ServiceTierCapabilities()
    try:
        parsed: Final = (
            tuple(_configured_tier(item) for item in _CONFIGURED_TIERS.validate_python(raw, strict=True))
            if configured
            else tuple(_PUBLISHED_TIERS.validate_python(raw, strict=True))
        )
    except ValidationError:
        return ServiceTierCapabilities(declared=True)
    kept: Final = (
        tuple(tier for index, tier in enumerate(parsed) if tier["id"] not in {seen["id"] for seen in parsed[:index]})
        if configured
        else parsed
    )
    ids: Final = tuple(tier["id"] for tier in kept)
    if any(not tier_id.strip() for tier_id in ids) or len(frozenset(ids)) != len(ids):
        return ServiceTierCapabilities(declared=True)
    if not configured and any("name" in tier and not tier["name"].strip() for tier in kept):
        return ServiceTierCapabilities(declared=True)
    default: Final = model_info.get("default_service_tier")
    return ServiceTierCapabilities(
        tiers=kept,
        default=default if isinstance(default, str) and default in ids else None,
        declared=True,
    )


def intersect_service_tier_capabilities(
    deployments: Sequence[ServiceTierCapabilities],
) -> ServiceTierCapabilities:
    if not deployments:
        return ServiceTierCapabilities()
    declared: Final = any(deployment.declared for deployment in deployments)
    if any(deployment.tiers == () for deployment in deployments):
        return ServiceTierCapabilities(tiers=(), declared=declared)
    if any(deployment.tiers is None for deployment in deployments):
        return ServiceTierCapabilities(declared=declared)
    first: Final = deployments[0]
    shared: Final = tuple(
        tier
        for tier in first.tiers or ()
        if all(any(other["id"] == tier["id"] for other in deployment.tiers or ()) for deployment in deployments)
    )
    default: Final = first.default
    return ServiceTierCapabilities(
        tiers=shared,
        default=(
            default
            if default is not None
            and any(tier["id"] == default for tier in shared)
            and all(deployment.default == default for deployment in deployments)
            else None
        ),
        declared=declared,
    )


def service_tier_metadata(capabilities: ServiceTierCapabilities) -> ServiceTierMetadata:
    if capabilities.tiers is None:
        return {}
    if capabilities.default is None:
        return {"service_tiers": capabilities.tiers}
    return {"service_tiers": capabilities.tiers, "default_service_tier": capabilities.default}
