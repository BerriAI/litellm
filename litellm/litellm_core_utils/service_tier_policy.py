from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.exceptions import PermissionDeniedError


class ServiceTierPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    allowed_service_tiers: tuple[str, ...] | None = None


class _AuthMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_api_key_auth: ServiceTierPolicy | None = None


@dataclass(frozen=True, slots=True)
class ServiceTierDenied:
    tier: str
    model: str


_REQUEST: Final = TypeAdapter(dict[str, object])


def normalize_service_tier(tier: str) -> str:
    return "priority" if tier == "fast" else tier


def service_tier_policy(
    request: Mapping[str, object], metadata_field: Literal["metadata", "litellm_metadata"] = "litellm_metadata"
) -> ServiceTierPolicy:
    metadata: Final = _AuthMetadata.model_validate(request.get(metadata_field) or {})
    return metadata.user_api_key_auth or ServiceTierPolicy()


def service_tier_denial(policy: ServiceTierPolicy, tier: str, model: str) -> ServiceTierDenied | None:
    if not policy.allowed_service_tiers:
        return None
    allowed: Final = frozenset(map(normalize_service_tier, policy.allowed_service_tiers))
    return None if normalize_service_tier(tier) in allowed else ServiceTierDenied(tier=tier, model=model)


def raise_service_tier_denial(denial: ServiceTierDenied | None) -> None:
    if denial is not None:
        raise PermissionDeniedError(
            message=f"This key is not allowed to use service_tier={denial.tier}",
            model=denial.model,
            llm_provider="",
            response=httpx.Response(403, request=httpx.Request("POST", "https://litellm.invalid")),
        )


def apply_service_tier_policy(request: Mapping[str, object], policy: ServiceTierPolicy) -> Mapping[str, object]:
    if not policy.allowed_service_tiers:
        return request
    extra_body: Final = _REQUEST.validate_python(request.get("extra_body") or {})
    raw_tier: Final = extra_body.get("service_tier", request.get("service_tier"))
    tier: Final = normalize_service_tier(raw_tier) if isinstance(raw_tier, str) else "default"
    raw_model: Final = request.get("model")
    raise_service_tier_denial(service_tier_denial(policy, tier, raw_model if isinstance(raw_model, str) else ""))
    return {
        **request,
        "service_tier": tier,
        **({"extra_body": {**extra_body, "service_tier": tier}} if "service_tier" in extra_body else {}),
    }
