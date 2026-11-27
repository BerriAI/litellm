from collections.abc import Mapping
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.constants import SEMANTIC_CACHE_PARAMS_KEY, SEMANTIC_CACHE_POLICY_KEY
from litellm.litellm_core_utils.core_helpers import as_str_mapping


class SemanticCacheParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    enabled: bool | None = None
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    ttl: int | None = Field(default=None, gt=0)


CachePolicyScope: TypeAlias = Literal["key", "user", "team", "global"]


class SemanticCachePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope: CachePolicyScope
    params: SemanticCacheParams | None = None
    key_opted_out: bool = False


def semantic_cache_params(metadata: object) -> SemanticCacheParams | None:
    fields: Final = as_str_mapping(metadata)
    if fields is None:
        return None
    raw: Final = fields.get(SEMANTIC_CACHE_PARAMS_KEY)
    if raw is None:
        return None
    try:
        params: Final = SemanticCacheParams.model_validate(raw)
    except ValidationError:
        values: Final = as_str_mapping(raw)
        return SemanticCacheParams(enabled=False) if values is not None and values.get("enabled") is False else None
    if params.enabled is None and params.similarity_threshold is None and params.ttl is None:
        return None
    return params


def semantic_cache_policy(request: Mapping[str, object]) -> SemanticCachePolicy | None:
    return _policy_from_controls(request.get("cache"))


def response_cache_policy_ttl(request: Mapping[str, object]) -> int | None:
    policy: Final = semantic_cache_policy(request)
    return policy.params.ttl if policy is not None and policy.params is not None else None


def _policy_from_controls(controls: object) -> SemanticCachePolicy | None:
    fields: Final = as_str_mapping(controls)
    policy: Final = fields.get(SEMANTIC_CACHE_POLICY_KEY) if fields is not None else None
    return policy if isinstance(policy, SemanticCachePolicy) else None


def semantic_similarity_override(controls: object) -> float | None:
    policy: Final = _policy_from_controls(controls)
    return policy.params.similarity_threshold if policy is not None and policy.params is not None else None
