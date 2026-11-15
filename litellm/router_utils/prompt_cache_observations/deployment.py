import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Final, TypeAlias

from pydantic import JsonValue

from litellm.utils import ModelInfoLookupMiss, lookup_model_info_for_registration

from .types import DeploymentCacheIdentity

MinPrefixTokens: TypeAlias = Callable[[str, str], int | None]

_CACHE_RELEVANT_PARAMS: Final = (
    "api_base",
    "api_version",
    "aws_region_name",
    "aws_bedrock_runtime_endpoint",
    "vertex_project",
    "vertex_location",
    "cache_control_injection_points",
)


def exact_min_prefix_tokens(provider_model: str, provider: str) -> int | None:
    """The model's own minimum from the cost map. Unlike `utils.get_prompt_cache_min_tokens` it never
    answers with the operator override, the provider-agnostic default or a generalized fallback row,
    none of which can prove a prefix is long enough for this model."""
    info: Final = lookup_model_info_for_registration(provider_model, provider)
    if isinstance(info, ModelInfoLookupMiss):
        return None
    minimum: Final = info.get("prompt_cache_min_tokens")
    return minimum if isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0 else None


def deployment_cache_identity(
    deployment_id: str,
    provider: str,
    provider_model: str,
    base_model: str | None,
    litellm_params: Mapping[str, JsonValue],
    fingerprint_key: bytes,
    min_prefix_tokens: MinPrefixTokens = exact_min_prefix_tokens,
) -> DeploymentCacheIdentity:
    relevant: Final = {key: litellm_params.get(key) for key in _CACHE_RELEVANT_PARAMS}
    canonical: Final = json.dumps(
        {"provider": provider, "model": provider_model, "base_model": base_model, **relevant},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return DeploymentCacheIdentity(
        deployment_id=deployment_id,
        config_fingerprint=hashlib.blake2b(canonical, key=fingerprint_key, digest_size=16).hexdigest(),
        provider=provider,
        provider_model=provider_model,
        min_prefix_tokens=min_prefix_tokens(base_model or provider_model, provider),
        gateway_injects_breakpoints=bool(relevant["cache_control_injection_points"]),
    )
