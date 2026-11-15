import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.model_cost_table import published

from .types import DeploymentCacheIdentity

MinPrefixTokens: TypeAlias = Callable[[str, str], int | None]

_COST_MAP_PROVIDERS: Final = MappingProxyType(
    {
        "anthropic": frozenset(("anthropic",)),
        "bedrock": frozenset(("bedrock", "bedrock_converse")),
        "vertex_ai": frozenset(("vertex_ai-anthropic_models",)),
        "azure_ai": frozenset(("azure_ai",)),
    }
)
_BEDROCK_ROUTE_PREFIXES: Final = ("invoke/", "converse/")
_OPERATION_SEGMENTS: Final = frozenset(("converse", "converse-stream", "invoke", "invoke-with-response-stream"))
_VERTEX_OPERATIONS: Final = (":rawPredict", ":streamRawPredict")


class _DeploymentModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    prompt_cache_observation: bool = False
    base_model: str | None = None


@dataclass(frozen=True, slots=True)
class SelectedDeployment:
    deployment_id: str
    base_model: str | None


def selected_deployment(model_info: object) -> SelectedDeployment | None:
    """The attempt's deployment, when its `model_info` opts it into observation. Reading it from the attempt
    rather than the router's current list attributes a request to the deployment that actually served it."""
    try:
        parsed: Final = _DeploymentModelInfo.model_validate(model_info)
    except ValidationError:
        return None
    if not parsed.prompt_cache_observation:
        return None
    return SelectedDeployment(deployment_id=parsed.id, base_model=parsed.base_model)


def _cost_map_model(provider: str, provider_model: str) -> str:
    if provider != "bedrock":
        return provider_model
    prefix: Final = next((prefix for prefix in _BEDROCK_ROUTE_PREFIXES if provider_model.startswith(prefix)), "")
    return provider_model[len(prefix) :]


def exact_min_prefix_tokens(provider_model: str, provider: str) -> int | None:
    """The minimum from the cost-map row keyed by exactly this model for this provider. No alias, base_model,
    region-stripped name, generalized row, operator override or provider-agnostic default qualifies, since
    any of them can describe a different model than the one that served the request."""
    allowed: Final = _COST_MAP_PROVIDERS.get(provider, frozenset())
    model: Final = _cost_map_model(provider, provider_model)
    table: Final = published()
    rows: Final = (as_str_mapping(table.get(key)) for key in (f"{provider}/{model}", model))
    minimum: Final = next(
        (
            row.get("prompt_cache_min_tokens")
            for row in rows
            if row is not None and row.get("litellm_provider") in allowed
        ),
        None,
    )
    return minimum if isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0 else None


def effective_destination(api_base: str | None) -> str | None:
    """Scheme, host and path of the URL the request was actually sent to, which carries the region, project
    and endpoint the environment resolved. The trailing operation is dropped so streaming and non-streaming
    calls to one destination agree, and the query is dropped since it can carry credentials."""
    if not api_base:
        return None
    parts: Final = urlsplit(api_base)
    segments: Final = parts.path.rstrip("/").split("/")
    kept: Final = segments[:-1] if segments and segments[-1] in _OPERATION_SEGMENTS else segments
    path: Final = "/".join(kept)
    operation: Final = next((suffix for suffix in _VERTEX_OPERATIONS if path.endswith(suffix)), "")
    return f"{parts.scheme}://{parts.netloc}{path[: len(path) - len(operation)]}"


def deployment_cache_identity(
    selected: SelectedDeployment,
    provider: str,
    provider_model: str,
    destination: str | None,
    fingerprint_key: bytes,
    min_prefix_tokens: MinPrefixTokens = exact_min_prefix_tokens,
) -> DeploymentCacheIdentity:
    canonical: Final = json.dumps(
        {
            "provider": provider,
            "model": provider_model,
            "destination": destination,
            "base_model": selected.base_model,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return DeploymentCacheIdentity(
        deployment_id=selected.deployment_id,
        config_fingerprint=hashlib.blake2b(canonical, key=fingerprint_key, digest_size=16).hexdigest(),
        provider=provider,
        provider_model=provider_model,
        min_prefix_tokens=min_prefix_tokens(provider_model, provider),
    )
