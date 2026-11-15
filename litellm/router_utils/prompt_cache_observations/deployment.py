import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.model_cost_table import published

from .types import DeploymentCacheIdentity, WireShape

MinPrefixTokens: TypeAlias = Callable[[str, str], int | None]

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_CACHE_RELEVANT_PARAMS: Final = (
    "api_version",
    "aws_region_name",
    "aws_bedrock_runtime_endpoint",
    "vertex_project",
    "vertex_location",
    "cache_control_injection_points",
)
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


@dataclass(frozen=True, slots=True)
class DeploymentCacheSettings:
    """The cache-relevant parameters an attempt was sent with, captured before the gateway consumes
    `cache_control_injection_points`. Credentials are never among them. Kept as plain tuples, since the call's
    logging details are deep-copied for some callbacks."""

    selected: SelectedDeployment
    settings: tuple[tuple[str, JsonValue], ...]


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


def _json_or_none(value: object) -> JsonValue:
    try:
        return _JSON.validate_python(value)
    except ValidationError:
        return None


def merged_cache_settings(earlier: object, later: DeploymentCacheSettings) -> DeploymentCacheSettings:
    """An attempt can pass the deployment hook twice, once per layer of a translated call, and the inner layer
    may already have consumed settings the outer one saw, so each setting keeps the first value reported."""
    if not isinstance(earlier, DeploymentCacheSettings) or earlier.selected != later.selected:
        return later
    reported: Final = dict(later.settings)
    return DeploymentCacheSettings(
        selected=later.selected,
        settings=tuple((key, value if value is not None else reported.get(key)) for key, value in earlier.settings),
    )


def deployment_cache_settings(kwargs: Mapping[str, object]) -> DeploymentCacheSettings | None:
    selected: Final = selected_deployment(kwargs.get("model_info"))
    if selected is None:
        return None
    return DeploymentCacheSettings(
        selected=selected, settings=tuple((key, _json_or_none(kwargs.get(key))) for key in _CACHE_RELEVANT_PARAMS)
    )


def served_model(provider: str, provider_model: str) -> str:
    """The model id the provider served, without the `provider/` and Bedrock `invoke/` or `converse/` route
    prefixes the standard log keeps."""
    unprefixed: Final = provider_model.removeprefix(f"{provider}/")
    if provider != "bedrock":
        return unprefixed
    route: Final = next((prefix for prefix in _BEDROCK_ROUTE_PREFIXES if unprefixed.startswith(prefix)), "")
    return unprefixed[len(route) :]


def exact_min_prefix_tokens(provider_model: str, provider: str) -> int | None:
    """The minimum from the cost-map row keyed by exactly this model for this provider. No alias, base_model,
    region-stripped name, generalized row, operator override or provider-agnostic default qualifies, since
    any of them can describe a different model than the one that served the request."""
    allowed: Final = _COST_MAP_PROVIDERS.get(provider, frozenset())
    model: Final = served_model(provider, provider_model)
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
    wire: WireShape | None,
    settings: DeploymentCacheSettings | None,
    min_prefix_tokens: MinPrefixTokens = exact_min_prefix_tokens,
) -> DeploymentCacheIdentity:
    """The fingerprint is a plain hash of configuration, not of prompt content, so it needs no secret and every
    worker derives the same identity for the same deployment and settings."""
    canonical: Final = json.dumps(
        {
            "provider": provider,
            "model": served_model(provider, provider_model),
            "destination": destination,
            "wire": wire.value if wire is not None else None,
            "base_model": selected.base_model,
            "settings": dict(settings.settings) if settings is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return DeploymentCacheIdentity(
        deployment_id=selected.deployment_id,
        config_fingerprint=hashlib.blake2b(canonical, digest_size=16).hexdigest(),
        provider=provider,
        provider_model=provider_model,
        min_prefix_tokens=min_prefix_tokens(provider_model, provider),
    )
