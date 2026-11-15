import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from .types import DeploymentCacheIdentity

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_CACHE_RELEVANT_PARAMS: Final = ("cache_control_injection_points",)


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
    """The model id the provider served, without the `provider/` prefix the standard log keeps."""
    return provider_model.removeprefix(f"{provider}/")


def effective_destination(api_base: str | None) -> str | None:
    """Scheme, host and path of the URL the request was actually sent to. The query is dropped since it can
    carry credentials."""
    if not api_base:
        return None
    parts: Final = urlsplit(api_base)
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def deployment_cache_identity(
    selected: SelectedDeployment,
    provider: str,
    provider_model: str,
    destination: str | None,
    settings: DeploymentCacheSettings | None,
) -> DeploymentCacheIdentity:
    """The fingerprint is a plain hash of configuration, not of prompt content, so it needs no secret and every
    worker derives the same identity for the same deployment and settings."""
    canonical: Final = json.dumps(
        {
            "provider": provider,
            "model": served_model(provider, provider_model),
            "destination": destination,
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
    )
