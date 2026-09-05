"""Sync a LiteLLM provider into pi's models.json.

pi ignores ANTHROPIC_BASE_URL/OPENAI_BASE_URL, so `lite pi` routes it through the
proxy by writing a provider entry instead. The key is stored as a $-reference so
the short-lived login token never lands on disk.
"""

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import requests
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

PI_CONFIG_DIR_ENV: Final = "PI_CODING_AGENT_DIR"
PI_PROVIDER_NAME: Final = "litellm"
LITELLM_PROXY_API_KEY_ENV: Final = "LITELLM_PROXY_API_KEY"


@dataclass(frozen=True, slots=True)
class PiSyncError:
    message: str


@dataclass(frozen=True, slots=True)
class ModelLimits:
    context_window: int | None
    max_tokens: int | None


class _Model(BaseModel):
    id: str


class _ModelList(BaseModel):
    data: tuple[_Model, ...]


class _ModelGroup(BaseModel):
    model_group: str
    max_input_tokens: float | None = None
    max_output_tokens: float | None = None


class _ModelGroupList(BaseModel):
    data: tuple[_ModelGroup, ...]


def fetch_model_ids(
    base_url: str,
    api_key: str,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> tuple[str, ...] | PiSyncError:
    url: Final = base_url.rstrip("/") + "/v1/models"
    try:
        resp: Final = get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},  # mutable-ok: requests headers require a dict
            timeout=10,
        )
    except requests.RequestException as e:
        return PiSyncError(f"Could not list models from the proxy: {e}")
    if resp.status_code != 200:
        return PiSyncError(f"The proxy returned HTTP {resp.status_code} for /v1/models; cannot build pi's model list.")
    try:
        listing: Final = _ModelList.model_validate(resp.json())
    except (ValueError, ValidationError) as e:
        return PiSyncError(f"Unexpected /v1/models response from the proxy: {e}")
    ids: Final = tuple(dict.fromkeys(model.id for model in listing.data))
    if not ids:
        return PiSyncError("The proxy returned no models for your key, so pi would have nothing to run.")
    return ids


_NO_LIMITS: Final[Mapping[str, ModelLimits]] = MappingProxyType({})


def fetch_model_limits(
    base_url: str,
    api_key: str,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> Mapping[str, ModelLimits]:
    """Best effort: pi falls back to its own defaults for models without limits,
    so an unavailable /model_group/info must not block the launch."""
    url: Final = base_url.rstrip("/") + "/model_group/info"
    try:
        resp: Final = get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},  # mutable-ok: requests headers require a dict
            timeout=10,
        )
        if resp.status_code != 200:
            return _NO_LIMITS
        listing: Final = _ModelGroupList.model_validate(resp.json())
    except (requests.RequestException, ValueError, ValidationError):
        return _NO_LIMITS
    return MappingProxyType(
        {
            group.model_group: ModelLimits(
                context_window=int(group.max_input_tokens) if group.max_input_tokens else None,
                max_tokens=int(group.max_output_tokens) if group.max_output_tokens else None,
            )
            for group in listing.data
        }
    )


def models_json_path(env: Mapping[str, str]) -> Path:
    override: Final = env.get(PI_CONFIG_DIR_ENV)
    root: Final = Path(override) if override else Path.home() / ".pi" / "agent"
    return root / "models.json"


def _model_entry(
    model_id: str, limits: Mapping[str, ModelLimits]
) -> dict[str, JsonValue]:  # mutable-ok: JSON object is serialized
    limit: Final = limits.get(model_id)
    context: Final[dict[str, JsonValue]] = (  # mutable-ok: JSON field
        {"contextWindow": limit.context_window} if limit and limit.context_window else {}  # mutable-ok: JSON field
    )
    output: Final[dict[str, JsonValue]] = (  # mutable-ok: JSON field
        {"maxTokens": limit.max_tokens} if limit and limit.max_tokens else {}
    )  # mutable-ok: JSON field
    return {"id": model_id, **context, **output}  # mutable-ok: JSON serialization requires a mutable object


def provider_block(
    base_url: str,
    model_ids: tuple[str, ...],
    limits: Mapping[str, ModelLimits] = _NO_LIMITS,
) -> dict[str, JsonValue]:  # mutable-ok: JSON object is serialized
    """openai-completions is the one API shape every LiteLLM model serves.

    Real contextWindow/maxTokens matter: pi otherwise assumes 128k/16384, which
    breaks compaction thresholds and over-asks models with smaller output caps.
    """
    return {  # mutable-ok: JSON serialization requires a mutable object
        "baseUrl": base_url.rstrip("/") + "/v1",
        "api": "openai-completions",
        "apiKey": f"${LITELLM_PROXY_API_KEY_ENV}",
        "models": [_model_entry(model_id, limits) for model_id in model_ids],  # mutable-ok: JSON array
    }


_MODELS_FILE_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


def sync_models_json(
    path: Path,
    base_url: str,
    model_ids: tuple[str, ...],
    limits: Mapping[str, ModelLimits] = _NO_LIMITS,
) -> PiSyncError | None:
    """Replace only the litellm provider entry, leaving the rest of the file intact."""
    try:
        current: Final = (  # mutable-ok: JSON object default
            _MODELS_FILE_ADAPTER.validate_json(path.read_text()) if path.exists() else {}
        )
    except (OSError, ValidationError) as e:
        return PiSyncError(f"Could not read {path} as a JSON object: {e}. Fix or move the file, then retry.")
    existing_providers: Final = current.get("providers", {})  # mutable-ok: JSON object default
    if not isinstance(existing_providers, dict):
        return PiSyncError(f'"providers" in {path} is not an object; fix or move the file, then retry.')
    updated: Final = {  # mutable-ok: JSON serialization requires a mutable object
        **current,
        "providers": {  # mutable-ok: JSON serialization requires a mutable object
            **existing_providers,
            PI_PROVIDER_NAME: provider_block(base_url, model_ids, limits),
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return PiSyncError(f"Could not write {path}: {e}")
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    except OSError as e:
        return PiSyncError(f"Could not write {path}: {e}")
    try:
        with os.fdopen(fd, "w") as file:
            file.write(json.dumps(updated, indent=2) + "\n")
        os.replace(tmp_name, path)
    except OSError as e:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        return PiSyncError(f"Could not write {path}: {e}")
    return None


__all__ = (
    "LITELLM_PROXY_API_KEY_ENV",
    "PI_CONFIG_DIR_ENV",
    "PI_PROVIDER_NAME",
    "ModelLimits",
    "PiSyncError",
    "fetch_model_ids",
    "fetch_model_limits",
    "models_json_path",
    "provider_block",
    "sync_models_json",
)
