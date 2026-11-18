import hashlib
from collections.abc import Mapping
from typing import Annotated, Final, Literal, Protocol, TypeAlias
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter, ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import AuthenticationError, ServiceUnavailableError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared factory leaves params untyped
)
from litellm.llms.ollama.common_utils import OllamaModelInfo
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.credential_hydration import (
    hydrate_named_credential_authoritative,
    stored_credential_provider,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import credential_api_key_secret_reference
from litellm.utils import aresolve_api_key_secret_reference


class _ModelCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: tuple[str, ...]) -> None: ...


def _new_model_cache() -> _ModelCache:
    return InMemoryCache(default_ttl=60)


router: Final = APIRouter()
_CACHE: Final = _new_model_cache()
_MODELS: Final = TypeAdapter(tuple[str, ...])
OllamaDiscoveryProvider: TypeAlias = Literal["ollama", "ollama_chat"]

_DEFAULT_API_BASE: Final = "https://integrate.api.nvidia.com/v1"


class _CredentialMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    credential_info: Mapping[str, object]


class _ListedModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    id: str


class _ModelListing(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_ListedModel, ...]


class _OllamaModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    name: str | None = None
    model: str | None = None


class _OllamaListing(BaseModel):
    model_config = ConfigDict(frozen=True)

    models: tuple[_OllamaModel, ...]


class UpstreamModelDiscoveryConnection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr | None = None
    api_base: str | None = None


class UpstreamModelDiscoveryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    credential_name: str | None = None
    credential_values: UpstreamModelDiscoveryConnection | None = None


class UpstreamModelDiscoveryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["success", "unavailable"]
    models: tuple[str, ...] = ()


def _resolved_secret(value: str | None) -> str | None:
    if value is not None and value.startswith("os.environ/"):
        return get_secret_str(value.removeprefix("os.environ/"))
    return value


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


async def nim_http_client() -> httpx.AsyncClient:
    return get_async_httpx_client(llm_provider="nvidia_nim", params={"follow_redirects": False, "timeout": 10}).client


async def ollama_http_client() -> httpx.AsyncClient:
    return get_async_httpx_client(llm_provider="ollama", params={"follow_redirects": False, "timeout": 10}).client


@router.post(
    "/credentials/nvidia_nim/models",
    tags=["credential management"],
    response_model=UpstreamModelDiscoveryResponse,
)
async def discover_nim_models(
    body: UpstreamModelDiscoveryRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    http_client: Annotated[httpx.AsyncClient, Depends(nim_http_client)],
) -> UpstreamModelDiscoveryResponse:
    """List upstream NIM models using a saved credential or submitted connection settings.

    Requires a proxy administrator. Returns model IDs only, without asserting inference access,
    pricing or capabilities. Upstream failures return unavailable so the UI can use its catalog.
    """
    return await _discover_models("nvidia_nim", body, user_api_key_dict, http_client)


@router.post(
    "/credentials/{provider}/models",
    tags=["credential management"],
    response_model=UpstreamModelDiscoveryResponse,
)
async def discover_ollama_models(
    provider: OllamaDiscoveryProvider,
    body: UpstreamModelDiscoveryRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    http_client: Annotated[httpx.AsyncClient, Depends(ollama_http_client)],
) -> UpstreamModelDiscoveryResponse:
    """List model IDs on the selected Ollama server for the Add Model picker.

    Requires a proxy administrator. Uses a saved credential or explicitly submitted connection settings.
    Returns unavailable when the server cannot list models so the UI can use its catalog.
    """
    return await _discover_models(provider, body, user_api_key_dict, http_client)


async def _discover_models(
    provider: Literal["nvidia_nim"] | OllamaDiscoveryProvider,
    body: UpstreamModelDiscoveryRequest,
    user_api_key_dict: UserAPIKeyAuth,
    http_client: httpx.AsyncClient,
) -> UpstreamModelDiscoveryResponse:
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can discover upstream models")
    if body.credential_name and body.credential_values is not None:
        raise HTTPException(status_code=400, detail="Saved credentials cannot be combined with connection overrides")
    credential: Final = (
        await hydrate_named_credential_authoritative(body.credential_name, prisma_client)
        if body.credential_name
        else None
    )
    if body.credential_name and credential is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    credential_provider: Final = (
        stored_credential_provider(
            _CredentialMetadata.model_validate(credential).credential_info.get("custom_llm_provider")
        )
        if credential is not None
        else None
    )
    if credential_provider is not None and credential_provider != provider:
        raise HTTPException(status_code=400, detail=f"Select a {provider} credential")
    submitted: Final = body.credential_values or UpstreamModelDiscoveryConnection()
    literal_key: Final = submitted.api_key.get_secret_value() if submitted.api_key is not None else None
    if any(
        value is not None and value.startswith(("os.environ/", "oidc/")) for value in (submitted.api_base, literal_key)
    ):
        raise HTTPException(
            status_code=400, detail="Connection settings must use literal values, not secret references"
        )
    values: Final = credential.credential_values if credential is not None else {}
    stored_api_base: Final = _text(values.get("api_base"))
    saved_api_base: Final = _resolved_secret(stored_api_base)
    if stored_api_base is not None and not saved_api_base:
        return UpstreamModelDiscoveryResponse(status="unavailable")
    selection_base: Final = saved_api_base or submitted.api_base
    api_base: Final = (
        selection_base or get_secret_str("NVIDIA_NIM_API_BASE") or _DEFAULT_API_BASE
        if provider == "nvidia_nim"
        else OllamaModelInfo.get_server_api_base(selection_base)
    )
    try:
        parsed: Final = urlsplit(api_base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid API base") from None
    has_user_info: Final = bool(parsed.username or parsed.password)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or has_user_info:
        raise HTTPException(
            status_code=400, detail="API base must be an HTTP or HTTPS endpoint without user information"
        )
    if parsed.query or parsed.fragment:
        raise HTTPException(status_code=400, detail="API base cannot contain a query or fragment")
    saved_values: Final = credential_api_key_secret_reference(
        {"api_key": values.get("api_key"), "api_base": saved_api_base}
    )
    try:
        saved_api_key: Final = await aresolve_api_key_secret_reference(
            _resolved_secret(_text(saved_values.get("api_key"))), None, provider, {"api_base": api_base}
        )
    except (AuthenticationError, ServiceUnavailableError):
        return UpstreamModelDiscoveryResponse(status="unavailable")
    api_key: Final = (
        saved_api_key
        or literal_key
        or _default_api_key(provider, credential is not None or submitted.api_base is None, selection_base)
    )
    cache_key: Final = hashlib.sha256(f"{provider}\0{api_base.rstrip('/')}\0{api_key or ''}".encode()).hexdigest()
    cached: Final = _CACHE.get_cache(cache_key)
    if cached is not None:
        return UpstreamModelDiscoveryResponse(status="success", models=_MODELS.validate_python(cached))
    try:
        ids: Final = await _list_model_ids(provider, api_base, api_key, http_client)
    except (OpenAIError, httpx.HTTPError, ValidationError, ValueError):
        return UpstreamModelDiscoveryResponse(status="unavailable")
    models: Final = tuple(sorted(frozenset(f"{provider}/{model_id}" for model_id in ids if model_id)))
    if not models:
        return UpstreamModelDiscoveryResponse(status="unavailable")
    _CACHE.set_cache(cache_key, models)
    return UpstreamModelDiscoveryResponse(status="success", models=models)


def _default_api_key(
    provider: Literal["nvidia_nim"] | OllamaDiscoveryProvider,
    use_nim_default: bool,
    selection_base: str | None,
) -> str | None:
    if provider == "nvidia_nim":
        return get_secret_str("NVIDIA_NIM_API_KEY") if use_nim_default else None
    return get_secret_str("OLLAMA_API_KEY") if not selection_base else None


async def _list_model_ids(
    provider: Literal["nvidia_nim"] | OllamaDiscoveryProvider,
    api_base: str,
    api_key: str | None,
    http_client: httpx.AsyncClient,
) -> tuple[str, ...]:
    if provider != "nvidia_nim":
        response: Final = await http_client.get(
            join_api_base(api_base, "api/tags"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            follow_redirects=False,
        )
        response.raise_for_status()
        tags: Final = _OllamaListing.model_validate_json(response.content)
        return tuple((model.name or model.model or "").removeprefix(f"{provider}/") for model in tags.models)
    client: Final = AsyncOpenAI(
        api_key=api_key or "not-required",
        base_url=api_base,
        timeout=10,
        max_retries=0,
        http_client=http_client,
    )
    page: Final = await client.models.list()
    listing: Final = _ModelListing.model_validate(page.model_dump())
    return tuple(item.id for item in listing.data)
