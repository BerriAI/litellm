import hashlib
from collections.abc import Mapping
from typing import Annotated, Final, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter, ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared factory leaves params untyped
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.credential_hydration import (
    hydrate_named_credential_authoritative,
    stored_credential_provider,
)
from litellm.secret_managers.main import get_secret_str


class _ModelCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: tuple[str, ...]) -> None: ...


def _new_model_cache() -> _ModelCache:
    return InMemoryCache(default_ttl=60)


router: Final = APIRouter()
_CACHE: Final = _new_model_cache()
_MODELS: Final = TypeAdapter(tuple[str, ...])
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


class NimModelDiscoveryConnection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr | None = None
    api_base: str | None = None


class NimModelDiscoveryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    credential_name: str | None = None
    credential_values: NimModelDiscoveryConnection | None = None


class NimModelDiscoveryResponse(BaseModel):
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


@router.post(
    "/credentials/nvidia_nim/models",
    tags=["credential management"],
    response_model=NimModelDiscoveryResponse,
)
async def discover_nim_models(
    body: NimModelDiscoveryRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    http_client: Annotated[httpx.AsyncClient, Depends(nim_http_client)],
) -> NimModelDiscoveryResponse:
    """List upstream NIM models using a saved credential or submitted connection settings.

    Requires a proxy administrator. Returns model IDs only, without asserting inference access,
    pricing or capabilities. Upstream failures return unavailable so the UI can use its catalog.
    """
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can discover NIM models")
    if body.credential_name and body.credential_values is not None:
        raise HTTPException(status_code=400, detail="Saved credentials cannot be combined with connection overrides")
    credential: Final = (
        await hydrate_named_credential_authoritative(body.credential_name, prisma_client)
        if body.credential_name
        else None
    )
    if body.credential_name and credential is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    provider: Final = (
        stored_credential_provider(
            _CredentialMetadata.model_validate(credential).credential_info.get("custom_llm_provider")
        )
        if credential is not None
        else None
    )
    if provider is not None and provider != "nvidia_nim":
        raise HTTPException(status_code=400, detail="Select an NVIDIA NIM credential")
    submitted: Final = body.credential_values or NimModelDiscoveryConnection()
    literal_key: Final = submitted.api_key.get_secret_value() if submitted.api_key is not None else None
    if any(
        value is not None and value.startswith(("os.environ/", "oidc/")) for value in (submitted.api_base, literal_key)
    ):
        raise HTTPException(
            status_code=400, detail="Connection settings must use literal values, not secret references"
        )
    values: Final = credential.credential_values if credential is not None else {}
    api_base: Final = (
        _resolved_secret(_text(values.get("api_base")))
        or submitted.api_base
        or (get_secret_str("NVIDIA_NIM_API_BASE") or _DEFAULT_API_BASE)
    )
    try:
        parsed: Final = urlsplit(api_base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid NIM API base") from None
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(
            status_code=400, detail="API base must be an HTTP or HTTPS endpoint without user information"
        )
    if parsed.query or parsed.fragment:
        raise HTTPException(status_code=400, detail="API base cannot contain a query or fragment")
    api_key: Final = (
        _resolved_secret(_text(values.get("api_key")))
        or literal_key
        or (get_secret_str("NVIDIA_NIM_API_KEY") if credential is not None or submitted.api_base is None else None)
    )
    cache_key: Final = hashlib.sha256(f"{api_base.rstrip('/')}\0{api_key or ''}".encode()).hexdigest()
    cached: Final = _CACHE.get_cache(cache_key)
    if cached is not None:
        return NimModelDiscoveryResponse(status="success", models=_MODELS.validate_python(cached))
    try:
        client: Final = AsyncOpenAI(
            api_key=api_key or "not-required",
            base_url=api_base,
            timeout=10,
            max_retries=0,
            http_client=http_client,
        )
        page: Final = await client.models.list()
        listing: Final = _ModelListing.model_validate(page.model_dump())
        models: Final = tuple(sorted(frozenset(f"nvidia_nim/{item.id}" for item in listing.data if item.id)))
        if not models:
            return NimModelDiscoveryResponse(status="unavailable")
        _CACHE.set_cache(cache_key, models)
        return NimModelDiscoveryResponse(status="success", models=models)
    except (OpenAIError, httpx.HTTPError, ValidationError):
        return NimModelDiscoveryResponse(status="unavailable")
