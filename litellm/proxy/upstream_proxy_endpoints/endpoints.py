from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.llms.custom_httpx.upstream_proxy import UPSTREAM_PROXY_PARAM, UpstreamProxyProtocol
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.config_repository import ConfigRepository
from litellm.types.llms.base import LiteLLMBaseModel

from .store import (
    ConfigRowStore,
    StoredUpstreamProxy,
    UpstreamProxyAddress,
    encrypt_password,
    load_stored_upstream_proxies,
    save_upstream_proxies,
)

_PARAMS: Final = TypeAdapter(Mapping[str, object])

router: Final = APIRouter(prefix="/upstream_proxies", tags=["upstream proxy management"])


class UpstreamProxyView(LiteLLMBaseModel):
    name: str
    protocol: UpstreamProxyProtocol
    host: str
    port: int
    username: str | None
    has_password: bool
    description: str | None
    used_by_models: tuple[str, ...]
    used_by_credentials: tuple[str, ...]


class UpstreamProxyList(LiteLLMBaseModel):
    proxies: tuple[UpstreamProxyView, ...]


class CreateUpstreamProxy(UpstreamProxyAddress):
    password: str | None = None


class UpdateUpstreamProxy(LiteLLMBaseModel):
    """Fields left out keep their stored value; an empty `password` clears it."""

    model_config = ConfigDict(frozen=True)

    protocol: UpstreamProxyProtocol | None = None
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None
    description: str | None = None


async def _read_admin(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        raise HTTPException(status_code=403, detail="Only proxy admins can view upstream proxies.")
    return user_api_key_dict


async def _write_admin(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UserAPIKeyAuth:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only proxy admins can change upstream proxies.")
    return user_api_key_dict


def config_repository() -> ConfigRowStore:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return ConfigRepository(prisma_client, use_writer=True)


def _deployment_references() -> tuple[tuple[str, object], ...]:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return ()
    return tuple(
        (deployment["model_name"], _PARAMS.validate_python(deployment["litellm_params"]).get(UPSTREAM_PROXY_PARAM))
        for deployment in llm_router.registry().deployments() or ()
    )


def _credential_references() -> tuple[tuple[str, object], ...]:
    return tuple(
        (credential.credential_name, credential.credential_values.get(UPSTREAM_PROXY_PARAM))
        for credential in litellm.credential_list
    )


@dataclass(frozen=True, slots=True)
class ProxyReferences:
    models: tuple[tuple[str, object], ...]
    credentials: tuple[tuple[str, object], ...]


def current_references() -> ProxyReferences:
    return ProxyReferences(models=_deployment_references(), credentials=_credential_references())


def _users_of(name: str, references: Iterable[tuple[str, object]]) -> tuple[str, ...]:
    return tuple(sorted({owner for owner, configured in references if configured == name}))


def _views(proxies: Iterable[StoredUpstreamProxy], references: ProxyReferences) -> UpstreamProxyList:
    return UpstreamProxyList(
        proxies=tuple(
            UpstreamProxyView(
                name=proxy.name,
                protocol=proxy.protocol,
                host=proxy.host,
                port=proxy.port,
                username=proxy.username,
                has_password=proxy.encrypted_password is not None,
                description=proxy.description,
                used_by_models=_users_of(proxy.name, references.models),
                used_by_credentials=_users_of(proxy.name, references.credentials),
            )
            for proxy in proxies
        )
    )


def _not_found(name: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Upstream proxy {name!r} does not exist.")


@router.get("", dependencies=[Depends(_read_admin)])
async def list_upstream_proxies(
    repository: Annotated[ConfigRowStore, Depends(config_repository)],
    references: Annotated[ProxyReferences, Depends(current_references)],
) -> UpstreamProxyList:
    return _views((await load_stored_upstream_proxies(repository)).proxies, references)


@router.post("", dependencies=[Depends(_write_admin)])
async def create_upstream_proxy(
    proxy: CreateUpstreamProxy,
    repository: Annotated[ConfigRowStore, Depends(config_repository)],
    references: Annotated[ProxyReferences, Depends(current_references)],
) -> UpstreamProxyList:
    stored: Final = await load_stored_upstream_proxies(repository)
    if any(existing.name == proxy.name for existing in stored.proxies):
        raise HTTPException(status_code=409, detail=f"Upstream proxy {proxy.name!r} already exists.")
    created: Final = StoredUpstreamProxy.model_validate(
        {**proxy.model_dump(exclude={"password"}), "encrypted_password": encrypt_password(proxy.password)}
    )
    saved: Final = await save_upstream_proxies(repository, stored, (*stored.proxies, created))
    return _views(saved.proxies, references)


@router.patch("/{name}", dependencies=[Depends(_write_admin)])
async def update_upstream_proxy(
    name: Annotated[str, Path()],
    update: UpdateUpstreamProxy,
    repository: Annotated[ConfigRowStore, Depends(config_repository)],
    references: Annotated[ProxyReferences, Depends(current_references)],
) -> UpstreamProxyList:
    stored: Final = await load_stored_upstream_proxies(repository)
    current: Final = next((proxy for proxy in stored.proxies if proxy.name == name), None)
    if current is None:
        raise _not_found(name)
    changes: Final = update.model_dump(include=update.model_fields_set - {"password"})
    password: Final = (
        {"encrypted_password": encrypt_password(update.password)} if "password" in update.model_fields_set else {}
    )
    try:
        updated: Final = StoredUpstreamProxy.model_validate({**current.model_dump(), **changes, **password})
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False, include_context=False)) from None
    saved: Final = await save_upstream_proxies(
        repository, stored, tuple(updated if proxy.name == name else proxy for proxy in stored.proxies)
    )
    return _views(saved.proxies, references)


@router.delete("/{name}", dependencies=[Depends(_write_admin)])
async def delete_upstream_proxy(
    name: Annotated[str, Path()],
    repository: Annotated[ConfigRowStore, Depends(config_repository)],
    references: Annotated[ProxyReferences, Depends(current_references)],
) -> UpstreamProxyList:
    stored: Final = await load_stored_upstream_proxies(repository)
    if not any(proxy.name == name for proxy in stored.proxies):
        raise _not_found(name)
    models: Final = _users_of(name, references.models)
    credentials: Final = _users_of(name, references.credentials)
    if models or credentials:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Upstream proxy {name!r} is still used by models {list(models)} and credentials {list(credentials)}. "
                "Point them at another proxy first."
            ),
        )
    saved: Final = await save_upstream_proxies(
        repository, stored, tuple(proxy for proxy in stored.proxies if proxy.name != name)
    )
    return _views(saved.proxies, references)
