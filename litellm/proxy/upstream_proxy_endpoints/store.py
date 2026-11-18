from typing import Final, Protocol

from fastapi import HTTPException
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.upstream_proxy import (
    UPSTREAM_PROXY_REGISTRY,
    NamedUpstreamProxy,
    UpstreamProxyProtocol,
    UpstreamProxyRegistry,
    validate_upstream_proxy_host,
    validate_upstream_proxy_name,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.repositories.config_repository import ConfigParam
from litellm.types.llms.base import LiteLLMBaseModel

UPSTREAM_PROXIES_PARAM: Final = "upstream_proxies"


class ConfigRowStore(Protocol):
    async def get_param(self, param_name: str) -> ConfigParam | None: ...

    async def set_param_if_revision(self, param_name: str, param_value: object, revision: int) -> bool: ...


_ENCRYPTED: Final = TypeAdapter(str)


class UpstreamProxyAddress(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    protocol: UpstreamProxyProtocol
    host: str
    port: int = Field(ge=1, le=65535)
    username: str | None = None
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        return validate_upstream_proxy_name(value)

    @field_validator("host")
    @classmethod
    def _valid_host(cls, value: str) -> str:
        return validate_upstream_proxy_host(value)

    @field_validator("username", "description")
    @classmethod
    def _blank_is_unset(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class StoredUpstreamProxy(UpstreamProxyAddress):
    encrypted_password: str | None = None


class StoredUpstreamProxies(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    revision: int = 0
    proxies: tuple[StoredUpstreamProxy, ...] = Field(default_factory=tuple)


def encrypt_password(password: str | None) -> str | None:
    return None if not password else _ENCRYPTED.validate_python(encrypt_value_helper(password))


def _named(stored: StoredUpstreamProxy) -> NamedUpstreamProxy:
    return NamedUpstreamProxy(
        name=stored.name,
        protocol=stored.protocol,
        host=stored.host,
        port=stored.port,
        username=stored.username,
        password=None
        if stored.encrypted_password is None
        else decrypt_value_helper(stored.encrypted_password, key=f"upstream_proxy:{stored.name}"),
        description=stored.description,
    )


async def load_stored_upstream_proxies(repository: ConfigRowStore) -> StoredUpstreamProxies:
    parameter: Final = await repository.get_param(UPSTREAM_PROXIES_PARAM)
    if parameter is None:
        return StoredUpstreamProxies()
    try:
        return StoredUpstreamProxies.model_validate(parameter.param_value)
    except ValidationError:
        raise HTTPException(status_code=500, detail="Stored upstream proxies are invalid.") from None


async def save_upstream_proxies(
    repository: ConfigRowStore,
    previous: StoredUpstreamProxies,
    proxies: tuple[StoredUpstreamProxy, ...],
    registry: UpstreamProxyRegistry = UPSTREAM_PROXY_REGISTRY,
) -> StoredUpstreamProxies:
    updated: Final = StoredUpstreamProxies(revision=previous.revision + 1, proxies=proxies)
    if not await repository.set_param_if_revision(
        UPSTREAM_PROXIES_PARAM, updated.model_dump(mode="json"), previous.revision
    ):
        raise HTTPException(
            status_code=409, detail="Upstream proxies changed while you were editing. Reload and retry."
        )
    registry.replace(_named(proxy) for proxy in updated.proxies)
    return updated


async def sync_upstream_proxy_registry(
    repository: ConfigRowStore, registry: UpstreamProxyRegistry = UPSTREAM_PROXY_REGISTRY
) -> None:
    try:
        stored: Final = await load_stored_upstream_proxies(repository)
    except HTTPException:
        verbose_proxy_logger.exception("Skipping upstream proxy sync: the stored row does not validate")
        return
    registry.replace(_named(proxy) for proxy in stored.proxies)
