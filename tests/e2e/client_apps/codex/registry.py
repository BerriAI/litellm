from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from proxy_client import ProxyClient

from client_apps.codex.columns import Deployment


@dataclass(frozen=True, slots=True)
class RegistrationError:
    model_name: str
    error: str


def _register(proxy: ProxyClient, deployment: Deployment) -> str | RegistrationError:
    try:
        return proxy.create_model(deployment.model_name, deployment.litellm_params, provider_live=True)
    except AssertionError as error:
        return RegistrationError(model_name=deployment.model_name, error=str(error))


class ColumnRegistry:
    __slots__ = ("_columns", "_proxy", "_registered")

    def __init__(self, proxy: ProxyClient, columns: Mapping[str, tuple[Deployment, ...]]) -> None:
        self._proxy: Final = proxy
        self._columns: Final = columns
        self._registered: Mapping[str, tuple[str | RegistrationError, ...]] = MappingProxyType({})

    def ensure(self, provider: str) -> tuple[RegistrationError, ...]:
        if provider not in self._registered:
            self._registered = MappingProxyType(
                {
                    **self._registered,
                    provider: tuple(_register(self._proxy, deployment) for deployment in self._columns[provider]),
                }
            )
        return tuple(result for result in self._registered[provider] if isinstance(result, RegistrationError))

    def release(self) -> None:
        for results in self._registered.values():
            for model_id in (result for result in results if isinstance(result, str)):
                self._proxy.delete_model(model_id)
