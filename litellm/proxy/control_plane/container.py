"""Long-lived objects built once per process and handed to the app. Tests build their own."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import Final

import httpx
from opentelemetry.sdk.trace import TracerProvider

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # params is an untyped dict upstream
)
from litellm.proxy.control_plane.audit.log import AuditLog, build_audit_log
from litellm.proxy.control_plane.auth.jwt import JwksKeyLookup, SessionTokenCodec, TokenVerifier, build_token_verifier
from litellm.proxy.control_plane.auth.login import LoginService, LoginThrottle
from litellm.proxy.control_plane.auth.rbac import (
    RoleDefinition,
    RolePolicy,
    RolePolicyError,
    UnknownParent,
    UnknownScope,
    build_policy,
    parse_role_definition,
)
from litellm.proxy.control_plane.common.fanout import Fanout
from litellm.proxy.control_plane.common.tracing import Telemetry, build_telemetry
from litellm.proxy.control_plane.dataplanes.client import DataplaneClient, HttpTransport, RetryPolicy
from litellm.proxy.control_plane.dataplanes.credentials import TokenExchanger, http_token_exchanger
from litellm.proxy.control_plane.dataplanes.registry import DataplaneRegistry, build_registry
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from litellm.types.llms.custom_http import httpxSpecialProvider


@dataclass(frozen=True, slots=True)
class Container:
    settings: ControlPlaneSettings
    policy: RolePolicy
    token_verifier: TokenVerifier
    login_service: LoginService
    registry: DataplaneRegistry
    dataplane_client: DataplaneClient
    fanout: Fanout
    audit_log: AuditLog
    telemetry: Telemetry


class InvalidRolePolicy(ValueError):
    def __init__(self, error: RolePolicyError) -> None:
        match error:
            case UnknownScope(role=role, scope=scope):
                super().__init__(f"role {role!r} grants unknown scope {scope!r}")
            case UnknownParent(role=role, parent=parent):
                super().__init__(f"role {role!r} inherits unknown role {parent!r}")


def _parse_roles(settings: ControlPlaneSettings) -> Mapping[str, RoleDefinition]:
    definitions: Final = MappingProxyType(
        {name: parse_role_definition(name, spec.scopes, spec.inherits) for name, spec in settings.roles.items()}
    )
    for error in definitions.values():
        if isinstance(error, UnknownScope):
            raise InvalidRolePolicy(error)
    return MappingProxyType(
        {name: definition for name, definition in definitions.items() if isinstance(definition, RoleDefinition)}
    )


def build_role_policy(settings: ControlPlaneSettings) -> RolePolicy:
    policy: Final = build_policy(_parse_roles(settings), include_defaults=settings.include_default_roles)
    if isinstance(policy, RolePolicy):
        return policy
    raise InvalidRolePolicy(policy)


def _shared_http_handler(settings: ControlPlaneSettings) -> AsyncHTTPHandler:
    return get_async_httpx_client(
        llm_provider=httpxSpecialProvider.ControlPlane, params={"timeout": settings.dataplane_timeout_seconds}
    )


@dataclass(frozen=True, slots=True)
class HandlerTransport:
    """Plain httpx requests over the shared handler's pool, re-read per call because the handler cache closes and
    recreates the pool on eviction. No handler retries, only httpx errors."""

    handler: AsyncHTTPHandler

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,  # mutable-ok: httpx takes dict
        json: dict[str, object] | None = None,  # mutable-ok: httpx takes dict
        headers: dict[str, str] | None = None,  # mutable-ok: httpx takes dict
        timeout: float,
    ) -> httpx.Response:
        return await self.handler.client.request(
            method, url, params=params, json=json, headers=headers, timeout=timeout, follow_redirects=False
        )


def _shared_http_client(settings: ControlPlaneSettings) -> HttpTransport:
    return HandlerTransport(_shared_http_handler(settings))


def build_container(
    settings: ControlPlaneSettings,
    jwks: JwksKeyLookup | None = None,
    transport: HttpTransport | None = None,
    exchanger: TokenExchanger | None = None,
    tracer_provider: TracerProvider | None = None,
) -> Container:
    master_key: Final = settings.master_key.get_secret_value()
    policy: Final = build_role_policy(settings)
    http: Final = transport if transport is not None else _shared_http_client(settings)
    exchange: Final = (
        exchanger
        if exchanger is not None
        else http_token_exchanger(_shared_http_handler(settings), settings.dataplane_timeout_seconds)
    )
    sessions: Final = SessionTokenCodec(
        master_key=master_key,
        issuer=settings.session_issuer,
        audience=settings.session_audience,
        ttl=timedelta(seconds=settings.session_ttl_seconds),
        policy=policy,
    )
    password: Final = settings.ui_password
    telemetry: Final = build_telemetry(settings, tracer_provider)
    return Container(
        settings=settings,
        policy=policy,
        token_verifier=build_token_verifier(settings, policy, jwks),
        login_service=LoginService(
            sessions=sessions,
            policy=policy,
            ttl_seconds=settings.session_ttl_seconds,
            admin_username=settings.ui_username,
            admin_password=None if password is None else password.get_secret_value(),
            throttle=LoginThrottle(time.monotonic),
        ),
        registry=build_registry(settings.dataplanes, exchange),
        dataplane_client=DataplaneClient(
            transport=http,
            timeout_seconds=settings.dataplane_timeout_seconds,
            retry=RetryPolicy(attempts=settings.dataplane_retry_attempts),
            tracing=telemetry.recorder,
        ),
        fanout=Fanout(concurrency=settings.fanout_concurrency),
        audit_log=build_audit_log(settings.audit_capacity),
        telemetry=telemetry,
    )
