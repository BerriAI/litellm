"""Long-lived objects built once per process and handed to the app. Tests build their own."""

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Final

import httpx
from opentelemetry.sdk.trace import TracerProvider
from pydantic import TypeAdapter

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # params is an untyped dict upstream
)
from litellm.proxy.control_plane.audit.log import AuditLog, build_audit_log
from litellm.proxy.control_plane.audit.ocsf import Product
from litellm.proxy.control_plane.audit.pipeline import AuditPipeline, AuditSink, build_pipeline
from litellm.proxy.control_plane.audit.s3 import HttpxObjectStore, S3ObjectLockSink, SigV4Signer
from litellm.proxy.control_plane.audit.sinks import LogSink, OtlpLogSink
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
from litellm.proxy.control_plane.common.tracing import Telemetry, build_log_exporter, build_resource, build_telemetry
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


_REGION: Final = TypeAdapter(str)
_OPTIONAL_REGION: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)


def _s3_sink(settings: ControlPlaneSettings) -> AuditSink:
    s3: Final = settings.audit_s3
    if s3 is None:
        raise ValueError("CONTROL_PLANE_AUDIT_SINKS names s3 but CONTROL_PLANE_AUDIT_S3_BUCKET is not set.")
    aws: Final = BaseAWSLLM()
    region: Final = s3.region or _environment_region(aws)
    return S3ObjectLockSink(
        replace(s3, region=region), HttpxObjectStore(_shared_http_handler(settings)), SigV4Signer(region, aws)
    )


def _environment_region(aws: BaseAWSLLM) -> str:
    """The AWS SDK order: ``AWS_REGION``, then botocore's chain (``AWS_DEFAULT_REGION``, profile, config), then
    litellm's resolver (``AWS_REGION_NAME`` and its default)."""
    from botocore.session import Session

    explicit: Final = os.environ.get("AWS_REGION")
    configured: Final = explicit or _OPTIONAL_REGION.validate_python(Session().get_config_variable("region"))
    return configured or _REGION.validate_python(aws.get_aws_region_name_for_non_llm_api_calls())


def _sink(kind: str, settings: ControlPlaneSettings) -> AuditSink:
    match kind:
        case "otlp":
            return OtlpLogSink(build_log_exporter(settings), build_resource(settings))
        case "s3":
            return _s3_sink(settings)
        case _:
            return LogSink()


def build_audit_pipeline(settings: ControlPlaneSettings, sinks: tuple[AuditSink, ...] | None = None) -> AuditPipeline:
    chosen: Final = sinks if sinks is not None else tuple(_sink(kind, settings) for kind in settings.audit_sinks)
    return build_pipeline(chosen, settings.audit_queue_size, settings.audit_batch_size, settings.audit_strict)


def litellm_product() -> Product:
    try:
        return Product(version=version("litellm"))
    except PackageNotFoundError:
        return Product(version="unknown")


def build_container(
    settings: ControlPlaneSettings,
    jwks: JwksKeyLookup | None = None,
    transport: HttpTransport | None = None,
    exchanger: TokenExchanger | None = None,
    tracer_provider: TracerProvider | None = None,
    audit_sinks: tuple[AuditSink, ...] | None = None,
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
        audit_log=build_audit_log(
            settings.audit_capacity, pipeline=build_audit_pipeline(settings, audit_sinks), product=litellm_product()
        ),
        telemetry=telemetry,
    )
