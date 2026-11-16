"""Long-lived objects built once per process and handed to the app. Tests build their own."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import Final

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
from litellm.proxy.control_plane.settings import ControlPlaneSettings


@dataclass(frozen=True, slots=True)
class Container:
    settings: ControlPlaneSettings
    policy: RolePolicy
    token_verifier: TokenVerifier
    login_service: LoginService


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


def build_container(settings: ControlPlaneSettings, jwks: JwksKeyLookup | None = None) -> Container:
    master_key: Final = settings.master_key.get_secret_value()
    policy: Final = build_role_policy(settings)
    sessions: Final = SessionTokenCodec(
        master_key=master_key,
        issuer=settings.session_issuer,
        audience=settings.session_audience,
        ttl=timedelta(seconds=settings.session_ttl_seconds),
        policy=policy,
    )
    password: Final = settings.ui_password
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
    )
