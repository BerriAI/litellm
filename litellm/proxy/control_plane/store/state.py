from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, TypeAlias

from nacl.exceptions import CryptoError

from litellm.proxy.control_plane.audit.event import CONTROL_PLANE_ID
from litellm.proxy.control_plane.auth.rbac import (
    DEFAULT_ROLES,
    RoleDefinition,
    RolePolicy,
    Scope,
    build_policy,
    parse_role_definition,
)
from litellm.proxy.control_plane.dataplanes.credentials import TokenExchanger
from litellm.proxy.control_plane.dataplanes.registry import (
    Dataplane,
    StaticRegistry,
    build_dataplane,
    build_registry,
)
from litellm.proxy.control_plane.settings import ControlPlaneSettings, RoleSpec
from litellm.proxy.control_plane.store.crypto import SecretCodec
from litellm.proxy.control_plane.store.models import Entry, Kind, revision_key

Key: TypeAlias = tuple[Kind, str]


@dataclass(frozen=True, slots=True)
class Snapshot:
    entries: Mapping[Key, Entry]
    revisions: Mapping[Key, str]
    registry: StaticRegistry
    policy: RolePolicy
    roles: Mapping[str, RoleSpec]


@dataclass(frozen=True, slots=True)
class InvalidSnapshot:
    detail: str


class LiveState:
    def __init__(
        self,
        settings: ControlPlaneSettings,
        exchanger: TokenExchanger,
        policy: RolePolicy,
        codec: SecretCodec | None,
    ) -> None:
        self.settings = settings
        self.exchanger = exchanger
        self.codec = codec
        self.config_roles: Final = MappingProxyType(
            {
                **(
                    {
                        name: RoleSpec(scopes=tuple(scope.value for scope in scopes))
                        for name, scopes in DEFAULT_ROLES.items()
                    }
                    if settings.include_default_roles
                    else {}
                ),
                **settings.roles,
            }
        )
        self.view = Snapshot(
            MappingProxyType({}),
            MappingProxyType({}),
            build_registry(settings.dataplanes, exchanger),
            policy,
            self.config_roles,
        )

    def protected(self, kind: Kind, name: str) -> bool:
        return name in (self.settings.dataplanes if kind == "dataplane" else self.config_roles)

    def build(self, entries: tuple[Entry, ...], revisions: Mapping[Key, str]) -> Snapshot | InvalidSnapshot:
        try:
            return self._build(entries, revisions)
        except (ValueError, CryptoError):
            return InvalidSnapshot("Invalid runtime entity. Check credential strategy, scopes and role parents.")

    def _build(self, entries: tuple[Entry, ...], revisions: Mapping[Key, str]) -> Snapshot | InvalidSnapshot:
        active: Final[Mapping[Key, Entry]] = {
            (entry.change.kind, entry.change.name): entry
            for entry in entries
            if entry.change.op == "upsert" and not self.protected(entry.change.kind, entry.change.name)
        }
        role_specs: Final = {
            **{
                name: RoleSpec.model_validate(entry.change.entity)
                for (kind, name), entry in active.items()
                if kind == "role"
            },
            **self.config_roles,
        }
        parsed: Final = {
            name: parse_role_definition(name, spec.scopes, spec.inherits) for name, spec in role_specs.items()
        }
        if any(not isinstance(definition, RoleDefinition) for definition in parsed.values()):
            return InvalidSnapshot("Unknown role scope.")
        policy: Final = build_policy(
            {name: definition for name, definition in parsed.items() if isinstance(definition, RoleDefinition)},
            include_defaults=False,
        )
        if not isinstance(policy, RolePolicy):
            return InvalidSnapshot("Unknown inherited role.")
        codec: Final = self.codec
        if active and codec is None:
            return InvalidSnapshot("Runtime secret codec is not configured.")
        runtime_dataplanes: Final = {
            name: replace(build_dataplane(name, codec.decode(entry.change.entity), self.exchanger), source="runtime")
            for (kind, name), entry in active.items()
            if kind == "dataplane" and codec is not None
        }
        if CONTROL_PLANE_ID in runtime_dataplanes:
            return InvalidSnapshot("The control-plane dataplane id is reserved.")
        config_registry: Final = build_registry(self.settings.dataplanes, self.exchanger)
        return Snapshot(
            MappingProxyType(active),
            MappingProxyType(dict(revisions)),
            StaticRegistry(MappingProxyType({**runtime_dataplanes, **config_registry.entries})),
            policy,
            MappingProxyType(role_specs),
        )

    def reload(self, entries: tuple[Entry, ...]) -> bool:
        incoming: Final[Mapping[Key, str]] = {(row.change.kind, row.change.name): row.revision for row in entries}
        versions: Final[Mapping[Key, str]] = {
            key: max((self.view.revisions.get(key, "0-0"), incoming.get(key, "0-0")), key=revision_key)
            for key in self.view.revisions.keys() | incoming.keys()
        }
        snapshot: Final = self.build(entries, versions)
        if isinstance(snapshot, InvalidSnapshot):
            return False
        self.view = snapshot
        return True

    def apply(self, entry: Entry) -> bool:
        key: Final = (entry.change.kind, entry.change.name)
        if revision_key(entry.revision) <= revision_key(self.view.revisions.get(key, "0-0")):
            return False
        entries: Final = tuple(row for row_key, row in self.view.entries.items() if row_key != key)
        desired: Final = (*entries, entry) if entry.change.op == "upsert" else entries
        snapshot: Final = self.build(desired, {**self.view.revisions, key: entry.revision})
        if isinstance(snapshot, InvalidSnapshot):
            return False
        self.view = snapshot
        return True


@dataclass(frozen=True, slots=True)
class LiveRegistry:
    state: LiveState

    def get(self, dataplane_id: str) -> Dataplane | None:
        return self.state.view.registry.get(dataplane_id)

    def all(self) -> tuple[Dataplane, ...]:
        return self.state.view.registry.all()


@dataclass(frozen=True, slots=True)
class LivePolicy:
    state: LiveState

    @property
    def names(self) -> frozenset[str]:
        return self.state.view.policy.names

    def scopes_for(self, role: str) -> frozenset[Scope]:
        return self.state.view.policy.scopes_for(role)

    def known(self, raw: tuple[str, ...]) -> frozenset[str]:
        return self.state.view.policy.known(raw)

    def grant(self, roles: frozenset[str]) -> frozenset[Scope]:
        return self.state.view.policy.grant(roles)
