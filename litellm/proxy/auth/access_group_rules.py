from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache, reduce
from importlib import import_module
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.constants import AUTH_OBJECTS_TARGET

_TRUSTED_ATTRIBUTES: Final = frozenset(("jwt_claims", "user_id", "user_role"))

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.repositories.prisma_protocols import DatabaseClient


class AccessGroupRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    attribute: str
    contains: str | int | float | bool | None = None
    equals: str | int | float | bool | None = None
    access_group: str

    @model_validator(mode="after")
    def validate_operator(self) -> AccessGroupRule:
        if ("contains" in self.model_fields_set) == ("equals" in self.model_fields_set):
            raise ValueError("An access group rule requires exactly one of contains or equals")
        from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

        if not self.attribute or any(not part for part in self.attribute.split(".")) or not self.access_group:
            raise ValueError("An access group rule requires a nonempty attribute path and access_group")
        if self.attribute.split(".")[0] not in _TRUSTED_ATTRIBUTES:
            raise ValueError(
                "An access group rule attribute must start with an administrator- or IdP-controlled field: "
                + ", ".join(sorted(_TRUSTED_ATTRIBUTES))
            )
        if self.attribute.split(".")[0] not in UserAPIKeyAuth.model_fields:
            raise ValueError("An access group rule attribute must start with a UserAPIKeyAuth field")
        return self


class _Enforcer(Protocol):
    def enforce(self, subject: str, resource: str) -> bool: ...

    def add_policy(self, subject: str, resource: str) -> bool: ...


class _CasbinModel(Protocol):
    def load_model_from_text(self, text: str) -> None: ...


@runtime_checkable
class _Casbin(Protocol):
    Model: Callable[[], _CasbinModel]
    Enforcer: Callable[[_CasbinModel], _Enforcer]


_MODEL: Final = """
[request_definition]
r = sub, obj
[policy_definition]
p = sub, obj
[policy_effect]
e = some(where (p.eft == allow))
[matchers]
m = r.sub == p.sub && r.obj == p.obj
"""
_RULES: Final = TypeAdapter(tuple[AccessGroupRule, ...])
_ATTRIBUTES: Final = TypeAdapter(dict[str, object])
_LIST: Final = TypeAdapter(list[object])
_CASBIN: Final = TypeAdapter(_Casbin, config=ConfigDict(arbitrary_types_allowed=True))
_MISSING: Final = object()


def _subject(operator: Literal["contains", "equals"], claim: str, value: object) -> str:
    return json.dumps((operator, claim, value), ensure_ascii=True, separators=(",", ":"))


def _attribute_segment(value: object, segment: str) -> object:
    if not isinstance(value, Mapping):
        return _MISSING
    return _ATTRIBUTES.validate_python(value).get(segment, _MISSING)


def _claim_value(attributes: Mapping[str, object], path: Sequence[str]) -> object:
    return reduce(_attribute_segment, path, attributes)


def _subjects(rule: AccessGroupRule, identity: UserAPIKeyAuth) -> tuple[str, ...]:
    first, *nested = rule.attribute.split(".")
    root: Final[object] = getattr(identity, first, _MISSING) if first in type(identity).model_fields else _MISSING
    value: Final = (
        _claim_value(_ATTRIBUTES.validate_python(root), nested)
        if nested and isinstance(root, Mapping)
        else _MISSING
        if nested
        else root
    )
    if "contains" in rule.model_fields_set:
        if not isinstance(value, list):
            return ()
        values: Final = _LIST.validate_python(value)
        return tuple(_subject("contains", rule.attribute, item) for item in values if _is_scalar(item))
    if not _is_scalar(value):
        return ()
    return (_subject("equals", rule.attribute, value),)


def _is_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


@lru_cache(maxsize=1)
def _enforcer(policies: tuple[tuple[str, str], ...]) -> _Enforcer:
    casbin: Final = _CASBIN.validate_python(import_module("casbin"))
    model: Final = casbin.Model()
    model.load_model_from_text(_MODEL)
    enforcer: Final = casbin.Enforcer(model)
    for subject, resource in policies:
        enforcer.add_policy(subject, resource)
    return enforcer


def _policy(rule: AccessGroupRule) -> tuple[str, str]:
    operator: Final = "contains" if "contains" in rule.model_fields_set else "equals"
    value: Final = rule.contains if operator == "contains" else rule.equals
    return _subject(operator, rule.attribute, value), rule.access_group


def _validate_rules(value: object) -> tuple[AccessGroupRule, ...]:
    rules: Final = _RULES.validate_python(value)
    if rules:
        _enforcer(tuple(_policy(rule) for rule in rules))
    return rules


@with_service_target(AUTH_OBJECTS_TARGET)
async def _resolve_name(name: str, prisma_client: DatabaseClient | None, cache: UserApiKeyCache) -> str | None:
    from litellm.constants import DEFAULT_ACCESS_GROUP_CACHE_TTL
    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import get_access_group_tables
    from litellm.repositories.table_repositories import AccessGroupRepository

    if prisma_client is None:
        verbose_proxy_logger.debug("Access group rule cannot resolve %s without a database", name)
        return None
    cache_key: Final = f"access_group_rule_name:{name}"
    cached: Final = await cache.async_get_cache(key=cache_key, model_type=LiteLLM_AccessGroupTable)
    if cached is not None:
        groups: Final = await get_access_group_tables(
            (cached.access_group_id,), prisma_client=prisma_client, user_api_key_cache=cache
        )
        if groups and groups[0].access_group_name == name:
            return cached.access_group_id
    group: Final = await AccessGroupRepository(prisma_client).find_by_name(name)
    if group is None:
        verbose_proxy_logger.debug("Access group rule references unknown access group %s", name)
        return None
    await cache.async_set_cache(
        key=cache_key, value=group, model_type=LiteLLM_AccessGroupTable, ttl=DEFAULT_ACCESS_GROUP_CACHE_TTL
    )
    return group.access_group_id


async def access_group_ids_for_identity(
    identity: UserAPIKeyAuth,
    *,
    rules: tuple[AccessGroupRule, ...] | None = None,
    prisma_client: DatabaseClient | None = None,
    cache: UserApiKeyCache | None = None,
) -> tuple[str, ...]:
    if rules is None:
        from litellm.proxy import proxy_server

        settings: Final = _ATTRIBUTES.validate_python(proxy_server.general_settings)  # pyright: ignore[reportUnknownMemberType]  # legacy settings are validated at this boundary
        configured: Final = settings.get("access_group_rules")
        if not configured:
            return ()
        return await access_group_ids_for_identity(
            identity,
            rules=_validate_rules(configured),
            prisma_client=proxy_server.prisma_client,
            cache=proxy_server.user_api_key_cache,
        )
    if not rules or cache is None:
        return ()
    enforcer: Final = _enforcer(tuple(_policy(rule) for rule in rules))
    names: Final = frozenset(
        rule.access_group
        for rule in rules
        if any(enforcer.enforce(subject, rule.access_group) for subject in _subjects(rule, identity))
    )
    ids: Final = tuple([await _resolve_name(name, prisma_client, cache) for name in sorted(names)])
    return tuple(group_id for group_id in ids if group_id is not None)
