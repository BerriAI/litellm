from typing import Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class SupportsEnforce(Protocol):
    """Anything that can decide a casbin-style authorization check.

    Lets the authorizer depend on the capability, not the concrete
    ``CasbinEnforcer``, so test doubles and the loud-open sentinel type-check too.
    """

    def enforce(self, subject: str, domain: str, obj: str, action: str) -> bool: ...


@runtime_checkable
class CasbinRuleRow(Protocol):
    """A persisted casbin rule row (the LiteLLM_CasbinRule shape).

    Typed structurally so the policy store doesn't depend on the generated Prisma
    model class while still being fully typed over ``ptype`` and ``v0``..``v5``.
    """

    ptype: Optional[str]
    v0: Optional[str]
    v1: Optional[str]
    v2: Optional[str]
    v3: Optional[str]
    v4: Optional[str]
    v5: Optional[str]


class CasbinRuleTable(Protocol):
    """The LiteLLM_CasbinRule Prisma accessor surface the policy store reads."""

    async def find_many(self) -> List[CasbinRuleRow]: ...


class _PolicyDBNamespace(Protocol):
    litellm_casbinrule: CasbinRuleTable


class PolicyDB(Protocol):
    """A Prisma client narrowed to the casbin-rule table the policy store needs."""

    db: _PolicyDBNamespace


class CasbinRuleWriteTable(Protocol):
    """The casbin-rule table surface the policy-admin endpoints read and write."""

    async def find_many(self) -> List[CasbinRuleRow]: ...

    async def create(self, data: Dict[str, str]) -> object: ...

    async def delete_many(self, where: Dict[str, str]) -> int: ...


class _AdminDBNamespace(Protocol):
    litellm_casbinrule: CasbinRuleWriteTable


class PolicyAdminDB(Protocol):
    """A Prisma client narrowed to the casbin-rule writes the admin API performs."""

    db: _AdminDBNamespace
