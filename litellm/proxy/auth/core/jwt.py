from dataclasses import dataclass
from typing import Final

from litellm.proxy.auth.core.answers import (
    MappedKey,
    NoKeyMapping,
    NotRegistered,
    PendingAutoRegister,
    Registered,
    RegistrationFailed,
)
from litellm.proxy.auth.core.host import JwtHost
from litellm.proxy.auth.core.identity import (
    JwtAuthentication,
    JwtKey,
    JwtKeyMappingFailed,
    JwtOwnerDeactivated,
    JwtProxyAdmin,
    JwtUser,
    ManagedAgentNeedsDirectJwt,
)
from litellm.proxy.auth.core.inputs import JwtSettings
from litellm.proxy.auth.core.model_budgets import (
    check_user_model_budgets,
    skip_model_budgets,
    skips_model_budgets,
    within_model_budgets,
)
from litellm.proxy.auth.core.outcome import ModelBudgetExceeded


@dataclass(frozen=True, slots=True)
class _Unmapped:
    pending_auto_register: bool


def _needs_email_backfill(mapped: MappedKey) -> bool:
    return (
        mapped.key_user_id is not None
        and not mapped.key_has_email
        and mapped.claim_has_email
        and mapped.claim_user_id == mapped.key_user_id
    )


async def _map_to_virtual_key(host: JwtHost) -> JwtKey | ManagedAgentNeedsDirectJwt | _Unmapped:
    await host.decode_claims()
    if await host.is_managed_agent():
        return ManagedAgentNeedsDirectJwt()
    mapping: Final = await host.resolve_mapping()
    match mapping:
        case MappedKey():
            if _needs_email_backfill(mapping):
                await host.backfill_mapped_user_email()
            return JwtKey()
        case PendingAutoRegister():
            return _Unmapped(pending_auto_register=True)
        case NoKeyMapping():
            return _Unmapped(pending_auto_register=False)


async def _apply_policy(host: JwtHost, settings: JwtSettings, *, pending_auto_register: bool) -> JwtAuthentication:
    policy: Final = await host.run_jwt_policy()
    if policy.owner_deactivated:
        return JwtOwnerDeactivated(user_id=policy.user_id)
    if policy.is_proxy_admin:
        if pending_auto_register:
            await host.remember_proxy_admin()
        return JwtProxyAdmin()
    registration: Final = await host.auto_register() if pending_auto_register and settings.has_db else NotRegistered()
    match registration:
        case RegistrationFailed(reason=reason):
            return JwtKeyMappingFailed(reason=reason)
        case Registered(links_existing_key=links_existing_key):
            host.attach_user_model_budgets(policy.user_model_budgets)
            if links_existing_key and settings.master_key_set:
                return JwtKey()
        case NotRegistered():
            pass
    exemption: Final = await host.read_budget_exemption()
    await host.load_project()
    if skips_model_budgets(exemption):
        return JwtUser(budgets=skip_model_budgets())
    budgets: Final = within_model_budgets(await check_user_model_budgets(host.model_budgets()))
    if isinstance(budgets, ModelBudgetExceeded):
        return budgets
    return JwtUser(budgets=budgets)


async def authenticate(host: JwtHost, settings: JwtSettings) -> JwtAuthentication:
    mapped: Final = await _map_to_virtual_key(host) if settings.virtual_key_mapping else _Unmapped(False)
    if not isinstance(mapped, _Unmapped):
        return mapped
    return await _apply_policy(host, settings, pending_auto_register=mapped.pending_auto_register)
