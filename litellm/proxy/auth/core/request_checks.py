from typing import Final

from litellm.proxy.auth.core.host import RequestCheckHost
from litellm.proxy.auth.core.inputs import CheckFacts, CheckSettings
from litellm.proxy.auth.core.outcome import Checked, Exempt, ExemptionReason, RequestCheck
from litellm.proxy.auth.core.plan import is_public_route


def _exemption(facts: CheckFacts, settings: CheckSettings) -> ExemptionReason | None:
    route: Final = facts.request.route
    if facts.force_virtual_key_checks:
        return None
    if is_public_route(route, settings.plan):
        return ExemptionReason.PUBLIC_ROUTE
    if route in settings.auth_off_pass_through_routes:
        return ExemptionReason.PASS_THROUGH_AUTH_OFF
    if settings.no_auth_dev_mode:
        return ExemptionReason.NO_AUTH_DEV_MODE
    if settings.custom_auth_skips_checks:
        return ExemptionReason.CUSTOM_AUTH
    return None


async def check_request(host: RequestCheckHost, facts: CheckFacts, settings: CheckSettings) -> RequestCheck:
    exemption: Final = _exemption(facts, settings)
    if exemption is not None:
        return Exempt(exemption)
    end_user: Final = await host.resolve_end_user(resolve=not facts.token_has_end_user)
    role: Final = await host.load_check_objects()
    if role.is_proxy_admin:
        host.override_admin_user(keep_budget_row=not role.is_builtin_admin)
    host.apply_project()
    if end_user.has_end_user and end_user.has_key_end_user_budget and settings.has_db:
        await host.apply_end_user_default_budget()
    await host.load_end_user_model_budget(assign=end_user.has_end_user)
    skip_budget_checks: Final = host.skips_budget_checks()
    await host.load_team_member_default_model_budget(
        skip_budget_checks=skip_budget_checks or facts.route_skips_budget_checks
    )
    host.merge_request_tags()
    host.bind_admission_counters()
    try:
        await host.run_common_checks(skip_budget_checks=skip_budget_checks)
        if not skip_budget_checks:
            await _check_model_budgets(host, facts)
        if host.committed_spend_tracked():
            await host.carry_committed_spend(read_committed=not facts.route_skips_budget_checks)
        await host.reserve_budget(skip_budget_checks=skip_budget_checks or facts.sub_call)
    finally:
        host.release_spend_counter_batch()
    return Checked()


async def _check_model_budgets(host: RequestCheckHost, facts: CheckFacts) -> None:
    await host.check_team_model_budget()
    if not facts.route_skips_budget_checks:
        await host.check_team_member_model_budget()
        await host.check_end_user_model_budget()
    if facts.sub_call and facts.request.is_llm_api_route:
        await host.check_key_max_budget()
