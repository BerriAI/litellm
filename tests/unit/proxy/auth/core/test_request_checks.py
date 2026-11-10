from dataclasses import dataclass, replace
from typing import Final

import pytest

from litellm.proxy.auth.core.answers import BudgetExemption, EndUserScope, TokenRole
from litellm.proxy.auth.core.inputs import CheckFacts, CheckSettings, PlanSettings, RequestFacts
from litellm.proxy.auth.core.outcome import Checked, Exempt, ExemptionReason, RequestCheck
from litellm.proxy.auth.core.python_core import PythonCore

PLAN: Final = PlanSettings(
    custom_auth=False,
    premium_user=True,
    public_routes=frozenset({"/health/liveness", "/metrics"}),
    additional_public_routes=("/public/*",),
    metrics_require_auth=False,
    oauth2_introspection=False,
    oauth2_proxy=False,
    jwt_enabled=False,
    oauth2_routing_overrides=(),
    unauthenticated_pass_through_routes=frozenset(),
)
SETTINGS: Final = CheckSettings(
    plan=PLAN,
    auth_off_pass_through_routes=frozenset({"/open"}),
    no_auth_dev_mode=False,
    custom_auth_skips_checks=False,
    has_db=True,
)
FACTS: Final = CheckFacts(
    request=RequestFacts(route="/chat/completions", token="hash", is_llm_api_route=True),
    force_virtual_key_checks=False,
    sub_call=False,
    token_has_end_user=False,
    route_skips_budget_checks=False,
)
FULL_RUN: Final = (
    ("resolve_end_user", True),
    ("load_check_objects", None),
    ("apply_project", None),
    ("load_end_user_model_budget", False),
    ("read_budget_exemption", None),
    ("load_team_member_default_model_budget", False),
    ("merge_request_tags", None),
    ("bind_admission_counters", None),
    ("run_common_checks", False),
    ("check_team_model_budget", None),
    ("check_team_member_model_budget", None),
    ("check_end_user_model_budget", None),
    ("committed_spend_tracked", None),
    ("reserve_budget", False),
    ("release_spend_counter_batch", None),
)


class CommonCheckDenied(Exception):
    pass


@dataclass
class RecordingCheckHost:
    end_user: EndUserScope = EndUserScope(has_end_user=False, has_key_end_user_budget=False)
    role: TokenRole = TokenRole(is_proxy_admin=False, is_builtin_admin=False)
    skip_budget_checks: bool = False
    committed_spend: bool = False
    deny_common_checks: bool = False
    calls: tuple[tuple[str, object], ...] = ()

    def _record(self, name: str, payload: object = None) -> None:
        self.calls = (*self.calls, (name, payload))

    async def resolve_end_user(self, resolve: bool) -> EndUserScope:
        self._record("resolve_end_user", resolve)
        return self.end_user

    async def load_check_objects(self) -> TokenRole:
        self._record("load_check_objects")
        return self.role

    def override_admin_user(self, keep_budget_row: bool) -> None:
        self._record("override_admin_user", keep_budget_row)

    def apply_project(self) -> None:
        self._record("apply_project")

    async def apply_end_user_default_budget(self) -> None:
        self._record("apply_end_user_default_budget")

    async def load_end_user_model_budget(self, assign: bool) -> None:
        self._record("load_end_user_model_budget", assign)

    async def read_budget_exemption(self) -> BudgetExemption:
        self._record("read_budget_exemption")
        return BudgetExemption(route_skips_budgets=False, mcp_discovery=False, model_is_free=self.skip_budget_checks)

    async def load_team_member_default_model_budget(self, skip_budget_checks: bool) -> None:
        self._record("load_team_member_default_model_budget", skip_budget_checks)

    def merge_request_tags(self) -> None:
        self._record("merge_request_tags")

    def bind_admission_counters(self) -> None:
        self._record("bind_admission_counters")

    async def run_common_checks(self, skip_budget_checks: bool) -> None:
        self._record("run_common_checks", skip_budget_checks)
        if self.deny_common_checks:
            raise CommonCheckDenied

    async def check_team_model_budget(self) -> None:
        self._record("check_team_model_budget")

    async def check_team_member_model_budget(self) -> None:
        self._record("check_team_member_model_budget")

    async def check_end_user_model_budget(self) -> None:
        self._record("check_end_user_model_budget")

    async def check_key_max_budget(self) -> None:
        self._record("check_key_max_budget")

    def committed_spend_tracked(self) -> bool:
        self._record("committed_spend_tracked")
        return self.committed_spend

    async def carry_committed_spend(self, read_committed: bool) -> None:
        self._record("carry_committed_spend", read_committed)

    async def reserve_budget(self, skip_budget_checks: bool) -> None:
        self._record("reserve_budget", skip_budget_checks)

    def release_spend_counter_batch(self) -> None:
        self._record("release_spend_counter_batch")


async def _check(
    host: RecordingCheckHost, facts: CheckFacts = FACTS, settings: CheckSettings = SETTINGS
) -> RequestCheck:
    return await PythonCore().authorize().check_request(host, facts, settings)


def _route(route: str, *, force: bool = False) -> CheckFacts:
    return replace(FACTS, request=replace(FACTS.request, route=route), force_virtual_key_checks=force)


@pytest.mark.parametrize(
    ("route", "settings", "reason"),
    [
        ("/health/liveness", SETTINGS, ExemptionReason.PUBLIC_ROUTE),
        ("/metrics", SETTINGS, ExemptionReason.PUBLIC_ROUTE),
        ("/public/docs", SETTINGS, ExemptionReason.PUBLIC_ROUTE),
        ("/open", SETTINGS, ExemptionReason.PASS_THROUGH_AUTH_OFF),
        ("/chat/completions", replace(SETTINGS, no_auth_dev_mode=True), ExemptionReason.NO_AUTH_DEV_MODE),
        ("/chat/completions", replace(SETTINGS, custom_auth_skips_checks=True), ExemptionReason.CUSTOM_AUTH),
    ],
)
async def test_exempt_requests_skip_every_check_and_do_no_io(
    route: str, settings: CheckSettings, reason: ExemptionReason
) -> None:
    host: Final = RecordingCheckHost()
    assert await _check(host, _route(route), settings) == Exempt(reason)
    assert host.calls == ()


async def test_metrics_is_checked_when_the_proxy_requires_auth_for_it() -> None:
    settings: Final = replace(SETTINGS, plan=replace(PLAN, metrics_require_auth=True))
    host: Final = RecordingCheckHost()
    assert await _check(host, _route("/metrics"), settings) == Checked()
    assert ("run_common_checks", False) in host.calls


async def test_additional_public_routes_need_a_premium_proxy() -> None:
    settings: Final = replace(SETTINGS, plan=replace(PLAN, premium_user=False))
    assert await _check(RecordingCheckHost(), _route("/public/docs"), settings) == Checked()


@pytest.mark.parametrize("route", ["/health/liveness", "/open"])
async def test_forced_virtual_key_checks_override_every_exemption(route: str) -> None:
    settings: Final = replace(SETTINGS, no_auth_dev_mode=True, custom_auth_skips_checks=True)
    host: Final = RecordingCheckHost()
    assert await _check(host, _route(route, force=True), settings) == Checked()
    assert host.calls == FULL_RUN


async def test_a_request_runs_every_check_in_order() -> None:
    host: Final = RecordingCheckHost()
    assert await _check(host) == Checked()
    assert host.calls == FULL_RUN


async def test_the_end_user_is_resolved_only_when_the_token_has_none() -> None:
    host: Final = RecordingCheckHost()
    _ = await _check(host, replace(FACTS, token_has_end_user=True))
    assert host.calls[0] == ("resolve_end_user", False)


@pytest.mark.parametrize(("is_builtin_admin", "keep_budget_row"), [(False, True), (True, False)])
async def test_a_proxy_admin_token_overrides_the_user_row_keeping_its_budget_unless_built_in(
    is_builtin_admin: bool, keep_budget_row: bool
) -> None:
    host: Final = RecordingCheckHost(role=TokenRole(is_proxy_admin=True, is_builtin_admin=is_builtin_admin))
    _ = await _check(host)
    assert host.calls[1:4] == (
        ("load_check_objects", None),
        ("override_admin_user", keep_budget_row),
        ("apply_project", None),
    )


@pytest.mark.parametrize(
    ("end_user", "has_db", "applied"),
    [
        (EndUserScope(has_end_user=True, has_key_end_user_budget=True), True, True),
        (EndUserScope(has_end_user=True, has_key_end_user_budget=True), False, False),
        (EndUserScope(has_end_user=True, has_key_end_user_budget=False), True, False),
        (EndUserScope(has_end_user=False, has_key_end_user_budget=True), True, False),
    ],
)
async def test_the_key_end_user_default_budget_applies_only_to_a_resolved_end_user_with_a_database(
    end_user: EndUserScope, has_db: bool, applied: bool
) -> None:
    host: Final = RecordingCheckHost(end_user=end_user)
    _ = await _check(host, settings=replace(SETTINGS, has_db=has_db))
    assert (("apply_end_user_default_budget", None) in host.calls) == applied
    assert ("load_end_user_model_budget", end_user.has_end_user) in host.calls


async def test_budget_exempt_requests_skip_model_budgets_and_reservation() -> None:
    host: Final = RecordingCheckHost(skip_budget_checks=True)
    _ = await _check(host)
    names: Final = tuple(name for name, _ in host.calls)
    assert ("load_team_member_default_model_budget", True) in host.calls
    assert ("run_common_checks", True) in host.calls
    assert ("reserve_budget", True) in host.calls
    assert "check_team_model_budget" not in names


async def test_routes_that_skip_budgets_keep_only_the_team_model_budget() -> None:
    host: Final = RecordingCheckHost(committed_spend=True)
    _ = await _check(host, replace(FACTS, route_skips_budget_checks=True))
    names: Final = tuple(name for name, _ in host.calls)
    assert ("load_team_member_default_model_budget", True) in host.calls
    assert "check_team_model_budget" in names
    assert "check_team_member_model_budget" not in names
    assert "check_end_user_model_budget" not in names
    assert ("carry_committed_spend", False) in host.calls


@pytest.mark.parametrize(("is_llm_api_route", "checked"), [(True, True), (False, False)])
async def test_a_sub_call_checks_the_key_budget_on_llm_routes_and_reserves_nothing(
    is_llm_api_route: bool, checked: bool
) -> None:
    facts: Final = replace(FACTS, sub_call=True, request=replace(FACTS.request, is_llm_api_route=is_llm_api_route))
    host: Final = RecordingCheckHost()
    _ = await _check(host, facts)
    assert (("check_key_max_budget", None) in host.calls) == checked
    assert ("reserve_budget", True) in host.calls


async def test_committed_spend_is_carried_before_reservation_when_tracked() -> None:
    host: Final = RecordingCheckHost(committed_spend=True)
    _ = await _check(host)
    assert host.calls[-4:] == (
        ("committed_spend_tracked", None),
        ("carry_committed_spend", True),
        ("reserve_budget", False),
        ("release_spend_counter_batch", None),
    )


async def test_the_spend_counter_batch_is_released_when_a_check_denies() -> None:
    host: Final = RecordingCheckHost(deny_common_checks=True)
    with pytest.raises(CommonCheckDenied):
        _ = await _check(host)
    assert host.calls[-2:] == (("run_common_checks", False), ("release_spend_counter_batch", None))
