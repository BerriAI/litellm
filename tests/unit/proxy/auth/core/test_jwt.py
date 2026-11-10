from dataclasses import dataclass, replace
from typing import Final

import pytest

from litellm.proxy.auth.core.answers import (
    AutoRegistration,
    BudgetExemption,
    FallbackTarget,
    JwtPolicy,
    KeyMapping,
    MappedKey,
    ModelBudgetScope,
    NoKeyMapping,
    NotRegistered,
    PendingAutoRegister,
    Registered,
    RegistrationFailed,
    UserModelBudgets,
)
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
from litellm.proxy.auth.core.model_budgets import skip_model_budgets, within_model_budgets
from litellm.proxy.auth.core.outcome import ModelBudgetExceeded
from litellm.proxy.auth.core.python_core import PythonCore

MAPPING: Final = JwtSettings(virtual_key_mapping=True, has_db=True, master_key_set=True)
NO_MAPPING: Final = replace(MAPPING, virtual_key_mapping=False)
USER_BUDGETS: Final = UserModelBudgets(limits={"gpt-4o": {"budget_limit": 1.0, "time_period": "1mo"}})
USER_POLICY: Final = JwtPolicy(
    is_proxy_admin=False, user_id="u1", owner_deactivated=False, user_model_budgets=USER_BUDGETS
)
BUDGET_CHECK: Final = ("read_model_budget_scope", "check_user_model_budget")
POLICY_TAIL: Final = ("run_jwt_policy", "read_budget_exemption", "load_project", *BUDGET_CHECK)
CHECKED: Final = JwtUser(budgets=within_model_budgets(None))
SKIPPED: Final = JwtUser(budgets=skip_model_budgets())
EXCEEDED: Final = ModelBudgetExceeded(
    message="over", current_cost=2.0, max_budget=1.0, entity_type="user", entity_id="u1"
)
MAPPED_WITHOUT_EMAIL: Final = MappedKey(key_user_id="u1", key_has_email=False, claim_user_id="u1", claim_has_email=True)


@dataclass
class RecordingJwtHost:
    managed_agent: bool = False
    mapping: KeyMapping = NoKeyMapping()
    policy: JwtPolicy = USER_POLICY
    registration: AutoRegistration = NotRegistered()
    skip_budget_checks: bool = False
    user_budget_exceeded: ModelBudgetExceeded | None = None
    attached: tuple[UserModelBudgets | None, ...] = ()
    calls: tuple[str, ...] = ()

    def _record(self, name: str) -> None:
        self.calls = (*self.calls, name)

    async def decode_claims(self) -> None:
        self._record("decode_claims")

    async def is_managed_agent(self) -> bool:
        self._record("is_managed_agent")
        return self.managed_agent

    async def resolve_mapping(self) -> KeyMapping:
        self._record("resolve_mapping")
        return self.mapping

    async def backfill_mapped_user_email(self) -> None:
        self._record("backfill_mapped_user_email")

    async def run_jwt_policy(self) -> JwtPolicy:
        self._record("run_jwt_policy")
        return self.policy

    async def remember_proxy_admin(self) -> None:
        self._record("remember_proxy_admin")

    async def auto_register(self) -> AutoRegistration:
        self._record("auto_register")
        return self.registration

    def attach_user_model_budgets(self, budgets: UserModelBudgets | None) -> None:
        self._record("attach_user_model_budgets")
        self.attached = (*self.attached, budgets)

    async def read_budget_exemption(self) -> BudgetExemption:
        self._record("read_budget_exemption")
        return BudgetExemption(route_skips_budgets=False, mcp_discovery=False, model_is_free=self.skip_budget_checks)

    async def load_project(self) -> None:
        self._record("load_project")

    def model_budgets(self) -> "RecordingJwtHost":
        return self

    async def read_model_budget_scope(self) -> ModelBudgetScope:
        self._record("read_model_budget_scope")
        return ModelBudgetScope(models=("gpt-4o",), rewritable_model=None, key=None, user_has_model_budgets=True)

    async def check_key_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        self._record("check_key_model_budget")
        return None

    async def authorize_budget_fallback(self, model: str) -> bool:
        self._record("authorize_budget_fallback")
        return False

    async def rewrite_request_model(self, model: str) -> None:
        self._record("rewrite_request_model")

    async def check_user_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        self._record("check_user_model_budget")
        return self.user_budget_exceeded


async def _authenticate(host: RecordingJwtHost, settings: JwtSettings = MAPPING) -> JwtAuthentication:
    return await PythonCore().jwt().authenticate(host, settings)


async def test_without_key_mapping_the_jwt_policy_runs_and_enforces_the_user_model_budget() -> None:
    host: Final = RecordingJwtHost()
    assert await _authenticate(host, NO_MAPPING) == CHECKED
    assert host.calls == POLICY_TAIL


async def test_a_zero_cost_model_skips_the_user_model_budget_on_the_jwt_path() -> None:
    host: Final = RecordingJwtHost(skip_budget_checks=True)
    assert await _authenticate(host, NO_MAPPING) == SKIPPED
    assert host.calls == ("run_jwt_policy", "read_budget_exemption", "load_project")


async def test_a_jwt_user_over_its_model_budget_is_refused() -> None:
    host: Final = RecordingJwtHost(user_budget_exceeded=EXCEEDED)
    assert await _authenticate(host, NO_MAPPING) == EXCEEDED
    assert host.calls == POLICY_TAIL


async def test_a_managed_agent_cannot_use_key_mapping() -> None:
    host: Final = RecordingJwtHost(managed_agent=True, mapping=MAPPED_WITHOUT_EMAIL)
    assert await _authenticate(host) == ManagedAgentNeedsDirectJwt()
    assert host.calls == ("decode_claims", "is_managed_agent")


async def test_a_mapped_key_falls_through_to_the_key_checks_without_the_jwt_policy_or_budget_check() -> None:
    host: Final = RecordingJwtHost(mapping=replace(MAPPED_WITHOUT_EMAIL, claim_has_email=False))
    assert await _authenticate(host) == JwtKey()
    assert host.calls == ("decode_claims", "is_managed_agent", "resolve_mapping")
    assert host.attached == ()


@pytest.mark.parametrize(
    ("mapping", "backfilled"),
    [
        (MAPPED_WITHOUT_EMAIL, True),
        (replace(MAPPED_WITHOUT_EMAIL, key_has_email=True), False),
        (replace(MAPPED_WITHOUT_EMAIL, key_user_id=None, claim_user_id=None), False),
        (replace(MAPPED_WITHOUT_EMAIL, claim_user_id="someone-else"), False),
        (replace(MAPPED_WITHOUT_EMAIL, claim_has_email=False), False),
    ],
)
async def test_a_mapped_key_backfills_its_email_only_from_claims_for_the_same_user(
    mapping: MappedKey, backfilled: bool
) -> None:
    host: Final = RecordingJwtHost(mapping=mapping)
    assert await _authenticate(host) == JwtKey()
    assert ("backfill_mapped_user_email" in host.calls) == backfilled


async def test_an_unmapped_jwt_runs_the_policy_without_registering() -> None:
    host: Final = RecordingJwtHost()
    assert await _authenticate(host) == CHECKED
    assert host.calls == ("decode_claims", "is_managed_agent", "resolve_mapping", *POLICY_TAIL)


async def test_a_deactivated_owner_is_refused_before_admin_or_registration() -> None:
    policy: Final = replace(USER_POLICY, is_proxy_admin=True, owner_deactivated=True)
    host: Final = RecordingJwtHost(mapping=PendingAutoRegister(), policy=policy)
    assert await _authenticate(host) == JwtOwnerDeactivated(user_id="u1")
    assert host.calls[-1] == "run_jwt_policy"


@pytest.mark.parametrize(("mapping", "remembered"), [(PendingAutoRegister(), True), (NoKeyMapping(), False)])
async def test_a_proxy_admin_skips_registration_and_budgets_and_is_remembered_when_pending(
    mapping: KeyMapping, remembered: bool
) -> None:
    host: Final = RecordingJwtHost(mapping=mapping, policy=replace(USER_POLICY, is_proxy_admin=True))
    assert await _authenticate(host) == JwtProxyAdmin()
    assert host.calls[-1] == ("remember_proxy_admin" if remembered else "run_jwt_policy")


@pytest.mark.parametrize(("has_db", "registers"), [(True, True), (False, False)])
async def test_a_pending_mapping_auto_registers_only_with_a_database(has_db: bool, registers: bool) -> None:
    host: Final = RecordingJwtHost(mapping=PendingAutoRegister())
    assert await _authenticate(host, replace(MAPPING, has_db=has_db)) == CHECKED
    assert ("auto_register" in host.calls) == registers


async def test_a_failed_auto_registration_refuses_the_request() -> None:
    host: Final = RecordingJwtHost(mapping=PendingAutoRegister(), registration=RegistrationFailed(reason="taken"))
    assert await _authenticate(host) == JwtKeyMappingFailed(reason="taken")
    assert host.calls[-1] == "auto_register"


@pytest.mark.parametrize(
    ("registration", "master_key_set", "expected"),
    [
        (Registered(links_existing_key=True), True, JwtKey()),
        (Registered(links_existing_key=True), False, CHECKED),
        (Registered(links_existing_key=False), True, CHECKED),
        (NotRegistered(), True, CHECKED),
    ],
)
async def test_only_a_registration_linking_an_existing_key_falls_through_to_the_key_checks(
    registration: AutoRegistration, master_key_set: bool, expected: JwtAuthentication
) -> None:
    host: Final = RecordingJwtHost(mapping=PendingAutoRegister(), registration=registration)
    assert await _authenticate(host, replace(MAPPING, master_key_set=master_key_set)) == expected
    assert ("check_user_model_budget" in host.calls) == isinstance(expected, JwtUser)
    assert host.attached == ((USER_BUDGETS,) if isinstance(registration, Registered) else ())
