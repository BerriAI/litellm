from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import pytest

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.app_auth import app_alias_from_request, app_scope, resolve_app_hierarchy
from litellm.proxy.auth.customer_hierarchy_auth import check_customer_hierarchy_budget
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth
from litellm.types.proxy.management_endpoints.apps import AppCreate, AppFailure
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget
from tests.unit.proxy.management_helpers.test_apps import AppDatabase
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW

if TYPE_CHECKING:
    from prisma import Prisma

    from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
@pytest.mark.parametrize("with_customer", (False, True))
async def test_resolved_app_adds_only_scopes_with_resolved_customer_parents(with_customer: bool) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    app: Final = await store.create(AppCreate(alias="assistant", budget=Budget(max_budget=10)), "admin")
    assert not isinstance(app, AppFailure)
    hierarchy: Final = (
        CustomerHierarchyAuth(
            customer_org_id="org",
            customer_member_id="member",
            scopes=(CustomerScopeAuth(key="customer_org", entity_id="org", spend=1, max_budget=100),),
        )
        if with_customer
        else None
    )
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    result: Final = await resolve_app_hierarchy(
        hierarchy,
        {},
        {"app_alias": "assistant"},
        connection,
        UserApiKeyCache(),
        None,
        "request-model",
        "/v1/chat/completions",
    )
    assert result is not None and result.app_id == app.app_id
    assert tuple(scope.key for scope in result.scopes) == (
        ("customer_org", "app", "app_customer_org", "app_customer_member") if with_customer else ("app",)
    )
    assert result.customer_org_id == ("org" if with_customer else None)


@pytest.mark.asyncio
async def test_unknown_app_is_400_and_is_not_registered() -> None:
    database: Final = AppDatabase()
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    with pytest.raises(ProxyException) as rejected:
        await resolve_app_hierarchy(
            None,
            {"x-litellm-app": "unknown"},
            {},
            connection,
            UserApiKeyCache(),
            None,
            None,
            "/v1/chat/completions",
        )
    assert int(rejected.value.code) == 400
    assert not database.litellm_apptable.rows


@pytest.mark.parametrize(
    ("headers", "bound", "expected"),
    (
        ({}, {}, None),
        ({"x-litellm-app": "assistant"}, {}, "assistant"),
        ({}, {"app_alias": "assistant"}, "assistant"),
        ({"x-litellm-app": "assistant"}, {"app_alias": "assistant"}, "assistant"),
        (
            {"x-litellm-app": "other"},
            {"app_alias": "assistant"},
            AppFailure(403, "This virtual key is bound to a different app."),
        ),
    ),
)
def test_trusted_key_app_binding(
    headers: dict[str, str], bound: dict[str, str], expected: str | AppFailure | None
) -> None:
    assert app_alias_from_request(headers, bound) == expected


def test_app_org_denies_a_request_that_global_org_admits() -> None:
    global_scope: Final = CustomerScopeAuth(key="customer_org", entity_id="org", spend=4, max_budget=100)
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=(global_scope,))
    check_customer_hierarchy_budget(hierarchy)
    combined: Final = hierarchy.model_copy(
        update={
            "scopes": (
                global_scope,
                app_scope("app_customer_org", "app:org", 4, Budget(max_budget=3)),
            )
        }
    )
    with pytest.raises(litellm.BudgetExceededError) as rejected:
        check_customer_hierarchy_budget(combined)
    assert rejected.value.entity_type == "app_customer_org"
    assert hierarchy.scopes[0] == global_scope


@pytest.mark.parametrize("duration", ("24h", "1mo", "12mo"))
def test_app_duration_cap_uses_a_separate_current_window_not_lifetime_spend(duration: str) -> None:
    scope: Final = app_scope("app", "application", 1000, Budget(max_budget=10, budget_duration=duration))
    assert scope.max_budget is None
    counters: Final = customer_window_counters(scope.key, scope.entity_id, scope.budget_limits, now=NOW)
    assert len(counters) == 1 and counters[0].max_budget == 10
    assert counters[0].window_duration == duration
    assert counters[0].counter_key.startswith("spend:app:application:window:")
    assert counters[0].reset_at is not None
    next_period: Final = customer_window_counters(
        scope.key,
        scope.entity_id,
        scope.budget_limits,
        now=counters[0].reset_at,
    )
    assert next_period[0].counter_key != counters[0].counter_key


@pytest.mark.asyncio
async def test_bound_key_mismatch_is_403_without_a_database() -> None:
    with pytest.raises(ProxyException) as rejected:
        await resolve_app_hierarchy(
            None,
            {"x-litellm-app": "other"},
            {"app_alias": "assistant"},
            None,
            UserApiKeyCache(),
            None,
            None,
            "/v1/chat/completions",
        )
    assert int(rejected.value.code) == 403


@pytest.mark.asyncio
async def test_no_app_returns_the_same_hierarchy_without_a_database() -> None:
    hierarchy: Final = CustomerHierarchyAuth(customer_org_id="org", scopes=())
    before: Final = hierarchy.model_dump_json()
    result: Final = await resolve_app_hierarchy(
        hierarchy,
        {},
        {},
        None,
        UserApiKeyCache(),
        None,
        None,
        "/v1/chat/completions",
    )
    assert result is hierarchy and result.model_dump_json() == before
    assert "app_id" not in result.model_dump()


def test_spoofed_app_metadata_is_removed_when_auth_has_no_app() -> None:
    result: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        {"metadata": {"user_api_key_app_id": "spoof", "user_api_key_app_alias": "spoof"}},
        UserAPIKeyAuth(),
        "metadata",
    )
    assert "user_api_key_app_id" not in result["metadata"]
    assert "user_api_key_app_alias" not in result["metadata"]


def test_app_only_scopes_stamp_trusted_identity_without_a_customer_org() -> None:
    auth: Final = UserAPIKeyAuth(
        customer_hierarchy=CustomerHierarchyAuth(
            app_id="app",
            app_alias="assistant",
            scopes=(app_scope("app", "app", 0, Budget(max_budget=10)),),
        )
    )
    result: Final = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        {"metadata": {"user_api_key_app_id": "spoof"}},
        auth,
        "metadata",
    )
    assert result["metadata"]["user_api_key_app_id"] == "app"
    assert result["metadata"]["user_api_key_app_alias"] == "assistant"
    assert result["metadata"]["user_api_key_customer_org_id"] is None
