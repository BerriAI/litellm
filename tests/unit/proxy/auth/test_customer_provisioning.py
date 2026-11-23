from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # in-memory adapters implement the external Prisma edge

import pytest
from pydantic import ValidationError

import litellm
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import CustomerBudgetExceededError
from litellm.proxy.auth.customer_provisioning import (
    gated_registrars,
    refuse_unknown_customer,
    registration_refusal,
    resolve_registered_customer,
    unknown_customer_header,
)
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerRequestIdentity
from litellm.types.proxy.management_endpoints.apps import (
    AppCreate,
    AppDefinition,
    AppFailure,
    AppPatch,
    AppResourceType,
)
from tests.unit.proxy.management_helpers.test_apps import AppDatabase, seed_resources
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW

if TYPE_CHECKING:
    from prisma import Prisma

    from litellm.proxy.utils import PrismaClient

KNOWN: Final = CustomerHierarchyAuth(
    customer_org_id="org", customer_org_alias="org", customer_member_id="member", scopes=()
)
ORG_ONLY: Final = CustomerHierarchyAuth(customer_org_id="org", customer_org_alias="org", scopes=())


async def _refusal(
    app_may_create: bool | None, without_app_may_create: bool, authenticated: UserAPIKeyAuth
) -> str | None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await seed_resources(database)
    await store.put_definition(
        AppDefinition(resource_type=AppResourceType.TEAM, auto_create_customers_without_app=without_app_may_create)
    )
    app: Final = await store.create(AppCreate(resource_id="chat", auto_create_customers=app_may_create), "admin")
    assert not isinstance(app, AppFailure)
    connection: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    return await registration_refusal(connection, authenticated, None, "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("app_may_create", "without_app_may_create", "team_id", "refused"),
    (
        (None, True, "chat", False),
        (True, False, "chat", False),
        (False, True, "chat", True),
        (False, True, "search", False),
        (True, False, "search", True),
        (True, False, None, True),
        (False, True, None, False),
    ),
)
async def test_a_request_may_register_customers_only_when_its_app_or_the_no_app_setting_allows(
    app_may_create: bool | None, without_app_may_create: bool, team_id: str | None, refused: bool
) -> None:
    reason: Final = await _refusal(app_may_create, without_app_may_create, UserAPIKeyAuth(team_id=team_id))

    assert (reason is not None) is refused, reason


@pytest.mark.asyncio
async def test_a_request_may_register_customers_when_no_app_definition_was_ever_chosen() -> None:
    connection: Final = cast("PrismaClient", SimpleNamespace(db=AppDatabase()))  # cast-ok: in-memory Prisma edge

    assert await registration_refusal(connection, UserAPIKeyAuth(team_id="chat"), None, "") is None


@pytest.mark.asyncio
async def test_turning_registration_off_for_an_app_is_stored_and_survives_an_unrelated_patch() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    app: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert not isinstance(app, AppFailure)
    assert app.auto_create_customers is True

    off: Final = await store.patch(app.app_id, AppPatch(auto_create_customers=False), "admin")
    unrelated: Final = await store.patch(app.app_id, AppPatch(default_tier_budget_id=None), "admin")

    assert not isinstance(off, AppFailure) and off.auto_create_customers is False
    assert not isinstance(unrelated, AppFailure) and unrelated.auto_create_customers is False


@pytest.mark.asyncio
async def test_the_no_app_setting_survives_a_definition_write_that_omits_it_and_an_app_being_configured() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await seed_resources(database)
    await store.put_definition(
        AppDefinition(resource_type=AppResourceType.TEAM, auto_create_customers_without_app=False)
    )

    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    await store.create(AppCreate(resource_id="chat"), "admin")

    assert await store.definition() == AppDefinition(
        resource_type=AppResourceType.TEAM, auto_create_customers_without_app=False
    )
    changed: Final = await store.put_definition(
        AppDefinition(resource_type=AppResourceType.TEAM, auto_create_customers_without_app=True)
    )
    assert changed == AppDefinition(resource_type=AppResourceType.TEAM, auto_create_customers_without_app=True)
    assert await store.put_definition(AppDefinition(resource_type=AppResourceType.PROJECT)) == AppFailure(
        409, "The app resource definition cannot change after a resource has been configured."
    )


@pytest.mark.parametrize(
    ("identity", "hierarchy", "header"),
    (
        (CustomerRequestIdentity(customer_org_alias="org"), None, "x-litellm-customer-org-alias"),
        (CustomerRequestIdentity(customer_org_alias="org"), ORG_ONLY, None),
        (CustomerRequestIdentity(customer_member_id="member"), None, "x-litellm-customer-member-id"),
        (
            CustomerRequestIdentity(customer_org_alias="org", customer_member_id="member"),
            ORG_ONLY,
            "x-litellm-customer-member-id",
        ),
        (CustomerRequestIdentity(customer_org_alias="org", customer_member_id="member"), KNOWN, None),
        (
            CustomerRequestIdentity(customer_org_alias="another-org", customer_member_id="member"),
            KNOWN,
            "x-litellm-customer-org-alias",
        ),
    ),
)
def test_the_unknown_customer_header_is_the_one_the_resolved_hierarchy_lacks(
    identity: CustomerRequestIdentity, hierarchy: CustomerHierarchyAuth | None, header: str | None
) -> None:
    assert unknown_customer_header(identity, hierarchy) == header


async def _refused() -> str | None:
    return "App chat may not create customers."


async def _allowed() -> str | None:
    return None


@pytest.mark.asyncio
async def test_an_unknown_customer_is_refused_with_403_naming_its_header_when_registration_is_off() -> None:
    with pytest.raises(ProxyException) as refused:
        await refuse_unknown_customer(CustomerRequestIdentity(customer_org_alias="new-tenant"), None, _refused)

    assert (refused.value.code, refused.value.param) == ("403", "x-litellm-customer-org-alias")
    assert refused.value.message == (
        "Unknown customer in x-litellm-customer-org-alias. App chat may not create customers."
    )


@pytest.mark.asyncio
async def test_an_unknown_member_of_a_known_organization_is_refused_from_the_organization_only_hierarchy() -> None:
    identity: Final = CustomerRequestIdentity(customer_org_alias="org", customer_member_id="new-member")

    with pytest.raises(ProxyException) as refused:
        await refuse_unknown_customer(identity, ORG_ONLY, _refused)

    assert (refused.value.code, refused.value.param) == ("403", "x-litellm-customer-member-id")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("identity", "hierarchy", "refusal"),
    (
        (CustomerRequestIdentity(customer_org_alias="org", customer_member_id="member"), KNOWN, _refused),
        (CustomerRequestIdentity(customer_org_alias="new-tenant"), None, _allowed),
    ),
)
async def test_a_known_customer_or_an_allowed_registration_is_not_refused(
    identity: CustomerRequestIdentity, hierarchy: CustomerHierarchyAuth | None, refusal: object
) -> None:
    assert await refuse_unknown_customer(identity, hierarchy, refusal) is None  # pyright: ignore[reportArgumentType]  # parametrized callables


@pytest.mark.asyncio
async def test_registrars_create_nothing_when_registration_is_off() -> None:
    member, organization = gated_registrars(_refused)
    no_database: Final = cast("PrismaClient", None)  # cast-ok: a refused registrar must never reach the database

    assert await organization("new-tenant", no_database) == Failure(ErrorCode.NOT_FOUND)
    assert await member("org", "new-user", no_database) == Failure(ErrorCode.NOT_FOUND)


@pytest.mark.asyncio
async def test_an_apps_surcharge_is_stored_survives_an_unrelated_patch_and_clears_with_null() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    app: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert not isinstance(app, AppFailure)
    assert app.customer_margin is None

    set_to_five: Final = await store.patch(app.app_id, AppPatch(customer_margin=0.05), "admin")
    unrelated: Final = await store.patch(app.app_id, AppPatch(auto_create_customers=False), "admin")
    cleared: Final = await store.patch(app.app_id, AppPatch(customer_margin=None), "admin")

    assert not isinstance(set_to_five, AppFailure) and set_to_five.customer_margin == 0.05
    assert not isinstance(unrelated, AppFailure) and unrelated.customer_margin == 0.05
    assert not isinstance(cleared, AppFailure) and cleared.customer_margin is None


@pytest.mark.parametrize("value", [-0.01, 10.01])
def test_an_apps_surcharge_outside_zero_to_ten_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        AppPatch(customer_margin=value)


@pytest.mark.asyncio
async def test_an_over_budget_organization_does_not_hide_that_its_new_member_may_not_register() -> None:
    identity: Final = CustomerRequestIdentity(customer_org_alias="org", customer_member_id="new-member")

    async def over_budget(register: object, register_organization: object) -> CustomerHierarchyAuth | None:
        raise CustomerBudgetExceededError(ORG_ONLY, litellm.BudgetExceededError(current_cost=2, max_budget=1))

    with pytest.raises(ProxyException) as refused:
        await resolve_registered_customer(identity, _refused, over_budget)
    assert (refused.value.code, refused.value.param) == ("403", "x-litellm-customer-member-id")

    with pytest.raises(CustomerBudgetExceededError):
        await resolve_registered_customer(identity, _allowed, over_budget)


@pytest.mark.asyncio
async def test_a_known_customer_resolves_to_its_hierarchy_when_registration_is_off() -> None:
    identity: Final = CustomerRequestIdentity(customer_org_alias="org", customer_member_id="member")

    async def known(register: object, register_organization: object) -> CustomerHierarchyAuth | None:
        return KNOWN

    assert await resolve_registered_customer(identity, _refused, known) is KNOWN
