from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final, cast  # noqa: TID251  # the fake exposes only the Prisma lookup methods exercised here

import pytest

from litellm.proxy.common_utils.customer_usage import (
    CustomerUsageAttribution,
    CustomerUsageDatabase,
    CustomerUsageError,
    CustomerUsageFailure,
    resolve_customer_usage,
)


@dataclass(frozen=True, slots=True)
class Table:
    identifier: str
    rows: Mapping[str, Mapping[str, object]]

    async def find_unique(self, where: Mapping[str, object]) -> Mapping[str, object] | None:
        return self.rows.get(str(where[self.identifier]))


@dataclass(frozen=True, slots=True)
class Database:
    litellm_customerorganizationtable: Table
    litellm_customerteamtable: Table
    litellm_customermembertable: Table


DATABASE: Final = cast(  # cast-ok: the external Prisma fake implements the lookup subset used by attribution
    CustomerUsageDatabase,
    Database(
        Table("customer_org_id", {"org-1": {}, "org-2": {}}),
        Table("customer_team_id", {"team-1": {"parent_org_id": "org-1"}, "team-2": {"parent_org_id": "org-2"}}),
        Table("customer_member_id", {"member-1": {"parent_team_id": "team-1"}}),
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        (
            {"customer_member_id": "member-1"},
            CustomerUsageAttribution(customer_org_id="org-1", customer_team_id="team-1", customer_member_id="member-1"),
        ),
        ({"customer_team_id": "team-1"}, CustomerUsageAttribution(customer_org_id="org-1", customer_team_id="team-1")),
        ({"customer_org_id": "org-1"}, CustomerUsageAttribution(customer_org_id="org-1")),
    ),
)
async def test_resolves_ancestors_without_requiring_redundant_request_ids(
    metadata: Mapping[str, object], expected: CustomerUsageAttribution
) -> None:
    actual: Final = await resolve_customer_usage(
        metadata=metadata, key_metadata={}, administrator=True, database=DATABASE, ledger_enabled=True
    )
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "key_metadata", "administrator", "expected"),
    (
        ({"customer_member_id": "member-1", "customer_team_id": "team-2"}, {}, True, CustomerUsageError.MISMATCH),
        ({"customer_member_id": "member-1"}, {"customer_org_id": "org-2"}, False, CustomerUsageError.FORBIDDEN),
        ({"customer_team_id": "team-2", "customer_org_id": "org-1"}, {}, True, CustomerUsageError.MISMATCH),
        ({"customer_org_id": "org-2"}, {"customer_org_id": "org-1"}, False, CustomerUsageError.FORBIDDEN),
        ({"customer_org_id": "org-1"}, {}, False, CustomerUsageError.UNBOUND),
        ({"customer_member_id": "missing"}, {}, True, CustomerUsageError.NOT_FOUND),
        ({"customer_team_id": "missing"}, {}, True, CustomerUsageError.NOT_FOUND),
        ({"customer_org_id": "missing"}, {}, True, CustomerUsageError.NOT_FOUND),
        ({"customer_org_id": 123}, {}, True, CustomerUsageError.INVALID),
    ),
)
async def test_rejects_cross_organization_attribution_and_invalid_identifiers(
    metadata: Mapping[str, object],
    key_metadata: Mapping[str, object],
    administrator: bool,
    expected: CustomerUsageError,
) -> None:
    actual: Final = await resolve_customer_usage(
        metadata=metadata,
        key_metadata=key_metadata,
        administrator=administrator,
        database=DATABASE,
        ledger_enabled=True,
    )
    assert actual == CustomerUsageFailure(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "key_metadata"),
    (
        ({"customer_member_id": "missing"}, {"customer_org_id": "org-1"}),
        ({"customer_member_id": "member-1"}, {"customer_org_id": "org-2"}),
        (
            {"customer_member_id": "member-1"},
            {"customer_org_id": "org-2", "customer_team_id": "team-2"},
        ),
        ({"customer_team_id": "missing"}, {"customer_org_id": "org-1"}),
        ({"customer_team_id": "team-2"}, {"customer_org_id": "org-1"}),
        ({}, {"customer_org_id": "missing"}),
    ),
)
async def test_non_admin_cannot_distinguish_missing_and_other_organization_identifiers(
    metadata: Mapping[str, object], key_metadata: Mapping[str, object]
) -> None:
    actual: Final = await resolve_customer_usage(
        metadata=metadata,
        key_metadata=key_metadata,
        administrator=False,
        database=DATABASE,
        ledger_enabled=True,
    )
    assert actual == CustomerUsageFailure(CustomerUsageError.FORBIDDEN)


@pytest.mark.asyncio
@pytest.mark.parametrize("administrator", (False, True))
async def test_invalid_identifier_types_keep_validation_errors(administrator: bool) -> None:
    actual: Final = await resolve_customer_usage(
        metadata={"customer_member_id": 123},
        key_metadata={"customer_org_id": "org-1"},
        administrator=administrator,
        database=DATABASE,
        ledger_enabled=True,
    )
    assert actual == CustomerUsageFailure(CustomerUsageError.INVALID)


@pytest.mark.asyncio
async def test_key_binding_tracks_requests_with_no_repeated_customer_metadata() -> None:
    actual: Final = await resolve_customer_usage(
        metadata={},
        key_metadata={"customer_org_id": "org-1"},
        administrator=False,
        database=DATABASE,
        ledger_enabled=True,
    )
    assert actual == CustomerUsageAttribution(customer_org_id="org-1")


@pytest.mark.asyncio
async def test_requires_accounting_before_accepting_a_tracked_request() -> None:
    actual: Final = await resolve_customer_usage(
        metadata={"customer_org_id": "org-1"},
        key_metadata={},
        administrator=True,
        database=DATABASE,
        ledger_enabled=False,
    )
    assert actual == CustomerUsageFailure(CustomerUsageError.UNAVAILABLE)


@pytest.mark.asyncio
async def test_untracked_requests_do_not_require_a_database_or_ledger() -> None:
    actual: Final = await resolve_customer_usage(
        metadata={}, key_metadata={}, administrator=False, database=None, ledger_enabled=False
    )
    assert actual is None


@pytest.mark.asyncio
async def test_untracked_request_enrichment_does_not_touch_customer_database_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import Request

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import ProxyConfig

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=object()))
    request: Final = Request(
        {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": (), "query_string": b""}
    )
    enriched: Final = await add_litellm_data_to_request(
        data={"model": "usage-model", "metadata": {"application": "support"}},
        request=request,
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
        proxy_config=ProxyConfig(),
        general_settings={},
    )
    assert enriched["metadata"]["application"] == "support"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
async def test_request_enrichment_replaces_forged_stamps_and_preserves_valid_attribution(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    from fastapi import Request
    from pydantic import TypeAdapter

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.customer_usage import CUSTOMER_USAGE_METADATA_KEY
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request, get_metadata_variable_name
    from litellm.proxy.proxy_server import ProxyConfig

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=DATABASE))
    request: Final = Request({"type": "http", "path": path, "method": "POST", "headers": (), "query_string": b""})
    metadata_name: Final = get_metadata_variable_name(request)
    forged: Final = CustomerUsageAttribution(customer_org_id="org-2").model_dump_json()
    metadata: Final = {"customer_member_id": "member-1", CUSTOMER_USAGE_METADATA_KEY: forged}
    enriched: Final = await add_litellm_data_to_request(
        data={
            "model": "usage-model",
            "metadata": {CUSTOMER_USAGE_METADATA_KEY: forged},
            "litellm_metadata": {CUSTOMER_USAGE_METADATA_KEY: forged},
            metadata_name: metadata,
        },
        request=request,
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key", metadata={"customer_org_id": "org-1"}),
        proxy_config=ProxyConfig(),
        general_settings={},
    )
    stamp: Final = TypeAdapter(str).validate_python(enriched[metadata_name][CUSTOMER_USAGE_METADATA_KEY])
    assert CustomerUsageAttribution.model_validate_json(stamp) == CustomerUsageAttribution(
        customer_org_id="org-1", customer_team_id="team-1", customer_member_id="member-1"
    )
