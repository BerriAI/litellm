import asyncio
import importlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final, cast  # noqa: TID251  # the fake exposes only the Prisma lookup methods exercised here
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy import proxy_server
from litellm.proxy._types import PassThroughGenericEndpoint, ProxyException, UserAPIKeyAuth
from litellm.proxy.common_utils.customer_usage import (
    CustomerUsageAttribution,
    CustomerUsageDatabase,
    CustomerUsageError,
    CustomerUsageFailure,
    ResolvedCustomerUsage,
    resolve_customer_usage,
)
from tests._master_key import MASTER_KEY


@dataclass(frozen=True, slots=True)
class Table:
    identifier: str
    rows: Mapping[str, Mapping[str, object]]

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, bool] | None = None
    ) -> Mapping[str, object] | None:
        row: Final = self.rows.get(str(where[self.identifier]))
        if row is None:
            return None
        return {
            field: value
            for field, value in row.items()
            if field not in ("budget", "budget_override") or (include is not None and include.get(field))
        }


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
        Table("customer_member_id", {"member-1": {"parent_team_ids": ["team-1"]}}),
    ),
)


def _budget_database(
    *,
    organization: tuple[str, ...] | None = None,
    team: tuple[str, ...] | None = None,
    template: tuple[str, ...] | None = None,
    override: tuple[str, ...] | None = None,
    expires_at: datetime | None = None,
) -> CustomerUsageDatabase:
    return cast(  # cast-ok: the fake implements the existing hierarchy lookup subset
        CustomerUsageDatabase,
        Database(
            Table("customer_org_id", {"org-1": {"budget": {"allowed_models": organization}}}),
            Table("customer_team_id", {"team-1": {"parent_org_id": "org-1", "budget": {"allowed_models": team}}}),
            Table(
                "customer_member_id",
                {
                    "member-1": {
                        "parent_team_ids": ["team-1"],
                        "budget": {"allowed_models": template},
                        "budget_override": {"allowed_models": override},
                        "budget_override_expires_at": expires_at,
                    }
                },
            ),
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
    assert actual == ResolvedCustomerUsage(expected, ())


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
    assert actual == ResolvedCustomerUsage(CustomerUsageAttribution(customer_org_id="org-1"), ())


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
@pytest.mark.parametrize(
    ("organization", "team", "template", "override", "expires_at", "expected"),
    (
        (None, (), None, (), None, ()),
        (("a", "b"), ("b", "c"), ("b",), None, None, (("a", "b"), ("b", "c"), ("b",))),
        (None, None, ("a",), (), None, (("a",),)),
        (None, None, ("a",), ("b",), None, (("b",),)),
        (None, None, ("a",), ("b",), datetime(2000, 1, 1, tzinfo=timezone.utc), (("a",),)),
        (None, None, ("a",), ("b",), datetime(2100, 1, 1, tzinfo=timezone.utc), (("b",),)),
    ),
)
async def test_resolves_model_scopes_with_member_inheritance_and_override_expiry(
    organization: tuple[str, ...] | None,
    team: tuple[str, ...] | None,
    template: tuple[str, ...] | None,
    override: tuple[str, ...] | None,
    expires_at: datetime | None,
    expected: tuple[tuple[str, ...], ...],
) -> None:
    resolved: Final = await resolve_customer_usage(
        metadata={"customer_member_id": "member-1"},
        key_metadata={"customer_org_id": "org-1"},
        administrator=False,
        database=_budget_database(
            organization=organization, team=team, template=template, override=override, expires_at=expires_at
        ),
        ledger_enabled=True,
    )
    assert isinstance(resolved, ResolvedCustomerUsage)
    assert resolved.model_scopes == expected
    assert resolved.attribution.model_dump() == {
        "customer_org_id": "org-1",
        "customer_team_id": "team-1",
        "customer_member_id": "member-1",
    }


def test_customer_model_policy_is_private_and_request_local() -> None:
    from litellm.proxy._types import UserAPIKeyAuth

    original: Final = UserAPIKeyAuth(models=["a", "b"])
    first: Final = original.with_customer_model_scopes((("a",),))
    second: Final = original.with_customer_model_scopes((("b",),))
    assert original.customer_model_scopes == ()
    assert first.customer_model_scopes == (("a",),)
    assert second.customer_model_scopes == (("b",),)
    assert "customer_model_scopes" not in first.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("key-only", "customer-only"))
async def test_customer_model_selection_cannot_grant_models_outside_the_key(model: str) -> None:
    from litellm.proxy._types import ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import can_key_call_resolved_model

    auth: Final = UserAPIKeyAuth(models=["key-only"]).with_customer_model_scopes((("customer-only",),))
    with pytest.raises(ProxyException) as denied:
        await can_key_call_resolved_model(model, None, auth, None)
    assert denied.value.code == "403"


@pytest.mark.asyncio
async def test_empty_customer_policy_inherits_existing_key_permissions() -> None:
    from litellm.proxy._types import ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import can_key_call_resolved_model

    auth: Final = UserAPIKeyAuth(models=["key-only"]).with_customer_model_scopes(())
    await can_key_call_resolved_model("key-only", None, auth, None)
    with pytest.raises(ProxyException) as denied:
        await can_key_call_resolved_model("other-model", None, auth, None)
    assert denied.value.code == "403"


@pytest.mark.asyncio
async def test_customer_models_intersect_each_parent_and_reuse_aliases_wildcards_and_access_groups() -> None:
    from litellm import Router
    from litellm.proxy._types import ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import can_key_call_resolved_model

    router: Final = Router(
        model_list=[
            {
                "model_name": "allowed-model",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "unused-local"},
                "model_info": {"access_groups": ["model-group"]},
            }
        ]
    )
    auth: Final = UserAPIKeyAuth(
        models=["model-group"], aliases={"friendly": "allowed-model"}
    ).with_customer_model_scopes((("allowed-*",), ("model-group",), ("allowed-model",)))
    await can_key_call_resolved_model("friendly", None, auth, router)
    denied: Final = auth.with_customer_model_scopes((("allowed-*",), ("other-model",), ("allowed-model",)))
    with pytest.raises(ProxyException) as error:
        await can_key_call_resolved_model("friendly", None, denied, router)
    assert error.value.code == "403"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/v1/chat/completions", "/v1/responses"))
@pytest.mark.parametrize(
    ("level", "requested_model", "rewrite_model"),
    (
        ("organization", "blocked-model", False),
        ("team", "blocked-model", False),
        ("template", "blocked-model", False),
        ("organization", "allowed-model", True),
    ),
)
async def test_registered_inference_routes_reject_hierarchy_model_denials_before_provider_calls(
    monkeypatch: pytest.MonkeyPatch, path: str, level: str, requested_model: str, rewrite_model: bool
) -> None:
    from litellm import Router
    from litellm.proxy import proxy_server as server
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    organization: Final = ("allowed-model",) if level == "organization" else None
    team: Final = ("allowed-model",) if level == "team" else None
    template: Final = ("allowed-model",) if level == "template" else None
    database: Final = _budget_database(organization=organization, team=team, template=template)
    auth: Final = UserAPIKeyAuth(
        api_key="controlled-hash",
        user_role=LitellmUserRoles.INTERNAL_USER,
        metadata={"customer_org_id": "org-1"},
        models=["allowed-model", "blocked-model"],
    )
    completion: Final = AsyncMock(side_effect=AssertionError("provider must not run"))
    responses: Final = AsyncMock(side_effect=AssertionError("provider must not run"))
    monkeypatch.setattr(server, "prisma_client", SimpleNamespace(db=database))
    monkeypatch.setattr(server, "general_settings", {"use_spend_logs_ledger": True})
    router: Final = (
        Router(
            model_list=[
                {
                    "model_name": name,
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "unused-local"},
                }
                for name in ("allowed-model", "blocked-model")
            ]
        )
        if rewrite_model
        else None
    )
    monkeypatch.setattr(server, "llm_router", router)
    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr("litellm.aresponses", responses)
    monkeypatch.setitem(server.app.dependency_overrides, user_api_key_auth, lambda: auth)
    if rewrite_model:

        async def rewrite(**kwargs: object) -> Mapping[str, object]:
            return {
                **cast(Mapping[str, object], kwargs["data"]),
                "model": "blocked-model",
            }  # cast-ok: hook data is the request mapping

        monkeypatch.setattr(server.proxy_logging_obj, "pre_call_hook", rewrite)
    metadata_field: Final = "litellm_metadata" if path.endswith("responses") else "metadata"
    body: Final = {"model": requested_model, metadata_field: {"customer_member_id": "member-1"}}
    payload: Final = (
        {**body, "input": "controlled"}
        if path.endswith("responses")
        else {**body, "messages": [{"role": "user", "content": "controlled"}]}
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False), base_url="http://controlled-proxy"
    ) as client:
        response: Final = await client.post(path, json=payload)
    assert response.status_code == 403
    assert response.json()["error"]["param"] == "model"
    assert auth.customer_model_scopes == ()
    completion.assert_not_awaited()
    responses.assert_not_awaited()


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


@dataclass(frozen=True, slots=True)
class CustomerPassthrough:
    client: TestClient
    token: UserAPIKeyAuth
    headers: Mapping[str, str]


@pytest.fixture
def customer_passthrough(monkeypatch: pytest.MonkeyPatch, httpx_transport: None) -> Iterator[CustomerPassthrough]:
    from litellm.proxy.pass_through_endpoints import pass_through_endpoints
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import router
    from litellm.proxy.pass_through_endpoints.openai_passthrough_endpoints import router as dedicated_router

    auth_module: Final = importlib.import_module("litellm.proxy.auth.user_api_key_auth")
    token: Final = UserAPIKeyAuth(
        api_key="hashed-customer-test-key",
        models=["gpt-4.1-mini", "gpt-5.5"],
        metadata={"customer_org_id": "org-1", "customer_member_id": "member-1"},
    )

    async def authenticate(request: Request, **kwargs: object) -> UserAPIKeyAuth:
        request.state.requires_key_model_check = True
        return token

    monkeypatch.setattr(auth_module, "user_api_key_auth_builder", authenticate)
    monkeypatch.setattr(auth_module, "run_centralized_common_checks", AsyncMock())
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(
        proxy_server, "prisma_client", SimpleNamespace(db=_budget_database(organization=("gpt-4.1-mini",)))
    )
    monkeypatch.setenv("OPENAI_API_BASE", "https://customer-provider.test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-key")
    app: Final = FastAPI(exception_handlers={ProxyException: proxy_server.openai_exception_handler})
    app.include_router(router)
    app.include_router(dedicated_router)
    monkeypatch.setattr(
        pass_through_endpoints,
        "_registered_pass_through_routes",
        dict(pass_through_endpoints._registered_pass_through_routes),
    )
    asyncio.run(
        pass_through_endpoints._register_pass_through_endpoint(
            PassThroughGenericEndpoint(
                path="/customer-configured", target="https://customer-provider.test/external", methods=["POST"]
            ),
            app=app,
            premium_user=False,
            visited_endpoints=set(),
        )
    )
    with TestClient(app) as client:
        yield CustomerPassthrough(client, token, {"Authorization": f"Bearer {MASTER_KEY}"})


@pytest.mark.parametrize(
    "database",
    [
        pytest.param(_budget_database(organization=("gpt-4.1-mini",)), id="organization"),
        pytest.param(_budget_database(team=("gpt-4.1-mini",)), id="team"),
        pytest.param(_budget_database(template=("gpt-4.1-mini",)), id="member-template"),
        pytest.param(_budget_database(override=("gpt-4.1-mini",)), id="member-override"),
    ],
)
@pytest.mark.parametrize("prefix", ["openai", "openai_passthrough"])
def test_passthrough_rejects_customer_model_without_contacting_provider(
    customer_passthrough: CustomerPassthrough,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    database: CustomerUsageDatabase,
    prefix: str,
) -> None:
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=database))
    response: Final = customer_passthrough.client.post(
        f"/{prefix}/v1/chat/completions",
        headers=customer_passthrough.headers,
        json={"model": "gpt-5.5", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 403, response.text
    assert response.json()["error"]["type"] == "customer_model_access_denied"
    assert not respx_mock.calls
    assert customer_passthrough.token.customer_model_scopes == ()


def test_passthrough_allows_customer_model_and_preserves_provider_body(
    customer_passthrough: CustomerPassthrough, respx_mock: respx.MockRouter
) -> None:
    body: Final = {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hello"}]}
    upstream: Final = respx_mock.post("https://customer-provider.test/v1/chat/completions").respond(
        json={"id": "chat-test"}
    )
    response: Final = customer_passthrough.client.post(
        "/openai/v1/chat/completions", headers=customer_passthrough.headers, json=body
    )
    assert response.status_code == 200, response.text
    assert upstream.call_count == 1
    assert json.loads(upstream.calls.last.request.content) == body
    assert customer_passthrough.token.customer_model_scopes == ()


def test_passthrough_customer_models_do_not_grant_key_access(
    customer_passthrough: CustomerPassthrough, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    customer_passthrough.token.models = ["gpt-4.1-mini"]
    monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace(db=_budget_database(organization=("gpt-5.5",))))
    response: Final = customer_passthrough.client.post(
        "/openai/v1/chat/completions", headers=customer_passthrough.headers, json={"model": "gpt-5.5", "messages": []}
    )
    assert response.status_code == 403, response.text
    assert response.json()["error"]["type"] == "key_model_access_denied"
    assert not respx_mock.calls


def test_passthrough_checks_customer_model_after_hook_rewrite(
    customer_passthrough: CustomerPassthrough, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    async def rewrite(*, data: Mapping[str, object], **kwargs: object) -> dict[str, object]:
        return {**data, "model": "gpt-5.5"}

    monkeypatch.setattr(proxy_server.proxy_logging_obj, "pre_call_hook", rewrite)
    response: Final = customer_passthrough.client.post(
        "/openai/v1/chat/completions",
        headers=customer_passthrough.headers,
        json={"model": "gpt-4.1-mini", "messages": []},
    )
    assert response.status_code == 403, response.text
    assert not respx_mock.calls


def test_passthrough_allows_model_less_retrieval(
    customer_passthrough: CustomerPassthrough, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.get("https://customer-provider.test/v1/files").respond(json={"data": []})
    response: Final = customer_passthrough.client.get("/openai/v1/files", headers=customer_passthrough.headers)
    assert response.status_code == 200, response.text
    assert upstream.call_count == 1


def test_configured_passthrough_preserves_upstream_model_semantics(
    customer_passthrough: CustomerPassthrough, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.post("https://customer-provider.test/external").respond(json={"id": "external-test"})
    response: Final = customer_passthrough.client.post(
        "/customer-configured",
        headers=customer_passthrough.headers,
        json={"model": "external-model"},
    )
    assert response.status_code == 200, response.text
    assert upstream.call_count == 1
