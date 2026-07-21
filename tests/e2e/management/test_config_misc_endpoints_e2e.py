"""Live e2e: the config and miscellaneous Management/UI routes.

One method per registry cell, each asserting the real contract against a live
proxy: read-only inventory routes return their documented shape, stateless
validators compute their verdict from the request, and the write routes persist
so a read-back reflects the change. The two routes that mutate global proxy state
(cache settings and router settings, both driven from the admin UI) are exercised
with a benign, self-restoring change so a shared proxy is left as it was found.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, Success, unwrap, unwrap_status
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import KeyGenerateBody, LiteLLMParamsBody, TeamNewBody

pytestmark = pytest.mark.e2e


def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.proxy.poll_interval)
    pytest.fail(failure)


# ---- callbacks -------------------------------------------------------------


class CallbacksListResponse(BaseModel):
    success: list[str]
    failure: list[str]
    success_and_failure: list[str]


# ---- cost estimate ---------------------------------------------------------


class CostEstimateBody(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    num_requests_per_day: int | None = None


class CostEstimateResponse(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost_per_request: float
    input_cost_per_request: float
    output_cost_per_request: float
    margin_cost_per_request: float
    daily_cost: float | None = None
    provider: str | None = None


# ---- credential migration check --------------------------------------------


class MigrationReport(BaseModel):
    residual_legacy: int
    total_undecryptable: int


class MigrationCheckResponse(BaseModel):
    status: str
    report: MigrationReport


# ---- tool + workflow inventories -------------------------------------------


class ToolListEntry(BaseModel):
    name: str | None = None


class ToolListResponse(BaseModel):
    tools: list[ToolListEntry]
    total: int


class WorkflowRunEntry(BaseModel):
    workflow_id: str | None = None


class WorkflowRunsResponse(BaseModel):
    runs: list[WorkflowRunEntry]
    count: int


# ---- compliance ------------------------------------------------------------


class ComplianceGdprBody(BaseModel):
    request_id: str
    user_id: str
    model: str
    timestamp: str


class ComplianceCheck(BaseModel):
    check_name: str
    article: str
    passed: bool
    detail: str


class ComplianceResponse(BaseModel):
    compliant: bool
    regulation: str
    checks: list[ComplianceCheck]


# ---- cache settings --------------------------------------------------------


class CacheSettingsValue(BaseModel):
    type: str
    host: str = ""
    port: str = ""


class CacheSettingsUpdateBody(BaseModel):
    cache_settings: CacheSettingsValue


class CacheCurrentValues(BaseModel):
    type: str | None = None
    host: str | None = None
    port: str | None = None


class CacheGetResponse(BaseModel):
    current_values: CacheCurrentValues


class CacheUpdateResponse(BaseModel):
    status: str
    settings: CacheSettingsValue


# ---- fallback management ---------------------------------------------------


class FallbackShape(BaseModel):
    model: str
    fallback_models: list[str]
    fallback_type: str


class FallbackCreateBody(FallbackShape):
    pass


class FallbackResponse(FallbackShape):
    message: str


class FallbackGetParams(BaseModel):
    fallback_type: str


class FallbackGetResponse(FallbackShape):
    pass


# ---- jwt key mapping -------------------------------------------------------


class JwtKeyMappingNewBody(BaseModel):
    jwt_claim_name: str
    jwt_claim_value: str
    key: str
    description: str


class JwtInfoParams(BaseModel):
    id: str


class JwtDeleteBody(BaseModel):
    id: str


class JwtKeyMappingResponse(BaseModel):
    id: str
    jwt_claim_name: str
    jwt_claim_value: str
    is_active: bool
    description: str | None = None


# ---- router settings via /config/update ------------------------------------


class RouterSettingsPatch(BaseModel):
    num_retries: int


class ConfigUpdateBody(BaseModel):
    router_settings: RouterSettingsPatch


class ConfigUpdateResponse(BaseModel):
    message: str


class RouterCurrentValues(BaseModel):
    num_retries: int | None = None


class RouterSettingsResponse(BaseModel):
    current_values: RouterCurrentValues


# ---- mcp server submission -------------------------------------------------


class McpRegisterBody(BaseModel):
    server_name: str
    url: str
    transport: str
    description: str


class McpServerResponse(BaseModel):
    server_id: str
    server_name: str | None = None
    approval_status: str
    transport: str
    url: str | None = None


class TestInventoryRoutes:
    @pytest.mark.covers("mgmt.callback.list.happy_path")
    def test_callbacks_list_reports_active_logging_callbacks(self, client: ManagementClient) -> None:
        listing = unwrap(
            client.proxy.transport.get(
                "/callbacks/list",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=CallbacksListResponse,
            )
        )
        every = [*listing.success, *listing.failure, *listing.success_and_failure]
        assert every, "/callbacks/list reported no active logging callbacks; the proxy always runs the db logger"
        assert "_ProxyDBLogger" in every, (
            f"/callbacks/list omitted the always-on _ProxyDBLogger spend logger; got {every}"
        )

    @pytest.mark.covers("mgmt.tool_management.list.happy_path")
    def test_tool_list_returns_catalog_with_consistent_total(self, client: ManagementClient) -> None:
        listing = unwrap(
            client.proxy.transport.get(
                "/v1/tool/list",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=ToolListResponse,
            )
        )
        assert listing.total == len(listing.tools), (
            f"/v1/tool/list total {listing.total} disagrees with the {len(listing.tools)} tools returned"
        )

    @pytest.mark.covers("mgmt.workflow.list.happy_path")
    def test_workflow_runs_list_returns_consistent_count(self, client: ManagementClient) -> None:
        listing = unwrap(
            client.proxy.transport.get(
                "/v1/workflows/runs",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=WorkflowRunsResponse,
            )
        )
        assert listing.count == len(listing.runs), (
            f"/v1/workflows/runs count {listing.count} disagrees with the {len(listing.runs)} runs returned"
        )

    @pytest.mark.covers("mgmt.credential_migration.check.happy_path")
    def test_credential_migration_check_reports_residual_scan(self, client: ManagementClient) -> None:
        report = unwrap(
            client.proxy.transport.get(
                "/credentials/migrate-encryption/check",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=MigrationCheckResponse,
            )
        )
        assert report.status == "success", f"migrate-encryption/check status {report.status!r}, expected 'success'"
        assert report.report.residual_legacy >= 0, (
            f"residual_legacy count is negative ({report.report.residual_legacy}); the scan is broken"
        )
        assert report.report.total_undecryptable >= 0, (
            f"total_undecryptable count is negative ({report.report.total_undecryptable}); the scan is broken"
        )


class TestCostEstimate:
    @pytest.mark.covers("mgmt.cost_tracking.estimate.happy_path")
    def test_estimate_computes_cost_from_token_counts(self, client: ManagementClient) -> None:
        estimate = unwrap(
            client.proxy.transport.post(
                "/cost/estimate",
                headers=client.proxy.transport.master,
                json=CostEstimateBody(
                    model="gpt-4o-mini", input_tokens=1000, output_tokens=500, num_requests_per_day=100
                ),
                response_type=CostEstimateResponse,
            )
        )
        assert estimate.input_cost_per_request > 0, (
            f"input cost per request is {estimate.input_cost_per_request}; a priced model must cost more than zero"
        )
        assert estimate.output_cost_per_request > 0, (
            f"output cost per request is {estimate.output_cost_per_request}; a priced model must cost more than zero"
        )
        expected_per_request = (
            estimate.input_cost_per_request + estimate.output_cost_per_request + estimate.margin_cost_per_request
        )
        assert math.isclose(estimate.cost_per_request, expected_per_request, rel_tol=1e-9), (
            f"cost_per_request {estimate.cost_per_request} != input+output+margin {expected_per_request}"
        )
        assert estimate.daily_cost is not None and math.isclose(
            estimate.daily_cost, estimate.cost_per_request * 100, rel_tol=1e-9
        ), f"daily_cost {estimate.daily_cost} != cost_per_request * 100 requests {estimate.cost_per_request * 100}"


class TestComplianceRoutes:
    @pytest.mark.covers("mgmt.compliance.gdpr.happy_path")
    def test_gdpr_check_derives_verdict_from_the_request(self, client: ManagementClient) -> None:
        result = unwrap(
            client.proxy.transport.post(
                "/compliance/gdpr",
                headers=client.proxy.transport.master,
                json=ComplianceGdprBody(
                    request_id=f"e2e-gdpr-{unique_marker()}",
                    user_id=f"e2e-user-{unique_marker()}",
                    model="gpt-4o-mini",
                    timestamp="2026-07-21T00:00:00Z",
                ),
                response_type=ComplianceResponse,
            )
        )
        assert result.regulation == "GDPR", (
            f"/compliance/gdpr reported regulation {result.regulation!r}, expected 'GDPR'"
        )
        articles = {check.article for check in result.checks}
        assert articles == {"Art. 32", "Art. 5(1)(c)", "Art. 30"}, (
            f"/compliance/gdpr returned articles {articles}, expected the three GDPR articles"
        )
        assert result.compliant == all(check.passed for check in result.checks), (
            "the overall compliant verdict must be the conjunction of the individual checks"
        )
        assert all(check.check_name and check.detail for check in result.checks), (
            "every compliance check must carry a name and a human-readable detail"
        )


class TestCacheSettings:
    @pytest.mark.covers("mgmt.cache_settings.update.happy_path")
    def test_update_persists_cache_backend_to_get(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        """Exercise the update route without changing global state: capture the live
        cache backend and write exactly that back, so the config the proxy ends on is
        byte-for-byte the one it started with. A teardown restore of the same captured
        settings is the safety net if the body fails partway. The update route is only
        meaningful against a configured cache, so an unconfigured proxy fails loudly
        here rather than being silently switched to redis."""
        before = self._read_settings(client)
        assert before.type is not None, (
            "GET /cache/settings reported no cache type; refusing to invent one and mutate the shared proxy"
        )
        captured = CacheSettingsValue(type=before.type, host=before.host or "", port=before.port or "")
        resources.defer(lambda: self._write_settings(client, captured))

        updated = unwrap(
            client.proxy.transport.post(
                "/cache/settings",
                headers=client.proxy.transport.master,
                json=CacheSettingsUpdateBody(cache_settings=captured),
                response_type=CacheUpdateResponse,
            )
        )
        assert updated.status == "success", f"/cache/settings update status {updated.status!r}, expected 'success'"
        assert updated.settings.type == captured.type, (
            f"/cache/settings echoed type {updated.settings.type!r}, wrote {captured.type!r}"
        )

        def reflected() -> CacheCurrentValues | None:
            current = self._read_settings(client)
            return current if current.type == captured.type else None

        after = _poll(client, reflected, f"/cache/settings never reported type {captured.type!r} after the update")
        assert after.host == captured.host and after.port == captured.port, (
            f"/cache/settings persisted host/port {after.host!r}/{after.port!r}, "
            f"wrote {captured.host!r}/{captured.port!r}"
        )

    @staticmethod
    def _read_settings(client: ManagementClient) -> CacheCurrentValues:
        return unwrap(
            client.proxy.transport.get(
                "/cache/settings",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=CacheGetResponse,
            )
        ).current_values

    @staticmethod
    def _write_settings(client: ManagementClient, settings: CacheSettingsValue) -> None:
        _ = unwrap(
            client.proxy.transport.post(
                "/cache/settings",
                headers=client.proxy.transport.master,
                json=CacheSettingsUpdateBody(cache_settings=settings),
                response_type=CacheUpdateResponse,
            )
        )


class TestFallbackManagement:
    @pytest.mark.covers("mgmt.fallback_management.update.happy_path")
    def test_create_persists_and_is_read_back(self, client: ManagementClient, resources: ResourceManager) -> None:
        primary = f"e2e-fallback-primary-{unique_marker()}"
        secondary = f"e2e-fallback-secondary-{unique_marker()}"
        params = LiteLLMParamsBody(model="openai/gpt-5.5", api_key="e2e-dummy-key")
        primary_id = client.proxy.create_model(primary, params)
        resources.defer(lambda: client.proxy.delete_model(primary_id))
        secondary_id = client.proxy.create_model(secondary, params)
        resources.defer(lambda: client.proxy.delete_model(secondary_id))
        resources.defer(lambda: self._delete_fallback(client, primary))

        created = unwrap(
            client.proxy.transport.post(
                "/fallback",
                headers=client.proxy.transport.master,
                json=FallbackCreateBody(model=primary, fallback_models=[secondary], fallback_type="general"),
                response_type=FallbackResponse,
            )
        )
        assert created.model == primary and created.fallback_models == [secondary], (
            f"/fallback echoed model={created.model!r} fallbacks={created.fallback_models}, "
            f"configured {primary!r} -> [{secondary!r}]"
        )

        def read_back() -> FallbackGetResponse | None:
            result = client.proxy.transport.get(
                f"/fallback/{primary}",
                headers=client.proxy.transport.master,
                params=FallbackGetParams(fallback_type="general"),
                response_type=FallbackGetResponse,
            )
            match result:
                case Success(data=data) if secondary in data.fallback_models:
                    return data
                case _:
                    return None

        got = _poll(client, read_back, f"GET /fallback/{primary} never reported {secondary} after /fallback")
        assert got.fallback_models == [secondary], (
            f"GET /fallback/{primary} reports fallbacks {got.fallback_models}, configured [{secondary!r}]"
        )

    @staticmethod
    def _delete_fallback(client: ManagementClient, model: str) -> None:
        _ = client.proxy.transport.delete(
            f"/fallback/{model}",
            headers=client.proxy.transport.master,
            json=NoBody(),
            params=FallbackGetParams(fallback_type="general"),
            response_type=NoBody,
        )


class TestJwtKeyMapping:
    @pytest.mark.covers("mgmt.jwt_key_mapping.new.happy_path")
    def test_new_persists_mapping_and_is_read_back(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = client.proxy.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.proxy.delete_key(key))
        claim_value = f"e2e_jwt_{unique_marker()}"

        created = unwrap(
            client.proxy.transport.post(
                "/jwt/key/mapping/new",
                headers=client.proxy.transport.master,
                json=JwtKeyMappingNewBody(
                    jwt_claim_name="team_id",
                    jwt_claim_value=claim_value,
                    key=key,
                    description="e2e coverage mapping",
                ),
                response_type=JwtKeyMappingResponse,
            )
        )
        resources.defer(lambda: self._delete_mapping(client, created.id))
        assert created.jwt_claim_value == claim_value and created.is_active, (
            f"/jwt/key/mapping/new returned claim_value={created.jwt_claim_value!r} active={created.is_active}, "
            f"configured {claim_value!r} active=True"
        )

        info = unwrap(
            client.proxy.transport.get(
                "/jwt/key/mapping/info",
                headers=client.proxy.transport.master,
                params=JwtInfoParams(id=created.id),
                response_type=JwtKeyMappingResponse,
            )
        )
        assert info.id == created.id and info.jwt_claim_name == "team_id" and info.jwt_claim_value == claim_value, (
            f"/jwt/key/mapping/info reports {info.jwt_claim_name!r}={info.jwt_claim_value!r} for id {info.id}, "
            f"created team_id={claim_value!r}"
        )

    @staticmethod
    def _delete_mapping(client: ManagementClient, mapping_id: str) -> None:
        _ = client.proxy.transport.post(
            "/jwt/key/mapping/delete",
            headers=client.proxy.transport.master,
            json=JwtDeleteBody(id=mapping_id),
            response_type=NoBody,
        )


class TestRouterSettings:
    @pytest.mark.covers("mgmt.router_settings.update.happy_path")
    def test_config_update_persists_router_setting_to_get(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        """/config/update is the only write path for router_settings (there is no
        dedicated router-settings write route). The change is restored on teardown so
        the shared proxy keeps its original retry policy."""
        original = self._read_num_retries(client)
        assert original is not None, "GET /router/settings did not report num_retries; cannot prove a change"
        resources.defer(lambda: self._write_num_retries(client, original))

        target = original + 5
        response = unwrap(
            client.proxy.transport.post(
                "/config/update",
                headers=client.proxy.transport.master,
                json=ConfigUpdateBody(router_settings=RouterSettingsPatch(num_retries=target)),
                response_type=ConfigUpdateResponse,
            )
        )
        assert "success" in response.message.lower(), (
            f"/config/update reported {response.message!r}, expected a success message"
        )

        _ = _poll(
            client,
            lambda: True if self._read_num_retries(client) == target else None,
            f"GET /router/settings never reported num_retries {target} after /config/update",
        )

        self._write_num_retries(client, original)
        restored = _poll(
            client,
            lambda: original if self._read_num_retries(client) == original else None,
            f"GET /router/settings never returned to the original num_retries {original} after the restore",
        )
        assert restored == original, f"router num_retries left at {restored}, expected the original {original}"

    @staticmethod
    def _read_num_retries(client: ManagementClient) -> int | None:
        return unwrap(
            client.proxy.transport.get(
                "/router/settings",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=RouterSettingsResponse,
            )
        ).current_values.num_retries

    @staticmethod
    def _write_num_retries(client: ManagementClient, value: int) -> None:
        _ = unwrap(
            client.proxy.transport.post(
                "/config/update",
                headers=client.proxy.transport.master,
                json=ConfigUpdateBody(router_settings=RouterSettingsPatch(num_retries=value)),
                response_type=ConfigUpdateResponse,
            )
        )


class TestMcpServerSubmission:
    @pytest.mark.covers("mgmt.mcp_server.register.happy_path")
    def test_register_submits_pending_server(self, client: ManagementClient, resources: ResourceManager) -> None:
        """A non-admin, team-scoped key submits an MCP server for review; the proxy
        stores it as pending_review without loading it into the runtime registry."""
        team_id = client.create_team(TeamNewBody(team_alias=f"e2e-mcp-team-{unique_marker()}"))
        resources.defer(lambda: client.delete_team(team_id))
        team_key = client.proxy.generate_key(KeyGenerateBody(team_id=team_id))
        resources.defer(lambda: client.proxy.delete_key(team_key))

        server_name = f"e2e_mcp_{unique_marker()}"
        submitted = unwrap_status(
            client.proxy.transport.post(
                "/v1/mcp/server/register",
                headers=client.proxy.transport.bearer(team_key),
                json=McpRegisterBody(
                    server_name=server_name,
                    url="https://example.com/mcp",
                    transport="sse",
                    description="e2e coverage submission",
                ),
                response_type=McpServerResponse,
            ),
            201,
        )
        resources.defer(lambda: self._delete_server(client, submitted.server_id))
        assert submitted.approval_status == "pending_review", (
            f"a user submission must be pending_review, got {submitted.approval_status!r}"
        )
        assert submitted.server_name == server_name and submitted.transport == "sse", (
            f"/v1/mcp/server/register echoed name={submitted.server_name!r} transport={submitted.transport!r}, "
            f"configured {server_name!r}/sse"
        )

    @pytest.mark.covers("mgmt.mcp_server.approve.persists")
    def test_approve_activates_submission_and_persists(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        """An admin approving a pending submission flips it to active, and the change
        persists to a fresh read of the server."""
        team_id = client.create_team(TeamNewBody(team_alias=f"e2e-mcp-team-{unique_marker()}"))
        resources.defer(lambda: client.delete_team(team_id))
        team_key = client.proxy.generate_key(KeyGenerateBody(team_id=team_id))
        resources.defer(lambda: client.proxy.delete_key(team_key))

        submitted = unwrap(
            client.proxy.transport.post(
                "/v1/mcp/server/register",
                headers=client.proxy.transport.bearer(team_key),
                json=McpRegisterBody(
                    server_name=f"e2e_mcp_{unique_marker()}",
                    url="https://example.com/mcp",
                    transport="sse",
                    description="e2e coverage submission",
                ),
                response_type=McpServerResponse,
            )
        )
        resources.defer(lambda: self._delete_server(client, submitted.server_id))
        assert submitted.approval_status == "pending_review", (
            f"a fresh submission must be pending_review before approval, got {submitted.approval_status!r}"
        )

        approved = unwrap(
            client.proxy.transport.put(
                f"/v1/mcp/server/{submitted.server_id}/approve",
                headers=client.proxy.transport.master,
                json=NoBody(),
                response_type=McpServerResponse,
            )
        )
        assert approved.approval_status == "active", (
            f"approve must flip the submission to active, got {approved.approval_status!r}"
        )

        fetched = unwrap(
            client.proxy.transport.get(
                f"/v1/mcp/server/{submitted.server_id}",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=McpServerResponse,
            )
        )
        assert fetched.server_id == submitted.server_id and fetched.approval_status == "active", (
            f"GET /v1/mcp/server/{submitted.server_id} reports approval_status {fetched.approval_status!r} "
            "after approve, expected 'active'"
        )

    @staticmethod
    def _delete_server(client: ManagementClient, server_id: str) -> None:
        _ = client.proxy.transport.delete(
            f"/v1/mcp/server/{server_id}",
            headers=client.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )
