"""Live e2e: deployment configuration parity and UI response-shape parity.

The deployment silently drifts from what production needs and from what the
dashboard expects, and nothing fails when it does: the model cost map falls
back to the bundled backup, driver model groups vanish from the config, or a
management route changes its wire shape out from under the UI. This suite is
the pin, in two halves:

- TestConfigParity asserts the deployed proxy's effective configuration
  matches the checked-in expectations in STAGE; updating the parity contract
  is a one-line diff to that frozen constants block, reviewed in a PR
- TestUiShapeParity asserts the dashboard's top read routes answer with the
  shapes the UI reads, validated by strict pydantic models whose fields are
  required, so shape drift fails validation instead of passing vacuously

Plane notes, verified live against the split stage deployment: /cache/ping,
/active/callbacks, /model/cost_map/source, and every management read route are
served only by the control plane (the data-plane gateway 404s them; see
transport.CONTROL_PLANE_PREFIXES). /health/readiness and
/health/readiness/details are served by both planes, so readiness is asserted
per plane. The public /health/readiness answers only status+db; the litellm
version lives on the authenticated /health/readiness/details.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel, Field, RootModel

from e2e_config import CONTROL_PLANE_BASE_URL, MASTER_KEY, PROXY_BASE_URL, REQUEST_TIMEOUT, unique_marker
from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, KeyGenerateBody
from transport import HttpTransport

pytestmark = pytest.mark.e2e


@dataclass(frozen=True, slots=True)
class ParityExpectations:
    """The checked-in parity contract for the deployment under test.

    required_model_groups: every model group the e2e suites and production
    traffic depend on; a group missing from /model/info fails with its name.
    cost_map_source pins where pricing data must come from ("remote" on
    stage): a proxy that silently fell back to the bundled backup map reports
    "local" here and fails loudly. cost_map_min_models is a sanity floor on
    the loaded map. cache_type pins the response-cache backend.
    """

    required_model_groups: tuple[str, ...]
    cost_map_source: str
    cost_map_min_models: int
    cache_type: str


STAGE = ParityExpectations(
    required_model_groups=(
        "gemini-2.5-flash",
        "gpt-5.5",
        "claude-haiku-4-5",
        "openai-text-embedding-3-small",
        "custom-priced-flash",
        "openai-realtime",
    ),
    cost_map_source="remote",
    cost_map_min_models=2000,
    cache_type="redis",
)

SEED_MODEL = "gemini-2.5-flash"


class ReadinessDetails(BaseModel):
    status: str
    db: str
    cache: str
    litellm_version: str
    success_callbacks: list[str]


class CachePing(BaseModel):
    status: str
    cache_type: str
    ping_response: bool
    set_cache_response: str


class CostMapSource(BaseModel):
    source: str
    is_env_forced: bool
    fallback_reason: str | None
    model_count: int


class ActiveCallbacks(BaseModel):
    num_callbacks: int
    success_callback: list[str] = Field(alias="litellm.success_callback")


class UserRow(BaseModel):
    user_id: str
    user_role: str


class UserListPage(BaseModel):
    users: list[UserRow]
    total: int
    page: int
    page_size: int
    total_pages: int


class UserListFilter(BaseModel):
    user_ids: str


class UserNewBody(BaseModel):
    user_email: str
    user_role: str


class UserNewResponse(BaseModel):
    user_id: str


class UserDeleteBody(BaseModel):
    user_ids: list[str]


class KeyRow(BaseModel):
    token: str
    key_alias: str
    user_id: str
    spend: float


class KeyListPage(BaseModel):
    keys: list[KeyRow]
    total_count: int
    current_page: int
    total_pages: int


class KeyListFilter(BaseModel):
    key_alias: str
    return_full_object: bool


class TeamNewBody(BaseModel):
    team_alias: str


class TeamNewResponse(BaseModel):
    team_id: str


class TeamDeleteBody(BaseModel):
    team_ids: list[str]


class TeamListEntry(BaseModel):
    team_id: str
    team_alias: str | None


class TeamList(RootModel[list[TeamListEntry]]):
    pass


class UiModelParams(BaseModel):
    model: str


class UiModelEntry(BaseModel):
    model_name: str
    litellm_params: UiModelParams


class UiModelInfo(BaseModel):
    data: list[UiModelEntry]


class UiSpendLogRow(BaseModel):
    request_id: str
    api_key: str
    model: str
    spend: float
    status: str
    start_time: str = Field(alias="startTime")


class UiSpendLogsPage(BaseModel):
    data: list[UiSpendLogRow]
    total: int
    page: int
    page_size: int
    total_pages: int


class UiSpendLogsParams(BaseModel):
    start_date: str
    end_date: str
    page: int
    page_size: int
    api_key: str | None = None


class DailyMetrics(BaseModel):
    spend: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    api_requests: int
    successful_requests: int
    failed_requests: int


class DailyMetricWithMetadata(BaseModel):
    metrics: DailyMetrics


class DailyBreakdown(BaseModel):
    models: dict[str, DailyMetricWithMetadata]
    api_keys: dict[str, DailyMetricWithMetadata]
    providers: dict[str, DailyMetricWithMetadata]


class DailySpendRow(BaseModel):
    date: str
    metrics: DailyMetrics
    breakdown: DailyBreakdown


class DailyActivityMetadata(BaseModel):
    total_spend: float
    total_api_requests: int
    total_successful_requests: int
    total_failed_requests: int
    total_tokens: int
    page: int
    total_pages: int
    has_more: bool


class DailyActivity(BaseModel):
    results: list[DailySpendRow]
    metadata: DailyActivityMetadata


class DailyActivityParams(BaseModel):
    start_date: str
    end_date: str
    page: int
    page_size: int


@dataclass(frozen=True, slots=True)
class Plane:
    name: str
    transport: HttpTransport


@dataclass(frozen=True, slots=True)
class ParityClient:
    gateway: Gateway
    planes: tuple[Plane, ...]

    @property
    def data_plane(self) -> HttpTransport:
        return self.planes[0].transport


def build_client() -> ParityClient:
    data = Plane(
        name="data",
        transport=HttpTransport(base_url=PROXY_BASE_URL, master_key=MASTER_KEY, request_timeout=REQUEST_TIMEOUT),
    )
    control = Plane(
        name="control",
        transport=HttpTransport(
            base_url=CONTROL_PLANE_BASE_URL, master_key=MASTER_KEY, request_timeout=REQUEST_TIMEOUT
        ),
    )
    planes = (data,) if CONTROL_PLANE_BASE_URL == PROXY_BASE_URL else (data, control)
    return ParityClient(gateway=build_gateway(), planes=planes)


@pytest.fixture(scope="session")
def client() -> ParityClient:
    return build_client()


@dataclass(frozen=True, slots=True)
class SeededRequest:
    """A fresh key's hashed token after exactly one real chat call went through
    it. Spend rows record the hashed token as api_key, and the dashboard reads
    that same token off /key/list to filter /spend/logs/ui, so the token is the
    wire-level correlation between the seeded call and its spend row."""

    token: str


@pytest.fixture(scope="module")
def seeded_request(client: ParityClient) -> Iterator[SeededRequest]:
    """One real chat call through the data plane, shared by the spend-facing
    shape tests so their read routes have a fresh row to validate against."""
    alias = f"e2e-parity-seed-{unique_marker()}"
    key = client.gateway.generate_key(KeyGenerateBody(models=[], user_id="e2e-test-user", key_alias=alias))
    try:
        outcome = client.gateway.transport.send(
            "/chat/completions",
            headers=client.gateway.transport.bearer(key),
            json=ChatBody(
                model=SEED_MODEL,
                messages=[ChatMessage(role="user", content=f"reply with one word {unique_marker()}")],
                max_tokens=16,
            ),
        )
        require_successful_call(outcome)
        listed = unwrap(
            client.gateway.transport.get(
                "/key/list",
                headers=client.gateway.transport.master,
                params=KeyListFilter(key_alias=alias, return_full_object=True),
                response_type=KeyListPage,
            )
        )
        assert listed.keys, f"/key/list answered no row for the seeded key alias {alias}"
        yield SeededRequest(token=listed.keys[0].token)
    finally:
        client.gateway.delete_key(key)


def _poll[T](timeout: float, interval: float, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(interval)
    pytest.fail(failure)


def _day_window() -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    return now - timedelta(days=1), now + timedelta(days=1)


def _readiness_details(transport: HttpTransport) -> ReadinessDetails:
    return unwrap(
        transport.get(
            "/health/readiness/details",
            headers=transport.master,
            params=NoBody(),
            response_type=ReadinessDetails,
        )
    )


def _delete_user(client: ParityClient, user_id: str) -> None:
    _ = client.gateway.transport.post(
        "/user/delete",
        headers=client.gateway.transport.master,
        json=UserDeleteBody(user_ids=[user_id]),
        response_type=NoBody,
    )


def _delete_team(client: ParityClient, team_id: str) -> None:
    _ = client.gateway.transport.post(
        "/team/delete",
        headers=client.gateway.transport.master,
        json=TeamDeleteBody(team_ids=[team_id]),
        response_type=NoBody,
    )


class TestConfigParity:
    @pytest.mark.covers("other.config.readiness.db_connected")
    def test_readiness_reports_db_and_version_on_every_plane(self, client: ParityClient) -> None:
        for plane in client.planes:
            details = _readiness_details(plane.transport)
            assert details.status == "healthy", f"{plane.name} plane readiness status is {details.status!r}"
            assert details.db == "connected", f"{plane.name} plane reports db {details.db!r}, expected 'connected'"
            assert details.litellm_version, f"{plane.name} plane reports an empty litellm version"

    @pytest.mark.covers("other.config.cache.ping_healthy")
    def test_cache_ping_reports_a_working_response_cache(self, client: ParityClient) -> None:
        """A deployment without cache config must fail this loudly: /cache/ping
        503s with no cache, and the data plane's readiness would report no
        cache backend."""
        ping = unwrap(
            client.gateway.transport.get(
                "/cache/ping",
                headers=client.gateway.transport.master,
                params=NoBody(),
                response_type=CachePing,
            )
        )
        assert ping.status == "healthy", f"/cache/ping status {ping.status!r}"
        assert ping.cache_type == STAGE.cache_type, (
            f"/cache/ping cache_type {ping.cache_type!r}, expected {STAGE.cache_type!r}"
        )
        assert ping.ping_response is True, "/cache/ping could not ping the cache backend"
        assert ping.set_cache_response == "success", f"/cache/ping set_cache_response {ping.set_cache_response!r}"
        data_details = _readiness_details(client.data_plane)
        assert data_details.cache == STAGE.cache_type, (
            f"data plane cache backend is {data_details.cache!r}, expected {STAGE.cache_type!r}"
        )

    @pytest.mark.covers("other.config.cost_map.remote_loaded")
    def test_cost_map_loaded_remotely_without_fallback(self, client: ParityClient) -> None:
        source = unwrap(
            client.gateway.transport.get(
                "/model/cost_map/source",
                headers=client.gateway.transport.master,
                params=NoBody(),
                response_type=CostMapSource,
            )
        )
        assert source.source == STAGE.cost_map_source, (
            f"model cost map came from {source.source!r}, expected {STAGE.cost_map_source!r}; "
            f"fallback_reason={source.fallback_reason!r}"
        )
        assert source.fallback_reason is None, f"cost map loaded with fallback_reason {source.fallback_reason!r}"
        assert not source.is_env_forced, "LITELLM_LOCAL_MODEL_COST_MAP forces the bundled backup map on stage"
        assert source.model_count >= STAGE.cost_map_min_models, (
            f"cost map holds {source.model_count} models, below the {STAGE.cost_map_min_models} sanity floor"
        )

    @pytest.mark.covers("other.config.model_groups.present")
    def test_required_model_groups_are_deployed(self, client: ParityClient) -> None:
        info = unwrap(
            client.gateway.transport.get(
                "/model/info",
                headers=client.gateway.transport.master,
                params=NoBody(),
                response_type=UiModelInfo,
            )
        )
        deployed = frozenset(entry.model_name for entry in info.data)
        missing = tuple(group for group in STAGE.required_model_groups if group not in deployed)
        assert not missing, (
            f"model groups missing from the deployed config: {', '.join(missing)}; deployed groups: {sorted(deployed)}"
        )

    @pytest.mark.covers("other.config.callbacks.cache_active")
    def test_expected_callbacks_are_active(self, client: ParityClient) -> None:
        """The data plane serves no callback-introspection route
        (/active/callbacks is control-plane only), so the data plane is
        checked through its readiness success_callbacks and the control plane
        through /active/callbacks. Logging integrations declared in the stage
        gateway config (datadog and friends) are not observable on either
        surface and are deliberately not asserted here."""
        details = _readiness_details(client.data_plane)
        assert "cache" in details.success_callbacks, (
            f"response-cache callback is not active on the data plane; active: {details.success_callbacks}"
        )
        active = unwrap(
            client.gateway.transport.get(
                "/active/callbacks",
                headers=client.gateway.transport.master,
                params=NoBody(),
                response_type=ActiveCallbacks,
            )
        )
        assert active.num_callbacks > 0, "control plane reports zero active callbacks"
        assert "cache" in active.success_callback, (
            "response-cache callback is not in the control plane's success callbacks"
        )


class TestUiShapeParity:
    @pytest.mark.covers("mgmt.user.list.happy_path")
    def test_user_list_answers_users_and_total(self, client: ParityClient, resources: ResourceManager) -> None:
        """surface=ui: the Users table reads {users, total} with user_id and
        user_role on every row (plus page/page_size/total_pages paging)."""
        marker = unique_marker()
        user_id = unwrap(
            client.gateway.transport.post(
                "/user/new",
                headers=client.gateway.transport.master,
                json=UserNewBody(user_email=f"e2e-parity-{marker}@example.com", user_role="internal_user"),
                response_type=UserNewResponse,
            )
        ).user_id
        resources.defer(lambda: _delete_user(client, user_id))
        page = unwrap(
            client.gateway.transport.get(
                "/user/list",
                headers=client.gateway.transport.master,
                params=UserListFilter(user_ids=user_id),
                response_type=UserListPage,
            )
        )
        assert page.total == 1, f"/user/list filtered to one user reports total {page.total}"
        assert page.users[0].user_id == user_id, f"/user/list row carries user_id {page.users[0].user_id!r}"
        assert page.users[0].user_role == "internal_user", (
            f"/user/list row carries user_role {page.users[0].user_role!r}"
        )

    @pytest.mark.covers("mgmt.key.list.happy_path")
    def test_key_list_answers_keys_and_total_count(self, client: ParityClient, resources: ResourceManager) -> None:
        """surface=ui: the Keys table reads {keys, total_count} from
        /key/list?return_full_object=true, each row carrying its token."""
        alias = f"e2e-parity-key-{unique_marker()}"
        key = client.gateway.generate_key(KeyGenerateBody(models=[], user_id="e2e-test-user", key_alias=alias))
        resources.defer(lambda: client.gateway.delete_key(key))
        page = unwrap(
            client.gateway.transport.get(
                "/key/list",
                headers=client.gateway.transport.master,
                params=KeyListFilter(key_alias=alias, return_full_object=True),
                response_type=KeyListPage,
            )
        )
        assert page.total_count == 1, f"/key/list filtered to one alias reports total_count {page.total_count}"
        assert page.keys[0].key_alias == alias, f"/key/list row carries key_alias {page.keys[0].key_alias!r}"
        assert page.keys[0].token, "/key/list row carries an empty token"

    @pytest.mark.covers("mgmt.team.list.happy_path")
    def test_team_list_entries_carry_team_id(self, client: ParityClient, resources: ResourceManager) -> None:
        """surface=ui: the Teams table reads a bare array whose entries carry
        team_id."""
        alias = f"e2e-parity-team-{unique_marker()}"
        team_id = unwrap(
            client.gateway.transport.post(
                "/team/new",
                headers=client.gateway.transport.master,
                json=TeamNewBody(team_alias=alias),
                response_type=TeamNewResponse,
            )
        ).team_id
        resources.defer(lambda: _delete_team(client, team_id))
        teams = unwrap(
            client.gateway.transport.get(
                "/team/list",
                headers=client.gateway.transport.master,
                params=NoBody(),
                response_type=TeamList,
            )
        ).root
        listed = next((team for team in teams if team.team_id == team_id), None)
        assert listed is not None, f"team {team_id} missing from /team/list after /team/new"
        assert listed.team_alias == alias, f"/team/list entry carries team_alias {listed.team_alias!r}"

    @pytest.mark.covers("mgmt.model.info.happy_path")
    def test_model_info_entries_carry_name_and_params(self, client: ParityClient) -> None:
        """surface=ui: the Models table reads model_name and litellm_params
        off every /model/info entry."""
        info = unwrap(
            client.gateway.transport.get(
                "/model/info",
                headers=client.gateway.transport.master,
                params=NoBody(),
                response_type=UiModelInfo,
            )
        )
        assert info.data, "/model/info answered zero deployments"
        unnamed = tuple(
            entry.model_name for entry in info.data if not (entry.model_name and entry.litellm_params.model)
        )
        assert not unnamed, f"/model/info entries with empty model_name or litellm_params.model: {unnamed}"

    @pytest.mark.covers("mgmt.spend.logs_ui.happy_path")
    def test_spend_logs_ui_answers_the_paginated_shape(
        self, client: ParityClient, seeded_request: SeededRequest
    ) -> None:
        """surface=ui: the Logs page reads {data, total, page, page_size,
        total_pages}, filtering by the hashed key token it read off /key/list;
        spend rows land on the batch-write cadence, so the read polls to a
        deadline for the seeded key's row."""
        start, end = _day_window()

        def fetch() -> UiSpendLogsPage | None:
            page = unwrap(
                client.gateway.transport.get(
                    "/spend/logs/ui",
                    headers=client.gateway.transport.master,
                    params=UiSpendLogsParams(
                        start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                        end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                        page=1,
                        page_size=10,
                        api_key=seeded_request.token,
                    ),
                    response_type=UiSpendLogsPage,
                )
            )
            return page if page.data else None

        page = _poll(
            client.gateway.poll_timeout,
            client.gateway.poll_interval,
            fetch,
            "spend log row for the seeded key never appeared on /spend/logs/ui",
        )
        assert page.total >= 1, f"/spend/logs/ui reports total {page.total} despite returning rows"
        assert page.page == 1 and page.total_pages >= 1, (
            f"/spend/logs/ui paging is page={page.page} total_pages={page.total_pages}"
        )
        foreign = tuple(row.request_id for row in page.data if row.api_key != seeded_request.token)
        assert not foreign, f"/spend/logs/ui filtered by api_key returned rows for other keys: {foreign}"
        assert page.data[0].request_id, "/spend/logs/ui row carries an empty request_id"

    @pytest.mark.covers("mgmt.user.daily_activity.happy_path")
    def test_user_daily_activity_answers_results_and_metadata(
        self, client: ParityClient, seeded_request: SeededRequest
    ) -> None:
        """surface=ui: the Usage page reads the SpendAnalyticsPaginatedResponse
        shape (results rows with date/metrics/breakdown plus paging metadata);
        the seeded request guarantees the window is non-empty once daily
        aggregation lands, so the read polls to a deadline."""
        start, end = _day_window()

        def fetch() -> DailyActivity | None:
            activity = unwrap(
                client.gateway.transport.get(
                    "/user/daily/activity",
                    headers=client.gateway.transport.master,
                    params=DailyActivityParams(
                        start_date=start.strftime("%Y-%m-%d"),
                        end_date=end.strftime("%Y-%m-%d"),
                        page=1,
                        page_size=50,
                    ),
                    response_type=DailyActivity,
                )
            )
            return activity if activity.results and activity.metadata.total_api_requests >= 1 else None

        activity = _poll(
            client.gateway.poll_timeout,
            client.gateway.poll_interval,
            fetch,
            "no daily-activity row appeared for the seeded request before the deadline",
        )
        assert activity.metadata.page == 1, f"/user/daily/activity metadata reports page {activity.metadata.page}"
        assert any(row.metrics.api_requests >= 1 for row in activity.results), (
            "no daily-activity row carries a positive api_requests count"
        )
        assert any(row.breakdown.models for row in activity.results), "daily-activity rows carry no per-model breakdown"
