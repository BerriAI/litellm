import time
from collections.abc import Mapping
from typing import Final
from uuid import uuid4

import pytest
from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from e2e_metadata import Domain, Subject, meta
from models import UserNewBody, UserNewResponse
from pydantic import BaseModel

from .conftest import signed_in
from .stack import Stack, until

pytestmark: Final = [pytest.mark.e2e, pytest.mark.lens_versions]

UPDATE_NOTICE: Final = "The connected Lens does not serve a UI this gateway can show. Update Lens to use it here"


class Connection(BaseModel):
    connected: bool
    configured: bool


class Accepted(BaseModel):
    pass


class Text(BaseModel):
    stringValue: str


class Attribute(BaseModel):
    key: str
    value: Text


class Attributes(BaseModel):
    attributes: tuple[Attribute, ...]


class Scope(BaseModel):
    name: str


class Span(Attributes):
    traceId: str
    spanId: str
    name: str
    kind: int
    startTimeUnixNano: str
    endTimeUnixNano: str


class ScopeSpans(BaseModel):
    scope: Scope
    spans: tuple[Span, ...]


class ResourceSpans(BaseModel):
    resource: Attributes
    scopeSpans: tuple[ScopeSpans, ...]


class Export(BaseModel):
    resourceSpans: tuple[ResourceSpans, ...]


def _agent_run(trace_id: str, marker: str) -> Export:
    ended: Final = time.time_ns()
    span: Final = Span(
        traceId=trace_id,
        spanId=uuid4().hex[:16],
        name=f"agent run {marker}",
        kind=1,
        startTimeUnixNano=str(ended - 1_000_000_000),
        endTimeUnixNano=str(ended),
        attributes=(
            Attribute(key="gen_ai.operation.name", value=Text(stringValue="invoke_agent")),
            Attribute(key="gen_ai.agent.name", value=Text(stringValue=marker)),
        ),
    )
    return Export(
        resourceSpans=(
            ResourceSpans(
                resource=Attributes(attributes=(Attribute(key="service.name", value=Text(stringValue=marker)),)),
                scopeSpans=(ScopeSpans(scope=Scope(name="lens-versions"), spans=(span,)),),
            ),
        )
    )


class ToolList(BaseModel):
    jsonrpc: str = "2.0"
    id: int = 1
    method: str = "tools/list"


class Tool(BaseModel):
    name: str


class Tools(BaseModel):
    tools: tuple[Tool, ...]


class ToolListResult(BaseModel):
    result: Tools


class TestGatewayAcrossLensVersions:
    @pytest.mark.parametrize("release", ("older", "newest"))
    @meta(Subject(domain=Domain.OBSERVABILITY))
    def test_dashboard_login_reaches_lens(self, releases: Mapping[str, Stack], release: str) -> None:
        either: Final = releases[release]
        session: Final = signed_in(either)
        connection: Final = Connection.model_validate_json(either.page("/lens/service", session).body)
        assert connection == Connection(connected=True, configured=True)

    @pytest.mark.parametrize("release", ("older", "newest"))
    @meta(Subject(domain=Domain.OBSERVABILITY))
    def test_spans_an_agent_sends_to_the_gateway_are_stored_in_lens(
        self, releases: Mapping[str, Stack], release: str
    ) -> None:
        either: Final = releases[release]
        marker: Final = unique_marker()
        trace_id: Final = uuid4().hex
        unwrap(
            either.gateway.post(
                "/v1/traces",
                headers=either.gateway.master,
                json=_agent_run(trace_id, marker),
                response_type=Accepted,
            )
        )
        until("the trace to be readable in Lens", lambda: marker in either.stored_trace(trace_id).body, 90)

    @meta(Subject(domain=Domain.OBSERVABILITY))
    def test_newest_lens_serves_its_ui_and_pages_through_the_gateway(self, newest: Stack) -> None:
        session: Final = signed_in(newest)
        ui: Final = newest.page("/lens/ui/", session)
        traces: Final = newest.page("/lens/v1/traces", session)
        assert (ui.status_code, "<html" in ui.body.lower(), traces.status_code) == (200, True, 200), (ui, traces)

    @meta(Subject(domain=Domain.OBSERVABILITY))
    def test_newest_lens_admin_pages_are_blocked_for_other_users_at_the_gateway(self, newest: Stack) -> None:
        marker: Final = unique_marker()
        member: Final = unwrap(
            newest.gateway.post(
                "/user/new",
                headers=newest.gateway.master,
                json=UserNewBody(user_email=f"{marker}@example.com", user_role="internal_user", auto_create_key=True),
                response_type=UserNewResponse,
            )
        ).key
        assert member, "the gateway created the user without a key"
        admin: Final = newest.gateway.probe("/lens/admin/signals", params=NoBody())
        other: Final = newest.gateway.probe(
            "/lens/admin/signals", params=NoBody(), headers=newest.gateway.bearer(member)
        )
        assert (admin.status_code, other.status_code) == (200, 403), (admin, other)

    @meta(Subject(domain=Domain.MCP))
    def test_newest_lens_mcp_lists_its_tools_through_the_gateway(self, newest: Stack) -> None:
        listed: Final = unwrap(
            newest.gateway.post(
                "/lens/mcp", headers=newest.gateway.master, json=ToolList(), response_type=ToolListResult
            )
        )
        assert listed.result.tools, "Lens listed no MCP tools through the gateway"

    @meta(Subject(domain=Domain.UI))
    def test_older_lens_shows_an_update_notice_in_place_of_its_ui(self, older: Stack) -> None:
        ui: Final = older.page("/lens/ui/", signed_in(older))
        assert (ui.status_code, UPDATE_NOTICE in ui.body) == (404, True), ui
