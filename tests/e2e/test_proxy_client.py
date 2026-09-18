"""Harness coverage for the barriers that gate on every replica.

No proxy needed and no ``e2e`` marker: this pins that a model registered through
the control plane only counts as servable once every configured replica lists it
on /v1/models, and that a management write only counts as read back once every
replica's read satisfies the caller's predicate, which is what keeps a two-gateway
stack from handing a test a model or a key that one gateway has not caught up on
yet. The fakes are plain pollers standing in for each replica's transport plus an
injected clock, so nothing here monkeypatches anything.
"""

from __future__ import annotations

import json
from builtins import ExceptionGroup
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain, repeat
from queue import SimpleQueue
from threading import Thread
from types import MappingProxyType
from typing import Final, cast

import pytest
from e2e_config import parse_replica_urls
from e2e_http import NoBody, Result, Success, without_retries
from idp import Keycloak
from lifecycle import ResourceManager
from management.jwt_actors import ActorFactory
from management.management_client import ManagementClient
from models import (
    ConnectionTestBody,
    CredentialCreateBody,
    KeyGenerateBody,
    KeyInfo,
    KeyInfoResponse,
    KeyUpdateBody,
    LiteLLMParamsBody,
    McpServerCreateBody,
    McpServerUpdateBody,
    ModelListEntry,
    ModelsListResponse,
    OrgNewBody,
    OrgUpdateBody,
    SpendLogsParams,
    TagNewBody,
    TeamNewBody,
    TeamUpdateBody,
    ToolsetCreateBody,
    ToolsetUpdateBody,
    UserNewBody,
    UserUpdateBody,
)
from proxy_client import (
    Caller,
    Converged,
    ConvergeOutcome,
    CredentialKind,
    EverywhereConverged,
    ModelsPoller,
    NeverConvergedOn,
    NotConverged,
    NotServableOn,
    Poller,
    ProxyClient,
    ReplicaRead,
    Servable,
    await_converged_everywhere,
    await_everywhere,
    await_servable_everywhere,
    build_proxy_client,
    converge_timeout_message,
    first_lagging_replica,
)
from transport import Transport


@contextmanager
def caller_boundary(
    status: int = 200, bodies: SimpleQueue[bytes] | None = None, *, delete_status: int | None = None
) -> Generator[tuple[ManagementClient, SimpleQueue[str]]]:
    received: Final[SimpleQueue[str]] = SimpleQueue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            received.put(self.headers.get("Authorization", ""))
            self.send_response(delete_status if self.path == "/key/delete" and delete_status is not None else status)
            self.end_headers()
            self.wfile.write(
                b'{"key":"owned","info":{"key_alias":"owned"},"data":[{"id":"owned"}],"team_id":"owned","team_info":{},"model_id":"owned"}'
            )

        def do_POST(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if bodies is not None:
                bodies.put(body)
            self.do_GET()

        do_PATCH = do_POST
        do_PUT = do_POST
        do_DELETE = do_POST

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    url: Final = f"http://127.0.0.1:{server.server_port}"
    proxy: Final = build_proxy_client(
        base_url=url, control_plane_base_url=url, replica_urls=(url,), master_key="bootstrap"
    )
    try:
        yield ManagementClient(proxy=proxy, master_key="bootstrap"), received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestBoundManagementCaller:
    def test_strict_key_cleanup_accepts_missing_only_when_requested(self) -> None:
        with caller_boundary(delete_status=404) as (bootstrap, received), without_retries():
            with pytest.raises(AssertionError):
                bootstrap.delete_key_strict("owned")
            bootstrap.delete_key_strict("owned", missing_ok=True)
            assert (received.get_nowait(), received.get_nowait()) == ("Bearer bootstrap", "Bearer bootstrap")

    def test_actor_key_cleanup_reports_failure_and_continues(self) -> None:
        with caller_boundary(delete_status=500) as (bootstrap, received), without_retries():
            resources: Final = ResourceManager(client=bootstrap.proxy, strict_cleanup=True)
            remaining: SimpleQueue[str] = SimpleQueue()
            resources.defer(lambda: remaining.put("cleaned"))
            factory: Final = ActorFactory(
                bootstrap=bootstrap,
                idp=Keycloak(base_url="http://unused.test", realm="test", admin_username="test", admin_password="test"),
                resources=resources,
            )
            assert factory.key().key == "owned"
            with pytest.raises(ExceptionGroup, match="Resource cleanup failed") as failure:
                resources.teardown()
            assert len(failure.value.exceptions) == 1
            assert remaining.get_nowait() == "cleaned"
            assert (received.get_nowait(), received.get_nowait()) == ("Bearer bootstrap", "Bearer bootstrap")

    @pytest.mark.parametrize("kind", ("direct_jwt", "virtual_key", "dashboard_session"))
    def test_direct_delegated_and_replica_reads_keep_the_bound_caller(self, kind: CredentialKind) -> None:
        with caller_boundary() as (bootstrap, received):
            caller: Final = Caller(credential="synthetic-caller", kind=kind, role="internal_user", tenant="tenant-a")
            bound: Final = bootstrap.with_caller(caller)
            bound.update_key(KeyUpdateBody(key="owned", key_alias="updated"))
            bound.proxy.key_info("owned")
            bound.proxy.read_back_everywhere(
                "/key/info",
                params=KeyUpdateBody(key="owned"),
                response_type=KeyInfoResponse,
                converged=lambda result: isinstance(result, Success),
            )
            bound.proxy.read_body_back_everywhere(
                "/key/info", KeyInfoResponse, settled=lambda result: result.info.key_alias == "owned"
            )
            assert tuple(received.get_nowait() for _ in range(4)) == ("Bearer synthetic-caller",) * 4
            assert received.empty()
            bootstrap.proxy.key_info("owned")
            assert received.get_nowait() == "Bearer bootstrap"

    def test_explicit_override_wins_without_rebinding_or_changing_master(self) -> None:
        with caller_boundary() as (bootstrap, received):
            bound: Final = bootstrap.with_caller(Caller(credential="bound", kind="direct_jwt", role="internal_user"))
            bound.update_key(KeyUpdateBody(key="owned"), caller_key="override")
            bound.proxy.key_info("owned")
            assert received.get_nowait() == "Bearer override"
            assert received.get_nowait() == "Bearer bound"
            assert bound.master_key == "bootstrap"

    def test_credentials_are_absent_from_binding_and_header_diagnostics(self) -> None:
        with caller_boundary() as (bootstrap, _):
            caller: Final = Caller(credential="private-value", kind="direct_jwt", role="internal_user")
            bound: Final = bootstrap.with_caller(caller)
            assert "private-value" not in repr(caller)
            assert "private-value" not in repr(bound)
            assert "private-value" not in repr(bound.proxy.management_headers())
            assert "bootstrap" not in repr(bound)


MODEL: Final = "gpt-under-test"
_NO_TRANSPORTS: Final = cast(Transport, None)
TIMEOUT: Final = 10.0
INTERVAL: Final = 2.0
RPM_BEFORE_UPDATE: Final = 100
RPM_AFTER_UPDATE: Final = 200


@dataclass
class FakeClock:
    elapsed: float = 0.0

    def now(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


def _listing(*model_ids: str) -> Success[ModelsListResponse]:
    entries: Final = tuple(ModelListEntry(id=model_id) for model_id in model_ids)
    return Success(status_code=200, data=ModelsListResponse(data=entries))


def _poller(results: Iterable[Success[ModelsListResponse]]) -> ModelsPoller:
    it: Final = iter(results)
    return lambda _timeout: next(it)


def _await(pollers: Mapping[str, ModelsPoller]) -> Servable | NotServableOn:
    clock: Final = FakeClock()
    return await_servable_everywhere(
        pollers,
        model_name=MODEL,
        timeout=TIMEOUT,
        interval=INTERVAL,
        request_timeout=5.0,
        db_sync_seconds=0.0,
        now=clock.now,
        sleep=clock.sleep,
    )


class TestAwaitServableEverywhere:
    @pytest.mark.parametrize("missing", ["gateway-1", "gateway-2"])
    def test_fails_on_the_replica_that_never_lists_the_model(self, missing: str) -> None:
        pollers: Final = {
            "gateway-1": _poller(repeat(_listing(MODEL))),
            "gateway-2": _poller(repeat(_listing(MODEL))),
        } | {missing: _poller(repeat(_listing()))}
        assert _await(pollers) == NotServableOn(replica=missing, last_result=_listing())

    def test_passes_once_every_replica_lists_the_model(self) -> None:
        pollers: Final = {
            "gateway-1": _poller(repeat(_listing(MODEL))),
            "gateway-2": _poller(chain(repeat(_listing(), 2), repeat(_listing(MODEL)))),
        }
        assert _await(pollers) == Servable()


def _key_info(rpm_limit: int) -> Success[KeyInfoResponse]:
    return Success(status_code=200, data=KeyInfoResponse(info=KeyInfo(rpm_limit=rpm_limit)))


def _reads(results: Iterable[Result[KeyInfoResponse]]) -> Poller[Result[KeyInfoResponse]]:
    it: Final = iter(results)
    return lambda: next(it)


def _updated(result: Result[KeyInfoResponse]) -> bool:
    return isinstance(result, Success) and result.data.info.rpm_limit == RPM_AFTER_UPDATE


def _converge(
    pollers: Mapping[str, Poller[Result[KeyInfoResponse]]], clock: FakeClock
) -> Mapping[str, ConvergeOutcome[Result[KeyInfoResponse]]]:
    return await_converged_everywhere(
        pollers,
        converged=_updated,
        timeout=TIMEOUT,
        interval=INTERVAL,
        now=clock.now,
        sleep=clock.sleep,
    )


class TestAwaitConvergedEverywhere:
    def test_waits_for_the_replica_that_lags_behind_the_write(self) -> None:
        clock: Final = FakeClock()
        pollers: Final = MappingProxyType(
            {
                "gateway-1": _reads(repeat(_key_info(RPM_AFTER_UPDATE))),
                "gateway-2": _reads(
                    chain(repeat(_key_info(RPM_BEFORE_UPDATE), 2), repeat(_key_info(RPM_AFTER_UPDATE)))
                ),
            }
        )
        outcomes: Final = _converge(pollers, clock)
        assert outcomes == {
            "gateway-1": Converged(result=_key_info(RPM_AFTER_UPDATE)),
            "gateway-2": Converged(result=_key_info(RPM_AFTER_UPDATE)),
        }
        assert first_lagging_replica(outcomes) is None
        assert clock.elapsed == 2 * INTERVAL

    def test_names_the_replica_that_never_converges_with_its_last_read(self) -> None:
        clock: Final = FakeClock()
        pollers: Final = MappingProxyType(
            {
                "gateway-1": _reads(repeat(_key_info(RPM_AFTER_UPDATE))),
                "gateway-2": _reads(repeat(_key_info(RPM_BEFORE_UPDATE))),
            }
        )
        outcomes: Final = _converge(pollers, clock)
        assert first_lagging_replica(outcomes) == (
            "gateway-2",
            NotConverged(last_result=_key_info(RPM_BEFORE_UPDATE)),
        )
        assert clock.elapsed == TIMEOUT
        message: Final = converge_timeout_message(
            what="GET /key/info",
            replica="gateway-2",
            timeout=TIMEOUT,
            last_result=_key_info(RPM_BEFORE_UPDATE),
        )
        assert "gateway-2" in message and "/key/info" in message and str(RPM_BEFORE_UPDATE) in message

    def test_each_replica_gets_its_own_full_budget(self) -> None:
        """A replica that converges late must not eat into the next replica's budget: both
        need most of the timeout here, so one shared deadline would starve the second."""
        clock: Final = FakeClock()
        slow: Final = chain(repeat(_key_info(RPM_BEFORE_UPDATE), 3), repeat(_key_info(RPM_AFTER_UPDATE)))
        pollers: Final = MappingProxyType(
            {
                "gateway-1": _reads(slow),
                "gateway-2": _reads(
                    chain(repeat(_key_info(RPM_BEFORE_UPDATE), 3), repeat(_key_info(RPM_AFTER_UPDATE)))
                ),
            }
        )
        outcomes: Final = _converge(pollers, clock)
        assert first_lagging_replica(outcomes) is None
        assert clock.elapsed == 2 * 3 * INTERVAL


class TestParseReplicaUrls:
    def test_splits_and_trims_the_gateway_addresses(self) -> None:
        raw: Final = " http://127.0.0.1:4010/, http://127.0.0.1:4011 "
        assert parse_replica_urls(raw, "http://lb") == ("http://127.0.0.1:4010", "http://127.0.0.1:4011")

    def test_falls_back_to_the_data_plane_address_when_unset(self) -> None:
        assert parse_replica_urls("", "http://lb") == ("http://lb",)


def _answers(answers: Iterable[str]) -> ReplicaRead[str]:
    it: Final = iter(answers)
    return lambda _timeout: next(it)


def _await_everywhere(reads: Mapping[str, ReplicaRead[str]]) -> EverywhereConverged[str] | NeverConvergedOn[str]:
    clock: Final = FakeClock()
    return await_everywhere(
        reads,
        settled=lambda answer: answer == "renamed",
        timeout=TIMEOUT,
        interval=INTERVAL,
        request_timeout=5.0,
        now=clock.now,
        sleep=clock.sleep,
    )


class TestAwaitEverywhere:
    def test_waits_for_the_lagging_replica_and_returns_every_settled_answer(self) -> None:
        reads: Final = {
            "gateway-1": _answers(repeat("renamed")),
            "gateway-2": _answers(chain(repeat("stale", 2), repeat("renamed"))),
        }
        outcome: Final = _await_everywhere(reads)
        assert isinstance(outcome, EverywhereConverged)
        assert dict(outcome.answers) == {"gateway-1": "renamed", "gateway-2": "renamed"}

    def test_names_the_replica_that_never_converges_with_what_it_last_served(self) -> None:
        reads: Final = {
            "gateway-1": _answers(repeat("renamed")),
            "gateway-2": _answers(repeat("stale")),
        }
        assert _await_everywhere(reads) == NeverConvergedOn(replica="gateway-2", last="stale")

    def test_polls_until_the_deadline_before_giving_up(self) -> None:
        lagging: Final = chain(repeat("stale", int(TIMEOUT / INTERVAL)), repeat("renamed"))
        outcome: Final = _await_everywhere({"gateway-1": _answers(lagging)})
        assert isinstance(outcome, EverywhereConverged), outcome


class TestReplicasFor:
    def test_split_deployment_reads_management_routes_back_from_the_control_plane(self) -> None:
        client: Final = build_proxy_client(
            base_url="http://lb",
            control_plane_base_url="http://backend",
            replica_urls=("http://gateway-1", "http://gateway-2"),
        )
        assert set(client.replicas_for("/key/info")) == {"http://backend"}
        assert set(client.replicas_for("/project/info")) == {"http://backend"}
        assert set(client.replicas_for("/v1/models")) == {"http://gateway-1", "http://gateway-2"}

    def test_monolith_reads_management_routes_back_from_every_replica(self) -> None:
        client: Final = build_proxy_client(
            base_url="http://lb",
            control_plane_base_url="http://lb",
            replica_urls=("http://pod-1", "http://pod-2"),
        )
        assert set(client.replicas_for("/key/info")) == {"http://pod-1", "http://pod-2"}

    def test_mcp_admin_routes_read_back_from_every_data_plane_replica(self) -> None:
        """/v1/mcp/* is a lazily mounted feature, so a data-plane replica serves it
        too and answers from its own in-memory registry. Routing it to the control
        plane would leave every replica but that one unproven, and would move the
        tools/list barrier in mcp_client off the plane that serves tools/list."""
        client: Final = build_proxy_client(
            base_url="http://lb",
            control_plane_base_url="http://backend",
            replica_urls=("http://gateway-1", "http://gateway-2"),
        )
        assert set(client.replicas_for("/v1/mcp/server/abc")) == {"http://gateway-1", "http://gateway-2"}
        assert set(client.replicas_for("/v1/mcp/toolset/abc")) == {"http://gateway-1", "http://gateway-2"}

    def test_a_route_no_replica_serves_is_refused_rather_than_read_back_vacuously(self) -> None:
        """A read-back over zero replicas would satisfy every predicate and assert
        nothing, so asking for one fails instead of passing silently."""
        client: Final = ProxyClient(transport=_NO_TRANSPORTS, replicas={}, control_replicas={})
        with pytest.raises(AssertionError, match="no replica is configured"):
            _ = client.replicas_for("/v1/models")


MANAGEMENT_OPERATIONS: Final[tuple[tuple[str, Callable[[ManagementClient], object]], ...]] = (
    ("generate_key", lambda c: c.generate_key(KeyGenerateBody())),
    ("llm_only_key", lambda c: c.llm_only_key()),
    ("update_key", lambda c: c.update_key(KeyUpdateBody(key="owned"))),
    ("update_key_models", lambda c: c.update_key_models("owned", [])),
    ("key_info", lambda c: c.key_info_as("owned")),
    ("delete_key_strict", lambda c: c.delete_key_strict("owned")),
    ("delete_model_strict", lambda c: c.delete_model_strict("owned")),
    (
        "connection_test",
        lambda c: c.connection_test(
            ConnectionTestBody(litellm_params=LiteLLMParamsBody(model="synthetic"), mode="chat")
        ),
    ),
    ("block_key", lambda c: c.block_key("owned")),
    ("regenerate_key", lambda c: c.regenerate_key("owned")),
    ("reset_key_spend", lambda c: c.reset_key_spend("owned", 0)),
    ("key_list", lambda c: c.key_list("owned")),
    ("key_alias_count", lambda c: c.key_alias_count("owned")),
    ("create_team", lambda c: c.create_team(TeamNewBody(team_alias="owned"))),
    ("update_team", lambda c: c.update_team(TeamUpdateBody(team_id="owned", team_alias="updated"))),
    ("delete_team", lambda c: c.delete_team("owned")),
    ("team_info", lambda c: c.team_info("owned")),
    ("team_list_ids", lambda c: c.team_list_ids()),
    ("team_info_status", lambda c: c.team_info_status("owned")),
    ("add_team_member", lambda c: c.add_team_member("owned", "user")),
    ("delete_team_member", lambda c: c.delete_team_member("owned", "user")),
    ("create_user", lambda c: c.create_user(UserNewBody(user_email="actor@example.com", user_role="internal_user"))),
    ("create_customer", lambda c: c.create_customer("owned")),
    ("customer_info", lambda c: c.customer_info("owned")),
    ("delete_customer", lambda c: c.delete_customer("owned")),
    ("update_user", lambda c: c.update_user(UserUpdateBody(user_id="owned", user_role="internal_user"))),
    ("delete_user", lambda c: c.delete_user("owned")),
    ("delete_user_strict", lambda c: c.delete_user_strict("owned")),
    ("user_info", lambda c: c.user_info("owned")),
    ("user_count", lambda c: c.user_count("owned")),
    ("user_list_ids", lambda c: c.user_list_ids("owned")),
    ("create_org", lambda c: c.create_org(OrgNewBody(organization_alias="owned"))),
    ("update_org", lambda c: c.update_org(OrgUpdateBody(organization_id="owned", organization_alias="updated"))),
    ("delete_org", lambda c: c.delete_org("owned")),
    ("org_info", lambda c: c.org_info("owned")),
    ("org_info_status", lambda c: c.org_info_status("owned")),
    ("create_tag", lambda c: c.create_tag(TagNewBody(name="owned"))),
    ("delete_tag", lambda c: c.delete_tag("owned")),
    ("tag_list", lambda c: c.tag_list()),
    ("create_mcp_server", lambda c: c.create_mcp_server(McpServerCreateBody(alias="owned", url="http://example.test"))),
    ("update_mcp_server", lambda c: c.update_mcp_server(McpServerUpdateBody(server_id="owned", alias=None))),
    ("delete_mcp_server", lambda c: c.delete_mcp_server("owned")),
    ("proxy.generate_key", lambda c: c.proxy.generate_key(KeyGenerateBody())),
    ("proxy.delete_key", lambda c: c.proxy.delete_key("owned")),
    ("proxy.delete_customers", lambda c: c.proxy.delete_customers(["owned"])),
    ("proxy.key_info", lambda c: c.proxy.key_info("owned")),
    ("proxy.memory_summary", lambda c: c.proxy.memory_summary_everywhere()),
    ("proxy.model_info", lambda c: c.proxy.model_info()),
    ("proxy.model_cost_map", lambda c: c.proxy.model_cost_map()),
    ("proxy.create_model", lambda c: c.proxy.create_model("owned", LiteLLMParamsBody(model="synthetic"))),
    ("proxy.update_model", lambda c: c.proxy.update_model("owned", LiteLLMParamsBody(model="synthetic"))),
    ("proxy.delete_model", lambda c: c.proxy.delete_model("owned")),
    ("proxy.create_toolset", lambda c: c.proxy.create_toolset(ToolsetCreateBody(toolset_name="owned", tools=[]))),
    ("proxy.update_toolset", lambda c: c.proxy.update_toolset(ToolsetUpdateBody(toolset_id="owned", description=None))),
    ("proxy.delete_toolset", lambda c: c.proxy.delete_toolset("owned")),
    (
        "proxy.create_credential",
        lambda c: c.proxy.create_credential(CredentialCreateBody(credential_name="owned", credential_values={})),
    ),
    ("proxy.delete_credential", lambda c: c.proxy.delete_credential("owned")),
    ("proxy.create_team", lambda c: c.proxy.create_team(TeamNewBody(team_alias="owned"))),
    ("proxy.delete_team", lambda c: c.proxy.delete_team("owned")),
    ("proxy.delete_user", lambda c: c.proxy.delete_user("owned")),
    ("proxy.spend_logs", lambda c: c.proxy.spend_logs(SpendLogsParams(api_key="owned"))),
    ("proxy.probe", lambda c: c.proxy.probe("/user/info", params=NoBody())),
)


@pytest.mark.parametrize(
    ("name", "operation"), MANAGEMENT_OPERATIONS, ids=tuple(name for name, _ in MANAGEMENT_OPERATIONS)
)
@pytest.mark.parametrize("kind", ("master", "direct_jwt", "virtual_key", "dashboard_session"))
def test_management_operations_send_the_selected_credential(
    name: str,
    operation: Callable[[ManagementClient], object],
    kind: CredentialKind,
) -> None:
    with caller_boundary(status=401) as (bootstrap, received), without_retries():
        client: Final = (
            bootstrap
            if kind == "master"
            else bootstrap.with_caller(Caller(credential=f"synthetic-{kind}", kind=kind, role="internal_user"))
        )
        try:
            operation(client)
        except AssertionError:
            pass
        expected: Final = "Bearer bootstrap" if kind == "master" else f"Bearer synthetic-{kind}"
        assert received.get_nowait() == expected, name
        assert received.empty(), "an unauthorized request must not be retried"


class TestSplitCallerPropagation:
    def test_control_and_data_replica_readers_keep_the_caller(self) -> None:
        with caller_boundary() as (data, data_headers), caller_boundary() as (control, control_headers):
            data_url: Final = next(iter(data.proxy.replicas))
            control_url: Final = next(iter(control.proxy.replicas))
            proxy: Final = build_proxy_client(
                base_url=data_url,
                control_plane_base_url=control_url,
                replica_urls=(data_url,),
                master_key="bootstrap",
            ).with_caller(Caller(credential="tenant-token", kind="direct_jwt", role="team_member"))
            proxy.key_info("owned")
            proxy.read_body_back_everywhere(
                "/key/info", KeyInfoResponse, settled=lambda info: info.info.key_alias == "owned"
            )
            proxy.read_back_everywhere(
                "/key/info",
                params=NoBody(),
                response_type=KeyInfoResponse,
                converged=lambda result: isinstance(result, Success),
            )
            assert control_headers.get_nowait() == "Bearer tenant-token"
            assert control_headers.get_nowait() == "Bearer tenant-token"
            assert data_headers.get_nowait() == "Bearer tenant-token"
            assert control_headers.empty() and data_headers.empty()

    def test_successful_team_and_model_polling_uses_the_bound_caller(self) -> None:
        with caller_boundary() as (bootstrap, received):
            bound: Final = bootstrap.with_caller(Caller(credential="caller", kind="direct_jwt", role="proxy_admin"))
            bound.create_team(TeamNewBody(team_alias="owned"))
            bound.proxy.create_model("owned", LiteLLMParamsBody(model="synthetic"))
            assert tuple(received.get_nowait() for _ in range(4)) == ("Bearer caller",) * 4
            assert received.empty()

    def test_expired_shaped_token_is_sent_once_without_renewal(self) -> None:
        with caller_boundary(status=401) as (bootstrap, received):
            bound: Final = bootstrap.with_caller(
                Caller(credential="expired.payload.signature", kind="direct_jwt", role="internal_user")
            )
            result: Final = bound.key_info_as("owned")
            assert not isinstance(result, Success)
            assert received.get_nowait() == "Bearer expired.payload.signature"
            assert received.empty()


@pytest.mark.parametrize("operation", ("server", "toolset"))
def test_partial_updates_preserve_explicit_null_at_the_http_boundary(operation: str) -> None:
    bodies: Final[SimpleQueue[bytes]] = SimpleQueue()
    with caller_boundary(status=401, bodies=bodies) as (bootstrap, _):
        try:
            if operation == "server":
                bootstrap.update_mcp_server(McpServerUpdateBody(server_id="owned", alias=None))
            else:
                bootstrap.proxy.update_toolset(ToolsetUpdateBody(toolset_id="owned", description=None))
        except AssertionError:
            pass
        expected: Final = (
            {"server_id": "owned", "alias": None}
            if operation == "server"
            else {"toolset_id": "owned", "description": None}
        )
        assert json.loads(bodies.get_nowait()) == expected
        assert bodies.empty()
