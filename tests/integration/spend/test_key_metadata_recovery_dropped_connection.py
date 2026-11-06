import csv
import io
import json
import re
import socket
import socketserver
import threading
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, object_value, string_value
from integration._support.daily_activity import (
    DAY,
    ROUTES,
    TEAM_SPEND,
    USER_SPEND,
    Route,
    activity_of_key,
    assert_key_reported,
    daily_rows,
    key_metadata,
    key_no_key_table_holds,
    seeded_metrics,
    seeded_row,
    user_row,
)
from integration._support.database import scratch_database
from integration._support.database_relay import dropped_connection_relay
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

WORKERS: Final = 2
PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
REVERSE_HASH_TRIGGER: Final = b"encode(sha256(convert_to(token, 'UTF8')), 'hex') = ANY("
OWNER_RECOVERY_TRIGGER: Final = b"MIN(user_id) AS first_owner"
CLOUDZERO_HOST: Final = "api.cloudzero.com"
CLOUDZERO_AUTHORITY: Final = f"{CLOUDZERO_HOST}:443"
EXPORT_FILENAME: Final = re.compile(r"usage_\d{8}T\d{6}Z_\d{8}T\d{6}Z\.csv")
FILENAME_STAMP: Final = "%Y%m%dT%H%M%SZ"
VANTAGE_DONE: Final = "Vantage export completed successfully"
CLOUDZERO_DONE: Final = "CloudZero export completed successfully"
NO_ENVIRONMENT: Final[Mapping[str, str]] = MappingProxyType({})
ENTITY_TABLES: Final = tuple(
    dict.fromkeys((route.table, route.entity_column) for route in ROUTES if route.table != USER_SPEND)
)
FOCUS_ALIAS_FIELDS: Final = frozenset({"BillingAccountName", "Tags"})
CBF_ALIAS_FIELDS: Final = frozenset({"resource/account", "resource/tag:api_key_alias"})


@dataclass(frozen=True, slots=True)
class Owned:
    owner: str
    email: str
    alias: str
    key: str
    token: str
    double: str


@dataclass(frozen=True, slots=True)
class Sink:
    forbidden: threading.Event
    missing: threading.Event

    def respond(self, request: Request) -> Reply:
        if self.forbidden.is_set():
            return Reply(status=403, body=b'{"error":"forbidden"}')
        if self.missing.is_set():
            return Reply(status=404, body=b'{"error":"missing"}')
        return Reply()


@dataclass(frozen=True, slots=True)
class Upload:
    target: str
    authorization: str
    filename: str
    row: Mapping[str, str]


def _identity(label: str) -> Owned:
    stamp: Final = uuid.uuid4().hex[:8]
    key: Final = f"sk-{uuid.uuid4().hex}"
    token: Final = sha256(key.encode()).hexdigest()
    return Owned(
        owner=f"{label}-owner-{stamp}",
        email=f"{label}-{uuid.uuid4().hex[:8]}@example.com",
        alias=f"{label}-key-{stamp}",
        key=key,
        token=token,
        double=sha256(token.encode()).hexdigest(),
    )


def _register(proxy: Gateway, owned: Owned) -> None:
    proxy.post("/user/new", {"user_id": owned.owner, "user_email": owned.email, "auto_create_key": False})
    proxy.post("/key/generate", {"key": owned.key, "user_id": owned.owner, "key_alias": owned.alias})


@contextmanager
def _relayed_proxy(
    gateway: Gateway,
    directory: Path,
    relayed_url: str,
    environment: Mapping[str, str] = NO_ENVIRONMENT,
    config: Path | None = None,
) -> Generator[OwnedProxy]:
    with owned_proxy_process(
        gateway,
        directory,
        {
            "DATABASE_URL": relayed_url,
            "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
            "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
            **environment,
        },
        config=config,
        remove_environment=("DATABASE_URL_READ_REPLICA",),
        workers=WORKERS,
    ) as owned:
        yield owned


@contextmanager
def _reader(owned: OwnedProxy) -> Generator[Gateway]:
    with httpx.Client(base_url=str(owned.gateway.client.base_url), timeout=90, trust_env=False) as client:
        yield Gateway(client, owned.gateway.key, owned.gateway.upstream_url)


def _filters(route: Route, entity: str) -> dict[str, str]:
    return {} if route.entity_filter is None else {route.entity_filter: entity}


def _config_allowing_a_base_url_in_the_body(directory: Path) -> Path:
    config: Final = directory / "client_side_credentials.yaml"
    config.write_text(
        PROXY_CONFIG.read_text().replace(
            "general_settings:\n", "general_settings:\n  allow_client_side_credentials: true\n", 1
        )
    )
    return config


def _records(value: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    assert isinstance(value, list), value
    return tuple(object_value(item) for item in value)


def _first(body: Mapping[str, JsonValue], name: str) -> dict[str, JsonValue]:
    records: Final = _records(body[name])
    assert len(records) == 1, body
    return records[0]


def _without(record: Mapping[str, JsonValue], names: frozenset[str]) -> dict[str, JsonValue]:
    return {name: value for name, value in record.items() if name not in names}


def _tags(record: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(json.loads(string_value(record["Tags"])))


def _dry_run(reader: Gateway, path: str) -> dict[str, JsonValue]:
    return object_value(reader.post(path, {})["dry_run_data"])


def _focus_tags(owned: Owned, *, alias: bool) -> dict[str, str]:
    return {
        **({"api_key_alias": owned.alias} if alias else {}),
        "user_id": owned.owner,
        "user_email": owned.email,
        "model": "gpt-4o-mini",
        "model_group": "gpt-4o-mini",
        "custom_llm_provider": "openai",
    }


def _assert_vantage_dry_run(body: Mapping[str, JsonValue], owned: Owned, *, alias: str | None) -> None:
    usage: Final = _first(body, "usage_data")
    assert {name: usage.get(name) for name in ("api_key", "api_key_alias", "user_id", "user_email", "spend")} == {
        "api_key": owned.double,
        "api_key_alias": alias,
        "user_id": owned.owner,
        "user_email": owned.email,
        "spend": 0.25,
    }, body
    focus: Final = _first(body, "normalized_data")
    assert {
        name: focus.get(name)
        for name in ("BillingAccountName", "BillingAccountId", "BilledCost", "ChargePeriodStart", "ChargePeriodEnd")
    } == {
        "BillingAccountName": alias,
        "BillingAccountId": owned.double,
        "BilledCost": 0.25,
        "ChargePeriodStart": "2026-02-03T00:00:00Z",
        "ChargePeriodEnd": "2026-02-04T00:00:00Z",
    }, body
    assert _tags(focus) == _focus_tags(owned, alias=alias is not None), body


def _assert_cloudzero_dry_run(body: Mapping[str, JsonValue], owned: Owned, *, alias: str | None) -> None:
    usage: Final = _first(body, "usage_data")
    assert {name: usage.get(name) for name in ("api_key", "api_key_alias", "user_id", "user_email")} == {
        "api_key": owned.double,
        "api_key_alias": alias,
        "user_id": owned.owner,
        "user_email": owned.email,
    }, body
    cbf: Final = _first(body, "cbf_data")
    assert {name: cbf.get(name) for name in ("resource/account", "resource/tag:api_key_alias", "cost/cost")} == {
        "resource/account": f"{alias}|{owned.double[:8]}" if alias else owned.double[:8],
        "resource/tag:api_key_alias": str(alias),
        "cost/cost": 0.25,
    }, body


def _cbf_record(owned: Owned, *, alias: str | None) -> dict[str, str]:
    prefix: Final = owned.double[:8]
    return {
        "time/usage_start": "2026-02-03T00:00:00Z",
        "cost/cost": "0.25",
        "resource/id": "czrn:litellm:openai:cross-region:unknown:llm-usage:gpt-4o-mini",
        "usage/amount": "15",
        "usage/units": "tokens",
        "resource/service": "gpt-4o-mini",
        "resource/account": f"{alias}|{prefix}" if alias else prefix,
        "resource/region": "cross-region",
        "resource/usage_family": "openai",
        "action/operation": "",
        "lineitem/type": "Usage",
        "resource/tag:provider": "openai",
        "resource/tag:model": "gpt-4o-mini",
        "resource/tag:entity_type": "team",
        "resource/tag:model_group": "gpt-4o-mini",
        "resource/tag:api_key_prefix": prefix,
        "resource/tag:api_key_alias": str(alias),
        "resource/tag:user_email": owned.email,
        "resource/tag:api_requests": "1",
        "resource/tag:successful_requests": "1",
        "resource/tag:failed_requests": "0",
        "resource/tag:cache_creation_tokens": "0",
        "resource/tag:cache_read_tokens": "0",
        "resource/tag:prompt_tokens": "10",
        "resource/tag:completion_tokens": "5",
    }


def _window() -> tuple[datetime, datetime]:
    now: Final = datetime.now(UTC).replace(microsecond=0)
    return now - timedelta(hours=1), now + timedelta(hours=1)


def _window_body(window: tuple[datetime, datetime]) -> dict[str, JsonValue]:
    return {"start_time_utc": window[0].isoformat(), "end_time_utc": window[1].isoformat()}


def _window_filename(window: tuple[datetime, datetime]) -> str:
    return f"usage_{window[0].strftime(FILENAME_STAMP)}_{window[1].strftime(FILENAME_STAMP)}.csv"


def _last_received(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert received, "The sink received nothing"
    return received[-1]


def _csv_part(request: Request) -> Message:
    message: Final = BytesParser(policy=HTTP).parsebytes(
        b"content-type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.body
    )
    parts: Final = message.get_payload()
    assert isinstance(parts, list) and len(parts) == 1, request.body
    part: Final = parts[0]
    assert isinstance(part, Message), request.body
    assert part.get_param("name", header="content-disposition") == "csv", request.body
    return part


def _upload(wire: Wire) -> Upload:
    request: Final = _last_received(wire)
    part: Final = _csv_part(request)
    payload: Final = part.get_payload(decode=True)
    assert isinstance(payload, bytes), request.body
    rows: Final = tuple(csv.DictReader(io.StringIO(payload.decode())))
    assert len(rows) == 1, payload
    filename: Final = part.get_filename()
    assert filename is not None, request.body
    return Upload(request.target, request.headers["authorization"], filename, rows[0])


def _assert_csv_row(row: Mapping[str, str], owned: Owned, *, alias: str | None) -> None:
    assert {
        name: row.get(name)
        for name in (
            "BilledCost",
            "BillingAccountId",
            "BillingAccountName",
            "ChargeDescription",
            "ChargePeriodStart",
            "ChargePeriodEnd",
        )
    } == {
        "BilledCost": "0.25",
        "BillingAccountId": owned.double,
        "BillingAccountName": alias or "",
        "ChargeDescription": "gpt-4o-mini",
        "ChargePeriodStart": "2026-02-03T00:00:00Z",
        "ChargePeriodEnd": "2026-02-04T00:00:00Z",
    }, row
    assert json.loads(row["Tags"]) == _focus_tags(owned, alias=alias is not None), row


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@contextmanager
def _connect_tunnel(destination: Wire, authority: str) -> Generator[str]:
    destination_port: Final = int(destination.url.rsplit(":", 1)[1])

    class Tunnel(socketserver.StreamRequestHandler):
        rbufsize = 0
        request: socket.socket

        def handle(self) -> None:
            requested: Final = self.rfile.readline().decode().split()[1]
            while self.rfile.readline() not in (b"\r\n", b""):
                pass
            if requested != authority:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.request.settimeout(10)
            with socket.create_connection(("127.0.0.1", destination_port), timeout=10) as upstream:
                outbound: Final = threading.Thread(target=_pipe, args=(self.request, upstream))
                outbound.start()
                _pipe(upstream, self.request)
                outbound.join(timeout=12)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Tunnel) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=6)


@pytest.mark.timeout(600)
def test_every_usage_route_survives_a_dropped_database_connection_during_reverse_hash_recovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    owned_key: Final = _identity("dropped-reverse-hash")
    entity: Final = f"dropped-reverse-hash-{uuid.uuid4().hex[:8]}"
    rows: Final = (
        user_row(None, owned_key.double, DAY),
        *(seeded_row(table, column, entity, owned_key.double, DAY) for table, column in ENTITY_TABLES),
    )
    with (
        scratch_database() as database_url,
        dropped_connection_relay(database_url, REVERSE_HASH_TRIGGER) as (relay, relayed_url),
        _relayed_proxy(gateway, tmp_path, relayed_url) as owned,
        _reader(owned) as reader,
        daily_rows(rows, database_url=database_url),
    ):
        _register(owned.gateway, owned_key)
        relay.arm()
        for route in ROUTES:
            relay.dropped.clear()
            dropped: Final = activity_of_key(reader, route.path, owned_key.double, **_filters(route, entity))
            assert relay.dropped.is_set(), f"{route.path}: {dropped.text}"
            assert_key_reported(dropped, owned_key.double, DAY, key_metadata(), seeded_metrics(1))
        relay.disarm()
        named: Final = key_metadata(alias=owned_key.alias, user=owned_key.owner, email=owned_key.email)
        for route in ROUTES:
            recovered: Final = activity_of_key(reader, route.path, owned_key.double, **_filters(route, entity))
            assert_key_reported(recovered, owned_key.double, DAY, named, seeded_metrics(1))


@pytest.mark.timeout(300)
def test_usage_page_survives_a_dropped_database_connection_during_user_detail_recovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    owned_key: Final = _identity("dropped-user-detail")
    with (
        scratch_database() as database_url,
        dropped_connection_relay(database_url, owned_key.owner.encode()) as (relay, relayed_url),
        _relayed_proxy(gateway, tmp_path, relayed_url) as owned,
        _reader(owned) as reader,
        daily_rows((user_row(None, owned_key.token, DAY),), database_url=database_url),
    ):
        _register(owned.gateway, owned_key)
        relay.arm()
        dropped: Final = activity_of_key(reader, "/user/daily/activity", owned_key.token)
        assert relay.dropped.is_set(), dropped.text
        assert_key_reported(
            dropped,
            owned_key.token,
            DAY,
            key_metadata(alias=owned_key.alias, user=owned_key.owner, exists=True),
            seeded_metrics(1),
        )
        relay.disarm()
        recovered: Final = activity_of_key(reader, "/user/daily/activity", owned_key.token)
        assert_key_reported(
            recovered,
            owned_key.token,
            DAY,
            key_metadata(alias=owned_key.alias, user=owned_key.owner, email=owned_key.email, exists=True),
            seeded_metrics(1),
        )


@pytest.mark.timeout(300)
def test_team_usage_page_survives_a_dropped_database_connection_during_owner_recovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    owner: Final = f"dropped-owner-{uuid.uuid4().hex[:8]}"
    email: Final = f"owner-{uuid.uuid4().hex[:8]}@example.com"
    team: Final = f"dropped-owner-team-{uuid.uuid4().hex[:8]}"
    api_key: Final = key_no_key_table_holds()
    rows: Final = (user_row(owner, api_key, DAY), seeded_row(TEAM_SPEND, "team_id", team, api_key, DAY))
    with (
        scratch_database() as database_url,
        dropped_connection_relay(database_url, OWNER_RECOVERY_TRIGGER) as (relay, relayed_url),
        _relayed_proxy(gateway, tmp_path, relayed_url) as owned,
        _reader(owned) as reader,
        daily_rows(rows, database_url=database_url),
    ):
        owned.gateway.post("/user/new", {"user_id": owner, "user_email": email, "auto_create_key": False})
        relay.arm()
        dropped: Final = activity_of_key(reader, "/team/daily/activity", api_key, team_ids=team)
        assert relay.dropped.is_set(), dropped.text
        assert_key_reported(dropped, api_key, DAY, key_metadata(), seeded_metrics(1))
        relay.disarm()
        recovered: Final = activity_of_key(reader, "/team/daily/activity", api_key, team_ids=team)
        assert_key_reported(recovered, api_key, DAY, key_metadata(user=owner, email=email), seeded_metrics(1))


@pytest.mark.timeout(300)
def test_vantage_and_cloudzero_dry_runs_survive_a_dropped_database_connection_during_reverse_hash_recovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    owned_key: Final = _identity("dropped-dry-run")
    with (
        scratch_database() as database_url,
        dropped_connection_relay(database_url, owned_key.double.encode()) as (relay, relayed_url),
        _relayed_proxy(gateway, tmp_path, relayed_url) as owned,
        _reader(owned) as reader,
        daily_rows((user_row(owned_key.owner, owned_key.double, DAY),), database_url=database_url),
    ):
        _register(owned.gateway, owned_key)
        relay.arm()
        vantage_dropped: Final = _dry_run(reader, "/vantage/dry-run")
        assert relay.dropped.is_set(), vantage_dropped
        _assert_vantage_dry_run(vantage_dropped, owned_key, alias=None)
        relay.dropped.clear()
        cloudzero_dropped: Final = _dry_run(reader, "/cloudzero/dry-run")
        assert relay.dropped.is_set(), cloudzero_dropped
        _assert_cloudzero_dry_run(cloudzero_dropped, owned_key, alias=None)
        relay.disarm()
        vantage: Final = _dry_run(reader, "/vantage/dry-run")
        _assert_vantage_dry_run(vantage, owned_key, alias=owned_key.alias)
        cloudzero: Final = _dry_run(reader, "/cloudzero/dry-run")
        _assert_cloudzero_dry_run(cloudzero, owned_key, alias=owned_key.alias)
        assert _without(_first(vantage_dropped, "normalized_data"), FOCUS_ALIAS_FIELDS) == _without(
            _first(vantage, "normalized_data"), FOCUS_ALIAS_FIELDS
        )
        assert _without(_first(cloudzero_dropped, "cbf_data"), CBF_ALIAS_FIELDS) == _without(
            _first(cloudzero, "cbf_data"), CBF_ALIAS_FIELDS
        )


@pytest.mark.timeout(300)
def test_vantage_and_cloudzero_dry_runs_survive_a_dropped_database_connection_during_user_detail_recovery(
    gateway: Gateway, tmp_path: Path
) -> None:
    owned_key: Final = _identity("dropped-dry-run-detail")
    with (
        scratch_database() as database_url,
        dropped_connection_relay(database_url, owned_key.owner.encode()) as (relay, relayed_url),
        _relayed_proxy(gateway, tmp_path, relayed_url) as owned,
        _reader(owned) as reader,
        daily_rows((user_row(owned_key.owner, owned_key.double, DAY),), database_url=database_url),
    ):
        _register(owned.gateway, owned_key)
        relay.arm()
        vantage_dropped: Final = _dry_run(reader, "/vantage/dry-run")
        assert relay.dropped.is_set(), vantage_dropped
        _assert_vantage_dry_run(vantage_dropped, owned_key, alias=owned_key.alias)
        relay.dropped.clear()
        cloudzero_dropped: Final = _dry_run(reader, "/cloudzero/dry-run")
        assert relay.dropped.is_set(), cloudzero_dropped
        _assert_cloudzero_dry_run(cloudzero_dropped, owned_key, alias=owned_key.alias)
        relay.disarm()
        vantage: Final = _dry_run(reader, "/vantage/dry-run")
        cloudzero: Final = _dry_run(reader, "/cloudzero/dry-run")
        assert _first(vantage_dropped, "normalized_data") == _first(vantage, "normalized_data")
        assert _first(cloudzero_dropped, "cbf_data") == _first(cloudzero, "cbf_data")


@pytest.mark.timeout(420)
def test_vantage_export_delivers_a_blank_alias_under_a_dropped_database_connection_and_reports_sink_errors(
    gateway: Gateway, tmp_path: Path
) -> None:
    owned_key: Final = _identity("dropped-vantage-export")
    sink: Final = Sink(threading.Event(), threading.Event())
    api_key: Final = f"vantage-api-key-{uuid.uuid4().hex}"
    integration_token: Final = f"vantage-token-{uuid.uuid4().hex}"
    costs_target: Final = f"/v2/integrations/{integration_token}/costs.csv"
    with (
        scratch_database() as database_url,
        dropped_connection_relay(database_url, owned_key.double.encode()) as (relay, relayed_url),
        wire_server(sink.respond) as wire,
        _relayed_proxy(
            gateway, tmp_path, relayed_url, config=_config_allowing_a_base_url_in_the_body(tmp_path)
        ) as owned,
        _reader(owned) as reader,
        daily_rows((user_row(owned_key.owner, owned_key.double, DAY),), database_url=database_url),
    ):
        _register(owned.gateway, owned_key)
        owned.gateway.post(
            "/vantage/init", {"api_key": api_key, "integration_token": integration_token, "base_url": wire.url}
        )
        relay.arm()
        dropped: Final = reader.post("/vantage/export", {})
        assert relay.dropped.is_set(), dropped
        assert dropped["message"] == VANTAGE_DONE, dropped
        blank: Final = _upload(wire)
        assert (blank.target, blank.authorization) == (costs_target, f"Bearer {api_key}"), blank
        assert EXPORT_FILENAME.fullmatch(blank.filename), blank
        _assert_csv_row(blank.row, owned_key, alias=None)
        relay.dropped.clear()
        window: Final = _window()
        windowed: Final = reader.post("/vantage/export", _window_body(window))
        assert relay.dropped.is_set(), windowed
        assert windowed["message"] == VANTAGE_DONE, windowed
        bounded: Final = _upload(wire)
        assert bounded.filename == _window_filename(window), bounded
        _assert_csv_row(bounded.row, owned_key, alias=None)
        relay.disarm()
        healthy: Final = reader.post("/vantage/export", {})
        assert healthy["message"] == VANTAGE_DONE, healthy
        named: Final = _upload(wire)
        _assert_csv_row(named.row, owned_key, alias=owned_key.alias)
        assert _without(dict(blank.row), FOCUS_ALIAS_FIELDS) == _without(dict(named.row), FOCUS_ALIAS_FIELDS)
        sink.forbidden.set()
        forbidden: Final = reader.request("POST", "/vantage/export", {})
        assert forbidden.status_code == 500 and "Failed to perform Vantage export" in forbidden.text, forbidden.text
        assert _last_received(wire).target == costs_target
        sink.forbidden.clear()
        sink.missing.set()
        missing: Final = reader.request("POST", "/vantage/export", {})
        assert missing.status_code == 500 and "Failed to perform Vantage export" in missing.text, missing.text
        assert _last_received(wire).target == costs_target
        alive: Final = reader.request("GET", "/health/liveliness")
        assert alive.status_code == 200, alive.text


@pytest.mark.timeout(420)
def test_cloudzero_export_delivers_a_blank_alias_under_a_dropped_database_connection(
    gateway: Gateway, tmp_path: Path
) -> None:
    owned_key: Final = _identity("dropped-cloudzero-export")
    api_key: Final = f"cloudzero-api-key-{uuid.uuid4().hex}"
    connection_id: Final = f"cloudzero-connection-{uuid.uuid4().hex[:8]}"
    drops_target: Final = f"/v2/connections/billing/anycost/{connection_id}/billing_drops"
    cert, key = write_self_signed_cert(tmp_path, (CLOUDZERO_HOST,))
    with (
        scratch_database() as database_url,
        dropped_connection_relay(database_url, owned_key.double.encode()) as (relay, relayed_url),
        wire_server(lambda request: Reply(), tls=server_context(cert, key)) as wire,
        _connect_tunnel(wire, CLOUDZERO_AUTHORITY) as tunnel_url,
        _relayed_proxy(
            gateway, tmp_path, relayed_url, {"HTTPS_PROXY": tunnel_url, "SSL_CERT_FILE": str(cert)}
        ) as owned,
        _reader(owned) as reader,
        daily_rows((user_row(owned_key.owner, owned_key.double, DAY),), database_url=database_url),
    ):
        _register(owned.gateway, owned_key)
        owned.gateway.post("/cloudzero/init", {"api_key": api_key, "connection_id": connection_id, "timezone": "UTC"})
        relay.arm()
        dropped: Final = reader.post("/cloudzero/export", {})
        assert relay.dropped.is_set(), dropped
        assert dropped["message"] == CLOUDZERO_DONE, dropped
        blank: Final = _last_received(wire)
        assert (blank.method, blank.target) == ("POST", drops_target), blank
        assert blank.headers["authorization"] == f"Bearer {api_key}", blank.headers
        assert json.loads(blank.body) == {
            "month": "2026-02",
            "operation": "replace_hourly",
            "data": [_cbf_record(owned_key, alias=None)],
        }, blank.body
        relay.dropped.clear()
        windowed: Final = reader.post("/cloudzero/export", _window_body(_window()))
        assert relay.dropped.is_set(), windowed
        assert windowed["message"] == CLOUDZERO_DONE, windowed
        assert json.loads(_last_received(wire).body) == json.loads(blank.body)
        relay.disarm()
        healthy: Final = reader.post("/cloudzero/export", {})
        assert healthy["message"] == CLOUDZERO_DONE, healthy
        named: Final = _last_received(wire)
        assert json.loads(named.body) == {
            "month": "2026-02",
            "operation": "replace_hourly",
            "data": [_cbf_record(owned_key, alias=owned_key.alias)],
        }, named.body
