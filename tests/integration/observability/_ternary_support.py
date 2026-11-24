"""Owned Ternary peers for the FOCUS export contracts: the cost-import API and the signed-URL storage host.

The API answers `upload-url` with a signed `https://storage.googleapis.com/...` PUT target, the way Ternary
does. The proxy only PUTs to an HTTPS GCS host, so the storage peer serves TLS under that name and the
proxy reaches it through an owned tunnelling forward proxy that trusts the peer's own certificate.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import psycopg
import yaml
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from psycopg import sql

TERNARY_API_KEY: Final = "synthetic-ternary-api-key"
CONNECTION_ID: Final = "ternary-integration-connection"
GCS_HOST: Final = "storage.googleapis.com"
BUCKET: Final = "ternary-integration-bucket"
UPLOAD_URL_PATH: Final = f"/external-cost-sources/v1/{CONNECTION_ID}/upload-url"
SIGNED_HEADERS: Final = MappingProxyType(
    {
        "Content-Encoding": "identity",
        "Content-Type": "text/csv",
        "x-goog-content-length-range": "0,67108864",
        "x-goog-meta-ecs-connection-id": CONNECTION_ID,
        "x-goog-meta-ecs-provider": "litellm",
    }
)
_SCRUBBED: Final = frozenset(
    {"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE"}
)


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def signed_target(day: str) -> str:
    return f"/{BUCKET}/{CONNECTION_ID}/{day}.csv?X-Goog-Signature=synthetic-{day}"


@dataclass(frozen=True, slots=True)
class Upload:
    day: str
    headers: Mapping[str, str]
    body: bytes

    def rows(self) -> tuple[Mapping[str, str], ...]:
        return tuple(MappingProxyType(row) for row in csv.DictReader(io.StringIO(self.body.decode("utf-8"))))

    def cost_by_service(self) -> Mapping[str, float]:
        return MappingProxyType({row["ServiceName"]: float(row["BilledCost"]) for row in self.rows()})


@dataclass(slots=True)
class TernaryPeers:
    api: Wire
    storage: Wire
    environment: Mapping[str, str]
    _upload_url_requests: list[Request] = field(default_factory=list)  # mutable-ok: drain() consumes the wire queue
    _puts: list[Request] = field(default_factory=list)  # mutable-ok: drain() consumes the wire queue

    def upload_url_requests(self) -> tuple[Request, ...]:
        self._upload_url_requests.extend(self.api.drain())
        return tuple(self._upload_url_requests)

    def puts(self) -> tuple[Request, ...]:
        self._puts.extend(self.storage.drain())
        return tuple(self._puts)

    def uploads(self) -> tuple[Upload, ...]:
        return tuple(
            Upload(day=urlsplit(put.target).path.rsplit("/", 1)[-1].removesuffix(".csv"), headers=put.headers, body=put.body)
            for put in self.puts()
            if put.method == "PUT"
        )

    def latest_by_day(self) -> Mapping[str, Upload]:
        return MappingProxyType({upload.day: upload for upload in self.uploads()})

    def outbound(self) -> int:
        return len(self.upload_url_requests()) + len(self.puts())


def _api(request: Request) -> Reply:
    if (request.method, request.target) != ("POST", UPLOAD_URL_PATH):
        return Reply(status=404, body=b'{"message": "unexpected Ternary request"}')
    if request.headers.get("authorization") != f"Bearer {TERNARY_API_KEY}":
        return Reply(status=401, body=b'{"message": "invalid secret"}')
    day: Final = str(json.loads(request.body)["day"])
    return Reply(
        body=json.dumps(
            {
                "url": f"https://{GCS_HOST}{signed_target(day)}",
                "method": "PUT",
                "headers": dict(SIGNED_HEADERS),
                "expiresAt": (datetime.now(timezone.utc) + timedelta(seconds=899)).isoformat(),
                "day": day,
            }
        ).encode()
    )


def _storage(request: Request) -> Reply:
    if request.method != "PUT" or not request.target.startswith(f"/{BUCKET}/{CONNECTION_ID}/"):
        return Reply(status=404, body=b"<Error><Code>NoSuchKey</Code></Error>", content_type="application/xml")
    return Reply(body=b"", content_type="text/plain")


@contextmanager
def ternary_peers(directory: Path) -> Iterator[TernaryPeers]:
    cert, key = write_self_signed_cert(directory, (GCS_HOST,))
    with (
        wire_server(_storage, tls=server_context(cert, key)) as storage,
        tunnelling_forward_proxy(int(storage.url.rsplit(":", 1)[1])) as tunnel,
        wire_server(_api) as api,
    ):
        yield TernaryPeers(
            api=api,
            storage=storage,
            environment=MappingProxyType(
                {
                    "TERNARY_API_KEY": TERNARY_API_KEY,
                    "TERNARY_CONNECTION_ID": CONNECTION_ID,
                    "TERNARY_BASE_URL": api.url,
                    "HTTPS_PROXY": tunnel.url,
                    "NO_PROXY": "127.0.0.1,localhost",
                    "SSL_CERT_FILE": str(cert),
                }
            ),
        )


def scrubbed_environment(inherited: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(name for name in inherited if name.upper() in _SCRUBBED)


def callbacks_config(directory: Path, callbacks: tuple[str, ...]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["callbacks"] = list(callbacks)
    path: Final = directory / "ternary-callbacks.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def set_database_timezone(database_url: str, zone: str) -> None:
    name: Final = urlsplit(database_url).path.lstrip("/")
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(sql.SQL("ALTER DATABASE {} SET timezone TO {}").format(sql.Identifier(name), sql.Literal(zone)))


@dataclass(frozen=True, slots=True)
class SpendRow:
    """One `LiteLLM_DailyUserSpend` row; `service` becomes the FOCUS `ServiceName` (the model group)."""

    day: date
    service: str
    spend: float
    updated_at: datetime
    prompt_tokens: int = 7
    completion_tokens: int = 3


def seed_daily_spend(database_url: str, rows: tuple[SpendRow, ...]) -> None:
    """Insert every row in one transaction so no export run can read a partial seed."""
    with psycopg.connect(database_url) as connection, connection.transaction():
        for row in rows:
            naive_utc: Final = row.updated_at.astimezone(timezone.utc).replace(tzinfo=None)
            connection.execute(
                """
                INSERT INTO "LiteLLM_DailyUserSpend"
                    (id, user_id, date, api_key, model, model_group, custom_llm_provider,
                     prompt_tokens, completion_tokens, spend, api_requests, successful_requests,
                     failed_requests, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'openai', %s, %s, %s, 1, 1, 0, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    "ternary-integration-user",
                    row.day.isoformat(),
                    f"ternary-integration-key-{row.service}",
                    "gpt-4o-mini",
                    row.service,
                    row.prompt_tokens,
                    row.completion_tokens,
                    row.spend,
                    naive_utc,
                    naive_utc,
                ),
            )


@contextmanager
def refreshing_updated_at(database_url: str, service: str) -> Iterator[None]:
    """Keep a seeded row's `updated_at` at the current UTC instant, as ongoing traffic would."""
    stop: Final = threading.Event()

    def refresh() -> None:
        with psycopg.connect(database_url, autocommit=True) as connection:
            while not stop.wait(0.2):
                connection.execute(
                    """UPDATE "LiteLLM_DailyUserSpend" SET updated_at = (now() AT TIME ZONE 'UTC') WHERE model_group = %s""",
                    (service,),
                )

    worker: Final = threading.Thread(target=refresh, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=5)
        assert not worker.is_alive(), "updated_at refresher survived cleanup"
