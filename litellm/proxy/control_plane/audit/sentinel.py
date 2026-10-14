"""Microsoft Sentinel sink through the Azure Monitor Logs Ingestion API: an Entra client-credentials token, then one
JSON array of flattened rows per POST to the data collection rule's stream, split at the API's 1 MB limit
(https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview)."""

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import accumulate, groupby
from typing import Final, Protocol, TypeAlias
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.proxy.control_plane.audit.ocsf import AuditRecord, section
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult, Undelivered
from litellm.proxy.control_plane.audit.sinks import ocsf_line
from litellm.proxy.control_plane.audit.transport import Poster
from litellm.proxy.control_plane.settings import SentinelSettings

API_VERSION: Final = "2023-01-01"
MAX_PAYLOAD_BYTES: Final = 1_000_000
FIXED_WIDTH_COLUMNS: Final = frozenset(
    {
        "TimeGenerated",
        "ClassUid",
        "ActivityId",
        "StatusId",
        "SeverityId",
        "ResponseCode",
        "Sequence",
        "Hash",
        "PrevHash",
        "TraceId",
    }
)
TOKEN_SCOPE: Final = "https://monitor.azure.com/.default"
REFRESH_MARGIN_SECONDS: Final = 60.0
Row: TypeAlias = Mapping[str, object]


class TokenSource(Protocol):
    async def __call__(self) -> str: ...


class TokenRefused(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status} from the token endpoint")
        self.status = status


class TokenReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    access_token: str
    expires_in: float = 3600.0


_TOKEN_REPLY: Final = TypeAdapter(TokenReply)


@dataclass(slots=True)
class ClientCredentials:
    """Entra ``client_credentials`` grant for the Azure Monitor scope. The secret only ever travels in the form body."""

    settings: SentinelSettings
    post: Poster
    now: Callable[[], float] = time.time
    token: str = ""
    expires_at: float = 0.0

    async def __call__(self) -> str:
        if self.token and self.now() < self.expires_at - REFRESH_MARGIN_SECONDS:
            return self.token
        form: Final = urlencode(
            {
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret.get_secret_value(),
                "scope": TOKEN_SCOPE,
                "grant_type": "client_credentials",
            }
        ).encode()
        url: Final = f"{self.settings.authority_host}/{self.settings.tenant_id}/oauth2/v2.0/token"
        response: Final = await self.post.post(
            url, form, {"Content-Type": "application/x-www-form-urlencoded"}, self.settings.timeout_seconds
        )
        if response.status_code != 200:
            raise TokenRefused(response.status_code)
        reply: Final = _TOKEN_REPLY.validate_json(response.content)
        self.token = reply.access_token
        self.expires_at = self.now() + reply.expires_in
        return self.token


def sentinel_row(record: AuditRecord) -> Row:
    """One flat row per record for a custom table, plus the whole OCSF object in ``Ocsf`` for KQL ``parse_json``."""
    event: Final = record.event
    body: Final = record.body
    litellm_block: Final = section(body, "unmapped", "litellm")
    return {
        "TimeGenerated": event.at.isoformat(),
        "ClassUid": body.get("class_uid"),
        "ClassName": body.get("class_name"),
        "ActivityId": body.get("activity_id"),
        "ActivityName": body.get("activity_name"),
        "StatusId": body.get("status_id"),
        "SeverityId": body.get("severity_id"),
        "ActorUid": event.actor,
        "ActorEmail": event.actor_email,
        "DataplaneId": event.dataplane_id,
        "HttpMethod": event.method,
        "UrlPath": event.path,
        "ResponseCode": event.status,
        "Decision": event.decision,
        "Detail": event.detail,
        "TraceId": event.trace_id,
        "Sequence": record.sequence,
        "Hash": record.hash,
        "PrevHash": litellm_block.get("prev_hash"),
        "Ocsf": ocsf_line(record),
    }


def _encoded(row: Row) -> bytes:
    return json.dumps(row, separators=(",", ":"), default=str).encode()


def _fitted(row: Row, limit: int) -> bytes:
    """A row that alone would exceed the request limit first loses ``Ocsf`` and ``Detail``, the free text, and if a flat
    column is still too long keeps only its fixed-width columns, so the batch can never be stuck behind one record."""
    whole: Final = _encoded(row)
    if len(whole) + 2 <= limit:
        return whole
    trimmed: Final = _encoded({**row, "Detail": None, "Ocsf": None, "Truncated": True})
    if len(trimmed) + 2 <= limit:
        return trimmed
    return _encoded({**{key: value for key, value in row.items() if key in FIXED_WIDTH_COLUMNS}, "Truncated": True})


def chunks(rows: Iterable[Row], limit: int = MAX_PAYLOAD_BYTES) -> tuple[bytes, ...]:
    """JSON arrays of rows, each at most ``limit`` bytes."""
    encoded: Final = tuple(_fitted(row, limit) for row in rows)
    sizes: Final = tuple(len(item) + 1 for item in encoded)
    fills: Final = tuple(accumulate(sizes, _step(limit), initial=(0, 1)))[1:]
    groups: Final = groupby(zip((chunk for chunk, _ in fills), encoded, strict=True), key=lambda pair: pair[0])
    return tuple(b"[" + b",".join(item for _, item in members) + b"]" for _, members in groups)


def _step(limit: int) -> Callable[[tuple[int, int], int], tuple[int, int]]:
    def step(fill: tuple[int, int], size: int) -> tuple[int, int]:
        chunk, used = fill
        return (chunk, used + size) if used + size <= limit else (chunk + 1, size + 1)

    return step


@dataclass(frozen=True, slots=True)
class SentinelSink:
    """Posts are at least once: a batch whose later chunk fails is retried whole, so Sentinel may hold a duplicate of
    the earlier chunks, deduplicable on ``Hash``."""

    settings: SentinelSettings
    post: Poster
    token: TokenSource
    name: str = "sentinel"

    @property
    def url(self) -> str:
        return (
            f"{self.settings.endpoint.rstrip('/')}/dataCollectionRules/{self.settings.dcr_immutable_id}"
            f"/streams/{self.settings.stream_name}?api-version={API_VERSION}"
        )

    async def deliver(self, batch: Batch) -> DeliveryResult:
        try:
            bearer: Final = await self.token()
        except TokenRefused as refused:
            return Undelivered(f"entra token refused: {refused}")
        except (httpx.HTTPError, ValidationError) as error:
            return Undelivered(f"{type(error).__name__} getting the entra token")
        headers: Final = {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}
        for payload in chunks(sentinel_row(record) for record in batch):
            try:
                response = await self.post.post(self.url, payload, headers, self.settings.timeout_seconds)
            except httpx.HTTPError as error:
                return Undelivered(f"{type(error).__name__} posting to the logs ingestion api")
            if response.status_code != 204:
                return Undelivered(f"HTTP {response.status_code} from the logs ingestion api")
        return Delivered(len(batch))

    async def close(self) -> None:
        return None
