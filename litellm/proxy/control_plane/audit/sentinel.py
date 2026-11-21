import json
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby
from typing import Final, Protocol, TypeAlias
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.control_plane.audit.ocsf import AuditRecord, section
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult, Rejected, Undelivered
from litellm.proxy.control_plane.audit.sinks import ocsf_line
from litellm.proxy.control_plane.audit.transport import Poster
from litellm.proxy.control_plane.settings import SentinelSettings

API_VERSION: Final = "2023-01-01"
MAX_PAYLOAD_BYTES: Final = 1_000_000
MAX_FIELD_BYTES: Final = 64 * 1024
TRUNCATION_MARKER: Final = "...[truncated]"
_COMPACT_PATH_BYTES: Final = MAX_FIELD_BYTES // 2
_COMPACT_ACTOR_VALUE_BYTES: Final = MAX_FIELD_BYTES // 16
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
REFRESH_MARGIN_SECONDS: Final = 60.0
Row: TypeAlias = Mapping[str, object]
_JSON_MAPPING: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class TokenRefused:
    status: int


class TokenSource(Protocol):
    async def __call__(self) -> str | TokenRefused: ...


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

    async def __call__(self) -> str | TokenRefused:
        if self.token and self.now() < self.expires_at - REFRESH_MARGIN_SECONDS:
            return self.token
        form: Final = urlencode(
            {
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret.get_secret_value(),
                "scope": self.settings.token_scope,
                "grant_type": "client_credentials",
            }
        ).encode()
        url: Final = f"{self.settings.authority_host}/{self.settings.tenant_id}/oauth2/v2.0/token"
        response: Final = await self.post.post(
            url, form, {"Content-Type": "application/x-www-form-urlencoded"}, self.settings.timeout_seconds
        )
        if response.status_code != 200:
            return TokenRefused(response.status_code)
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
    return json.dumps(row, separators=(",", ":"), default=str, ensure_ascii=False).encode()


def _truncate_text(value: str, limit: int = MAX_FIELD_BYTES) -> str:
    encoded: Final = value.encode("utf-8")
    marker: Final = TRUNCATION_MARKER.encode()
    if len(encoded) <= limit:
        return value
    prefix: Final = encoded[: limit - len(marker)].decode("utf-8", errors="ignore")
    return prefix + TRUNCATION_MARKER


def _compact_actor_text(value: str) -> str:
    return _truncate_text(value, _COMPACT_ACTOR_VALUE_BYTES)


_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _compact_actor_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return _compact_actor_text(value)
    if isinstance(value, dict):
        actor_fields: Final[Mapping[str, JsonValue]] = value
        return {key: _compact_actor_value(item) for key, item in actor_fields.items()}
    if isinstance(value, list):
        actor_items: Final[Sequence[JsonValue]] = value
        return [_compact_actor_value(item) for item in actor_items]
    return value


def _compact_actor(value: object) -> object:
    try:
        actor: Final = _JSON_MAPPING.validate_python(value)
        compact: Final[JsonValue] = _JSON_VALUE.validate_python(actor)
    except ValidationError:
        return {}
    if not isinstance(compact, dict):
        return {}
    return _compact_actor_value(compact)


def _parse_json_mapping(value: str) -> Mapping[str, object]:
    try:
        return _JSON_MAPPING.validate_json(value)
    except ValidationError:
        return {}


def _compact_ocsf(value: str, row: Row) -> str:
    body: Final = _parse_json_mapping(value)
    litellm: Final = section(body, "unmapped", "litellm")
    request: Final = section(body, "http_request", "url")
    path_value: Final = request.get("path", row.get("UrlPath", ""))
    path: Final = path_value if isinstance(path_value, str) else ""
    detail: Final = body.get("status_detail")
    status_detail: Final = _truncate_text(detail, _COMPACT_ACTOR_VALUE_BYTES) if isinstance(detail, str) else detail
    actor: Final = _compact_actor(body.get("actor"))
    compact: Final = {
        "activity_id": body.get("activity_id", row.get("ActivityId")),
        "activity_name": body.get("activity_name", row.get("ActivityName")),
        "actor": actor or {"user": {"uid": row.get("ActorUid")}},
        "http_request": {"url": {"path": _truncate_text(path, _COMPACT_PATH_BYTES)}},
        "status": body.get("status", row.get("Decision")),
        "status_code": body.get("status_code", row.get("ResponseCode")),
        "status_detail": status_detail,
        "status_id": body.get("status_id", row.get("StatusId")),
        "unmapped": {
            "litellm": {
                "hash": litellm.get("hash", row.get("Hash")),
                "prev_hash": litellm.get("prev_hash", row.get("PrevHash")),
                "sequence": litellm.get("sequence", row.get("Sequence")),
            }
        },
    }
    serialized: Final = _encoded(compact)
    if len(serialized) <= MAX_FIELD_BYTES:
        return serialized.decode("utf-8")
    metadata: Final = {
        key: compact[key] for key in ("activity_id", "activity_name", "status", "status_code", "status_id", "unmapped")
    }
    return _encoded({**metadata, "Truncated": True}).decode("utf-8")


def _fitted(row: Row, limit: int) -> bytes:
    """Bound every string field and keep one row from blocking a batch larger than the ingestion limit."""
    oversized: Final = any(
        isinstance(value, str) and len(value.encode("utf-8")) > MAX_FIELD_BYTES for value in row.values()
    )
    bounded: Final = {
        key: _compact_ocsf(value, row)
        if key == "Ocsf" and isinstance(value, str) and len(value.encode("utf-8")) > MAX_FIELD_BYTES
        else _truncate_text(value)
        if isinstance(value, str)
        else value
        for key, value in row.items()
    }
    fitted: Final = {**bounded, **({"Truncated": True} if oversized else {})}
    whole: Final = _encoded(fitted)
    if len(whole) + 2 <= limit:
        return whole
    trimmed: Final = _encoded({**fitted, "Detail": None, "Truncated": True})
    if len(trimmed) + 2 <= limit:
        return trimmed
    without_ocsf: Final = _encoded({**fitted, "Detail": None, "Ocsf": None, "Truncated": True})
    if len(without_ocsf) + 2 <= limit:
        return without_ocsf
    return _encoded({**{key: value for key, value in fitted.items() if key in FIXED_WIDTH_COLUMNS}, "Truncated": True})


def chunks(rows: Iterable[Row], limit: int = MAX_PAYLOAD_BYTES) -> tuple[bytes, ...]:
    """JSON arrays of rows, each at most ``limit`` bytes."""
    return tuple(payload for payload, _ in _chunks_with_counts(rows, limit))


def _chunks_with_counts(rows: Iterable[Row], limit: int) -> tuple[tuple[bytes, int], ...]:
    encoded: Final = tuple(_fitted(row, limit) for row in rows)
    sizes: Final = tuple(len(item) + 1 for item in encoded)
    fills: Final = tuple(accumulate(sizes, _step(limit), initial=(0, 1)))[1:]
    groups: Final = groupby(zip((chunk for chunk, _ in fills), encoded, strict=True), key=lambda pair: pair[0])
    rows_by_chunk: Final = tuple(tuple(item for _, item in members) for _, members in groups)
    return tuple((b"[" + b",".join(group) + b"]", len(group)) for group in rows_by_chunk)


def _step(limit: int) -> Callable[[tuple[int, int], int], tuple[int, int]]:
    def step(fill: tuple[int, int], size: int) -> tuple[int, int]:
        chunk, used = fill
        return (chunk, used + size) if used + size <= limit else (chunk + 1, size + 1)

    return step


@dataclass(frozen=True, slots=True)
class SentinelSink:
    """A transient chunk failure leaves only the unsettled suffix for retry."""

    settings: SentinelSettings
    post: Poster
    token: TokenSource
    name: str = "sentinel"
    _chunk_limit: int = MAX_PAYLOAD_BYTES

    @property
    def url(self) -> str:
        return (
            f"{self.settings.endpoint.rstrip('/')}/dataCollectionRules/{self.settings.dcr_immutable_id}"
            f"/streams/{self.settings.stream_name}?api-version={API_VERSION}"
        )

    async def deliver(self, batch: Batch) -> DeliveryResult:
        try:
            bearer: Final = await self.token()
        except (httpx.HTTPError, ValidationError) as error:
            return Undelivered(f"{type(error).__name__} getting the entra token")
        if isinstance(bearer, TokenRefused):
            return Undelivered(f"entra token refused: HTTP {bearer.status} from the token endpoint")
        headers: Final = {"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"}
        uploads: Final = _chunks_with_counts(
            (sentinel_row(record) for record in batch),
            self._chunk_limit,
        )
        outcomes: Final = tuple([outcome async for outcome in self._post_chunks(uploads, headers)])
        rejected: Final = sum(count for status_code, _, count in outcomes if status_code in (400, 413, 422))
        for status_code, _, _ in outcomes:
            if status_code in (400, 413, 422):
                verbose_proxy_logger.error("control_plane.audit sentinel rejected chunk with HTTP %d", status_code)
        retryable: Final = next(
            (
                reason
                for status_code, reason, _ in outcomes
                if status_code is None or status_code not in (204, 400, 413, 422)
            ),
            None,
        )
        if retryable is not None:
            settled: Final = sum(count for status_code, _, count in outcomes if status_code in (204, 400, 413, 422))
            return Undelivered(retryable, settled=settled, rejected=rejected)
        if rejected:
            return Rejected("chunks were permanently rejected", rejected)
        return Delivered(sum(count for status_code, _, count in outcomes if status_code == 204))

    async def _post_chunks(
        self, uploads: tuple[tuple[bytes, int], ...], headers: Mapping[str, str]
    ) -> AsyncIterator[tuple[int | None, str | None, int]]:
        for payload, count in uploads:
            status_code, reason, posted_count = await self._post_chunk(payload, count, headers)
            yield status_code, reason, posted_count
            if status_code not in (204, 400, 413, 422):
                return

    async def _post_chunk(
        self, payload: bytes, count: int, headers: Mapping[str, str]
    ) -> tuple[int | None, str | None, int]:
        try:
            response: Final = await self.post.post(self.url, payload, headers, self.settings.timeout_seconds)
        except httpx.HTTPError as error:
            return None, f"{type(error).__name__} posting to the logs ingestion api", count
        status_code: Final = response.status_code
        if status_code == 204 or status_code in (400, 413, 422):
            return status_code, None, count
        return status_code, f"HTTP {status_code} from the logs ingestion api", count

    async def close(self) -> None:
        return None
