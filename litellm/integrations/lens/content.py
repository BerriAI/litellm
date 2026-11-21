import asyncio
import hashlib
import hmac
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.integrations.s3_v2 import S3Logger

MAX_INLINE_RECORD_BYTES: Final = 1024 * 1024
MAX_CONTENT_OBJECT_BYTES: Final = 8 * 1024 * 1024
MAX_PENDING_CONTENT_BYTES: Final = 64 * 1024 * 1024
MAX_CONTENT_JOBS: Final = 32
MAX_NATIVE_UPLOADS: Final = 4
MAX_SERIALIZATIONS: Final = 2
UPLOAD_TIMEOUT_SECONDS: Final = 30.0
MAX_CONTENT_READS: Final = 2
MAX_CONTENT_REQUEST_BYTES: Final = 16 * 1024
REFERENCE_KEY: Final = "litellm_lens_content"
CONTENT_STORAGE_S3_V2: Final = "s3_v2"
_VERSION: Final = "1"
_STORE_SETTINGS: Final = TypeAdapter(Mapping[str, object])


class Unavailable(Enum):
    TOO_LARGE = "too_large"
    STORAGE_NOT_CONFIGURED = "storage_not_configured"
    UPLOAD_FAILED = "upload_failed"
    OVERLOADED = "overloaded"
    UNSERIALIZABLE = "unserializable"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Stored:
    object_key: str
    size: int
    sha256: str


ContentState: TypeAlias = Stored | Unavailable


@dataclass(frozen=True, slots=True)
class Owner:
    team_id: str
    api_key: str
    request_id: str


@dataclass(frozen=True, slots=True)
class ContentLimits:
    inline_bytes: int = MAX_INLINE_RECORD_BYTES
    object_bytes: int = MAX_CONTENT_OBJECT_BYTES


class StoredReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    v: Literal[1]
    state: Literal["stored"]
    object_key: str
    bytes: int
    sha256: str
    mac: str


def _framed(*fields: str) -> bytes:
    return b"".join(f"{len(encoded)}:".encode() + encoded for encoded in (field.encode() for field in fields))


def _mac(signing_key: bytes, owner: Owner, state: str, reason: str, object_key: str, size: str, sha256: str) -> str:
    message: Final = _framed(
        "litellm-lens-content", _VERSION, owner.team_id, owner.api_key, owner.request_id, state, reason, object_key,
        size, sha256,
    )  # fmt: skip
    return hmac.new(signing_key, message, hashlib.sha256).hexdigest()


def content_signing_key(service_token: str) -> bytes:
    return hmac.new(service_token.encode(), b"litellm-lens-content", hashlib.sha256).digest()


def signed_reference(signing_key: bytes, owner: Owner, state: ContentState) -> Mapping[str, str | int]:
    match state:
        case Stored(object_key=object_key, size=size, sha256=sha256):
            return {
                "v": 1,
                "state": "stored",
                "object_key": object_key,
                "bytes": size,
                "sha256": sha256,
                "mac": _mac(signing_key, owner, "stored", "", object_key, str(size), sha256),
            }
        case Unavailable() as reason:
            return {
                "v": 1,
                "state": "unavailable",
                "reason": reason.value,
                "mac": _mac(signing_key, owner, "unavailable", reason.value, "", "", ""),
            }


def verified_stored(signing_key: bytes, owner: Owner, reference: StoredReference) -> Stored | None:
    expected: Final = _mac(
        signing_key, owner, "stored", "", reference.object_key, str(reference.bytes), reference.sha256
    )
    if not hmac.compare_digest(expected, reference.mac):
        return None
    return Stored(object_key=reference.object_key, size=reference.bytes, sha256=reference.sha256)


def object_name(owner: Owner, start_time_ms: int, content_sha256: str) -> str | None:
    try:
        day: Final = datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return None
    digest: Final = hashlib.sha256(_framed(owner.team_id, owner.api_key, owner.request_id, content_sha256)).hexdigest()
    return f"lens-content/{day}/{digest}.json"


class ContentStore(Protocol):
    def object_key(self, name: str) -> str | None: ...

    def upload(self, key: str, body: str) -> Awaitable[bool]: ...

    def download(self, key: str, max_bytes: int) -> Awaitable[str | None]: ...


def _active_s3_v2_logger() -> CustomLogger | None:
    try:
        return litellm.logging_callback_manager.get_active_custom_logger_for_callback_name(CONTENT_STORAGE_S3_V2)
    except ValueError:
        return None


async def _missing() -> None:
    return None


async def _not_uploaded() -> bool:
    return False


@dataclass(frozen=True, slots=True)
class S3V2ContentStore:
    resolve: Callable[[], CustomLogger | None] = _active_s3_v2_logger

    def _logger(self) -> "S3Logger | None":
        from litellm.integrations.s3_v2 import S3Logger

        logger: Final = self.resolve()
        return logger if isinstance(logger, S3Logger) else None

    def object_key(self, name: str) -> str | None:
        logger: Final = self._logger()
        if logger is None:
            return None
        prefix: Final = logger.s3_path
        return (f"{prefix.rstrip('/')}/" if prefix else "") + name

    def upload(self, key: str, body: str) -> Awaitable[bool]:
        logger: Final = self._logger()
        if logger is None:
            return _not_uploaded()
        return logger.sink.upload_line_items(key, body, key.rsplit("/", 1)[-1])

    def download(self, key: str, max_bytes: int) -> Awaitable[str | None]:
        logger: Final = self._logger()
        if logger is None:
            return _missing()
        return logger.sink.download_bounded(key, max_bytes)


def _store_settings(settings: Mapping[str, object] | None) -> Mapping[str, object]:
    if settings is None:
        return {}
    try:
        return _STORE_SETTINGS.validate_python(settings.get("store") or {})
    except ValidationError:
        return {}


def _bounded(value: object, default: int, cap: int) -> int:
    return min(value, cap) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default


def content_store_from_settings(settings: Mapping[str, object] | None) -> ContentStore | None:
    return S3V2ContentStore() if _store_settings(settings).get("content_storage") == CONTENT_STORAGE_S3_V2 else None


def content_limits_from_settings(settings: Mapping[str, object] | None) -> ContentLimits:
    store: Final = _store_settings(settings)
    return ContentLimits(
        inline_bytes=_bounded(store.get("content_inline_bytes"), MAX_INLINE_RECORD_BYTES, MAX_INLINE_RECORD_BYTES),
        object_bytes=_bounded(store.get("content_max_bytes"), MAX_CONTENT_OBJECT_BYTES, MAX_CONTENT_OBJECT_BYTES),
    )


class ContentRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    team_id: str
    api_key: str
    request_id: str
    reference: StoredReference


class ReadFailure(Enum):
    NOT_FOUND = 404
    INTEGRITY = 409
    UNAVAILABLE = 503


@dataclass(frozen=True, slots=True)
class ContentReader:
    store: ContentStore
    service_token: str
    slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(MAX_CONTENT_READS))

    def authorized(self, authorization: str) -> bool:
        return hmac.compare_digest(authorization.encode(), f"Bearer {self.service_token}".encode())

    async def read(self, request: ContentRequest) -> bytes | ReadFailure:
        owner: Final = Owner(team_id=request.team_id, api_key=request.api_key, request_id=request.request_id)
        stored: Final = verified_stored(content_signing_key(self.service_token), owner, request.reference)
        if stored is None or not 0 <= stored.size <= MAX_CONTENT_OBJECT_BYTES:
            return ReadFailure.NOT_FOUND
        if self.slots.locked():
            return ReadFailure.UNAVAILABLE
        await self.slots.acquire()
        try:
            download: Final = asyncio.ensure_future(self.store.download(stored.object_key, stored.size))
        except Exception:  # noqa: BLE001  # a store that cannot start a read leaves the content unavailable
            self.slots.release()
            return ReadFailure.UNAVAILABLE
        download.add_done_callback(self._downloaded)
        try:
            body: Final = await asyncio.shield(download)
        except Exception:  # noqa: BLE001  # a failed storage read leaves the content unavailable
            return ReadFailure.UNAVAILABLE
        if body is None:
            return ReadFailure.UNAVAILABLE
        data: Final = body.encode()
        if len(data) != stored.size or not hmac.compare_digest(hashlib.sha256(data).hexdigest(), stored.sha256):
            return ReadFailure.INTEGRITY
        return data

    def _downloaded(self, download: "asyncio.Future[str | None]") -> None:
        self.slots.release()
        if not download.cancelled():
            download.exception()
