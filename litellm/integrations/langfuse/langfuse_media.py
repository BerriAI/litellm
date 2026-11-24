"""Base64 media in Langfuse observations: swapped for Langfuse media references and uploaded over the REST API."""

from __future__ import annotations

import json
import threading
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import chain
from queue import Empty, Queue
from time import monotonic, sleep
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from langfuse.api import GetMediaUploadUrlResponse, LangfuseAPI, MediaContentType
from langfuse.api.core.api_error import ApiError
from pydantic import JsonValue, TypeAdapter

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url

__all__ = ("Media", "MediaField", "MediaUpload", "MediaUploader", "extract_media")

MediaField: TypeAlias = Literal["input", "output", "metadata"]

_REFERENCE_PREFIX: Final = "@@@langfuseMedia:"
_MAX_DEPTH: Final = 10
_MAX_QUEUED_BYTES: Final = 100 * 1024 * 1024
_REFUSAL_PAUSE_SECONDS: Final = 600.0
_IDLE_POLL_SECONDS: Final = 1.0
_DRAIN_POLL_SECONDS: Final = 0.01
_URL_SAFE_TO_STANDARD: Final = str.maketrans("-_", "+/")
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_CONTENT_TYPE_KEY_BY_BLOCK_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {"base64": "media_type", "media": "mime_type"}
)


@dataclass(frozen=True, slots=True)
class Media:
    """One decoded payload, identified the way the Langfuse SDK identifies it: by its SHA-256."""

    content_type: MediaContentType
    content: bytes
    sha256_hash: str

    @property
    def media_id(self) -> str:
        return self.sha256_hash.replace("+", "-").replace("/", "_")[:22]

    @property
    def reference(self) -> str:
        return f"{_REFERENCE_PREFIX}type={self.content_type.value}|id={self.media_id}|source=base64_data_uri@@@"


@dataclass(frozen=True, slots=True)
class MediaUpload:
    media: Media
    trace_id: str
    observation_id: str | None
    field: MediaField


def _media(content_type: object, encoded: object) -> Media | None:
    """``None`` for a content type Langfuse does not store or a payload that is not base64."""
    if not isinstance(content_type, str) or not isinstance(encoded, str) or not encoded:
        return None
    try:
        supported: Final = MediaContentType(content_type)
        content: Final = b64decode(encoded.translate(_URL_SAFE_TO_STANDARD), validate=True)
    except ValueError:
        return None
    return Media(
        content_type=supported, content=content, sha256_hash=b64encode(sha256(content).digest()).decode("ascii")
    )


def _from_data_uri(value: str) -> Media | None:
    if not value.startswith("data:"):
        return None
    header, separator, encoded = value[len("data:") :].partition(",")
    content_type, *flags = header.split(";")
    if not separator or "base64" not in flags:
        return None
    return _media(content_type, encoded)


def _from_typed_block(block: Mapping[str, JsonValue]) -> Media | None:
    """An Anthropic ``{"type": "base64", "media_type", "data"}`` or Vertex ``{"type": "media", "mime_type", "data"}``."""
    block_type: Final = block.get("type")
    content_type_key: Final = _CONTENT_TYPE_KEY_BY_BLOCK_TYPE.get(block_type) if isinstance(block_type, str) else None
    data: Final = block.get("data")
    if content_type_key is None or not isinstance(data, str) or data.startswith(_REFERENCE_PREFIX):
        return None
    return _media(block.get(content_type_key), data)


def _extract_media_nodes(node: JsonValue, depth: int) -> tuple[JsonValue, tuple[Media, ...]]:
    if depth > _MAX_DEPTH:
        return node, ()
    if isinstance(node, str):
        inline: Final = _from_data_uri(node)
        return (node, ()) if inline is None else (inline.reference, (inline,))
    if isinstance(node, list):
        items: Final = tuple(_extract_media_nodes(item, depth + 1) for item in node)
        return [item for item, _ in items], tuple(chain.from_iterable(found for _, found in items))
    if isinstance(node, dict):
        block: Final = _from_typed_block(node)
        if block is not None:
            return {**node, "data": block.reference}, (block,)
        entries: Final = {key: _extract_media_nodes(value, depth + 1) for key, value in node.items()}
        return (
            {key: value for key, (value, _) in entries.items()},
            tuple(chain.from_iterable(found for _, found in entries.values())),
        )
    return node, ()


def extract_media(serialized: str) -> tuple[str, tuple[Media, ...]]:
    """``serialized`` with each base64 payload replaced by its Langfuse media reference, and the payloads replaced.

    Covers what the v2 SDK replaced before ingestion: a ``data:<type>;base64,`` string anywhere in the JSON
    value, and the Anthropic and Vertex typed blocks.
    """
    if "base64" not in serialized and "mime_type" not in serialized:
        return serialized, ()
    whole: Final = _from_data_uri(serialized)
    if whole is not None:
        return whole.reference, (whole,)
    try:
        parsed: Final = _JSON.validate_json(serialized)
    except ValueError:
        return serialized, ()
    replaced, found = _extract_media_nodes(parsed, 0)
    return (json.dumps(replaced), found) if found else (serialized, ())


def _retryable_status(status: int) -> bool:
    return status in (408, 429) or status >= 500


def _media_api_unavailable(status: int | None) -> bool:
    """The answers of a Langfuse with no media route or no media bucket, which hold for every payload."""
    return status is not None and (status == 404 or status >= 500)


def _storage_headers(upload_url: str, host: str, media: Media) -> Mapping[str, str]:
    """Google Cloud Storage rejects the S3 checksum and Azure blob headers, so its URLs get the content type only."""
    return {
        "Content-Type": media.content_type.value,
        **(
            {}
            if "storage.googleapis.com" in upload_url
            else {"x-ms-blob-type": "BlockBlob", "x-amz-checksum-sha256": media.sha256_hash}
        ),
        **({"Host": host} if host else {}),
    }


class MediaUploader:
    """Uploads queued media to Langfuse on one daemon thread, so storage latency never holds up span export."""

    def __init__(
        self,
        *,
        api: LangfuseAPI,
        client: httpx.Client,
        timeout: float,
        delays: Sequence[float],
        validate_upload_url: bool = False,
        max_queued_bytes: int = _MAX_QUEUED_BYTES,
        refusal_pause: float = _REFUSAL_PAUSE_SECONDS,
    ) -> None:
        self._api: Final = api
        self._client: Final = client
        self._timeout: Final = timeout
        self._delays: Final = tuple(delays)
        self._validate_upload_url: Final = validate_upload_url
        self._max_queued_bytes: Final = max_queued_bytes
        self._refusal_pause: Final = refusal_pause
        self._probe_at: float | None = None
        self._queue: Final[Queue[MediaUpload]] = Queue()
        self._lock: Final = threading.Lock()
        self._queued_bytes = 0
        self._closed: Final = threading.Event()
        self.failed_uploads = 0
        threading.Thread(target=self._run, name="langfuse-media-upload", daemon=True).start()

    def submit(self, uploads: Sequence[MediaUpload]) -> bool:
        """All of ``uploads`` or none: ``False`` when the caller has to keep them inline, because they do not fit
        next to the payloads still waiting or Langfuse is refusing media.

        While Langfuse refuses media, one upload per pause is still sent to find out whether it takes media again.
        """
        with self._lock:
            refusing: Final = self._probe_at is not None
            probing: Final = self._probe_at is not None and monotonic() >= self._probe_at
            accepted: Final = tuple(uploads[:1]) if probing else () if refusing else tuple(uploads)
            size: Final = sum(len(upload.media.content) for upload in accepted)
            if self._queued_bytes + size > self._max_queued_bytes:
                return False
            if probing:
                self._probe_at = monotonic() + self._refusal_pause
            self._queued_bytes += size
        for upload in accepted:
            self._queue.put_nowait(upload)
        return not refusing

    def drain(self, timeout_millis: int) -> bool:
        """``True`` once every submitted upload has been attempted, ``False`` if ``timeout_millis`` passes first."""
        deadline: Final = monotonic() + timeout_millis / 1000
        while self._queue.unfinished_tasks and monotonic() < deadline:
            sleep(_DRAIN_POLL_SECONDS)
        return not self._queue.unfinished_tasks

    def shutdown(self) -> None:
        """The thread ends once the uploads already queued have been attempted."""
        self._closed.set()

    def _run(self) -> None:
        while not (self._closed.is_set() and self._queue.empty()):
            try:
                self._attempt(self._queue.get(timeout=_IDLE_POLL_SECONDS))
            except Empty:
                continue

    def _attempt(self, upload: MediaUpload) -> None:
        try:
            failure: Final = self._failure(upload)
            if failure is not None:
                self.failed_uploads += 1
                verbose_logger.warning(
                    "Langfuse media %s of trace %s was not uploaded: %s",
                    upload.media.media_id,
                    upload.trace_id,
                    failure,
                )
        finally:
            with self._lock:
                self._queued_bytes -= len(upload.media.content)
            self._queue.task_done()

    def _failure(self, upload: MediaUpload) -> str | None:
        try:
            return self._upload(upload)
        except ApiError as error:
            return f"HTTP {error.status_code} from the media API"
        except Exception as error:  # noqa: BLE001  # one failed upload must not end the thread
            return str(error)

    def _upload(self, upload: MediaUpload) -> str | None:
        """Registers the media with its trace; Langfuse returns an upload URL only for content it does not hold yet.

        The upload URL is as trusted as the host that returned it. For a Langfuse host that came with the request
        credentials it goes through ``validate_url``, the PUT never follows a redirect, and Langfuse is told the
        storage status but never the storage response body.
        """
        media: Final = upload.media
        registered: Final = self._registered(upload)
        upload_url: Final = registered.upload_url or ""
        try:
            url, host = validate_url(upload_url) if self._validate_upload_url and upload_url else (upload_url, "")
        except SSRFError as error:
            self._keep_media_inline(f"its upload URL was refused: {error}")
            return f"its upload URL was refused: {error}"
        with self._lock:
            self._probe_at = None
        if not url:
            return None
        started: Final = monotonic()
        response: Final = self._put(url, _storage_headers(upload_url, host, media), media)
        if response is None:
            return "the media storage was not reached"
        self._api.media.patch(
            registered.media_id,
            uploaded_at=datetime.now(timezone.utc),
            upload_http_status=response.status_code,
            upload_http_error=None if response.is_success else response.reason_phrase,
            upload_time_ms=int((monotonic() - started) * 1000),
        )
        return None if response.is_success else f"the media storage refused it with HTTP {response.status_code}"

    def _keep_media_inline(self, reason: str) -> None:
        """Langfuse cannot store media for a reason that holds for every payload, so no reference is exported
        until a later upload gets through."""
        with self._lock:
            if self._probe_at is None:
                verbose_logger.warning(
                    "Langfuse media cannot be stored (%s), so media stays inline in traces until an upload gets through",
                    reason,
                )
            self._probe_at = monotonic() + self._refusal_pause

    def _registered(self, upload: MediaUpload) -> GetMediaUploadUrlResponse:
        """Re-sent after each delay while Langfuse is not reached, which is how long its span export keeps trying."""
        for delay in self._delays:
            try:
                return self._registered_once(upload)
            except httpx.TransportError as error:
                verbose_logger.warning("Langfuse media %s registration raised %s", upload.media.media_id, error)
                sleep(delay)
        return self._registered_once(upload)

    def _registered_once(self, upload: MediaUpload) -> GetMediaUploadUrlResponse:
        try:
            return self._api.media.get_upload_url(
                content_type=upload.media.content_type,
                content_length=len(upload.media.content),
                sha256hash=upload.media.sha256_hash,
                field=upload.field,
                trace_id=upload.trace_id,
                observation_id=upload.observation_id,
            )
        except ApiError as error:
            if _media_api_unavailable(error.status_code):
                self._keep_media_inline(
                    f"HTTP {error.status_code} from the media API, a self-hosted Langfuse needs a media bucket"
                )
            raise

    def _put(self, url: str, headers: Mapping[str, str], media: Media) -> httpx.Response | None:
        """The storage response, re-sent after each delay on a transport failure or a retryable status; ``None``
        when the storage was never reached."""
        for delay in self._delays:
            response: httpx.Response | None = self._put_once(url, headers, media)
            if response is not None and not _retryable_status(response.status_code):
                return response
            sleep(delay)
        return self._put_once(url, headers, media)

    def _put_once(self, url: str, headers: Mapping[str, str], media: Media) -> httpx.Response | None:
        try:
            return self._client.put(
                url, content=media.content, headers=dict(headers), timeout=self._timeout, follow_redirects=False
            )
        except httpx.TransportError as error:
            verbose_logger.warning("Langfuse media %s upload raised %s", media.media_id, error)
            return None
