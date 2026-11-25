"""Base64 media in Langfuse observations: swapped for Langfuse media references and uploaded over the REST API."""

from __future__ import annotations

import json
import os
import threading
import weakref
from base64 import b64decode, b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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

__all__ = ("Media", "MediaField", "MediaUpload", "MediaUploader", "extract_media", "restart_uploads_in_forked_child")

MediaField: TypeAlias = Literal["input", "output", "metadata"]

_REFERENCE_PREFIX: Final = "@@@langfuseMedia:"
_MAX_DEPTH: Final = 10
_MAX_QUEUED_BYTES: Final = 100 * 1024 * 1024
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


def _may_hold_media(serialized: str) -> bool:
    """Whether ``serialized`` carries the markers every payload ``extract_media`` replaces has: a base64 data URI, or
    a typed block's content type key next to a ``data`` key."""
    return ("data:" in serialized and ";base64" in serialized) or (
        '"data"' in serialized and ('"media_type"' in serialized or '"mime_type"' in serialized)
    )


def extract_media(serialized: str) -> tuple[str, tuple[Media, ...]]:
    """``serialized`` with each base64 payload replaced by its Langfuse media reference, and the payloads replaced.

    Covers what the v2 SDK replaced before ingestion: a ``data:<type>;base64,`` string anywhere in the JSON
    value, and the Anthropic and Vertex typed blocks.
    """
    if not _may_hold_media(serialized):
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


def _storage_headers(upload_url: str, media: Media) -> Mapping[str, str]:
    """Google Cloud Storage rejects the S3 checksum and Azure blob headers, so its URLs get the content type only."""
    return {
        "Content-Type": media.content_type.value,
        **(
            {}
            if "storage.googleapis.com" in upload_url
            else {"x-ms-blob-type": "BlockBlob", "x-amz-checksum-sha256": media.sha256_hash}
        ),
    }


@dataclass(slots=True)
class _Backlog:
    """What one upload thread works through. A forked child gets a new one, since it has none of the parent's threads
    and the parent still uploads what it queued."""

    queue: Queue[MediaUpload] = field(default_factory=Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: threading.Event = field(default_factory=threading.Event)
    queued_bytes: int = 0

    def reserve(self, size: int, limit: int) -> bool:
        with self.lock:
            if self.queued_bytes + size > limit:
                return False
            self.queued_bytes += size
            return True

    def release(self, size: int) -> None:
        with self.lock:
            self.queued_bytes -= size


class MediaUploader:
    """Uploads queued media to Langfuse on one daemon thread, so storage latency never holds up span export.

    ``client`` is asked for the HTTP client before every upload and ``api`` builds the REST client bound to it,
    because the owner of a pooled client closes it and opens another over the life of a process.
    """

    def __init__(
        self,
        *,
        api: Callable[[httpx.Client], LangfuseAPI],
        client: Callable[[], httpx.Client],
        timeout: float,
        delays: Sequence[float],
        max_queued_bytes: int = _MAX_QUEUED_BYTES,
    ) -> None:
        self._api_for: Final = api
        self._current_client: Final = client
        self._bound: tuple[httpx.Client, LangfuseAPI] | None = None
        self._timeout: Final = timeout
        self._delays: Final = tuple(delays)
        self._max_queued_bytes: Final = max_queued_bytes
        self._backlog = self._started_backlog()
        self.failed_uploads = 0
        _LIVE_UPLOADERS.add(self)

    def _started_backlog(self) -> _Backlog:
        backlog: Final = _Backlog()
        threading.Thread(target=self._run, args=(backlog,), name="langfuse-media-upload", daemon=True).start()
        return backlog

    def restart_in_forked_child(self) -> None:
        self._backlog.closed.set()
        self._backlog = self._started_backlog()

    def submit(self, uploads: Sequence[MediaUpload]) -> bool:
        """All of ``uploads`` or none: ``False`` when they do not fit next to the payloads still waiting, and the
        caller has to keep them inline."""
        size: Final = sum(len(upload.media.content) for upload in uploads)
        backlog: Final = self._backlog
        if not backlog.reserve(size, self._max_queued_bytes):
            return False
        for upload in uploads:
            backlog.queue.put_nowait(upload)
        return True

    def drain(self, timeout_millis: int) -> bool:
        """``True`` once every submitted upload has been attempted, ``False`` if ``timeout_millis`` passes first."""
        deadline: Final = monotonic() + timeout_millis / 1000
        queue: Final = self._backlog.queue
        while queue.unfinished_tasks and monotonic() < deadline:
            sleep(_DRAIN_POLL_SECONDS)
        return not queue.unfinished_tasks

    def shutdown(self) -> None:
        """The thread ends once the uploads already queued have been attempted."""
        self._backlog.closed.set()

    def _run(self, backlog: _Backlog) -> None:
        while not (backlog.closed.is_set() and backlog.queue.empty()):
            try:
                self._attempt(backlog, backlog.queue.get(timeout=_IDLE_POLL_SECONDS))
            except Empty:
                continue

    def _attempt(self, backlog: _Backlog, upload: MediaUpload) -> None:
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
            backlog.release(len(upload.media.content))
            backlog.queue.task_done()

    def _failure(self, upload: MediaUpload) -> str | None:
        try:
            return self._upload(upload)
        except ApiError as error:
            return f"HTTP {error.status_code} from the media API"
        except Exception as error:  # noqa: BLE001  # one failed upload must not end the thread
            return str(error)

    def _upload(self, upload: MediaUpload) -> str | None:
        """Registers the media with its trace; Langfuse returns an upload URL only for content it does not hold yet.

        The upload URL is as trusted as the Langfuse host that returned it. The PUT never follows a redirect, and
        Langfuse is told the storage status but never the storage response body.
        """
        media: Final = upload.media
        client, api = self._connection()
        registered: Final = self._registered(api, upload)
        upload_url: Final = registered.upload_url
        if not upload_url:
            return None
        started: Final = monotonic()
        response: Final = self._put(client, upload_url, _storage_headers(upload_url, media), media)
        if response is None:
            return "the media storage was not reached"
        api.media.patch(
            registered.media_id,
            uploaded_at=datetime.now(timezone.utc),
            upload_http_status=response.status_code,
            upload_http_error=None if response.is_success else response.reason_phrase,
            upload_time_ms=int((monotonic() - started) * 1000),
        )
        return None if response.is_success else f"the media storage refused it with HTTP {response.status_code}"

    def _connection(self) -> tuple[httpx.Client, LangfuseAPI]:
        """The HTTP client as its owner hands it out now, with the REST client bound to it."""
        client: Final = self._current_client()
        if self._bound is None or self._bound[0] is not client:
            self._bound = (client, self._api_for(client))
        return self._bound

    def _registered(self, api: LangfuseAPI, upload: MediaUpload) -> GetMediaUploadUrlResponse:
        """Re-sent after each delay while Langfuse is not reached, which is how long its span export keeps trying."""
        for delay in self._delays:
            try:
                return self._registered_once(api, upload)
            except httpx.TransportError as error:
                verbose_logger.warning("Langfuse media %s registration raised %s", upload.media.media_id, error)
                sleep(delay)
        return self._registered_once(api, upload)

    def _registered_once(self, api: LangfuseAPI, upload: MediaUpload) -> GetMediaUploadUrlResponse:
        return api.media.get_upload_url(
            content_type=upload.media.content_type,
            content_length=len(upload.media.content),
            sha256hash=upload.media.sha256_hash,
            field=upload.field,
            trace_id=upload.trace_id,
            observation_id=upload.observation_id,
        )

    def _put(self, client: httpx.Client, url: str, headers: Mapping[str, str], media: Media) -> httpx.Response | None:
        """The storage response, re-sent after each delay on a transport failure or a retryable status; ``None``
        when the storage was never reached."""
        for delay in self._delays:
            response: httpx.Response | None = self._put_once(client, url, headers, media)
            if response is not None and not _retryable_status(response.status_code):
                return response
            sleep(delay)
        return self._put_once(client, url, headers, media)

    def _put_once(
        self, client: httpx.Client, url: str, headers: Mapping[str, str], media: Media
    ) -> httpx.Response | None:
        """The response is closed with its body unread: only its status is used, and a storage URL may answer with
        a body of any size."""
        try:
            with client.stream(
                "PUT",
                url,
                content=media.content,
                headers=dict(headers),
                timeout=self._timeout,
                follow_redirects=False,
            ) as response:
                return response
        except httpx.TransportError as error:
            verbose_logger.warning("Langfuse media %s upload raised %s", media.media_id, error)
            return None


_LIVE_UPLOADERS: Final[weakref.WeakSet[MediaUploader]] = weakref.WeakSet()


def restart_uploads_in_forked_child() -> None:
    for uploader in tuple(_LIVE_UPLOADERS):
        uploader.restart_in_forked_child()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=restart_uploads_in_forked_child)
