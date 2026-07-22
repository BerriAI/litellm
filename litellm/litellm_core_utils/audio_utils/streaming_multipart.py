"""
Incremental multipart/form-data parsing for streaming transcription uploads.

Parses the request body off an async byte iterator (e.g. `request.stream()`) instead of
FastAPI's `File(...)`, which fully buffers the upload before the route handler runs. Small
non-file fields are collected eagerly; the (large) file part is exposed as a tee: it streams
upstream as bytes arrive while keeping a copy so retries, router fallbacks, and duration-based
cost still work.

Assumes the file part is the last part (as OpenAI SDK / curl `-F` clients send it). A form field
appearing after the file part raises MultipartOrderError so the caller can fall back to buffering
rather than silently sending an incomplete upstream body.
"""

from typing import AsyncIterator

FieldValue = str | list[str]


class FilePartTooLarge(Exception):
    pass


class MultipartOrderError(Exception):
    pass


def _parse_content_disposition(value: bytes) -> tuple[str | None, str | None]:
    name: str | None = None
    filename: str | None = None
    for raw in value.decode("latin-1").split(";"):
        part = raw.strip()
        if part.startswith("name="):
            name = part[len("name=") :].strip('"')
        elif part.startswith("filename="):
            filename = part[len("filename=") :].strip('"')
    return name, filename


class StreamingMultipartUpload:
    """A multipart upload parsed incrementally: eager non-file fields plus a teed file part.

    Build with `open_transcription_multipart`. `stream()` yields the file part as it arrives while
    buffering a copy; `getvalue()` returns the full bytes (draining any unread remainder), so a
    retry after the stream is consumed still has the whole file.
    """

    def __init__(self, body: AsyncIterator[bytes], boundary: bytes, max_file_bytes: int | None) -> None:
        self.fields: dict[str, FieldValue] = {}  # mutable-ok: filled by push-parser callbacks
        self.filename: str | None = None
        self.file_content_type: str | None = None
        self.exhausted = False
        self.max_file_bytes = max_file_bytes  # settable after open(), before the file part streams
        self._body = body.__aiter__()
        self._buffer = bytearray()
        self._gen: AsyncIterator[bytes] | None = None
        self._pending: list[bytes] = []  # mutable-ok: per-write file segments drained by the tee
        self._file_started = False
        self._file_ended = False
        self._ended = False
        self._trailing_field = False
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: dict[str, bytes] = {}
        self._cur_name: str | None = None
        self._cur_is_file = False
        self._cur_value = bytearray()
        # Imported lazily: python-multipart is a proxy/server dependency, not required to import litellm.
        from multipart.multipart import MultipartParser

        self._parser = MultipartParser(
            boundary,
            {
                "on_part_begin": self._on_part_begin,
                "on_header_field": self._on_header_field,
                "on_header_value": self._on_header_value,
                "on_header_end": self._on_header_end,
                "on_headers_finished": self._on_headers_finished,
                "on_part_data": self._on_part_data,
                "on_part_end": self._on_part_end,
                "on_end": self._on_end,
            },
        )

    def _on_part_begin(self) -> None:
        if self._file_ended:
            self._trailing_field = True
        self._headers = {}
        self._cur_name = None
        self._cur_is_file = False
        self._cur_value = bytearray()
        self._header_field = bytearray()
        self._header_value = bytearray()

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        self._headers[bytes(self._header_field).decode("latin-1").lower()] = bytes(self._header_value)
        self._header_field = bytearray()
        self._header_value = bytearray()

    def _on_headers_finished(self) -> None:
        name, filename = _parse_content_disposition(self._headers.get("content-disposition", b""))
        self._cur_name = name
        self._cur_is_file = filename is not None or name == "file"
        if self._cur_is_file:
            self._file_started = True
            self.filename = filename
            content_type = self._headers.get("content-type")
            self.file_content_type = content_type.decode("latin-1") if content_type is not None else None

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._cur_is_file:
            self._pending.append(bytes(data[start:end]))
        else:
            self._cur_value += data[start:end]

    def _on_part_end(self) -> None:
        if self._cur_is_file:
            self._file_ended = True
        elif self._cur_name is not None:
            value = bytes(self._cur_value).decode("latin-1")
            if self._cur_name.endswith("[]"):
                # OpenAI SDKs send repeated fields as `name[]`; collect into a list (mirrors get_form_data)
                key = self._cur_name[:-2]
                existing = self.fields.get(key)
                self.fields[key] = [*existing, value] if isinstance(existing, list) else [value]
            else:
                self.fields[self._cur_name] = value

    def _on_end(self) -> None:
        self._ended = True

    async def _next(self) -> bytes | None:
        try:
            return await self._body.__anext__()
        except StopAsyncIteration:
            return None

    async def _open(self) -> None:
        while not self._file_started and not self._ended:
            chunk = await self._next()
            if chunk is None:
                break
            self._parser.write(chunk)

    @property
    def started(self) -> bool:
        """True once the file part has begun streaming (so a retry must use the buffer, not re-stream)."""
        return self._gen is not None

    @property
    def buffered(self) -> bytes:
        """The file bytes captured so far; complete once `exhausted` is True."""
        return bytes(self._buffer)

    def stream(self) -> AsyncIterator[bytes]:
        if self._gen is None:
            self._gen = self._iter_file()
        return self._gen

    async def getvalue(self) -> bytes:
        async for _ in self.stream():
            pass
        return bytes(self._buffer)

    async def _iter_file(self) -> AsyncIterator[bytes]:
        while True:
            for seg in self._pending:
                self._buffer += seg
                if self.max_file_bytes is not None and len(self._buffer) > self.max_file_bytes:
                    raise FilePartTooLarge(
                        f"Uploaded file exceeds the maximum allowed size of {self.max_file_bytes} bytes"
                    )
                yield seg
            self._pending = []
            if self._trailing_field:
                raise MultipartOrderError("A form field appeared after the file part; cannot stream")
            if self._file_ended or self._ended:
                break
            chunk = await self._next()
            if chunk is None:
                break
            self._parser.write(chunk)
        self.exhausted = True


async def open_transcription_multipart(
    body: AsyncIterator[bytes],
    boundary: bytes,
    max_file_bytes: int | None,
) -> StreamingMultipartUpload:
    upload = StreamingMultipartUpload(body, boundary, max_file_bytes)
    await upload._open()
    return upload
