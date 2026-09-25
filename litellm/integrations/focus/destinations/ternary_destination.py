"""Ternary API destination for FOCUS export: uploads FOCUS CSV to Ternary's cost-ingestion endpoint."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final
from urllib.parse import quote, urlparse
from uuid import uuid4

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from .base import FocusDestination, FocusTimeWindow

if TYPE_CHECKING:
    import httpx

TERNARY_MAX_ROWS_PER_UPLOAD: Final = 100_000
TERNARY_MAX_BYTES_PER_UPLOAD: Final = 30 * 1024 * 1024
TERNARY_UPLOAD_TIMEOUT_SECONDS: Final = 120.0

_LOOPBACK_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})


def _encode_csv(rows: Sequence[Sequence[str]]) -> bytes:
    """Serialize CSV rows to UTF-8 bytes via the csv module (quoted fields stay intact)."""
    buffer: Final = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _require_secure_base_url(base_url: str) -> None:
    parsed: Final = urlparse(base_url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and (parsed.hostname or "").lower() in _LOOPBACK_HOSTS:
        return
    raise ValueError(f"base_url must be an HTTPS URL (got {base_url!r}); http is allowed only for loopback")


class _NoRedirectHTTPHandler(AsyncHTTPHandler):
    """AsyncHTTPHandler that disables redirects on every client it creates (SSRF guard)."""

    def create_client(self, *args: object, **kwargs: object) -> httpx.AsyncClient:
        client: Final = super().create_client(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # forwarded verbatim to the base
        client.follow_redirects = False
        return client


class FocusTernaryDestination(FocusDestination):
    """Upload FOCUS CSV exports to the Ternary cost-ingestion API."""

    def __init__(
        self,
        *,
        prefix: str,
        config: dict[str, str] | None = None,  # mutable-ok: FocusDestination(config) factory contract
    ) -> None:
        resolved_config: Final = config or {}  # mutable-ok: read-only local; empty fallback for absent config
        api_key: Final = resolved_config.get("api_key")
        connection_id: Final = resolved_config.get("connection_id")
        base_url: Final = resolved_config.get("base_url")
        if not api_key:
            raise ValueError(
                "api_key must be provided for Ternary destination "
                "(set TERNARY_API_KEY env var or pass in destination_config)"
            )
        if not connection_id:
            raise ValueError(
                "connection_id must be provided for Ternary destination "
                "(set TERNARY_CONNECTION_ID env var or pass in destination_config)"
            )
        if "/" in connection_id or ".." in connection_id or any(c.isspace() for c in connection_id):
            raise ValueError(f"connection_id must not contain '/', '..', or whitespace (got {connection_id!r})")
        if not base_url:
            raise ValueError(
                "base_url must be provided for Ternary destination "
                "(set TERNARY_BASE_URL env var or pass in destination_config)"
            )
        _require_secure_base_url(str(base_url))
        self.api_key = api_key
        self.connection_id = connection_id
        self.base_url = str(base_url).rstrip("/")
        self.prefix = prefix

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        """Upload CSV to the Ternary API, chunking an oversized backfill; aborts on the first failed chunk."""
        if not content:
            verbose_logger.debug("Ternary destination: empty content, skipping upload")
            return

        client: Final = _NoRedirectHTTPHandler(timeout=TERNARY_UPLOAD_TIMEOUT_SECONDS)

        chunks: Final = self._split_into_chunks(content)
        upload_id: Final = uuid4().hex
        total: Final = len(chunks)

        try:
            for index, chunk in enumerate(chunks):
                await self._upload_csv(
                    client,
                    chunk,
                    filename if total == 1 else f"{filename}.part{index + 1}",
                    upload_id=upload_id,
                    chunk_index=index,
                    chunk_total=total,
                )
        finally:
            await client.close()

    def _split_into_chunks(self, content: bytes) -> Sequence[bytes]:
        """Split CSV bytes into chunks within the row/byte limits (csv-aware; a single oversized row errors)."""
        newline_count: Final = content.count(b"\n")
        if len(content) <= TERNARY_MAX_BYTES_PER_UPLOAD and newline_count <= TERNARY_MAX_ROWS_PER_UPLOAD:
            return (content,)

        rows: Final = tuple(tuple(row) for row in csv.reader(io.StringIO(content.decode("utf-8"))))
        if len(rows) <= 1:
            return (content,)
        header: Final = rows[0]
        data_rows: Final = rows[1:]
        header_bytes: Final = len(_encode_csv((header,)))

        chunks: Final[list[bytes]] = []  # mutable-ok: local accumulator, appended to and returned frozen
        current: Final[list[Sequence[str]]] = []  # mutable-ok: rows buffered for the in-progress chunk
        current_size = header_bytes  # rebind-ok: running byte tally advanced across the loop

        for row in data_rows:
            row_bytes = len(_encode_csv((header, row))) - header_bytes
            if header_bytes + row_bytes > TERNARY_MAX_BYTES_PER_UPLOAD:
                raise ValueError(
                    f"Ternary destination: a single CSV row is {row_bytes} bytes, exceeding the "
                    f"{TERNARY_MAX_BYTES_PER_UPLOAD}-byte upload limit and cannot be split"
                )
            exceeds_rows = len(current) >= TERNARY_MAX_ROWS_PER_UPLOAD
            exceeds_bytes = current_size + row_bytes > TERNARY_MAX_BYTES_PER_UPLOAD
            if current and (exceeds_rows or exceeds_bytes):
                chunks.append(_encode_csv((header, *current)))
                current.clear()
                current_size = header_bytes  # rebind-ok: reset after flushing a chunk
            current.append(row)
            current_size += row_bytes

        if current:
            chunks.append(_encode_csv((header, *current)))
        return tuple(chunks) or (content,)

    async def _upload_csv(
        self,
        client: AsyncHTTPHandler,
        csv_bytes: bytes,
        filename: str,
        *,
        upload_id: str,
        chunk_index: int,
        chunk_total: int,
    ) -> None:
        url: Final = f"{self.base_url}/external-cost-sources/v1/{quote(self.connection_id, safe='')}/focus"
        headers: Final = {  # mutable-ok: request headers handed to the HTTP client
            "Authorization": f"Bearer {self.api_key}",
            "X-Ternary-Upload-Id": upload_id,
            "X-Ternary-Chunk-Index": str(chunk_index),
            "X-Ternary-Chunk-Total": str(chunk_total),
        }

        await client.post(
            url,
            headers=headers,
            files={"csv": (filename, csv_bytes, "text/csv")},  # mutable-ok: multipart payload for the client
            timeout=TERNARY_UPLOAD_TIMEOUT_SECONDS,
        )

        verbose_logger.debug(
            "Ternary destination: uploaded %d bytes (%s, upload_id=%s, chunk %d/%d)",
            len(csv_bytes),
            filename,
            upload_id,
            chunk_index,
            chunk_total,
        )
