"""Ternary API destination for Focus export.

Uploads FOCUS CSV exports to a Ternary cost-ingestion endpoint so LiteLLM
spend can be allocated in Ternary. A thin sink over the shared FOCUS
transformer/serializer: it adds no columns and forwards the FOCUS CSV as-is.

A large backfill may exceed the per-request limits and be split into several
chunks. Every chunk of one export carries a stable ``X-Ternary-Upload-Id`` plus
its 0-based index and total count, so the receiver stages the parts and swaps
the whole export window once, after all parts arrive.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Final
from urllib.parse import quote, urlparse
from uuid import uuid4

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base import FocusDestination, FocusTimeWindow

# Chunk oversized backfills; the byte cap stays under the receiver's 32 MB limit.
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
        """Upload CSV content to the Ternary API, chunking an oversized backfill.

        Aborts on the first failed chunk (raising) rather than continuing: under
        the receiver's stage-then-swap an incomplete upload is never committed, so
        stopping early leaves nothing partial landed and the next scheduled run
        retries the whole window. All chunks share one upload id.
        """
        if not content:
            verbose_logger.debug("Ternary destination: empty content, skipping upload")
            return

        client: Final = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback,
        )

        chunks: Final = self._split_into_chunks(content)
        upload_id: Final = uuid4().hex
        total: Final = len(chunks)

        for index, chunk in enumerate(chunks):
            await self._upload_csv(
                client,
                chunk,
                filename if total == 1 else f"{filename}.part{index + 1}",
                upload_id=upload_id,
                chunk_index=index,
                chunk_total=total,
            )

    def _split_into_chunks(self, content: bytes) -> Sequence[bytes]:
        """Split CSV bytes into chunks within the row and byte limits.

        A steady-state export fits in one chunk and is returned untouched (no
        re-encoding). Only an oversized backfill is parsed and repartitioned;
        parsing goes through the csv module so a quoted field containing a
        newline is not mis-split. A single row larger than the byte limit cannot
        be split and is a hard error rather than silently dropped.
        """
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
                current.clear()  # reset in place after flushing (no rebinding)
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
