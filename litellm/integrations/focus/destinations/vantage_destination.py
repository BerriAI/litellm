"""Vantage API destination for Focus export."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from litellm._logging import verbose_logger

from .base import FocusDestination, FocusTimeWindow

# Vantage enforces a 10,000-row / 2 MB limit per upload.
VANTAGE_MAX_ROWS_PER_UPLOAD = 10_000
VANTAGE_MAX_BYTES_PER_UPLOAD = 2 * 1024 * 1024  # 2 MB


class FocusVantageDestination(FocusDestination):
    """Upload FOCUS CSV exports to the Vantage cost-import API."""

    def __init__(
        self,
        *,
        prefix: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        api_key = config.get("api_key")
        integration_token = config.get("integration_token")
        if not api_key:
            raise ValueError(
                "api_key must be provided for Vantage destination "
                "(set VANTAGE_API_KEY env var or pass in destination_config)"
            )
        if not integration_token:
            raise ValueError(
                "integration_token must be provided for Vantage destination "
                "(set VANTAGE_INTEGRATION_TOKEN env var or pass in destination_config)"
            )
        self.api_key = api_key
        self.integration_token = integration_token
        self.base_url = config.get(
            "base_url", "https://api.vantage.sh"
        )
        self.prefix = prefix

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        """Upload CSV content to the Vantage API, batching if needed."""
        if not content:
            verbose_logger.debug("Vantage destination: empty content, skipping upload")
            return

        # Reuse a single HTTP client for the entire deliver() call
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Check both size and row-count limits before single-shot upload
            lines = content.split(b"\n")
            data_line_count = sum(1 for line in lines[1:] if line.strip())
            within_limits = (
                len(content) <= VANTAGE_MAX_BYTES_PER_UPLOAD
                and data_line_count <= VANTAGE_MAX_ROWS_PER_UPLOAD
            )
            if within_limits:
                await self._upload_csv(client, content, filename)
                return

            # Otherwise split into batches respecting both limits
            await self._upload_batched(client, content, filename)

    async def _upload_csv(
        self, client: httpx.AsyncClient, csv_bytes: bytes, filename: str
    ) -> None:
        url = (
            f"{self.base_url}/v2/integrations/"
            f"{self.integration_token}/costs.csv"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        response = await client.post(
            url,
            headers=headers,
            files={"file": (filename, csv_bytes, "text/csv")},
        )
        response.raise_for_status()

        verbose_logger.debug(
            "Vantage destination: uploaded %d bytes (%s)",
            len(csv_bytes),
            filename,
        )

    async def _upload_batched(
        self, client: httpx.AsyncClient, csv_bytes: bytes, filename: str
    ) -> None:
        """Split the CSV into batches and upload each."""
        lines = csv_bytes.split(b"\n")
        header = lines[0]
        data_lines = [line for line in lines[1:] if line.strip()]

        batch_num = 0
        for start in range(0, len(data_lines), VANTAGE_MAX_ROWS_PER_UPLOAD):
            batch_lines = data_lines[start : start + VANTAGE_MAX_ROWS_PER_UPLOAD]
            batch_csv = header + b"\n" + b"\n".join(batch_lines) + b"\n"

            # If a single batch still exceeds 2 MB, split further by size
            if len(batch_csv) > VANTAGE_MAX_BYTES_PER_UPLOAD:
                await self._upload_size_limited(
                    client, header, batch_lines, filename, batch_num
                )
            else:
                batch_filename = f"{filename}.part{batch_num}"
                await self._upload_csv(client, batch_csv, batch_filename)
            batch_num += 1

    async def _upload_size_limited(
        self,
        client: httpx.AsyncClient,
        header: bytes,
        data_lines: list[bytes],
        filename: str,
        batch_offset: int,
    ) -> None:
        """Upload lines in chunks that stay under the 2 MB size limit."""
        current_chunk: list[bytes] = []
        current_size = len(header) + 1  # header + newline
        sub_batch = 0

        for line in data_lines:
            line_size = len(line) + 1  # line + newline
            if current_size + line_size > VANTAGE_MAX_BYTES_PER_UPLOAD and current_chunk:
                batch_csv = header + b"\n" + b"\n".join(current_chunk) + b"\n"
                batch_filename = f"{filename}.part{batch_offset}_{sub_batch}"
                await self._upload_csv(client, batch_csv, batch_filename)
                current_chunk = []
                current_size = len(header) + 1
                sub_batch += 1
            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            batch_csv = header + b"\n" + b"\n".join(current_chunk) + b"\n"
            batch_filename = f"{filename}.part{batch_offset}_{sub_batch}"
            await self._upload_csv(client, batch_csv, batch_filename)
