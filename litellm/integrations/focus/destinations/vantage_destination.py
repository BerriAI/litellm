"""Vantage API destination for Focus export."""

from __future__ import annotations

import csv
import io
from typing import Any, Optional

import httpx  # noqa: F401 - used at runtime (AsyncClient, HTTPStatusError)

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base import FocusDestination, FocusTimeWindow

# Vantage enforces a 10,000-row / 2 MB limit per upload.
VANTAGE_MAX_ROWS_PER_UPLOAD = 10_000
VANTAGE_MAX_BYTES_PER_UPLOAD = 2 * 1024 * 1024  # 2 MB

# Columns that Vantage actually supports for custom provider CSV uploads.
# See: https://docs.vantage.sh/connecting_custom_providers
# Columns not in this set are silently dropped before upload so Vantage
# does not reject the file.
VANTAGE_SUPPORTED_COLUMNS = {
    # Required
    "ChargeCategory",
    "ChargePeriodStart",
    "BilledCost",
    "ServiceName",
    # Optional
    "BillingCurrency",
    "BillingAccountId",
    "BillingAccountName",
    "ChargePeriodEnd",
    "ChargeDescription",
    "ChargeFrequency",
    "ConsumedQuantity",
    "ConsumedUnit",
    "ContractedCost",
    "EffectiveCost",
    "ListCost",
    "RegionId",
    "RegionName",
    "ResourceId",
    "ResourceName",
    "ResourceType",
    "ServiceCategory",
    "ServiceSubcategory",
    "SubAccountId",
    "SubAccountName",
    "Tags",
}


def _strip_unsupported_columns(csv_bytes: bytes) -> bytes:
    """Remove CSV columns not in VANTAGE_SUPPORTED_COLUMNS.

    Parses the header row, identifies column indices to keep, and
    rebuilds the CSV with only those columns.
    """
    lines = csv_bytes.split(b"\n")
    if not lines:
        return csv_bytes

    header_cols = lines[0].decode("utf-8").split(",")
    keep_indices = [
        i
        for i, col in enumerate(header_cols)
        if col.strip('"') in VANTAGE_SUPPORTED_COLUMNS
    ]

    # If all columns are supported, return as-is
    if len(keep_indices) == len(header_cols):
        return csv_bytes

    dropped = [col for i, col in enumerate(header_cols) if i not in keep_indices]
    verbose_logger.debug(
        "Vantage destination: dropping unsupported columns: %s", dropped
    )

    output = io.StringIO()
    writer = csv.writer(output)
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    for row in reader:
        if not row:
            continue
        writer.writerow([row[i] for i in keep_indices])

    return output.getvalue().encode("utf-8")


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
        self.base_url = config.get("base_url", "https://api.vantage.sh")
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

        # Strip columns that Vantage does not support to avoid silent
        # rejection (e.g. InvoiceIssuerName, ProviderName, PublisherName).
        content = _strip_unsupported_columns(content)

        client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback,
        )

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
        self, client: AsyncHTTPHandler, csv_bytes: bytes, filename: str
    ) -> None:
        url = f"{self.base_url}/v2/integrations/" f"{self.integration_token}/costs.csv"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        await client.post(
            url,
            headers=headers,
            files={"csv": (filename, csv_bytes, "text/csv")},
        )

        verbose_logger.debug(
            "Vantage destination: uploaded %d bytes (%s)",
            len(csv_bytes),
            filename,
        )

    async def _upload_batched(
        self, client: AsyncHTTPHandler, csv_bytes: bytes, filename: str
    ) -> None:
        """Split the CSV into batches and upload each.

        Continues uploading remaining batches even if one fails, then raises
        the first error encountered so callers know the export was partial.
        """
        lines = csv_bytes.split(b"\n")
        header = lines[0]
        data_lines = [line for line in lines[1:] if line.strip()]

        first_error: Optional[Exception] = None
        batch_num = 0
        for start in range(0, len(data_lines), VANTAGE_MAX_ROWS_PER_UPLOAD):
            batch_lines = data_lines[start : start + VANTAGE_MAX_ROWS_PER_UPLOAD]
            batch_csv = header + b"\n" + b"\n".join(batch_lines) + b"\n"

            try:
                # If a single batch still exceeds 2 MB, split further by size
                if len(batch_csv) > VANTAGE_MAX_BYTES_PER_UPLOAD:
                    await self._upload_size_limited(
                        client, header, batch_lines, filename, batch_num
                    )
                else:
                    batch_filename = f"{filename}.part{batch_num}"
                    await self._upload_csv(client, batch_csv, batch_filename)
            except Exception as e:
                verbose_logger.error(
                    "Vantage destination: batch %d failed: %s", batch_num, e
                )
                if first_error is None:
                    first_error = e
            batch_num += 1

        if first_error is not None:
            raise first_error

    async def _upload_size_limited(
        self,
        client: AsyncHTTPHandler,
        header: bytes,
        data_lines: list[bytes],
        filename: str,
        batch_offset: int,
    ) -> None:
        """Upload lines in chunks that stay under the 2 MB size limit.

        Individual rows that exceed the limit on their own are skipped with
        a warning — they cannot be split further.  Sub-batch failures are
        recorded and the first error is re-raised after all sub-batches have
        been attempted, consistent with ``_upload_batched``.
        """
        current_chunk: list[bytes] = []
        current_size = len(header) + 1  # header + newline
        sub_batch = 0
        header_size = len(header) + 1
        first_error: Optional[Exception] = None

        for line in data_lines:
            line_size = len(line) + 1  # line + newline

            # Skip individual rows that exceed the limit on their own
            if header_size + line_size > VANTAGE_MAX_BYTES_PER_UPLOAD:
                verbose_logger.warning(
                    "Vantage destination: skipping oversized row (%d bytes)",
                    line_size,
                )
                continue

            if (
                current_size + line_size > VANTAGE_MAX_BYTES_PER_UPLOAD
                and current_chunk
            ):
                batch_csv = header + b"\n" + b"\n".join(current_chunk) + b"\n"
                batch_filename = f"{filename}.part{batch_offset}_{sub_batch}"
                try:
                    await self._upload_csv(client, batch_csv, batch_filename)
                except Exception as e:
                    verbose_logger.error(
                        "Vantage destination: sub-batch %s failed: %s",
                        batch_filename,
                        e,
                    )
                    if first_error is None:
                        first_error = e
                current_chunk = []
                current_size = header_size
                sub_batch += 1
            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            batch_csv = header + b"\n" + b"\n".join(current_chunk) + b"\n"
            batch_filename = f"{filename}.part{batch_offset}_{sub_batch}"
            try:
                await self._upload_csv(client, batch_csv, batch_filename)
            except Exception as e:
                verbose_logger.error(
                    "Vantage destination: sub-batch %s failed: %s",
                    batch_filename,
                    e,
                )
                if first_error is None:
                    first_error = e

        if first_error is not None:
            raise first_error
