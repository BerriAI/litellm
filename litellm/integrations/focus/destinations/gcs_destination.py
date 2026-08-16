"""GCS destination for Focus export — reuses GCSBucketBase auth and httpx client."""

from __future__ import annotations

from datetime import timezone
from typing import Any, Optional

from litellm._logging import verbose_logger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm.litellm_core_utils.cloud_storage_security import (
    encode_gcs_object_name_for_url,
)

from .base import FocusDestination, FocusTimeWindow


class FocusGCSDestination(GCSBucketBase, FocusDestination):
    """Upload serialized Focus exports to GCS using the GCS JSON API."""

    def __init__(
        self,
        *,
        prefix: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        bucket_name = config.get("bucket_name")
        if not bucket_name:
            raise ValueError("bucket_name must be provided for GCS destination")
        super().__init__(bucket_name=bucket_name)
        service_account_json = config.get("service_account_json")
        if service_account_json is not None:
            self.path_service_account_json = service_account_json
        self.prefix = prefix.rstrip("/")

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        object_name = self._build_object_key(time_window=time_window, filename=filename)
        headers = await self.construct_request_headers(service_account_json=self.path_service_account_json)
        headers["Content-Type"] = "application/octet-stream"
        encoded_name = encode_gcs_object_name_for_url(object_name)
        url = (
            f"https://storage.googleapis.com/upload/storage/v1/b/"
            f"{self.BUCKET_NAME}/o?uploadType=media&name={encoded_name}"
        )
        response = await self.async_httpx_client.post(url=url, headers=headers, data=content)
        if response.status_code != 200:
            raise RuntimeError(f"GCS upload failed: status={response.status_code} body={response.text}")
        verbose_logger.debug(
            "Focus GCS: uploaded %d bytes to gs://%s/%s",
            len(content),
            self.BUCKET_NAME,
            object_name,
        )

    def _build_object_key(self, *, time_window: FocusTimeWindow, filename: str) -> str:
        start_utc = time_window.start_time.astimezone(timezone.utc)
        date_component = f"date={start_utc.strftime('%Y-%m-%d')}"
        parts = [self.prefix, date_component]
        if time_window.frequency == "hourly":
            parts.append(f"hour={start_utc.strftime('%H')}")
        key_prefix = "/".join(filter(None, parts))
        return f"{key_prefix}/{filename}" if key_prefix else filename
