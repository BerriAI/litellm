"""S3 destination implementation for Focus export."""

from __future__ import annotations

import asyncio
from datetime import timezone
from typing import Any, Optional

import boto3

from .base import FocusDestination, FocusTimeWindow


class FocusS3Destination(FocusDestination):
    """Handles uploading serialized exports to S3 buckets."""

    def __init__(
        self,
        *,
        prefix: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        bucket_name = config.get("bucket_name")
        if not bucket_name:
            raise ValueError("bucket_name must be provided for S3 destination")
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self.config = config

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        object_key = self._build_object_key(time_window=time_window, filename=filename)
        await asyncio.to_thread(self._upload, content, object_key)

    def _build_object_key(self, *, time_window: FocusTimeWindow, filename: str) -> str:
        start_utc = time_window.start_time.astimezone(timezone.utc)
        date_component = f"date={start_utc.strftime('%Y-%m-%d')}"
        parts = [self.prefix, date_component]
        if time_window.frequency == "hourly":
            parts.append(f"hour={start_utc.strftime('%H')}")
        key_prefix = "/".join(filter(None, parts))
        return f"{key_prefix}/{filename}" if key_prefix else filename

    def _upload(self, content: bytes, object_key: str) -> None:
        client_kwargs: dict[str, Any] = {}
        region_name = self.config.get("region_name")
        if region_name:
            client_kwargs["region_name"] = region_name
        endpoint_url = self.config.get("endpoint_url")
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        session_kwargs: dict[str, Any] = {}
        for key in (
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
        ):
            if self.config.get(key):
                session_kwargs[key] = self.config[key]

        s3_client = boto3.client("s3", **client_kwargs, **session_kwargs)
        s3_client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=content,
            ContentType="application/octet-stream",
        )
