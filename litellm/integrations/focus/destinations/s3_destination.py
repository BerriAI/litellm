"""S3 destination implementation for Focus export."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import timezone
from typing import Final, TypedDict

import boto3
from typing_extensions import ReadOnly

from .base import FocusDestination, FocusTimeWindow


class _S3ClientKwargs(TypedDict, total=False):
    """Optional boto3 client arguments the destination config may supply."""

    region_name: ReadOnly[str]
    endpoint_url: ReadOnly[str]
    aws_access_key_id: ReadOnly[str]
    aws_secret_access_key: ReadOnly[str]
    aws_session_token: ReadOnly[str]


class FocusS3Destination(FocusDestination):
    """Handles uploading serialized exports to S3 buckets."""

    def __init__(
        self,
        *,
        prefix: str,
        config: Mapping[str, str] | None = None,
    ) -> None:
        config = config or {}
        bucket_name: Final = config.get("bucket_name")
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
        object_key: Final = self._build_object_key(time_window=time_window, filename=filename)
        await asyncio.to_thread(self._upload, content, object_key)

    def _build_object_key(self, *, time_window: FocusTimeWindow, filename: str) -> str:
        start_utc: Final = time_window.start_time.astimezone(timezone.utc)
        date_component: Final = f"date={start_utc.strftime('%Y-%m-%d')}"
        parts: Final = [self.prefix, date_component]
        if time_window.frequency == "hourly":
            parts.append(f"hour={start_utc.strftime('%H')}")
        key_prefix: Final = "/".join(filter(None, parts))
        return f"{key_prefix}/{filename}" if key_prefix else filename

    def _client_kwargs(self) -> _S3ClientKwargs:
        """Collect the boto3 client arguments the destination config provides."""
        region: Final = self.config.get("region_name")
        endpoint: Final = self.config.get("endpoint_url")
        key_id: Final = self.config.get("aws_access_key_id")
        secret: Final = self.config.get("aws_secret_access_key")
        token: Final = self.config.get("aws_session_token")
        return {
            **(_S3ClientKwargs(region_name=region) if region else _S3ClientKwargs()),
            **(_S3ClientKwargs(endpoint_url=endpoint) if endpoint else _S3ClientKwargs()),
            **(_S3ClientKwargs(aws_access_key_id=key_id) if key_id else _S3ClientKwargs()),
            **(_S3ClientKwargs(aws_secret_access_key=secret) if secret else _S3ClientKwargs()),
            **(_S3ClientKwargs(aws_session_token=token) if token else _S3ClientKwargs()),
        }

    def _upload(self, content: bytes, object_key: str) -> None:
        s3_client: Final = boto3.client("s3", **self._client_kwargs())
        s3_client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=content,
            ContentType="application/octet-stream",
        )
