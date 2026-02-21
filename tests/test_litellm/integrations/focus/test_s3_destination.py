"""Tests for FocusS3Destination behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import litellm.integrations.focus.destinations.s3_destination as s3_module
from litellm.integrations.focus.destinations.base import FocusTimeWindow
from litellm.integrations.focus.destinations.s3_destination import FocusS3Destination


def _window(freq: str = "hourly", hour: int = 5) -> FocusTimeWindow:
    start = datetime(2024, 1, 2, hour, tzinfo=timezone.utc)
    end = start.replace(hour=hour + 1)
    return FocusTimeWindow(start_time=start, end_time=end, frequency=freq)


def test_should_require_bucket_name():
    with pytest.raises(ValueError):
        FocusS3Destination(prefix="focus", config={})


def test_should_build_hourly_object_key():
    dest = FocusS3Destination(prefix="exports/", config={"bucket_name": "bucket"})
    key = dest._build_object_key(
        time_window=_window(freq="hourly", hour=3), filename="data.snappy"
    )
    assert key == "exports/date=2024-01-02/hour=03/data.snappy"


def test_should_build_daily_key_without_hour_segment():
    dest = FocusS3Destination(prefix="", config={"bucket_name": "bucket"})
    key = dest._build_object_key(
        time_window=_window(freq="daily", hour=0), filename="daily.parquet"
    )
    assert key == "date=2024-01-02/daily.parquet"


@pytest.mark.asyncio
async def test_should_dispatch_upload_via_thread(monkeypatch: pytest.MonkeyPatch):
    dest = FocusS3Destination(prefix="focus", config={"bucket_name": "bucket"})
    captured: Dict[str, Any] = {}

    async def fake_to_thread(func, *args, **kwargs):  # type: ignore[override]
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(s3_module.asyncio, "to_thread", fake_to_thread)

    window = _window(freq="hourly", hour=1)
    await dest.deliver(content=b"payload", time_window=window, filename="file.bin")

    assert captured["func"] == dest._upload
    assert captured["args"][0] == b"payload"
    assert captured["args"][1].endswith("/file.bin")


def test_should_upload_with_configured_client(monkeypatch: pytest.MonkeyPatch):
    config = {
        "bucket_name": "bucket",
        "region_name": "us-east-2",
        "endpoint_url": "http://localhost:4566",
        "aws_access_key_id": "key",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
    }
    dest = FocusS3Destination(prefix="focus", config=config)
    captured: Dict[str, Any] = {}

    def fake_client(service: str, **kwargs):
        assert service == "s3"
        captured["client_kwargs"] = kwargs

        def put_object(**put_kwargs):
            captured["put_kwargs"] = put_kwargs

        return SimpleNamespace(put_object=put_object)

    monkeypatch.setattr(s3_module.boto3, "client", fake_client)

    dest._upload(content=b"payload", object_key="path/file.bin")

    assert captured["client_kwargs"] == {
        "region_name": "us-east-2",
        "endpoint_url": "http://localhost:4566",
        "aws_access_key_id": "key",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
    }
    assert captured["put_kwargs"] == {
        "Bucket": "bucket",
        "Key": "path/file.bin",
        "Body": b"payload",
        "ContentType": "application/octet-stream",
    }
