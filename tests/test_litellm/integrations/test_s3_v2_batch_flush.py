import asyncio
from unittest.mock import Mock, patch

import pytest

from litellm.integrations.s3_v2 import S3Logger
from litellm.types.integrations.s3_v2 import s3BatchLoggingElement


def _s3_batch_element(key: str) -> s3BatchLoggingElement:
    return s3BatchLoggingElement(
        s3_object_key=f"2025-09-14/{key}.json",
        payload={"test": key},
        s3_object_download_filename=f"{key}.json",
    )


def _s3_logger() -> S3Logger:
    with (
        patch("asyncio.create_task"),
        patch(
            "litellm.integrations.s3_v2.CustomBatchLogger.periodic_flush",
            new=lambda self: None,
        ),
    ):
        return S3Logger(
            s3_bucket_name="test-bucket",
            s3_aws_access_key_id="test-key",
            s3_aws_secret_access_key="test-secret",
            s3_region_name="us-east-1",
        )


@pytest.mark.asyncio
async def test_async_send_batch_awaits_all_s3_uploads():
    logger = _s3_logger()
    first = _s3_batch_element("first")
    second = _s3_batch_element("second")
    logger.log_queue = [first, second]
    completed_uploads = []

    async def upload(payload, raise_on_error=False):
        assert raise_on_error is True
        await asyncio.sleep(0)
        completed_uploads.append(payload.s3_object_key)

    logger.async_upload_data_to_s3 = upload

    await logger.async_send_batch()

    assert set(completed_uploads) == {first.s3_object_key, second.s3_object_key}


@pytest.mark.asyncio
async def test_async_send_batch_noops_when_queue_is_empty():
    logger = _s3_logger()
    logger.async_upload_data_to_s3 = Mock()

    await logger.async_send_batch()

    logger.async_upload_data_to_s3.assert_not_called()


@pytest.mark.asyncio
async def test_flush_queue_preserves_failed_s3_uploads_for_retry():
    logger = _s3_logger()
    first = _s3_batch_element("first")
    failed = _s3_batch_element("failed")
    added_during_flush = _s3_batch_element("added-during-flush")
    logger.log_queue = [first, failed]

    async def upload(payload, raise_on_error=False):
        assert raise_on_error is True
        if payload is failed:
            logger.log_queue.append(added_during_flush)
            raise RuntimeError("s3 upload failed")

    logger.async_upload_data_to_s3 = upload

    await logger.flush_queue()

    assert logger.log_queue == [failed, added_during_flush]


@pytest.mark.asyncio
async def test_flush_queue_preserves_s3_events_added_during_successful_flush():
    logger = _s3_logger()
    first = _s3_batch_element("first")
    second = _s3_batch_element("second")
    added_during_flush = _s3_batch_element("added-during-flush")
    logger.log_queue = [first, second]

    async def upload(payload, raise_on_error=False):
        assert raise_on_error is True
        if payload is first:
            logger.log_queue.append(added_during_flush)

    logger.async_upload_data_to_s3 = upload

    await logger.flush_queue()

    assert logger.log_queue == [added_during_flush]


@pytest.mark.asyncio
async def test_async_upload_data_to_s3_reraises_without_callback_failure():
    logger = _s3_logger()
    logger.get_credentials = Mock(side_effect=RuntimeError("credential failure"))
    logger.handle_callback_failure = Mock()

    with pytest.raises(RuntimeError, match="credential failure"):
        await logger.async_upload_data_to_s3(
            _s3_batch_element("failed"),
            raise_on_error=True,
        )

    logger.handle_callback_failure.assert_not_called()
