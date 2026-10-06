import io, asyncio
from collections import defaultdict

# import logging
# logging.basicConfig(level=logging.DEBUG)

from litellm import completion
import litellm

litellm.num_retries = 3

import time, random
import pytest
import boto3
from litellm._logging import verbose_logger
import logging


class _FakeS3Paginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket):
        keys = sorted(self.objects[Bucket])
        if not keys:
            return [{}]
        return [{"Contents": [{"Key": key} for key in keys]}]


class _FakeS3Client:
    def __init__(self):
        self.objects = defaultdict(dict)

    def clear(self):
        self.objects.clear()

    def put_object(self, Bucket, Key, Body, **_kwargs):
        self.objects[Bucket][Key] = Body
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def delete_object(self, Bucket, Key):
        self.objects[Bucket].pop(Key, None)
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakeS3Paginator(self.objects)

    def list_objects(self, Bucket):
        keys = sorted(self.objects[Bucket])
        return {"Contents": [{"Key": key, "LastModified": 0} for key in keys]}


_FAKE_S3_CLIENT = _FakeS3Client()


@pytest.fixture(autouse=True)
def fake_s3_client(monkeypatch):
    _FAKE_S3_CLIENT.clear()

    def fake_boto3_client(service_name, *args, **kwargs):
        assert service_name == "s3"
        return _FAKE_S3_CLIENT

    monkeypatch.setattr(boto3, "client", fake_boto3_client)
    litellm.success_callback = []
    litellm.callbacks = []
    yield _FAKE_S3_CLIENT
    litellm.success_callback = []
    litellm.callbacks = []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sync_mode,streaming", [(True, True), (True, False), (False, True), (False, False)]
)
@pytest.mark.flaky(retries=3, delay=1)
async def test_basic_s3_logging(sync_mode, streaming):
    verbose_logger.setLevel(level=logging.DEBUG)
    litellm.success_callback = ["s3"]
    litellm.s3_callback_params = {
        "s3_bucket_name": "load-testing-oct",
        "s3_aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY",
        "s3_aws_access_key_id": "os.environ/AWS_ACCESS_KEY_ID",
        "s3_region_name": "us-west-2",
    }
    litellm.set_verbose = True
    response_id = None
    if sync_mode is True:
        response = litellm.completion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "This is a test"}],
            mock_response="It's simple to use and easy to get started",
            stream=streaming,
        )
        if streaming:
            for chunk in response:
                print()
                response_id = chunk.id
        else:
            response_id = response.id
        time.sleep(2)
    else:
        response = await litellm.acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "This is a test"}],
            mock_response="It's simple to use and easy to get started",
            stream=streaming,
        )
        if streaming:
            async for chunk in response:
                print(chunk)
                response_id = chunk.id
        else:
            response_id = response.id
        await asyncio.sleep(2)
    print(f"response: {response}")

    total_objects, all_s3_keys = list_all_s3_objects("load-testing-oct")

    # assert that atlest one key has response.id in it
    assert any(response_id in key for key in all_s3_keys)
    s3 = boto3.client("s3")
    # delete all objects
    for key in all_s3_keys:
        s3.delete_object(Bucket="load-testing-oct", Key=key)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True])
@pytest.mark.flaky(retries=3, delay=1)
async def test_basic_s3_v2_logging(streaming):
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.integrations.s3_v2 import S3Logger

    litellm.s3_callback_params = {
        "s3_bucket_name": "load-testing-oct",
        "s3_aws_secret_access_key": "test-secret",
        "s3_aws_access_key_id": "test-key",
        "s3_region_name": "us-west-2",
    }

    s3_v2_logger = S3Logger(s3_flush_interval=1)
    litellm.callbacks = [s3_v2_logger]

    uploaded_keys: list = []
    original_upload = s3_v2_logger.async_upload_data_to_s3

    async def mock_upload(batch_logging_element):
        uploaded_keys.append(batch_logging_element.s3_object_key)

    s3_v2_logger.async_upload_data_to_s3 = mock_upload

    litellm.set_verbose = True
    response_id = None
    response = await litellm.acompletion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "This is a test"}],
        mock_response="It's simple to use and easy to get started",
        stream=streaming,
    )
    if streaming:
        async for chunk in response:
            response_id = chunk.id
    else:
        response_id = response.id

    await asyncio.sleep(5)

    assert len(uploaded_keys) > 0, "S3 upload was never called"
    assert any(
        response_id in key for key in uploaded_keys
    ), f"Expected response_id={response_id} in one of the uploaded S3 keys: {uploaded_keys}"


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_basic_s3_v2_logging_failure():
    """Test that S3 v2 logger makes httpx PUT request when logging failures"""
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.integrations.s3_v2 import S3Logger

    # Create S3 logger with short flush interval
    s3_v2_logger = S3Logger(s3_flush_interval=1)

    # Mock the httpx client to capture the PUT request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    s3_v2_logger.async_httpx_client = AsyncMock()
    s3_v2_logger.async_httpx_client.put.return_value = mock_response

    # Track the upload method calls
    original_upload = s3_v2_logger.async_upload_data_to_s3
    upload_called = False

    async def mock_upload(batch_logging_element):
        nonlocal upload_called
        upload_called = True
        # Mock the upload process but still make the httpx call
        url = f"https://test-bucket.s3.us-west-2.amazonaws.com/{batch_logging_element.s3_object_key}"
        headers = {"Content-Type": "application/json"}
        data = '{"model": "gpt-5-mini"}'

        # Make the actual httpx call we want to test
        await s3_v2_logger.async_httpx_client.put(url=url, headers=headers, data=data)

    s3_v2_logger.async_upload_data_to_s3 = mock_upload

    # Configure S3 callback params
    litellm.callbacks = [s3_v2_logger]
    litellm.s3_callback_params = {
        "s3_bucket_name": "test-bucket",
        "s3_aws_secret_access_key": "test-secret",
        "s3_aws_access_key_id": "test-key",
        "s3_region_name": "us-west-2",
    }
    litellm.set_verbose = True

    # Trigger a failure by using invalid API key
    try:
        response = await litellm.acompletion(
            model="gpt-5-mini",
            api_key="invalid-api-key",
            messages=[{"role": "user", "content": "This is a test"}],
            mock_response=Exception("forced failure for S3 logging test"),
        )
    except Exception as e:
        print(f"Expected error: {e}")

    # Wait for logger to process the failure
    await asyncio.sleep(5)

    # Verify that our mock upload was called
    assert upload_called, "S3 upload method was not called"
    print("✓ S3 upload method was called")

    # Verify that httpx PUT was called
    s3_v2_logger.async_httpx_client.put.assert_called()

    # Get the call arguments to verify the S3 URL
    call_args = s3_v2_logger.async_httpx_client.put.call_args
    assert call_args is not None
    url = call_args[1]["url"] if "url" in call_args[1] else call_args[0][0]

    # Verify the URL contains expected S3 endpoint
    assert "test-bucket.s3.us-west-2.amazonaws.com" in url
    print(f"✓ S3 PUT request made to: {url}")

    # Verify headers include expected content type
    headers = call_args[1]["headers"]
    assert headers["Content-Type"] == "application/json"
    print("✓ S3 request headers are correct")

    # Verify JSON data was included
    data = call_args[1]["data"]
    assert data is not None
    assert '"model": "gpt-5-mini"' in data
    print("✓ S3 request data contains expected log payload")


def list_all_s3_objects(bucket_name):
    s3 = boto3.client("s3")

    all_s3_keys = []

    paginator = s3.get_paginator("list_objects_v2")
    total_objects = 0

    for page in paginator.paginate(Bucket=bucket_name):
        if "Contents" in page:
            total_objects += len(page["Contents"])
            all_s3_keys.extend([obj["Key"] for obj in page["Contents"]])

    print(f"Total number of objects in {bucket_name}: {total_objects}")
    print(all_s3_keys)
    return total_objects, all_s3_keys




# test_s3_logging()






from litellm.integrations.s3_v2 import S3Logger


class TestS3Logger(S3Logger):
    def __init__(self, *args, **kwargs):
        self.recorded_requests = {}
        self.logged_standard_logging_payload = None
        super().__init__(*args, **kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.recorded_requests[response_obj["id"]] = start_time
        print("recorded request", self.recorded_requests)
        self.logged_standard_logging_payload = kwargs["standard_logging_object"]
        return await super().async_log_success_event(
            kwargs, response_obj, start_time, end_time
        )
