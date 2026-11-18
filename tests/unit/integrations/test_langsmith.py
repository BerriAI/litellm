import asyncio
import importlib
import json
import os
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

import litellm
from litellm.constants import LOGGING_WORKER_MAX_TIME_PER_COROUTINE
from litellm.integrations.langsmith import CredentialsKey, LangsmithLogger, LangsmithQueueObject
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_get_credentials_from_env_does_not_use_env_for_dynamic_base_url(
    monkeypatch,
):
    monkeypatch.setenv("LANGSMITH_API_KEY", "global-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "global-project")
    monkeypatch.setenv("LANGSMITH_TENANT_ID", "global-tenant")
    logger = LangsmithLogger(
        langsmith_api_key="default-key",
        langsmith_project="default-project",
        langsmith_base_url="https://default.example",
    )

    credentials = logger.get_credentials_from_env(
        langsmith_base_url="https://attacker.example",
        allow_env_credentials=False,
    )

    assert credentials["LANGSMITH_API_KEY"] is None
    assert credentials["LANGSMITH_PROJECT"] == "litellm-completion"
    assert credentials["LANGSMITH_BASE_URL"] == "https://attacker.example"
    assert credentials["LANGSMITH_TENANT_ID"] is None


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_dynamic_langsmith_base_url_does_not_inherit_default_api_key(
    monkeypatch,
):
    monkeypatch.setenv("LANGSMITH_API_KEY", "global-key")
    logger = LangsmithLogger(
        langsmith_api_key="default-key",
        langsmith_project="default-project",
        langsmith_base_url="https://default.example",
    )

    credentials = logger._get_credentials_to_use_for_request(
        kwargs={"standard_callback_dynamic_params": {"langsmith_base_url": "https://attacker.example"}}
    )

    assert credentials["LANGSMITH_API_KEY"] is None
    assert credentials["LANGSMITH_BASE_URL"] == "https://attacker.example"


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest_asyncio.fixture(loop_scope="function")
async def drain_logging_worker(isolate_litellm_state: None) -> AsyncIterator[None]:
    yield
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS)


LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS: Final = LOGGING_WORKER_MAX_TIME_PER_COROUTINE + 5.0


@pytest.fixture(scope="function")
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm state to the true defaults captured at conftest import time,
    then restores after the test. This prevents module-level mutations (e.g.
    `litellm.num_retries = 3` at the top of test_langfuse_e2e_test.py) from
    leaking across tests within the same xdist worker.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging
    from litellm.proxy.management_helpers import audit_logs as ll_audit_logs

    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])


_LIST_ATTRS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
    "pre_call_rules",
    "post_call_rules",
)

_SCALAR_ATTRS = (
    "set_verbose",
    "cache",
    "num_retries",
    "num_retries_per_request",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "s3_audit_callback_params",
    "datadog_params",
    "vector_store_registry",
)

_DEFAULTS: dict = {}


@pytest.fixture(scope="module")
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield


# Test get_credentials_from_env
@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_get_credentials_from_env():
    # Test with direct parameters
    logger = LangsmithLogger(
        langsmith_api_key="test-key",
        langsmith_project="test-project",
        langsmith_base_url="http://test-url",
    )

    credentials = logger.get_credentials_from_env(
        langsmith_api_key="custom-key",
        langsmith_project="custom-project",
        langsmith_base_url="http://custom-url",
    )

    assert credentials["LANGSMITH_API_KEY"] == "custom-key"
    assert credentials["LANGSMITH_PROJECT"] == "custom-project"
    assert credentials["LANGSMITH_BASE_URL"] == "http://custom-url"

    # assert that the default api base is used if not provided
    credentials = logger.get_credentials_from_env()
    assert credentials["LANGSMITH_BASE_URL"] == "https://api.smith.langchain.com"

    # Test with tenant_id
    credentials = logger.get_credentials_from_env(langsmith_tenant_id="test-tenant-id")
    assert credentials["LANGSMITH_TENANT_ID"] == "test-tenant-id"

    # Test tenant_id from environment variable

    os.environ["LANGSMITH_TENANT_ID"] = "env-tenant-id"
    credentials = logger.get_credentials_from_env()
    assert credentials["LANGSMITH_TENANT_ID"] == "env-tenant-id"
    del os.environ["LANGSMITH_TENANT_ID"]


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_group_batches_by_credentials():

    logger = LangsmithLogger(langsmith_api_key="test-key")

    # Create test queue objects
    queue_obj1 = LangsmithQueueObject(
        data={"test": "data1"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    logger.log_queue = [queue_obj1, queue_obj2]

    grouped = logger._group_batches_by_credentials()

    # Check grouping
    assert len(grouped) == 1  # Should have one group since credentials are same
    key = list(grouped.keys())[0]
    assert isinstance(key, CredentialsKey)
    assert len(grouped[key].queue_objects) == 2


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_group_batches_by_credentials_multiple_credentials():

    # Test with multiple different credentials
    logger = LangsmithLogger(langsmith_api_key="test-key")

    queue_obj1 = LangsmithQueueObject(
        data={"test": "data1"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key2",  # Different API key
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    queue_obj3 = LangsmithQueueObject(
        data={"test": "data3"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj2",  # Different project
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    logger.log_queue = [queue_obj1, queue_obj2, queue_obj3]

    grouped = logger._group_batches_by_credentials()

    # Check grouping
    assert len(grouped) == 3  # Should have three groups since credentials differ
    for key, batch_group in grouped.items():
        assert isinstance(key, CredentialsKey)
        assert len(batch_group.queue_objects) == 1  # Each group should have one object


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_group_batches_by_credentials_with_tenant_id():

    # Test that different tenant_ids create separate groups
    logger = LangsmithLogger(langsmith_api_key="test-key")

    queue_obj1 = LangsmithQueueObject(
        data={"test": "data1"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": "tenant1",
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": "tenant2",  # Different tenant_id
        },
    )

    queue_obj3 = LangsmithQueueObject(
        data={"test": "data3"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": "tenant1",  # Same as queue_obj1
        },
    )

    logger.log_queue = [queue_obj1, queue_obj2, queue_obj3]

    grouped = logger._group_batches_by_credentials()

    # Should have two groups: one for tenant1 (queue_obj1 and queue_obj3), one for tenant2 (queue_obj2)
    assert len(grouped) == 2
    for key, batch_group in grouped.items():
        assert isinstance(key, CredentialsKey)
        assert key.tenant_id in ["tenant1", "tenant2"]
        if key.tenant_id == "tenant1":
            assert len(batch_group.queue_objects) == 2
        else:
            assert len(batch_group.queue_objects) == 1


# Test make_dot_order
@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_make_dot_order():
    logger = LangsmithLogger(langsmith_api_key="test-key")
    run_id = "729cff0e-f30c-4336-8b79-45d6b61c64b4"
    dot_order = logger.make_dot_order(run_id)

    print("dot_order=", dot_order)

    # Check format: YYYYMMDDTHHMMSSfffZ + run_id
    # Check the timestamp portion (first 23 characters)
    timestamp_part = dot_order[:-36]  # 36 is length of run_id
    assert len(timestamp_part) == 22
    assert timestamp_part[8] == "T"  # Check T separator
    assert timestamp_part[-1] == "Z"  # Check Z suffix

    # Verify timestamp format
    try:
        # Parse the timestamp portion (removing the Z)
        datetime.strptime(timestamp_part[:-1], "%Y%m%dT%H%M%S%f")
    except ValueError:
        pytest.fail("Timestamp portion is not in correct format")

    # Verify run_id portion
    assert dot_order[-36:] == run_id


# Test is_serializable
@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_is_serializable():
    from pydantic import BaseModel

    from litellm.integrations.langsmith import is_serializable

    # Test basic types
    assert is_serializable("string") is True
    assert is_serializable(123) is True
    assert is_serializable({"key": "value"}) is True

    # Test non-serializable types
    async def async_func():
        pass

    assert is_serializable(async_func) is False

    class TestModel(BaseModel):
        field: str

    assert is_serializable(TestModel(field="test")) is False


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_async_send_batch():
    logger = LangsmithLogger(langsmith_api_key="test-key")

    # Mock the httpx client
    mock_response = AsyncMock()
    mock_response.status_code = 200
    logger.async_httpx_client = AsyncMock()
    logger.async_httpx_client.post.return_value = mock_response

    # Add test data to queue
    logger.log_queue = [LangsmithQueueObject(data={"test": "data"}, credentials=logger.default_credentials)]

    await logger.async_send_batch()

    # Verify the API call
    logger.async_httpx_client.post.assert_called_once()
    call_args = logger.async_httpx_client.post.call_args
    assert "runs/batch" in call_args[1]["url"]
    assert "x-api-key" in call_args[1]["headers"]
    # tenant_id should not be in headers if not provided
    assert "x-tenant-id" not in call_args[1]["headers"]


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_async_send_batch_with_tenant_id():
    logger = LangsmithLogger(langsmith_api_key="test-key", langsmith_tenant_id="test-tenant-id")

    # Mock the httpx client
    mock_response = AsyncMock()
    mock_response.status_code = 200
    logger.async_httpx_client = AsyncMock()
    logger.async_httpx_client.post.return_value = mock_response

    # Add test data to queue
    logger.log_queue = [LangsmithQueueObject(data={"test": "data"}, credentials=logger.default_credentials)]

    await logger.async_send_batch()

    # Verify the API call includes tenant_id header
    logger.async_httpx_client.post.assert_called_once()
    call_args = logger.async_httpx_client.post.call_args
    assert "runs/batch" in call_args[1]["url"]
    assert "x-api-key" in call_args[1]["headers"]
    assert "x-tenant-id" in call_args[1]["headers"]
    assert call_args[1]["headers"]["x-tenant-id"] == "test-tenant-id"


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_langsmith_key_based_logging():
    """
    In key based logging langsmith_api_key and langsmith_project are passed directly to litellm.acompletion
    """
    try:
        # Mock the httpx post request
        # We need to mock get_async_httpx_client to return a mock AsyncHTTPHandler
        # because LangsmithLogger creates its own instance
        mock_async_httpx_handler = AsyncMock()
        mock_response = MagicMock()  # Use MagicMock for response to allow sync methods
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()  # raise_for_status is sync in httpx
        mock_response.text = ""
        mock_async_httpx_handler.post = AsyncMock(return_value=mock_response)

        mock_get_client = patch(
            "litellm.integrations.langsmith.get_async_httpx_client",
            return_value=mock_async_httpx_handler,
        )
        mock_get_client.start()

        litellm.set_verbose = True
        litellm.DEFAULT_FLUSH_INTERVAL_SECONDS = 1

        litellm.callbacks = [LangsmithLogger()]
        response = await litellm.acompletion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
            langsmith_api_key="fake_key_project2",
            langsmith_project="fake_project2",
        )
        print("Waiting for logs to be flushed to Langsmith.....")
        await asyncio.sleep(3)

        print("done sleeping 3 seconds...")

        # Verify the post request was made with correct parameters
        mock_async_httpx_handler.post.assert_called_once()
        call_args = mock_async_httpx_handler.post.call_args

        print("call_args", call_args)

        # Check URL contains /runs/batch
        assert "/runs/batch" in call_args[1]["url"]

        # Check headers contain the correct API key
        assert call_args[1]["headers"]["x-api-key"] == "fake_key_project2"
        # tenant_id should not be in headers if not provided
        assert "x-tenant-id" not in call_args[1]["headers"]

        assert call_args[1]["headers"]["Content-Type"] == "application/json"

        # Verify the request body contains the expected data
        request_body = json.loads(call_args[1]["content"])
        assert "post" in request_body
        assert len(request_body["post"]) == 1

        # EXPECTED BODY
        expected_body = {
            "post": [
                {
                    "name": "LLMRun",
                    "run_type": "llm",
                    "inputs": {
                        "id": "chatcmpl-82699ee4-7932-4fc0-9585-76abc8caeafa",
                        "call_type": "acompletion",
                        "model": "gpt-4.1-mini",
                        "messages": [{"role": "user", "content": "Test message"}],
                        "model_parameters": {
                            "temperature": 0.2,
                            "max_tokens": 10,
                        },
                    },
                    "outputs": {
                        "id": "chatcmpl-82699ee4-7932-4fc0-9585-76abc8caeafa",
                        "model": "gpt-4.1-mini",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "message": {
                                    "content": "This is a mock response",
                                    "role": "assistant",
                                    "tool_calls": None,
                                    "function_call": None,
                                },
                            }
                        ],
                        "usage": {
                            "completion_tokens": 20,
                            "prompt_tokens": 10,
                            "total_tokens": 30,
                        },
                    },
                    "session_name": "fake_project2",
                }
            ]
        }

        # Print both bodies for debugging
        actual_body = json.loads(call_args[1]["content"])
        print("\nExpected body:")
        print(json.dumps(expected_body, indent=2))
        print("\nActual body:")
        print(json.dumps(actual_body, indent=2))

        assert len(actual_body["post"]) == 1

        # Assert only the critical parts we care about
        assert actual_body["post"][0]["name"] == expected_body["post"][0]["name"]
        assert actual_body["post"][0]["run_type"] == expected_body["post"][0]["run_type"]
        assert actual_body["post"][0]["inputs"]["messages"] == expected_body["post"][0]["inputs"]["messages"]
        assert (
            actual_body["post"][0]["inputs"]["model_parameters"]
            == expected_body["post"][0]["inputs"]["model_parameters"]
        )
        assert actual_body["post"][0]["outputs"]["choices"] == expected_body["post"][0]["outputs"]["choices"]
        assert (
            actual_body["post"][0]["outputs"]["usage"]["completion_tokens"]
            == expected_body["post"][0]["outputs"]["usage"]["completion_tokens"]
        )
        assert (
            actual_body["post"][0]["outputs"]["usage"]["prompt_tokens"]
            == expected_body["post"][0]["outputs"]["usage"]["prompt_tokens"]
        )
        assert (
            actual_body["post"][0]["outputs"]["usage"]["total_tokens"]
            == expected_body["post"][0]["outputs"]["usage"]["total_tokens"]
        )
        assert actual_body["post"][0]["session_name"] == expected_body["post"][0]["session_name"]

        mock_get_client.stop()

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.usefixtures("_vcr_outcome_gate", "drain_logging_worker", "isolate_litellm_state", "setup_and_teardown")
@pytest.mark.asyncio
async def test_langsmith_queue_logging():
    try:
        # Initialize LangsmithLogger
        test_langsmith_logger = LangsmithLogger()

        litellm.callbacks = [test_langsmith_logger]
        test_langsmith_logger.batch_size = 6
        litellm.set_verbose = True

        # Make multiple calls to ensure we don't hit the batch size
        for _ in range(5):
            response = await litellm.acompletion(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        # Poll for async callbacks to complete (up to 10s)
        for _ in range(20):
            if len(test_langsmith_logger.log_queue) >= 5:
                break
            await asyncio.sleep(0.5)

        # Check that logs are in the queue
        assert len(test_langsmith_logger.log_queue) == 5

        # Now make calls to exceed the batch size
        for _ in range(3):
            response = await litellm.acompletion(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        # Poll for flush to complete (up to 10s)
        for _ in range(20):
            if len(test_langsmith_logger.log_queue) < 5:
                break
            await asyncio.sleep(0.5)

        print("Length of langsmith log queue: {}".format(len(test_langsmith_logger.log_queue)))
        # Check that the queue was flushed after exceeding batch size
        assert len(test_langsmith_logger.log_queue) < 5

        # Clean up
        for cb in litellm.callbacks:
            if isinstance(cb, LangsmithLogger):
                await cb.async_httpx_client.client.aclose()

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
