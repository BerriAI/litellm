import asyncio
import importlib
import logging
import os
import time
from collections.abc import Iterator
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.fixture(autouse=True)
def isolate_opik_logging_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    original_level = verbose_logger.level
    verbose_logger.setLevel(logging.DEBUG)
    monkeypatch.setattr(litellm, "set_verbose", True)
    yield
    verbose_logger.setLevel(original_level)


INTERVAL_TOO_LONG_TO_FIRE_DURING_THIS_TEST = 3600


@pytest.mark.asyncio
async def test_opik_logging_http_request():
    """
    - Test that HTTP requests are made to Opik
    - Traces and spans are batched correctly
    """
    from litellm.integrations.opik.opik import OpikLogger

    os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
    os.environ["OPIK_API_KEY"] = "anything"
    os.environ["OPIK_WORKSPACE"] = "anything"

    test_opik_logger = OpikLogger()
    test_opik_logger.flush_interval = INTERVAL_TOO_LONG_TO_FIRE_DURING_THIS_TEST
    test_opik_logger.batch_size = 12

    litellm.callbacks = [test_opik_logger]

    mock_post = AsyncMock(return_value=Mock(status_code=202, text="Accepted"))
    test_opik_logger.async_httpx_client.post = mock_post

    def opik_batch_calls():
        return [
            call for call in mock_post.call_args_list if "/traces/batch" in str(call) or "/spans/batch" in str(call)
        ]

    for _ in range(5):
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
        )
    await asyncio.sleep(1)

    assert opik_batch_calls() == [], "events below batch_size must stay queued"
    assert len(test_opik_logger.log_queue) == 10

    for _ in range(3):
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
        )
    await asyncio.sleep(1)

    assert opik_batch_calls(), "crossing batch_size must flush the queue"
    events_left_over_after_the_size_triggered_flush = len(test_opik_logger.log_queue)
    assert 0 < events_left_over_after_the_size_triggered_flush < test_opik_logger.batch_size

    calls_before_periodic_flush = len(opik_batch_calls())
    await test_opik_logger.flush_queue()

    assert len(opik_batch_calls()) > calls_before_periodic_flush
    assert len(test_opik_logger.log_queue) == 0


def test_sync_opik_logging_http_request():
    """
    - Test that HTTP requests are made to Opik
    - Traces and spans are batched correctly
    """
    try:
        from litellm.integrations.opik.opik import OpikLogger

        os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
        os.environ["OPIK_API_KEY"] = "anything"
        os.environ["OPIK_WORKSPACE"] = "anything"

        # Initialize OpikLogger
        test_opik_logger = OpikLogger()

        litellm.callbacks = [test_opik_logger]
        litellm.set_verbose = True

        # Create a mock for the clients's post method
        mock_post = Mock()
        mock_post.return_value.status_code = 204
        mock_post.return_value.text = "Accepted"
        test_opik_logger.sync_httpx_client.post = mock_post

        # Make multiple calls to ensure we don't hit the batch size
        for _ in range(5):
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        # Need to wait for a short amount of time as the log_success callback is called in a different thread. One or two seconds is often not enough.
        time.sleep(3)

        # Check that 5 spans and 5 traces were sent
        assert mock_post.call_count == 10, f"Expected 10 HTTP requests, but got {mock_post.call_count}"

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_opik_attach_to_existing_trace():
    """
    Test attaching spans to existing trace (regression fix for PR #14888)

    - When trace_id is provided via current_span_data, only create a span
    - Do NOT create a new trace (this was the bug)
    - Verify span has correct trace_id and parent_span_id
    """
    try:
        from litellm.integrations.opik.opik import OpikLogger

        os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
        os.environ["OPIK_API_KEY"] = "anything"
        os.environ["OPIK_WORKSPACE"] = "anything"

        # Initialize OpikLogger
        test_opik_logger = OpikLogger()
        litellm.callbacks = [test_opik_logger]
        litellm.set_verbose = True

        # Create a mock for the sync client's post method
        mock_post = Mock()
        mock_post.return_value.status_code = 204
        mock_post.return_value.text = "Accepted"
        test_opik_logger.sync_httpx_client.post = mock_post

        # Simulate an existing trace and parent span
        existing_trace_id = "existing-trace-12345"
        existing_parent_span_id = "existing-span-67890"

        # Make a completion call with existing trace_id
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
            metadata={
                "opik": {
                    "current_span_data": {
                        "trace_id": existing_trace_id,
                        "id": existing_parent_span_id,
                    },
                    "tags": ["test-attach-span"],
                }
            },
        )

        # Need to wait for a short amount of time as the log_success callback is called in a different thread
        time.sleep(3)

        # Check the calls made to the mock
        calls_made = mock_post.call_args_list
        trace_calls = [call for call in calls_made if "/traces/batch" in str(call)]
        span_calls = [call for call in calls_made if "/spans/batch" in str(call)]

        # With the fix, when trace_id is provided, we should NOT create a new trace
        assert len(trace_calls) == 0, (
            f"Expected 0 trace calls when attaching to existing trace, but got {len(trace_calls)}"
        )
        assert len(span_calls) == 1, f"Expected exactly 1 span call, but got {len(span_calls)}"

        # Verify span has correct trace_id and parent_span_id
        span_payload = span_calls[0][1]["json"]["spans"][0]
        assert span_payload["trace_id"] == existing_trace_id, (
            f"Expected trace_id to be {existing_trace_id}, but got {span_payload['trace_id']}"
        )
        assert span_payload["parent_span_id"] == existing_parent_span_id, (
            f"Expected parent_span_id to be {existing_parent_span_id}, but got {span_payload['parent_span_id']}"
        )
        assert "test-attach-span" in span_payload["tags"], f"Expected 'test-attach-span' tag in {span_payload['tags']}"

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_opik_create_new_trace():
    """
    Test normal trace creation when no trace_id is provided

    - When NO trace_id is provided, create both a new trace and a new span
    - Verify the span references the created trace
    - Verify tags are included in both trace and span
    """
    try:
        from litellm.integrations.opik.opik import OpikLogger

        os.environ["OPIK_URL_OVERRIDE"] = "https://fake.comet.com/opik/api"
        os.environ["OPIK_API_KEY"] = "anything"
        os.environ["OPIK_WORKSPACE"] = "anything"

        # Initialize OpikLogger
        test_opik_logger = OpikLogger()
        litellm.callbacks = [test_opik_logger]
        litellm.set_verbose = True

        # Create a mock for the sync client's post method
        mock_post = Mock()
        mock_post.return_value.status_code = 204
        mock_post.return_value.text = "Accepted"
        test_opik_logger.sync_httpx_client.post = mock_post

        # Make a completion call WITHOUT providing trace_id
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
            metadata={"opik": {"tags": ["test-new-trace"]}},
        )

        # Need to wait for a short amount of time as the log_success callback is called in a different thread
        time.sleep(3)

        # Check the calls made to the mock
        calls_made = mock_post.call_args_list
        trace_calls = [call for call in calls_made if "/traces/batch" in str(call)]
        span_calls = [call for call in calls_made if "/spans/batch" in str(call)]

        # Without trace_id provided, we should create both a new trace and a new span
        assert len(trace_calls) == 1, f"Expected exactly 1 trace call, but got {len(trace_calls)}"
        assert len(span_calls) == 1, f"Expected exactly 1 span call, but got {len(span_calls)}"

        # Verify the span references the created trace
        trace_payload = trace_calls[0][1]["json"]["traces"][0]
        span_payload = span_calls[0][1]["json"]["spans"][0]
        assert span_payload["trace_id"] == trace_payload["id"], "Span should reference the created trace"

        # Verify tags are included in both trace and span
        assert "test-new-trace" in trace_payload["tags"], f"Expected 'test-new-trace' tag in trace tags"
        assert "test-new-trace" in span_payload["tags"], f"Expected 'test-new-trace' tag in span tags"

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server as proxy_server

                importlib.reload(proxy_server)
        except Exception as e:
            print(f"Error reloading litellm.proxy.proxy_server: {e}")
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield
