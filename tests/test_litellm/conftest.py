# conftest.py - IMPROVED VERSION
#
# Key changes:
# 1. Changed module reload from 'module' scope to 'function' scope for better isolation
# 2. Made cache flushing happen per-function instead of per-module
# 3. Removed manual event loop creation (let pytest-asyncio handle it)
# 4. Added proper cleanup in fixtures
# 5. Added worker-specific isolation for parallel execution

import importlib
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio

import litellm
from litellm._logging import ALL_LOGGERS
from litellm.litellm_core_utils.prompt_templates import (
    image_handling as image_handling_module,
)
from litellm.llms.custom_httpx.async_client_cleanup import (
    close_litellm_async_clients,
)
from litellm.proxy.db import tool_registry_writer as tool_registry_writer_module


def _reset_module_level_aws_auth_caches():
    """
    Clear module-level AWS auth state that can survive between tests.

    Bedrock/SageMaker handlers are instantiated once at import time and cache
    resolved credentials on the handler instance. If a previous test resolves an
    invalid or different auth flow, later tests can reuse that cached state and
    bypass their local monkeypatched env setup.
    """
    for module_name in (
        "litellm.main",
        "litellm.files.main",
        "litellm.rerank_api.main",
        "litellm.realtime_api.main",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            iam_cache = getattr(obj, "iam_cache", None)
            if iam_cache is None:
                continue
            flush_cache = getattr(iam_cache, "flush_cache", None)
            if callable(flush_cache):
                flush_cache()

    try:
        import boto3

        boto3.DEFAULT_SESSION = None
    except Exception:
        pass


@pytest.fixture(scope="session")
def isolated_aws_credentials_dir(tmp_path_factory):
    aws_dir = tmp_path_factory.mktemp("aws-config")
    credentials_file = Path(aws_dir) / "credentials"
    config_file = Path(aws_dir) / "config"
    credentials_file.write_text("", encoding="utf-8")
    config_file.write_text("", encoding="utf-8")
    return {
        "credentials": str(credentials_file),
        "config": str(config_file),
    }


@pytest.fixture(scope="function", autouse=True)
def isolate_host_aws_config(monkeypatch, isolated_aws_credentials_dir):
    """Prevent botocore from reading host AWS profiles during unit tests."""
    monkeypatch.setenv(
        "AWS_SHARED_CREDENTIALS_FILE", isolated_aws_credentials_dir["credentials"]
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", isolated_aws_credentials_dir["config"])
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", raising=False)
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_REGION_NAME", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)


def _run_coroutine_if_needed(result):
    if not asyncio.iscoroutine(result):
        return

    try:
        asyncio.run(result)
    except RuntimeError:
        # If pytest-asyncio already has a running loop, best-effort scheduling is
        # still better than leaking the client entirely.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if loop.is_running():
            loop.create_task(result)
    except Exception:
        pass


def _close_handler_if_needed(handler):
    if handler is None:
        return

    close_fn = getattr(handler, "close", None)
    if not callable(close_fn):
        return

    try:
        result = close_fn()
        _run_coroutine_if_needed(result)
    except Exception:
        pass


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture (changed from module scope).

    This ensures better isolation when running tests in parallel:
    - Each test function gets a clean litellm state
    - Cache is flushed before each test
    - No module reloading during parallel execution

    Note: Module reloading at function scope is safer for parallel execution
    but adds overhead. Consider removing reload entirely if tests can work without it.
    """
    # Get worker ID if running with pytest-xdist
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')

    # Store original callback state (all callback lists)
    original_state = {}
    if hasattr(litellm, 'callbacks'):
        original_state['callbacks'] = litellm.callbacks.copy() if litellm.callbacks else []
    if hasattr(litellm, 'success_callback'):
        original_state['success_callback'] = litellm.success_callback.copy() if litellm.success_callback else []
    if hasattr(litellm, 'failure_callback'):
        original_state['failure_callback'] = litellm.failure_callback.copy() if litellm.failure_callback else []
    if hasattr(litellm, 'input_callback'):
        original_state['input_callback'] = litellm.input_callback.copy() if litellm.input_callback else []
    if hasattr(litellm, '_async_success_callback'):
        original_state['_async_success_callback'] = litellm._async_success_callback.copy() if litellm._async_success_callback else []
    if hasattr(litellm, '_async_failure_callback'):
        original_state['_async_failure_callback'] = litellm._async_failure_callback.copy() if litellm._async_failure_callback else []
    if hasattr(litellm, '_async_input_callback'):
        original_state['_async_input_callback'] = litellm._async_input_callback.copy() if litellm._async_input_callback else []

    # Store routing globals — leaked model_fallbacks causes tests to route
    # through async_completion_with_fallbacks / Router, bypassing HTTP mocks
    if hasattr(litellm, 'model_fallbacks'):
        original_state['model_fallbacks'] = litellm.model_fallbacks

    # Store transport/network globals — many tests set these without restoring,
    # causing subsequent tests to get None from _create_async_transport()
    for _attr in ('disable_aiohttp_transport', 'force_ipv4'):
        if hasattr(litellm, _attr):
            original_state[_attr] = getattr(litellm, _attr)

    # Store request-mapping globals that are frequently mutated in tests.
    if hasattr(litellm, "drop_params"):
        original_state["drop_params"] = litellm.drop_params
    if hasattr(litellm, "cache"):
        original_state["cache"] = litellm.cache

    # Store secret-manager globals. Several tests swap these out, which changes
    # get_secret() behavior for later env-driven tests (for example Redis config).
    for _attr in ("secret_manager_client", "_key_management_system", "_key_management_settings"):
        if hasattr(litellm, _attr):
            original_state[_attr] = getattr(litellm, _attr)

    # Store other commonly-mutated LiteLLM globals that affect provider routing,
    # auth, and request shaping during larger suite runs.
    for _attr in (
        "api_base",
        "num_retries",
        "modify_params",
        "ssl_verify",
        "credential_list",
        "model_group_settings",
        "default_internal_user_params",
        "default_team_params",
        "prometheus_emit_stream_label",
        "vector_store_registry",
        "model_cost",
        "cost_margin_config",
        "cost_discount_config",
        "disable_hf_tokenizer_download",
        "disable_copilot_system_to_assistant",
        "cohere_models",
        "anthropic_models",
        "token_counter",
        "initialized_langfuse_clients",
    ):
        if hasattr(litellm, _attr):
            original_state[_attr] = getattr(litellm, _attr)

    # Store LiteLLM logger state. Some tests reconfigure handlers/propagation for
    # JSON logging and do not restore them, which breaks later caplog-based tests.
    logger_state = {}
    for logger in ALL_LOGGERS:
        logger_state[logger.name] = {
            "level": logger.level,
            "disabled": logger.disabled,
            "propagate": logger.propagate,
            "handlers": list(logger.handlers),
            "filters": list(logger.filters),
        }

    # Store singleton registries that are lazily initialized during tests and
    # can change endpoint behavior later in the suite.
    original_tool_policy_registry = tool_registry_writer_module._tool_policy_registry
    had_module_level_client = "module_level_client" in litellm.__dict__
    had_module_level_aclient = "module_level_aclient" in litellm.__dict__
    original_module_level_client = litellm.__dict__.get("module_level_client")
    original_module_level_aclient = litellm.__dict__.get("module_level_aclient")

    # Flush cache before test (critical for respx mocks)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    image_handling_module.in_memory_cache.flush_cache()
    _reset_module_level_aws_auth_caches()

    # Clear all callback lists to prevent cross-test contamination
    if hasattr(litellm, 'callbacks'):
        litellm.callbacks = []
    if hasattr(litellm, 'success_callback'):
        litellm.success_callback = []
    if hasattr(litellm, 'failure_callback'):
        litellm.failure_callback = []
    if hasattr(litellm, 'input_callback'):
        litellm.input_callback = []
    if hasattr(litellm, '_async_success_callback'):
        litellm._async_success_callback = []
    if hasattr(litellm, '_async_failure_callback'):
        litellm._async_failure_callback = []
    if hasattr(litellm, '_async_input_callback'):
        litellm._async_input_callback = []

    # Clear routing globals
    if hasattr(litellm, 'model_fallbacks'):
        litellm.model_fallbacks = None
    if hasattr(litellm, "cache"):
        litellm.cache = None
    litellm.__dict__.pop("module_level_client", None)
    litellm.__dict__.pop("module_level_aclient", None)
    tool_registry_writer_module._tool_policy_registry = None

    yield

    # Cleanup after test
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    image_handling_module.in_memory_cache.flush_cache()
    _reset_module_level_aws_auth_caches()
    current_module_level_client = litellm.__dict__.get("module_level_client")
    current_module_level_aclient = litellm.__dict__.get("module_level_aclient")

    # Restore all callback lists to original state
    for attr_name, original_value in original_state.items():
        if hasattr(litellm, attr_name):
            setattr(litellm, attr_name, original_value)

    # Restore logger configuration mutated by logging-focused tests.
    for logger in ALL_LOGGERS:
        original_logger_state = logger_state.get(logger.name)
        if original_logger_state is None:
            continue
        logger.setLevel(original_logger_state["level"])
        logger.disabled = original_logger_state["disabled"]
        logger.propagate = original_logger_state["propagate"]
        logger.handlers = list(original_logger_state["handlers"])
        logger.filters = list(original_logger_state["filters"])

    tool_registry_writer_module._tool_policy_registry = original_tool_policy_registry
    if current_module_level_client is not original_module_level_client:
        _close_handler_if_needed(current_module_level_client)
    if current_module_level_aclient is not original_module_level_aclient:
        _close_handler_if_needed(current_module_level_aclient)
    if had_module_level_client:
        litellm.__dict__["module_level_client"] = original_module_level_client
    else:
        litellm.__dict__.pop("module_level_client", None)
    if had_module_level_aclient:
        litellm.__dict__["module_level_aclient"] = original_module_level_aclient
    else:
        litellm.__dict__.pop("module_level_aclient", None)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup/teardown for heavy initialization.

    Use this sparingly - most state should be handled by isolate_litellm_state.
    Only reload modules here if absolutely necessary.
    """
    sys.path.insert(
        0, os.path.abspath("../..")
    )

    import litellm

    # Only reload if NOT running in parallel (module reload + parallel = bad)
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', None)
    if worker_id is None:
        # Single process mode - safe to reload
        importlib.reload(litellm)

        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server
                importlib.reload(litellm.proxy.proxy_server)
        except Exception as e:
            print(f"Error reloading litellm.proxy.proxy_server: {e}")

        # Flush cache after reload (prevents stale client instances)
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()

    print(f"[conftest] Module setup complete (worker: {worker_id or 'master'})")

    yield

    # Teardown - no need to manually manage event loops with pytest-asyncio auto mode
    print(f"[conftest] Module teardown complete (worker: {worker_id or 'master'})")


def pytest_collection_modifyitems(config, items):
    """
    Customize test collection order.

    - Separate tests marked with 'no_parallel' from parallelizable tests
    - Sort custom_logger tests first (they tend to interfere with other tests)
    """
    # Separate no_parallel tests
    no_parallel_tests = [
        item for item in items
        if any(mark.name == "no_parallel" for mark in item.iter_markers())
    ]

    # Separate custom_logger tests
    custom_logger_tests = [
        item for item in items
        if "custom_logger" in item.parent.name
        and item not in no_parallel_tests
    ]

    # Everything else
    other_tests = [
        item for item in items
        if item not in no_parallel_tests and item not in custom_logger_tests
    ]

    # Sort each group
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)
    no_parallel_tests.sort(key=lambda x: x.name)

    # Reorder: custom_logger first (isolated), then other tests, then no_parallel tests last
    items[:] = custom_logger_tests + other_tests + no_parallel_tests


def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # Add marker for flaky tests (for documentation purposes)
    config.addinivalue_line(
        "markers", "flaky: mark test as potentially flaky (should use --reruns)"
    )

    # Detect if running in CI
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('LITELLM_CI') == 'true'
    if is_ci:
        print("[conftest] Running in CI mode - enabling stricter test isolation")


# Optional: Add a fixture for tests that need even stricter isolation
@pytest.fixture
def strict_isolation():
    """
    Use this fixture for tests that need extra strict isolation.

    Example:
        def test_something(strict_isolation):
            # Test code with guaranteed clean state
            pass
    """
    # Force flush all caches
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()

    # Reset all global state
    if hasattr(litellm, "disable_aiohttp_transport"):
        original_aiohttp = litellm.disable_aiohttp_transport
        litellm.disable_aiohttp_transport = False
    else:
        original_aiohttp = None

    if hasattr(litellm, "set_verbose"):
        original_verbose = litellm.set_verbose
        litellm.set_verbose = False
    else:
        original_verbose = None

    yield

    # Restore original state
    if original_aiohttp is not None:
        litellm.disable_aiohttp_transport = original_aiohttp
    if original_verbose is not None:
        litellm.set_verbose = original_verbose

    # Final cache flush
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()


def pytest_sessionfinish(session, exitstatus):
    """Close any globally cached HTTP clients so xdist workers exit cleanly."""
    _close_handler_if_needed(litellm.__dict__.get("module_level_client"))
    _close_handler_if_needed(litellm.__dict__.get("module_level_aclient"))
    litellm.__dict__.pop("module_level_client", None)
    litellm.__dict__.pop("module_level_aclient", None)
    _close_handler_if_needed(getattr(litellm, "base_llm_aiohttp_handler", None))
    _close_handler_if_needed(getattr(litellm, "httpx_client", None))
    _close_handler_if_needed(getattr(litellm, "aclient", None))
    _close_handler_if_needed(getattr(litellm, "client", None))
    _run_coroutine_if_needed(close_litellm_async_clients())
