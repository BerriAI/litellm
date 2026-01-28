# conftest.py

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
import litellm
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    curr_dir = os.getcwd()  # Get the current working directory
    sys.path.insert(0, os.path.abspath("../.."))  # Adds the project directory to the system path

    import litellm
    from litellm import Router

    importlib.reload(litellm)
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    # from litellm import Router, completion, aembedding, acompletion, embedding
    yield

    # Teardown code (executes after the yield point)
    # Ensure we stop background logging worker tasks bound to this event loop.
    # Otherwise, pytest may close the loop with pending tasks ("Event loop is closed").
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        loop.run_until_complete(GLOBAL_LOGGING_WORKER.stop())
    except Exception:
        pass

    loop.close()  # Close the loop created earlier
    asyncio.set_event_loop(None)  # Remove the reference to the loop


def pytest_collection_modifyitems(config, items):
    # Skip Anthropic-dependent tests when ANTHROPIC_API_KEY is not available.
    # Fork repos often don't have access to upstream secrets.
    if not os.getenv("ANTHROPIC_API_KEY"):
        skip_anthropic = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
        for item in items:
            if item.name.startswith("test_streaming_responses_api_with_mcp_tools") and "[anthropic]" in item.name:
                item.add_marker(skip_anthropic)

    # Skip OpenAI-dependent tests when OPENAI_API_KEY is not available.
    # This test makes REAL LLM calls.
    if not os.getenv("OPENAI_API_KEY"):
        skip_openai = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            if item.name.startswith("test_streaming_responses_api_with_mcp_tools") and "[openai]" in item.name:
                item.add_marker(skip_openai)

    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [item for item in items if "custom_logger" in item.parent.name]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests
