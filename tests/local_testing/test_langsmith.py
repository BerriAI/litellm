import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import logging
import uuid

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.langsmith import LangsmithLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

verbose_logger.setLevel(logging.DEBUG)

litellm.set_verbose = True
import time


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
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        await asyncio.sleep(3)

        # Check that logs are in the queue
        assert len(test_langsmith_logger.log_queue) == 5

        # Now make calls to exceed the batch size
        for _ in range(3):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        # Wait a short time for any asynchronous operations to complete
        await asyncio.sleep(1)

        print(
            "Length of langsmith log queue: {}".format(
                len(test_langsmith_logger.log_queue)
            )
        )
        # Check that the queue was flushed after exceeding batch size
        assert len(test_langsmith_logger.log_queue) < 5

        # Clean up
        for cb in litellm.callbacks:
            if isinstance(cb, LangsmithLogger):
                await cb.async_httpx_client.client.aclose()

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Flaky test. covered by unit tests on custom logger.")
@pytest.mark.asyncio()
async def test_async_langsmith_logging():
    try:
        test_langsmith_logger = LangsmithLogger()
        run_id = str(uuid.uuid4())
        litellm.set_verbose = True
        litellm.callbacks = ["langsmith"]
        response = await litellm.acompletion(
            model="claude-instant-1.2",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            metadata={
                "id": run_id,
                "tags": ["tag1", "tag2"],
                "user_api_key": "6eb81e014497d89f3cc1aa9da7c2b37bda6b7fea68e4b710d33d94201e68970c",
                "user_api_key_alias": "ishaans-langmsith-key",
                "user_api_end_user_max_budget": None,
                "litellm_api_version": "1.40.19",
                "global_max_parallel_requests": None,
                "user_api_key_user_id": "admin",
                "user_api_key_org_id": None,
                "user_api_key_team_id": "dbe2f686-a686-4896-864a-4c3924458709",
                "user_api_key_team_alias": "testing-team",
            },
        )
        print(response)
        await asyncio.sleep(3)

        print("run_id", run_id)
        logged_run_on_langsmith = test_langsmith_logger.get_run_by_id(run_id=run_id)

        print("logged_run_on_langsmith", logged_run_on_langsmith)

        print("fields in logged_run_on_langsmith", logged_run_on_langsmith.keys())

        input_fields_on_langsmith = logged_run_on_langsmith.get("inputs")
        extra_fields_on_langsmith = logged_run_on_langsmith.get("extra").get(
            "invocation_params"
        )

        print("\nLogged INPUT ON LANGSMITH", input_fields_on_langsmith)

        print("\nextra fields on langsmith", extra_fields_on_langsmith)

        assert isinstance(input_fields_on_langsmith, dict)
        assert "api_key" not in input_fields_on_langsmith
        assert "api_key" not in extra_fields_on_langsmith

        # assert user_api_key in extra_fields_on_langsmith
        assert "user_api_key" in extra_fields_on_langsmith
        assert "user_api_key_user_id" in extra_fields_on_langsmith
        assert "user_api_key_team_alias" in extra_fields_on_langsmith

        for cb in litellm.callbacks:
            if isinstance(cb, LangsmithLogger):
                await cb.async_httpx_client.client.aclose()
        # test_langsmith_logger.async_httpx_client.close()

    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


# test_langsmith_logging()


@pytest.mark.skip(reason="Flaky test. covered by unit tests on custom logger.")
def test_async_langsmith_logging_with_metadata():
    try:
        litellm.success_callback = ["langsmith"]
        litellm.set_verbose = True
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
        time.sleep(3)

        for cb in litellm.callbacks:
            if isinstance(cb, LangsmithLogger):
                cb.async_httpx_client.close()

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        print(e)


@pytest.mark.skip(reason="Flaky test. covered by unit tests on custom logger.")
@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_async_langsmith_logging_with_streaming_and_metadata(sync_mode):
    try:
        test_langsmith_logger = LangsmithLogger()
        litellm.success_callback = ["langsmith"]
        litellm.set_verbose = True
        run_id = str(uuid.uuid4())

        messages = [{"role": "user", "content": "what llm are u"}]
        if sync_mode is True:
            response = completion(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=10,
                temperature=0.2,
                stream=True,
                metadata={"id": run_id},
            )
            for cb in litellm.callbacks:
                if isinstance(cb, LangsmithLogger):
                    cb.async_httpx_client = AsyncHTTPHandler()
            for chunk in response:
                continue
            time.sleep(3)
        else:
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock request",
                stream=True,
                metadata={"id": run_id},
            )
            for cb in litellm.callbacks:
                if isinstance(cb, LangsmithLogger):
                    cb.async_httpx_client = AsyncHTTPHandler()
            async for chunk in response:
                continue
            await asyncio.sleep(3)

        print("run_id", run_id)
        logged_run_on_langsmith = test_langsmith_logger.get_run_by_id(run_id=run_id)

        print("logged_run_on_langsmith", logged_run_on_langsmith)

        print("fields in logged_run_on_langsmith", logged_run_on_langsmith.keys())

        input_fields_on_langsmith = logged_run_on_langsmith.get("inputs")

        extra_fields_on_langsmith = logged_run_on_langsmith.get("extra").get(
            "invocation_params"
        )

        assert logged_run_on_langsmith.get("run_type") == "llm"
        print("\nLogged INPUT ON LANGSMITH", input_fields_on_langsmith)

        print("\nextra fields on langsmith", extra_fields_on_langsmith)

        assert isinstance(input_fields_on_langsmith, dict)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        print(e)
