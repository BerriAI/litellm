import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import logging
from litellm._uuid import uuid

import pytest

import litellm
from litellm import completion
from litellm._logging import verbose_logger
from litellm.integrations.langsmith import LangsmithLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

verbose_logger.setLevel(logging.DEBUG)

litellm.set_verbose = True
import time


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
        litellm.DEFAULT_BATCH_SIZE = 1
        litellm.DEFAULT_FLUSH_INTERVAL_SECONDS = 1
        test_langsmith_logger = LangsmithLogger()
        litellm.success_callback = ["langsmith"]
        litellm.set_verbose = True
        run_id = "497f6eca-6276-4993-bfeb-53cbbbba6f08"
        run_name = "litellmRUN"
        test_metadata = {
            "run_name": run_name,  # langsmith run name
            "run_id": run_id,  # langsmith run id
        }

        messages = [{"role": "user", "content": "what llm are u"}]
        if sync_mode is True:
            response = completion(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=10,
                temperature=0.2,
                stream=True,
                metadata=test_metadata,
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
                metadata=test_metadata,
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

        extra_fields_on_langsmith = logged_run_on_langsmith.get("extra", {}).get(
            "invocation_params"
        )

        assert (
            logged_run_on_langsmith.get("run_type") == "llm"
        ), f"run_type should be llm. Got: {logged_run_on_langsmith.get('run_type')}"
        assert (
            logged_run_on_langsmith.get("name") == run_name
        ), f"run_type should be llm. Got: {logged_run_on_langsmith.get('run_type')}"
        print("\nLogged INPUT ON LANGSMITH", input_fields_on_langsmith)

        print("\nextra fields on langsmith", extra_fields_on_langsmith)

        assert isinstance(input_fields_on_langsmith, dict)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        print(e)
