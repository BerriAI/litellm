import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion
from litellm.integrations.langsmith import LangsmithLogger

litellm.set_verbose = True
import time

test_langsmith_logger = LangsmithLogger()


def test_langsmith_logging():
    try:
        import uuid

        run_id = str(uuid.uuid4())
        litellm.set_verbose = True
        litellm.success_callback = ["langsmith"]
        response = completion(
            model="claude-instant-1.2",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            metadata={"id": run_id},
        )
        print(response)
        time.sleep(3)

        print("run_id", run_id)
        logged_run_on_langsmith = test_langsmith_logger.get_run_by_id(run_id=run_id)

        print("logged_run_on_langsmith", logged_run_on_langsmith)

        print("fields in logged_run_on_langsmith", logged_run_on_langsmith.keys())

        input_fields_on_langsmith = logged_run_on_langsmith.get("inputs")
        extra_fields_on_langsmith = logged_run_on_langsmith.get("extra")

        print("\nLogged INPUT ON LANGSMITH", input_fields_on_langsmith)

        print("\nextra fields on langsmith", extra_fields_on_langsmith)

        assert input_fields_on_langsmith is not None
        assert "api_key" not in input_fields_on_langsmith
        assert "api_key" not in extra_fields_on_langsmith

    except Exception as e:
        print(e)


# test_langsmith_logging()


def test_langsmith_logging_with_metadata():
    try:
        litellm.success_callback = ["langsmith"]
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
        time.sleep(3)
    except Exception as e:
        print(e)


# test_langsmith_logging_with_metadata()


def test_langsmith_logging_with_streaming_and_metadata():
    try:
        litellm.success_callback = ["langsmith"]
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            stream=True,
        )
        for chunk in response:
            continue
        time.sleep(3)
    except Exception as e:
        print(e)


# test_langsmith_logging_with_streaming_and_metadata()
