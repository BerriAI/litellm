import sys
import os
import io

sys.path.insert(0, os.path.abspath("../.."))

from litellm import completion
import litellm

litellm.failure_callback = ["lunary"]
litellm.success_callback = ["lunary"]
litellm.set_verbose = True


def test_lunary_logging():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            user="test-user",
        )
        print(response)
    except Exception as e:
        print(e)


# test_lunary_logging()


def test_lunary_template():
    import lunary

    try:
        template = lunary.render_template("test-template", {"question": "Hello!"})
        response = completion(**template)
        print(response)
    except Exception as e:
        print(e)


# test_lunary_template()


def test_lunary_logging_with_metadata():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            metadata={
                "run_name": "litellmRUN",
                "project_name": "litellm-completion",
            },
        )
        print(response)
    except Exception as e:
        print(e)


# test_lunary_logging_with_metadata()


def test_lunary_logging_with_streaming_and_metadata():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what llm are u"}],
            max_tokens=10,
            temperature=0.2,
            metadata={
                "run_name": "litellmRUN",
                "project_name": "litellm-completion",
            },
            stream=True,
        )
        for chunk in response:
            continue
    except Exception as e:
        print(e)


# test_lunary_logging_with_streaming_and_metadata()
