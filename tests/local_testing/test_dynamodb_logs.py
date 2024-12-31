import sys
import os
import io, asyncio

# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

from litellm import completion
import litellm

litellm.num_retries = 3

import time, random
import pytest


def pre_request():
    file_name = f"dynamo.log"
    log_file = open(file_name, "a+")

    # Clear the contents of the file by truncating it
    log_file.truncate(0)

    # Save the original stdout so that we can restore it later
    original_stdout = sys.stdout
    # Redirect stdout to the file
    sys.stdout = log_file

    return original_stdout, log_file, file_name


import re


@pytest.mark.skip
def verify_log_file(log_file_path):
    with open(log_file_path, "r") as log_file:
        log_content = log_file.read()
        print(
            f"\nVerifying DynamoDB file = {log_file_path}. File content=", log_content
        )

        # Define the pattern to search for in the log file
        pattern = r"Response from DynamoDB:{.*?}"

        # Find all matches in the log content
        matches = re.findall(pattern, log_content)

        # Print the DynamoDB success log matches
        print("DynamoDB Success Log Matches:")
        for match in matches:
            print(match)

        # Print the total count of lines containing the specified response
        print(f"Total occurrences of specified response: {len(matches)}")

        # Count the occurrences of successful responses (status code 200 or 201)
        success_count = sum(
            1
            for match in matches
            if "'HTTPStatusCode': 200" in match or "'HTTPStatusCode': 201" in match
        )

        # Print the count of successful responses
        print(f"Count of successful responses from DynamoDB: {success_count}")
    assert success_count == 3  # Expect 3 success logs from dynamoDB


@pytest.mark.skip(reason="AWS Suspended Account")
def test_dynamo_logging():
    # all dynamodb requests need to be in one test function
    # since we are modifying stdout, and pytests runs tests in parallel
    try:
        # pre
        # redirect stdout to log_file

        litellm.success_callback = ["dynamodb"]
        litellm.dynamodb_table_name = "litellm-logs-1"
        litellm.set_verbose = True
        original_stdout, log_file, file_name = pre_request()

        print("Testing async dynamoDB logging")

        async def _test():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "This is a test"}],
                max_tokens=100,
                temperature=0.7,
                user="ishaan-2",
            )

        response = asyncio.run(_test())
        print(f"response: {response}")

        # streaming + async
        async def _test2():
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "This is a test"}],
                max_tokens=10,
                temperature=0.7,
                user="ishaan-2",
                stream=True,
            )
            async for chunk in response:
                pass

        asyncio.run(_test2())

        # aembedding()
        async def _test3():
            return await litellm.aembedding(
                model="text-embedding-ada-002", input=["hi"], user="ishaan-2"
            )

        response = asyncio.run(_test3())
        time.sleep(1)
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
    finally:
        # post, close log file and verify
        # Reset stdout to the original value
        sys.stdout = original_stdout
        # Close the file
        log_file.close()
        # verify_log_file(file_name)
        print("Passed! Testing async dynamoDB logging")


# test_dynamo_logging_async()
