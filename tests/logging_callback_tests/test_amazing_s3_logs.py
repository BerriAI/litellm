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
import boto3
from litellm._logging import verbose_logger
import logging


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
            model="gpt-3.5-turbo",
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
            model="gpt-3.5-turbo",
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
@pytest.mark.parametrize(
    "streaming", [(True)]
)
async def test_basic_s3_v2_logging(streaming):
    from blockbuster import BlockBuster
    from litellm.integrations.s3_v2 import S3Logger
    s3_v2_logger = S3Logger(s3_flush_interval=1)
    litellm.callbacks = [s3_v2_logger]
    blockbuster = BlockBuster()
    blockbuster.activate()

    litellm._turn_on_debug()
    litellm.callbacks = ["s3_v2"]
    litellm.s3_callback_params = {
        "s3_bucket_name": "load-testing-oct",
        "s3_aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY",
        "s3_aws_access_key_id": "os.environ/AWS_ACCESS_KEY_ID",
        "s3_region_name": "us-west-2",
    }
    litellm.set_verbose = True
    response_id = None
    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "This is a test"}],
        stream=streaming,
    )
    if streaming:
        async for chunk in response:
            print(chunk)
            response_id = chunk.id
    else:
        response_id = response.id

    await asyncio.sleep(30)
    print(f"response: {response}")

    # stop blockbuster
    blockbuster.deactivate()

    total_objects, all_s3_keys = list_all_s3_objects("load-testing-oct")

    print(f"all_s3_keys: {all_s3_keys}")

    #assert that atlest one key has response.id in it
    assert any(response_id in key for key in all_s3_keys)
    s3 = boto3.client("s3")
    # delete all objects
    for key in all_s3_keys:
        s3.delete_object(Bucket="load-testing-oct", Key=key)


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


list_all_s3_objects("load-testing-oct")


@pytest.mark.skip(reason="AWS Suspended Account")
def test_s3_logging():
    # all s3 requests need to be in one test function
    # since we are modifying stdout, and pytests runs tests in parallel
    # on circle ci - we only test litellm.acompletion()
    try:
        # redirect stdout to log_file
        litellm.cache = litellm.Cache(
            type="s3",
            s3_bucket_name="litellm-my-test-bucket-2",
            s3_region_name="us-east-1",
        )

        litellm.success_callback = ["s3"]
        litellm.s3_callback_params = {
            "s3_bucket_name": "litellm-logs-2",
            "s3_aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY",
            "s3_aws_access_key_id": "os.environ/AWS_ACCESS_KEY_ID",
        }
        litellm.set_verbose = True

        print("Testing async s3 logging")

        expected_keys = []

        import time

        curr_time = str(time.time())

        async def _test():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"This is a test {curr_time}"}],
                max_tokens=10,
                temperature=0.7,
                user="ishaan-2",
            )

        response = asyncio.run(_test())
        print(f"response: {response}")
        expected_keys.append(response.id)

        async def _test():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"This is a test {curr_time}"}],
                max_tokens=10,
                temperature=0.7,
                user="ishaan-2",
            )

        response = asyncio.run(_test())
        expected_keys.append(response.id)
        print(f"response: {response}")
        time.sleep(5)  # wait 5s for logs to land

        import boto3

        s3 = boto3.client("s3")
        bucket_name = "litellm-logs-2"
        # List objects in the bucket
        response = s3.list_objects(Bucket=bucket_name)

        # Sort the objects based on the LastModified timestamp
        objects = sorted(
            response["Contents"], key=lambda x: x["LastModified"], reverse=True
        )
        # Get the keys of the most recent objects
        most_recent_keys = [obj["Key"] for obj in objects]
        print(most_recent_keys)
        # for each key, get the part before "-" as the key. Do it safely
        cleaned_keys = []
        for key in most_recent_keys:
            split_key = key.split("_")
            if len(split_key) < 2:
                continue
            cleaned_keys.append(split_key[1])
        print("\n most recent keys", most_recent_keys)
        print("\n cleaned keys", cleaned_keys)
        print("\n Expected keys: ", expected_keys)
        matches = 0
        for key in expected_keys:
            key += ".json"
            assert key in cleaned_keys

            if key in cleaned_keys:
                matches += 1
                # remove the match key
                cleaned_keys.remove(key)
        # this asserts we log, the first request + the 2nd cached request
        print("we had two matches ! passed ", matches)
        assert matches == 2
        try:
            # cleanup s3 bucket in test
            for key in most_recent_keys:
                s3.delete_object(Bucket=bucket_name, Key=key)
        except Exception:
            # don't let cleanup fail a test
            pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
    finally:
        # post, close log file and verify
        # Reset stdout to the original value
        print("Passed! Testing async s3 logging")


# test_s3_logging()


@pytest.mark.skip(reason="AWS Suspended Account")
def test_s3_logging_async():
    # this tests time added to make s3 logging calls, vs just acompletion calls
    try:
        litellm.set_verbose = True
        # Make 5 calls with an empty success_callback
        litellm.success_callback = []
        start_time_empty_callback = asyncio.run(make_async_calls())
        print("done with no callback test")

        print("starting s3 logging load test")
        # Make 5 calls with success_callback set to "langfuse"
        litellm.success_callback = ["s3"]
        litellm.s3_callback_params = {
            "s3_bucket_name": "litellm-logs-2",
            "s3_aws_secret_access_key": "os.environ/AWS_SECRET_ACCESS_KEY",
            "s3_aws_access_key_id": "os.environ/AWS_ACCESS_KEY_ID",
        }
        start_time_s3 = asyncio.run(make_async_calls())
        print("done with s3 test")

        # Compare the time for both scenarios
        print(f"Time taken with success_callback='s3': {start_time_s3}")
        print(f"Time taken with empty success_callback: {start_time_empty_callback}")

        # assert the diff is not more than 1 second
        assert abs(start_time_s3 - start_time_empty_callback) < 1

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls():
    tasks = []
    for _ in range(5):
        task = asyncio.create_task(
            litellm.acompletion(
                model="azure/chatgpt-v-3",
                messages=[{"role": "user", "content": "This is a test"}],
                max_tokens=5,
                temperature=0.7,
                timeout=5,
                user="langfuse_latency_test_user",
                mock_response="It's simple to use and easy to get started",
            )
        )
        tasks.append(task)

    # Measure the start time before running the tasks
    start_time = asyncio.get_event_loop().time()

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    # Print the responses when tasks return
    for idx, response in enumerate(responses):
        print(f"Response from Task {idx + 1}: {response}")

    # Calculate the total time taken
    total_time = asyncio.get_event_loop().time() - start_time

    return total_time


@pytest.mark.skip(reason="flaky test on ci/cd")
def test_s3_logging_r2():
    # all s3 requests need to be in one test function
    # since we are modifying stdout, and pytests runs tests in parallel
    # on circle ci - we only test litellm.acompletion()
    try:
        # redirect stdout to log_file
        # litellm.cache = litellm.Cache(
        #     type="s3", s3_bucket_name="litellm-r2-bucket", s3_region_name="us-west-2"
        # )
        litellm.set_verbose = True
        from litellm._logging import verbose_logger
        import logging

        verbose_logger.setLevel(level=logging.DEBUG)

        litellm.success_callback = ["s3"]
        litellm.s3_callback_params = {
            "s3_bucket_name": "litellm-r2-bucket",
            "s3_aws_secret_access_key": "os.environ/R2_S3_ACCESS_KEY",
            "s3_aws_access_key_id": "os.environ/R2_S3_ACCESS_ID",
            "s3_endpoint_url": "os.environ/R2_S3_URL",
            "s3_region_name": "os.environ/R2_S3_REGION_NAME",
        }
        print("Testing async s3 logging")

        expected_keys = []

        import time

        curr_time = str(time.time())

        async def _test():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"This is a test {curr_time}"}],
                max_tokens=10,
                temperature=0.7,
                user="ishaan-2",
            )

        response = asyncio.run(_test())
        print(f"response: {response}")
        expected_keys.append(response.id)

        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_S3_URL"),
            region_name=os.getenv("R2_S3_REGION_NAME"),
            aws_access_key_id=os.getenv("R2_S3_ACCESS_ID"),
            aws_secret_access_key=os.getenv("R2_S3_ACCESS_KEY"),
        )

        bucket_name = "litellm-r2-bucket"
        # List objects in the bucket
        response = s3.list_objects(Bucket=bucket_name)

    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
    finally:
        # post, close log file and verify
        # Reset stdout to the original value
        print("Passed! Testing async s3 logging")
