import sys
import os
import io, asyncio
# import logging
# logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm
litellm.num_retries = 3
litellm.success_callback = ["langfuse"]
os.environ["LANGFUSE_DEBUG"] = "True"
import time
import pytest

def search_logs(log_file_path):
    """
    Searches the given log file for logs containing the "/api/public" string. 

    Parameters:
    - log_file_path (str): The path to the log file to be searched.

    Returns:
    - None

    Raises:
    - Exception: If there are any bad logs found in the log file.
    """
    import re
    print("\n searching logs")
    bad_logs = []
    good_logs = []
    all_logs = []
    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()
            print(f"searching logslines: {lines}")
            for line in lines:
                all_logs.append(line.strip())
                if "/api/public" in line:
                    print("Found log with /api/public:")
                    print(line.strip())
                    print("\n\n")
                    match = re.search(r'receive_response_headers.complete return_value=\(b\'HTTP/1.1\', (\d+),', line)
                    if match:
                        status_code = int(match.group(1))
                        if status_code != 200 and status_code != 201:
                            print("got a BAD log")
                            bad_logs.append(line.strip())
                        else:

                            good_logs.append(line.strip())
        print("\nBad Logs")
        print(bad_logs)
        if len(bad_logs)>0:
            raise Exception(f"bad logs, Bad logs = {bad_logs}")
        
        print("\nGood Logs")
        print(good_logs)
        if len(good_logs) <= 0:
            raise Exception(f"There were no Good Logs from Langfuse. No logs with /api/public status 200. \nAll logs:{all_logs}")

    except Exception as e:
        raise e

def pre_langfuse_setup():
    """
    Set up the logging for the 'pre_langfuse_setup' function.
    """
    # sends logs to langfuse.log
    import logging
    # Configure the logging to write to a file
    logging.basicConfig(filename="langfuse.log", level=logging.DEBUG)
    logger = logging.getLogger()
    
    # Add a FileHandler to the logger
    file_handler = logging.FileHandler("langfuse.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return

@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_async(): 
    try: 
        pre_langfuse_setup()
        litellm.set_verbose = True
        async def _test_langfuse():
            return await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content":"This is a test"}],
                max_tokens=100,
                temperature=0.7,
                timeout=5,
            )
        response = asyncio.run(_test_langfuse())
        print(f"response: {response}")

        time.sleep(2)
        # check langfuse.log to see if there was a failed response
        search_logs("langfuse.log")
    except litellm.Timeout as e: 
        pass
    except Exception as e: 
        pytest.fail(f"An exception occurred - {e}")

test_langfuse_logging_async()

@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging():
    try:
        pre_langfuse_setup()
        litellm.set_verbose = True
        response = completion(model="claude-instant-1.2",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm claude"
                              }],
                              max_tokens=10,
                              temperature=0.2,
                              )
        print(response)
        time.sleep(5)
        # check langfuse.log to see if there was a failed response
        search_logs("langfuse.log")

    except litellm.Timeout as e: 
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")

test_langfuse_logging()

@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_stream():
    try:
        litellm.set_verbose=True
        response = completion(model="anyscale/meta-llama/Llama-2-7b-chat-hf",
                              messages=[{
                                  "role": "user",
                                  "content": "this is a streaming test for llama2 + langfuse"
                              }],
                              max_tokens=20,
                              temperature=0.2,
                              stream=True
                              )
        print(response)
        for chunk in response:
            pass
            # print(chunk)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        print(e)

# test_langfuse_logging_stream()

@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_custom_generation_name():
    try:
        litellm.set_verbose=True
        response = completion(model="gpt-3.5-turbo",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm claude"
                              }],
                              max_tokens=10,
                              metadata = {
                                    "langfuse/foo": "bar", 
                                    "langsmith/fizz": "buzz", 
                                    "prompt_hash": "asdf98u0j9131123"
                                }
        )
        print(response)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
        print(e)

# test_langfuse_logging_custom_generation_name()
@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_function_calling():
    function1 = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    try:
        response = completion(model="gpt-3.5-turbo",
                              messages=[{
                                  "role": "user",
                                  "content": "what's the weather in boston"
                              }],
                              temperature=0.1,
                              functions=function1,
            )
        print(response)
    except litellm.Timeout as e: 
        pass
    except Exception as e:
        print(e)

# test_langfuse_logging_function_calling()



