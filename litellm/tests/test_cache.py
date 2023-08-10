# import sys, os
# import traceback
# from dotenv import load_dotenv
# load_dotenv()
# import os

# sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
# import pytest
# import litellm

# # set cache to True
# litellm.cache = True
# litellm.cache_similarity_threshold = 0.5

# user_message = "Hello, whats the weather in San Francisco??"
# messages = [{ "content": user_message,"role": "user"}]

# def test_completion_with_cache_gpt4():
#     try:
#         # in this test make the same call twice, measure the response time
#         # the 2nd response time should be less than half of the first, ensuring that the cache is working
#         import time
#         start = time.time()
#         print(litellm.cache)
#         response = litellm.completion(model="gpt-4", messages=messages)
#         end = time.time()
#         first_call_time = end-start
#         print(f"first call: {first_call_time}")

#         start = time.time()
#         response = litellm.completion(model="gpt-4", messages=messages)
#         end = time.time()
#         second_call_time = end-start
#         print(f"second call: {second_call_time}")

#         if second_call_time > 1:
#             # the 2nd call should be less than 1s
#             pytest.fail(f"Cache is not working")
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# litellm.cache = False