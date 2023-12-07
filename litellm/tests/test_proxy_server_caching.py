# #### What this tests ####
# #    This tests using caching w/ litellm which requires SSL=True  

# import sys, os
# import time
# import traceback
# from dotenv import load_dotenv

# load_dotenv()
# import os

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest
# import litellm
# from litellm import embedding, completion
# from litellm.caching import Cache

# messages = [{"role": "user", "content": f"who is ishaan {time.time()}"}]

# @pytest.mark.skip(reason="local proxy test")
# def test_caching_v2(): # test in memory cache
#     try:
#         response1 = completion(model="openai/gpt-3.5-turbo", messages=messages, api_base="http://0.0.0.0:8000")
#         response2 = completion(model="openai/gpt-3.5-turbo", messages=messages, api_base="http://0.0.0.0:8000")
#         print(f"response1: {response1}")
#         print(f"response2: {response2}")
#         litellm.cache = None # disable cache
#         if response2['choices'][0]['message']['content'] != response1['choices'][0]['message']['content']:
#             print(f"response1: {response1}")
#             print(f"response2: {response2}")
#             raise Exception()
#     except Exception as e:
#         print(f"error occurred: {traceback.format_exc()}")
#         pytest.fail(f"Error occurred: {e}")

# test_caching_v2()