import sys, os
import traceback
from dotenv import load_dotenv
load_dotenv()
import os
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion

litellm.caching = True
messages = [{"role": "user", "content": "Hey, how's it going?"}]



# test if response cached
try:
    response1 = completion(model="gpt-3.5-turbo", messages=messages)
    response2 = completion(model="gpt-3.5-turbo", messages=messages)
    if response2 != response1:
        print(f"response1: {response1}")
        print(f"response2: {response2}")
        raise Exception
except Exception as e:
    print(f"error occurred: {traceback.format_exc()}") 
    pytest.fail(f"Error occurred: {e}")
litellm.caching = False

