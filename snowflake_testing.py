import os
from litellm import completion 

os.environ["SNOWFLAKE_ACCOUNT_ID"] = "YOUR ACCOUNT"
os.environ["SNOWFLAKE_JWT"] = "YOUR JWT"

messages = [{"role": "user", "content": "Write me a poem about the blue sky"}]

completion(model="snowflake/mistral-7b", messages=messages)