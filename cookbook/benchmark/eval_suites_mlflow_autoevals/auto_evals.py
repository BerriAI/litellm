import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()

import litellm
from litellm import embedding, completion, completion_cost

from autoevals.llm import *

###################
import litellm

# litellm completion call
question = "which country has the highest population"
response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": question}],
)
print(response)
# use the auto eval Factuality() evaluator

print("calling evaluator")
evaluator = Factuality()
result = evaluator(
    output=response.choices[0]["message"][
        "content"
    ],  # response from litellm.completion()
    expected="India",  # expected output
    input=question,  # question passed to litellm.completion
)

print(result)
