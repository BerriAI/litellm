# 1. Select the python interpreter  (Ctrl + Shift + P -> Python: Select Interpreter)
# 2. Run and debug the code (F5)

import os

from litellm import completion

for x in range(1):
    completion(model="gpt-3.5-turbo", messages=[{ "content": "what's the weather in SF","role": "user"}])
