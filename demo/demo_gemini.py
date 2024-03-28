from litellm import completion
import os
import pprint
import litellm
litellm.set_verbose=True

response = completion(
    model="gemini/gemini-pro", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)

pprint.pprint(response)
print("------------------")
print(response.choices[0].message.content)
