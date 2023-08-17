import openai
import os

os.environ["OPENAI_API_KEY"] = ""

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base ="http://localhost:5000"

messages = [
    {
        "role": "user",
        "content": "write a 1 pg essay in liteLLM"
    }
]

response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True)
print("got response", response)
# response is a generator

for chunk in response:
    print(chunk)