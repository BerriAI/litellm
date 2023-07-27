from main import completion
import os

## Configs for Models ##
# OpenAI Configs

## End Configs for Models ##


messages = [{ "content": "Hello, how are you?","role": "user"}]
# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)
print("\nOpenAI call")
print(response)

# azure openai call
response = completion("chatgpt-test", messages, azure=True)
print("\nAzure call")
print(response)

# text davinci openai call
response = completion("text-davinci-003", messages)
print("\nDavinci call")
print(response)

# cohere call
response = completion("command-nightly", messages)
print("\nCohere call")
print(response)

