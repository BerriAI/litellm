from main import embedding
import os

## Configs for Models ##
# OpenAI Configs

## End Configs for Models ##


input = ["Who is ishaan"]
# openai call
response = embedding(model="text-embedding-ada-002", input=input)
print("\nOpenAI call")
print(response)

# azure openai call
response = embedding(model="azure-embedding-mode", input=input, azure=True)
print("\nAzure OpenAI call")
print(response)
