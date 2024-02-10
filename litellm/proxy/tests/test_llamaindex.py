import os, dotenv

from dotenv import load_dotenv

load_dotenv()

from llama_index.llms import AzureOpenAI


llm = AzureOpenAI(
    engine="azure-gpt-3.5",
    temperature=0.0,
    azure_endpoint="http://0.0.0.0:4000",
    api_key="sk-1234",
    api_version="2023-07-01-preview",
)


response = llm.complete("The sky is a beautiful blue and")
print(response)
