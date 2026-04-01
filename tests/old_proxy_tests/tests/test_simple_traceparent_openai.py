# mypy: ignore-errors
from litellm._uuid import uuid

import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")
example_traceparent = "00-80e1afed08e019fc1110464cfa66635c-02e80198930058d4-01"
extra_headers = {"traceparent": example_traceparent}
_trace_id = example_traceparent.split("-")[1]

print("EXTRA HEADERS: ", extra_headers)
print("Trace ID: ", _trace_id)

response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "this is a test request, write a short poem"}
    ],
    extra_headers=extra_headers,
)

print(response)
