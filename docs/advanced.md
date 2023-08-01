# Advanced - liteLLM client

## Use liteLLM client to send Output Data to Posthog, Sentry etc
liteLLM allows you to create `completion_client` and `embedding_client` to send successfull / error LLM API call data to Posthog, Sentry, Slack etc

### Quick Start
```python
from main import litellm_client
import os

## set env variables
os.environ['SENTRY_API_URL'] = ""
os.environ['POSTHOG_API_KEY'], os.environ['POSTHOG_API_URL'] = "api-key", "api-url"

# init liteLLM client
client = litellm_client(success_callback=["posthog"], failure_callback=["sentry", "posthog"])
completion = client.completion
embedding = client.embedding

response = completion(model="gpt-3.5-turbo", messages=messages) 
```

## Calling Embeddings and Sending Data to Sentry/Posthog/etc.
To call embeddings and send data to Sentry, Posthog, and other similar services, you need to initialize the `litellm_client` with the appropriate callbacks for success and failure. Here is an example of how to do this:

```python
# init liteLLM client with callbacks
client = litellm_client(success_callback=["posthog"], failure_callback=["sentry", "posthog"])

# use the embedding method of the client
embedding = client.embedding
response = embedding(model="gpt-3.5-turbo", input=messages) 
```

You also need to set the necessary environment variables for the services like Sentry and Posthog. Here is how you can do this:

```python
# set env variables for Sentry and Posthog
os.environ['SENTRY_API_URL'] = "your-sentry-api-url"
os.environ['POSTHOG_API_KEY'] = "your-posthog-api-key"
os.environ['POSTHOG_API_URL'] = "your-posthog-api-url"
```
