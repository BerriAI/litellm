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


