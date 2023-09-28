# Sentry Tutorial 
[Sentry](https://sentry.io/) provides error monitoring for production. LiteLLM can add breadcrumbs and send exceptions to Sentry with this integration

This works on normal, async and streaming completion calls

### usage 

```python
import litellm
from litellm import completion 
litellm.set_verbose = True

litellm.input_callback=["sentry"] # adds sentry breadcrumbing
litellm.failure_callback=["sentry"] # [OPTIONAL] if you want litellm to capture -> send exception to sentry

import os 
os.environ["SENTRY_API_URL"] = "your-sentry-url"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# set bad key to trigger error 
api_key="bad-key"
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey!"}], stream=True, api_key=api_key)

print(response)
```

[Let us know](https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+) if you need any additional options from Sentry. 

