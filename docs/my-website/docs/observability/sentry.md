import Image from '@theme/IdealImage';

# Sentry - Log LLM Exceptions
[Sentry](https://sentry.io/) provides error monitoring for production. LiteLLM can add breadcrumbs and send exceptions to Sentry with this integration

Track exceptions for:
- litellm.completion() - completion()for 100+ LLMs
- litellm.acompletion() - async completion()
- Streaming completion() & acompletion() calls

<Image img={require('../../img/sentry.png')} />


## Usage

### Set SENTRY_DSN & callback

```python
import litellm, os
os.environ["SENTRY_DSN"] = "your-sentry-url"
litellm.failure_callback=["sentry"]
```

### Sentry callback with completion
```python
import litellm
from litellm import completion 

litellm.input_callback=["sentry"] # adds sentry breadcrumbing
litellm.failure_callback=["sentry"] # [OPTIONAL] if you want litellm to capture -> send exception to sentry

import os 
os.environ["SENTRY_DSN"] = "your-sentry-url"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# set bad key to trigger error 
api_key="bad-key"
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hey!"}], stream=True, api_key=api_key)

print(response)
```

[Let us know](https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+) if you need any additional options from Sentry. 

