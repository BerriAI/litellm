import Image from '@theme/IdealImage';

# Raw Request/Response Logging

See the raw request/response sent by LiteLLM in your logging provider (OTEL/Langfuse/etc.).

**on SDK**
```python
# pip install langfuse 
import litellm
import os

# log raw request/response
litellm.log_raw_request_response = True

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
# Optional, defaults to https://cloud.langfuse.com
os.environ["LANGFUSE_HOST"] # optional

# LLM API Keys
os.environ['OPENAI_API_KEY']=""

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

**on Proxy**

```yaml
litellm_settings:
  log_raw_request_response: True
```

**Expected Log**

<Image img={require('../../img/raw_request_log.png')}/>