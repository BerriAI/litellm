import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Braintrust - Evals + Logging 

[Braintrust](https://www.braintrust.dev/) manages evaluations, logging, prompt playground, to data management for AI products.


## Quick Start

```python
# pip install langfuse 
import litellm
import os

# set env 
os.environ["BRAINTRUST_API_KEY"] = "" 
os.environ['OPENAI_API_KEY']=""

# set braintrust as a callback, litellm will send the data to braintrust
litellm.callbacks = ["braintrust"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```



## OpenAI Proxy Usage

1. Add keys to env 
```env
BRAINTRUST_API_KEY="" 
```

2. Add braintrust to callbacks 
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY


litellm_settings:
  callbacks: ["braintrust"]
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "model": "groq-llama3",
    "messages": [
        { "role": "system", "content": "Use your tools smartly"},
        { "role": "user", "content": "What time is it now? Use your tool"}
    ]
}'
```

## Advanced - pass Project ID or name

<Tabs>
<TabItem value="sdk" label="SDK">

```python
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ], 
  metadata={
    "project_id": "1234",
    # passing project_name will try to find a project with that name, or create one if it doesn't exist
    # if both project_id and project_name are passed, project_id will be used
    # "project_name": "my-special-project" 
  }
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

**Curl**

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "model": "groq-llama3",
    "messages": [
        { "role": "system", "content": "Use your tools smartly"},
        { "role": "user", "content": "What time is it now? Use your tool"}
    ],
    "metadata": {
        "project_id": "my-special-project"
    }
}'
```

**OpenAI SDK**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={ # pass in any provider-specific param, if not supported by openai, https://docs.litellm.ai/docs/completion/input#provider-specific-params
        "metadata": { # 👈 use for logging additional params (e.g. to langfuse)
            "project_id": "my-special-project"
        }
    }
)

print(response)
```

For more examples, [**Click Here**](../proxy/user_keys.md#chatcompletions)

</TabItem>
</Tabs>

## Full API Spec 

Here's everything you can pass in metadata for a braintrust request 

`braintrust_*` - any metadata field starting with `braintrust_` will be passed as metadata to the logging request 

`project_id`  - set the project id for a braintrust call. Default is `litellm`. 