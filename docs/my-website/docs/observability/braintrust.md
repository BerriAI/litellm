import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Braintrust - Evals + Logging

[Braintrust](https://www.braintrust.dev/) manages evaluations, logging, prompt playground, to data management for AI products.

## Quick Start

```python
# pip install braintrust
import litellm
import os

# set env
os.environ["BRAINTRUST_API_KEY"] = ""
os.environ["BRAINTRUST_API_BASE"] = "https://api.braintrustdata.com/v1"
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
BRAINTRUST_API_BASE="https://api.braintrustdata.com/v1"
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

It is recommended that you include the `project_id` or `project_name` to ensure your traces are being written out to the correct Braintrust project.

### Custom Span Names

You can customize the span name in Braintrust logging by passing `span_name` in the metadata. By default, the span name is set to "Chat Completion".

### Custom Span Attributes

You can customize the span id, root span name and span parents in Braintrust logging by passing `span_id`, `root_span_id` and `span_parents` in the metadata. 
`span_parents` should be a string containing a list of span ids, joined by ,


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
    # "project_name": "my-special-project",
    # custom span name for this operation (default: "Chat Completion")
    "span_name": "User Greeting Handler"
  }
)
```

Note: Other `metadata` can be included here as well when using the SDK.

```python
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata={
    "project_id": "1234",
    "span_name": "Custom Operation",
    "item1": "an item",
    "item2": "another item"
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
        "project_id": "my-special-project",
        "span_name": "Tool Usage Request"
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
        "metadata": { # 👈 use for logging additional params (e.g. to braintrust)
            "project_id": "my-special-project",
            "span_name": "Poetry Generation"
        }
    }
)

print(response)
```

For more examples, [**Click Here**](../proxy/user_keys.md#chatcompletions)

</TabItem>
</Tabs>

You can use `BRAINTRUST_API_BASE` to point to your self-hosted Braintrust data plane. Read more about this [here](https://www.braintrust.dev/docs/guides/self-hosting).

## Full API Spec

Here's everything you can pass in metadata for a braintrust request

`braintrust_*` - If you are adding metadata from _proxy request headers_, any metadata field starting with `braintrust_` will be passed as metadata to the logging request. If you are using the SDK, just pass your metadata like normal (e.g., `metadata={"project_name": "my-test-project", "item1": "an item", "item2": "another item"}`)

`project_id` - Set the project id for a braintrust call. Default is `litellm`.

`project_name` - Set the project name for a braintrust call. Will try to find a project with that name, or create one if it doesn't exist. If both `project_id` and `project_name` are passed, `project_id` will be used.

`span_name` - Set a custom span name for the operation. Default is `"Chat Completion"`. Use this to provide more descriptive names for different types of operations in your application (e.g., "User Query", "Document Summary", "Code Generation").
