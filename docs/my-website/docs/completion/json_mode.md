import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# JSON Mode

## Quick Start 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os 

os.environ["OPENAI_API_KEY"] = ""

response = completion(
  model="gpt-4o-mini",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)
print(response.choices[0].message.content)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "gpt-4o-mini",
    "response_format": { "type": "json_object" },
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON."
      },
      {
        "role": "user",
        "content": "Who won the world series in 2020?"
      }
    ]
  }'
```
</TabItem>
</Tabs>

## Check Model Support 

Call `litellm.get_supported_openai_params` to check if a model/provider supports `response_format`. 

```python
from litellm import get_supported_openai_params

params = get_supported_openai_params(model="anthropic.claude-3", custom_llm_provider="bedrock")

assert "response_format" in params
```

## Validate JSON Schema 

For VertexAI models, LiteLLM supports passing the `response_schema` and validating the JSON output.

This works across Gemini (`vertex_ai_beta/`) + Anthropic (`vertex_ai/`) models. 


<Tabs>
<TabItem value="sdk" label="SDK">

```python
# !gcloud auth application-default login - run this to add vertex credentials to your env

from litellm import completion 

messages = [{"role": "user", "content": "List 5 cookie recipes"}]

response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "recipe_name": {
                "type": "string",
            },
        },
        "required": ["recipe_name"],
    },
}

resp = completion(
    model="vertex_ai_beta/gemini-1.5-pro",
    messages=messages,
    response_format={
        "type": "json_object",
        "response_schema": response_schema,
        "enforce_validation": True, # client-side json schema validation
    },
    vertex_location="us-east5",
)

print("Received={}".format(resp))
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "vertex_ai_beta/gemini-1.5-pro",
    "messages": [{"role": "user", "content": "List 5 cookie recipes"}]
    "response_format": { 
        "type": "json_object",
        "enforce_validation: true, 
        "response_schema": { 
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                    },
                },
                "required": ["recipe_name"],
            },
        }
    },
  }'
```

</TabItem>
</Tabs>