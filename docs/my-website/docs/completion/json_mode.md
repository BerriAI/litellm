import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Structured Outputs (JSON Mode)

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


### 1. Check if model supports `response_format`

Call `litellm.get_supported_openai_params` to check if a model/provider supports `response_format`. 

```python
from litellm import get_supported_openai_params

params = get_supported_openai_params(model="anthropic.claude-3", custom_llm_provider="bedrock")

assert "response_format" in params
```

### 2. Check if model supports `json_schema`

This is used to check if you can pass 
- `response_format={ "type": "json_schema", "json_schema": … , "strict": true }`
- `response_format=<Pydantic Model>`

```python
from litellm import supports_response_schema

assert supports_response_schema(model="gemini-1.5-pro-preview-0215", custom_llm_provider="bedrock")
```

Check out [model_prices_and_context_window.json](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) for a full list of models and their support for `response_schema`.

## Pass in 'json_schema' 

To use Structured Outputs, simply specify

```
response_format: { "type": "json_schema", "json_schema": … , "strict": true }
```

Works for:
- OpenAI models 
- Azure OpenAI models
- xAI models (Grok-2 or later)
- Google AI Studio - Gemini models
- Vertex AI models (Gemini + Anthropic)
- Bedrock Models
- Anthropic API Models
- Groq Models
- Ollama Models
- Databricks Models

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion 
from pydantic import BaseModel

# add to env var 
os.environ["OPENAI_API_KEY"] = ""

messages = [{"role": "user", "content": "List 5 important events in the XIX century"}]

class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

class EventsList(BaseModel):
    events: list[CalendarEvent]

resp = completion(
    model="gpt-4o-2024-08-06",
    messages=messages,
    response_format=EventsList
)

print("Received={}".format(resp))

events_list = EventsList.model_validate_json(resp.choices[0].message.content)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add openai model to config.yaml

```yaml
model_list:
  - model_name: "gpt-4o"
    litellm_params:
      model: "gpt-4o-2024-08-06"
```

2. Start proxy with config.yaml

```bash
litellm --config /path/to/config.yaml
```

3. Call with OpenAI SDK / Curl!

Just replace the 'base_url' in the openai sdk, to call the proxy with 'json_schema' for openai models

**OpenAI SDK**
```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(
    api_key="anything", # 👈 PROXY KEY (can be anything, if master_key not set)
    base_url="http://0.0.0.0:4000" # 👈 PROXY BASE URL
)

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
        {"role": "user", "content": "how can I solve 8x + 7 = -23"}
    ],
    response_format=MathReasoning,
)

math_reasoning = completion.choices[0].message.parsed
```

**Curl**

```bash
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step."
      },
      {
        "role": "user",
        "content": "how can I solve 8x + 7 = -23"
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "math_reasoning",
        "schema": {
          "type": "object",
          "properties": {
            "steps": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "explanation": { "type": "string" },
                  "output": { "type": "string" }
                },
                "required": ["explanation", "output"],
                "additionalProperties": false
              }
            },
            "final_answer": { "type": "string" }
          },
          "required": ["steps", "final_answer"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  }'
```

</TabItem>
</Tabs>


## Validate JSON Schema 


Not all vertex models support passing the json_schema to them (e.g. `gemini-1.5-flash`). To solve this, LiteLLM supports client-side validation of the json schema. 

```
litellm.enable_json_schema_validation=True
```
If `litellm.enable_json_schema_validation=True` is set, LiteLLM will validate the json response using `jsonvalidator`. 

[**See Code**](https://github.com/BerriAI/litellm/blob/671d8ac496b6229970c7f2a3bdedd6cb84f0746b/litellm/litellm_core_utils/json_validation_rule.py#L4)


<Tabs>
<TabItem value="sdk" label="SDK">

```python
# !gcloud auth application-default login - run this to add vertex credentials to your env
import litellm, os
from litellm import completion 
from pydantic import BaseModel 


messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ]

litellm.enable_json_schema_validation = True
litellm.set_verbose = True # see the raw request made by litellm

class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

resp = completion(
    model="gemini/gemini-1.5-pro",
    messages=messages,
    response_format=CalendarEvent,
)

print("Received={}".format(resp))
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Create config.yaml
```yaml
model_list:
  - model_name: "gemini-1.5-flash"
    litellm_params:
      model: "gemini/gemini-1.5-flash"
      api_key: os.environ/GEMINI_API_KEY

litellm_settings:
  enable_json_schema_validation: True
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "gemini-1.5-flash",
    "messages": [
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    "response_format": { 
        "type": "json_schema",
        "json_schema": {
          "name": "math_reasoning",
          "schema": {
            "type": "object",
            "properties": {
              "steps": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "explanation": { "type": "string" },
                    "output": { "type": "string" }
                  },
                  "required": ["explanation", "output"],
                  "additionalProperties": false
                }
              },
              "final_answer": { "type": "string" }
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": false
          },
          "strict": true
        }
    },
  }'
```

</TabItem>
</Tabs>

## Gemini - Native JSON Schema Format (Gemini 2.0+)

Gemini 2.0+ models automatically use the native `responseJsonSchema` parameter, which provides better compatibility with standard JSON Schema format.

### Benefits (Gemini 2.0+):
- Standard JSON Schema format (lowercase types like `string`, `object`)
- Supports `additionalProperties: false` for stricter validation
- Better compatibility with Pydantic's `model_json_schema()`
- No `propertyOrdering` required

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Extract: John is 25 years old"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": False  # Supported on Gemini 2.0+
            }
        }
    }
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [
        {"role": "user", "content": "Extract: John is 25 years old"}
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"],
                "additionalProperties": false
            }
        }
    }
  }'
```

</TabItem>
</Tabs>

### Model Behavior

| Model | Format Used | `additionalProperties` Support |
|-------|-------------|-------------------------------|
| Gemini 2.0+ | `responseJsonSchema` (JSON Schema) | ✅ Yes |
| Gemini 1.5 | `responseSchema` (OpenAPI) | ❌ No |

LiteLLM automatically selects the appropriate format based on the model version.