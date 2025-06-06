import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# VertexAI [Anthropic, Gemini, Model Garden]

## Overview

| Property | Details |
|-------|-------|
| Description | Vertex AI is a fully-managed AI development platform for building and using generative AI. |
| Provider Route on LiteLLM | `vertex_ai/` |
| Link to Provider Doc | [Vertex AI ↗](https://cloud.google.com/vertex-ai) |
| Base URL | 1. Regional endpoints<br/>[https://{vertex_location}-aiplatform.googleapis.com/](https://{vertex_location}-aiplatform.googleapis.com/)<br/>2. Global endpoints (limited availability)<br/>[https://aiplatform.googleapis.com/](https://{aiplatform.googleapis.com/)|
| Supported Operations | [`/chat/completions`](#sample-usage), `/completions`, [`/embeddings`](#embedding-models), [`/audio/speech`](#text-to-speech-apis), [`/fine_tuning`](#fine-tuning-apis), [`/batches`](#batch-apis), [`/files`](#batch-apis), [`/images`](#image-generation-models) |


<br />
<br />

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_VertextAI_Example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## `vertex_ai/` route 

The `vertex_ai/` route uses uses [VertexAI's REST API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#syntax).

```python
from litellm import completion
import json 

## GET CREDENTIALS 
## RUN ## 
# !gcloud auth application-default login - run this to add vertex credentials to your env
## OR ## 
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

## COMPLETION CALL 
response = completion(
  model="vertex_ai/gemini-pro",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  vertex_credentials=vertex_credentials_json
)
```

### **System Message**

```python
from litellm import completion
import json 

## GET CREDENTIALS 
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)


response = completion(
  model="vertex_ai/gemini-pro",
  messages=[{"content": "You are a good bot.","role": "system"}, {"content": "Hello, how are you?","role": "user"}], 
  vertex_credentials=vertex_credentials_json
)
```

### **Function Calling**

Force Gemini to make tool calls with `tool_choice="required"`.

```python
from litellm import completion
import json 

## GET CREDENTIALS 
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)


messages = [
    {
        "role": "system",
        "content": "Your name is Litellm Bot, you are a helpful assistant",
    },
    # User asks for their name and weather in San Francisco
    {
        "role": "user",
        "content": "Hello, what is your name and can you tell me the weather?",
    },
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

data = {
    "model": "vertex_ai/gemini-1.5-pro-preview-0514"),
    "messages": messages,
    "tools": tools,
    "tool_choice": "required",
    "vertex_credentials": vertex_credentials_json
}

## COMPLETION CALL 
print(completion(**data))
```

### **JSON Schema**

From v`1.40.1+` LiteLLM supports sending `response_schema` as a param for Gemini-1.5-Pro on Vertex AI. For other models (e.g. `gemini-1.5-flash` or `claude-3-5-sonnet`), LiteLLM adds the schema to the message list with a user-controlled prompt.

**Response Schema**
<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion 
import json 

## SETUP ENVIRONMENT
# !gcloud auth application-default login - run this to add vertex credentials to your env

messages = [
    {
        "role": "user",
        "content": "List 5 popular cookie recipes."
    }
]

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


completion(
    model="vertex_ai/gemini-1.5-pro", 
    messages=messages, 
    response_format={"type": "json_object", "response_schema": response_schema} # 👈 KEY CHANGE
    )

print(json.loads(completion.choices[0].message.content))
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: gemini-pro
    litellm_params:
      model: vertex_ai/gemini-1.5-pro
      vertex_project: "project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json" # [OPTIONAL] Do this OR `!gcloud auth application-default login` - run this to add vertex credentials to your env
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
  "model": "gemini-pro",
  "messages": [
        {"role": "user", "content": "List 5 popular cookie recipes."}
    ],
  "response_format": {"type": "json_object", "response_schema": { 
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
    }}
}
'
```

</TabItem>
</Tabs>

**Validate Schema**

To validate the response_schema, set `enforce_validation: true`.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion, JSONSchemaValidationError
try: 
	completion(
    model="vertex_ai/gemini-1.5-pro", 
    messages=messages, 
    response_format={
        "type": "json_object", 
        "response_schema": response_schema,
        "enforce_validation": true # 👈 KEY CHANGE
    }
	)
except JSONSchemaValidationError as e: 
	print("Raw Response: {}".format(e.raw_response))
	raise e
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: gemini-pro
    litellm_params:
      model: vertex_ai/gemini-1.5-pro
      vertex_project: "project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json" # [OPTIONAL] Do this OR `!gcloud auth application-default login` - run this to add vertex credentials to your env
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request!

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
  "model": "gemini-pro",
  "messages": [
        {"role": "user", "content": "List 5 popular cookie recipes."}
    ],
  "response_format": {"type": "json_object", "response_schema": { 
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
    }, 
    "enforce_validation": true
    }
}
'
```

</TabItem>
</Tabs>

LiteLLM will validate the response against the schema, and raise a `JSONSchemaValidationError` if the response does not match the schema. 

JSONSchemaValidationError inherits from `openai.APIError` 

Access the raw response with `e.raw_response`

**Add to prompt yourself**

```python 
from litellm import completion 

## GET CREDENTIALS 
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

messages = [
    {
        "role": "user",
        "content": """
List 5 popular cookie recipes.

Using this JSON schema:

    Recipe = {"recipe_name": str}

Return a `list[Recipe]`
        """
    }
]

completion(model="vertex_ai/gemini-1.5-flash-preview-0514", messages=messages, response_format={ "type": "json_object" })
```

### **Google Hosted Tools (Web Search, Code Execution, etc.)**

#### **Web Search**

Add Google Search Result grounding to vertex ai calls. 

[**Relevant VertexAI Docs**](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/grounding#examples)

See the grounding metadata with `response_obj._hidden_params["vertex_ai_grounding_metadata"]`

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion 

## SETUP ENVIRONMENT
# !gcloud auth application-default login - run this to add vertex credentials to your env

tools = [{"googleSearch": {}}] # 👈 ADD GOOGLE SEARCH

resp = litellm.completion(
                    model="vertex_ai/gemini-1.0-pro-001",
                    messages=[{"role": "user", "content": "Who won the world cup?"}],
                    tools=tools,
                )

print(resp)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

```python showLineNumbers
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # pass litellm proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:4000/v1/" # point to litellm proxy
)

response = client.chat.completions.create(
    model="gemini-pro",
    messages=[{"role": "user", "content": "Who won the world cup?"}],
    tools=[{"googleSearch": {}}],
)

print(response)
```
</TabItem>
<TabItem value="curl" label="cURL">

```bash showLineNumbers
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-pro",
    "messages": [
      {"role": "user", "content": "Who won the world cup?"}
    ],
   "tools": [
        {
            "googleSearch": {} 
        }
    ]
  }'

```
</TabItem>
</Tabs>

</TabItem>
</Tabs>

#### **Url Context**
Using the URL context tool, you can provide Gemini with URLs as additional context for your prompt. The model can then retrieve content from the URLs and use that content to inform and shape its response.

[**Relevant Docs**](https://ai.google.dev/gemini-api/docs/url-context)

See the grounding metadata with `response_obj._hidden_params["vertex_ai_url_context_metadata"]`

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion
import os

os.environ["GEMINI_API_KEY"] = ".."

# 👇 ADD URL CONTEXT
tools = [{"urlContext": {}}]

response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Summarize this document: https://ai.google.dev/gemini-api/docs/models"}],
    tools=tools,
)

print(response)

# Access URL context metadata
url_context_metadata = response.model_extra['vertex_ai_url_context_metadata']
urlMetadata = url_context_metadata[0]['urlMetadata'][0]
print(f"Retrieved URL: {urlMetadata['retrievedUrl']}")
print(f"Retrieval Status: {urlMetadata['urlRetrievalStatus']}")
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml
```yaml
model_list:
  - model_name: gemini-2.0-flash
    litellm_params:
      model: gemini/gemini-2.0-flash
      api_key: os.environ/GEMINI_API_KEY
```

2. Start Proxy
```bash
$ litellm --config /path/to/config.yaml
```

3. Make Request!
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "Summarize this document: https://ai.google.dev/gemini-api/docs/models"}],
    "tools": [{"urlContext": {}}]
  }'
```
</TabItem>
</Tabs>

#### **Enterprise Web Search**

You can also use the `enterpriseWebSearch` tool for an [enterprise compliant search](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/web-grounding-enterprise).

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion 

## SETUP ENVIRONMENT
# !gcloud auth application-default login - run this to add vertex credentials to your env

tools = [{"enterpriseWebSearch": {}}] # 👈 ADD GOOGLE ENTERPRISE SEARCH

resp = litellm.completion(
                    model="vertex_ai/gemini-1.0-pro-001",
                    messages=[{"role": "user", "content": "Who won the world cup?"}],
                    tools=tools,
                )

print(resp)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

```python showLineNumbers
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # pass litellm proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:4000/v1/" # point to litellm proxy
)

response = client.chat.completions.create(
    model="gemini-pro",
    messages=[{"role": "user", "content": "Who won the world cup?"}],
    tools=[{"enterpriseWebSearch": {}}],
)

print(response)
```
</TabItem>
<TabItem value="curl" label="cURL">

```bash showLineNumbers
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-pro",
    "messages": [
      {"role": "user", "content": "Who won the world cup?"}
    ],
   "tools": [
        {
            "enterpriseWebSearch": {} 
        }
    ]
  }'

```
</TabItem>
</Tabs>

</TabItem>
</Tabs>

#### **Code Execution**



<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers
from litellm import completion
import os

## SETUP ENVIRONMENT
# !gcloud auth application-default login - run this to add vertex credentials to your env


tools = [{"codeExecution": {}}] # 👈 ADD CODE EXECUTION

response = completion(
    model="vertex_ai/gemini-2.0-flash",
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
    tools=tools,
)

print(response)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash showLineNumbers
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gemini-2.0-flash",
  "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
  "tools": [{"codeExecution": {}}]
}
'
```

</TabItem>
</Tabs>





#### **Moving from Vertex AI SDK to LiteLLM (GROUNDING)**


If this was your initial VertexAI Grounding code,

```python
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool, grounding


vertexai.init(project=project_id, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-001")

# Use Google Search for grounding
tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

prompt = "When is the next total solar eclipse in US?"
response = model.generate_content(
    prompt,
    tools=[tool],
    generation_config=GenerationConfig(
        temperature=0.0,
    ),
)

print(response)
```

then, this is what it looks like now

```python
from litellm import completion


# !gcloud auth application-default login - run this to add vertex credentials to your env

tools = [{"googleSearch": {"disable_attributon": False}}] # 👈 ADD GOOGLE SEARCH

resp = litellm.completion(
                    model="vertex_ai/gemini-1.0-pro-001",
                    messages=[{"role": "user", "content": "Who won the world cup?"}],
                    tools=tools,
                    vertex_project="project-id"
                )

print(resp)
```


### **Thinking / `reasoning_content`**

LiteLLM translates OpenAI's `reasoning_effort` to Gemini's `thinking` parameter. [Code](https://github.com/BerriAI/litellm/blob/620664921902d7a9bfb29897a7b27c1a7ef4ddfb/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L362)

Added an additional non-OpenAI standard "disable" value for non-reasoning Gemini requests.

**Mapping**

| reasoning_effort | thinking |
| ---------------- | -------- |
| "disable"        | "budget_tokens": 0    |
| "low"            | "budget_tokens": 1024 |
| "medium"         | "budget_tokens": 2048 |
| "high"           | "budget_tokens": 4096 |

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

# !gcloud auth application-default login - run this to add vertex credentials to your env

resp = completion(
    model="vertex_ai/gemini-2.5-flash-preview-04-17",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="low",
    vertex_project="project-id",
    vertex_location="us-central1"
)

```

</TabItem>

<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
- model_name: gemini-2.5-flash
  litellm_params:
    model: vertex_ai/gemini-2.5-flash-preview-04-17
    vertex_credentials: {"project_id": "project-id", "location": "us-central1", "project_key": "project-key"}
    vertex_project: "project-id"
    vertex_location: "us-central1"
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": "low"
  }'
```

</TabItem>
</Tabs>


**Expected Response**

```python
ModelResponse(
    id='chatcmpl-c542d76d-f675-4e87-8e5f-05855f5d0f5e',
    created=1740470510,
    model='claude-3-7-sonnet-20250219',
    object='chat.completion',
    system_fingerprint=None,
    choices=[
        Choices(
            finish_reason='stop',
            index=0,
            message=Message(
                content="The capital of France is Paris.",
                role='assistant',
                tool_calls=None,
                function_call=None,
                reasoning_content='The capital of France is Paris. This is a very straightforward factual question.'
            ),
        )
    ],
    usage=Usage(
        completion_tokens=68,
        prompt_tokens=42,
        total_tokens=110,
        completion_tokens_details=None,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=0,
            text_tokens=None,
            image_tokens=None
        ),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0
    )
)
```

#### Pass `thinking` to Gemini models

You can also pass the `thinking` parameter to Gemini models.

This is translated to Gemini's [`thinkingConfig` parameter](https://ai.google.dev/gemini-api/docs/thinking#set-budget).

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

# !gcloud auth application-default login - run this to add vertex credentials to your env

response = litellm.completion(
  model="vertex_ai/gemini-2.5-flash-preview-04-17",
  messages=[{"role": "user", "content": "What is the capital of France?"}],
  thinking={"type": "enabled", "budget_tokens": 1024},
  vertex_project="project-id",
  vertex_location="us-central1"
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "vertex_ai/gemini-2.5-flash-preview-04-17",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "thinking": {"type": "enabled", "budget_tokens": 1024}
  }'
```

</TabItem>
</Tabs>


### **Context Caching**

Use Vertex AI context caching is supported by calling provider api directly. (Unified Endpoint support coming soon.).

[**Go straight to provider**](../pass_through/vertex_ai.md#context-caching)


## Pre-requisites
* `pip install google-cloud-aiplatform` (pre-installed on proxy docker image)
* Authentication: 
    * run `gcloud auth application-default login` See [Google Cloud Docs](https://cloud.google.com/docs/authentication/external/set-up-adc)
    * Alternatively you can set `GOOGLE_APPLICATION_CREDENTIALS`

    Here's how: [**Jump to Code**](#extra)

      - Create a service account on GCP
      - Export the credentials as a json
      - load the json and json.dump the json as a string
      - store the json string in your environment as `GOOGLE_APPLICATION_CREDENTIALS`

## Sample Usage
```python
import litellm
litellm.vertex_project = "hardy-device-38811" # Your Project ID
litellm.vertex_location = "us-central1"  # proj location

response = litellm.completion(model="gemini-pro", messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}])
```

## Usage with LiteLLM Proxy Server

Here's how to use Vertex AI with the LiteLLM Proxy Server

1. Modify the config.yaml 

  <Tabs>

  <TabItem value="completion_param" label="Different location per model">

  Use this when you need to set a different location for each vertex model

  ```yaml
  model_list:
    - model_name: gemini-vision
      litellm_params:
        model: vertex_ai/gemini-1.0-pro-vision-001
        vertex_project: "project-id"
        vertex_location: "us-central1"
    - model_name: gemini-vision
      litellm_params:
        model: vertex_ai/gemini-1.0-pro-vision-001
        vertex_project: "project-id2"
        vertex_location: "us-east"
  ```

  </TabItem>

  <TabItem value="litellm_param" label="One location all vertex models">

  Use this when you have one vertex location for all models

  ```yaml
  litellm_settings: 
    vertex_project: "hardy-device-38811" # Your Project ID
    vertex_location: "us-central1" # proj location

  model_list: 
    -model_name: team1-gemini-pro
    litellm_params: 
      model: gemini-pro
  ```

  </TabItem>

  </Tabs>

2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="team1-gemini-pro",
      messages = [
          {
              "role": "user",
              "content": "what llm are you"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "team1-gemini-pro",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>


## Authentication - vertex_project, vertex_location, etc. 

Set your vertex credentials via:
- dynamic params
OR
- env vars 


### **Dynamic Params**

You can set:
- `vertex_credentials` (str) - can be a json string or filepath to your vertex ai service account.json
- `vertex_location` (str) - place where vertex model is deployed (us-central1, asia-southeast1, etc.). Some models support the global location, please see [Vertex AI documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#supported_models)
- `vertex_project` Optional[str] - use if vertex project different from the one in vertex_credentials

as dynamic params for a `litellm.completion` call. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import json 

## GET CREDENTIALS 
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)


response = completion(
  model="vertex_ai/gemini-pro",
  messages=[{"content": "You are a good bot.","role": "system"}, {"content": "Hello, how are you?","role": "user"}], 
  vertex_credentials=vertex_credentials_json,
  vertex_project="my-special-project", 
  vertex_location="my-special-location"
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
    - model_name: gemini-1.5-pro
      litellm_params:
        model: gemini-1.5-pro
        vertex_credentials: os.environ/VERTEX_FILE_PATH_ENV_VAR # os.environ["VERTEX_FILE_PATH_ENV_VAR"] = "/path/to/service_account.json" 
        vertex_project: "my-special-project"
        vertex_location: "my-special-location:
```

</TabItem>
</Tabs>




### **Environment Variables**

You can set:
- `GOOGLE_APPLICATION_CREDENTIALS` - store the filepath for your service_account.json in here (used by vertex sdk directly).
- VERTEXAI_LOCATION - place where vertex model is deployed (us-central1, asia-southeast1, etc.)
- VERTEXAI_PROJECT - Optional[str] - use if vertex project different from the one in vertex_credentials

1. GOOGLE_APPLICATION_CREDENTIALS

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
```

2. VERTEXAI_LOCATION

```bash
export VERTEXAI_LOCATION="us-central1" # can be any vertex location
```

3. VERTEXAI_PROJECT

```bash
export VERTEXAI_PROJECT="my-test-project" # ONLY use if model project is different from service account project
```


## Specifying Safety Settings 
In certain use-cases you may need to make calls to the models and pass [safety settings](https://ai.google.dev/docs/safety_setting_gemini) different from the defaults. To do so, simple pass the `safety_settings` argument to `completion` or `acompletion`. For example:

### Set per model/request

<Tabs>

<TabItem value="sdk" label="SDK">

```python
response = completion(
    model="vertex_ai/gemini-pro", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**Option 1: Set in config**
```yaml
model_list:
  - model_name: gemini-experimental
    litellm_params:
      model: vertex_ai/gemini-experimental
      vertex_project: litellm-epic
      vertex_location: us-central1
      safety_settings:
      - category: HARM_CATEGORY_HARASSMENT
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_HATE_SPEECH
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_SEXUALLY_EXPLICIT
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_DANGEROUS_CONTENT
        threshold: BLOCK_NONE
```

**Option 2: Set on call**

```python
response = client.chat.completions.create(
    model="gemini-experimental",
    messages=[
        {
            "role": "user",
            "content": "Can you write exploits?",
        }
    ],
    max_tokens=8192,
    stream=False,
    temperature=0.0,

    extra_body={
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ],
    }
)
```
</TabItem>
</Tabs>

### Set Globally

<Tabs>

<TabItem value="sdk" label="SDK">

```python
import litellm 

litellm.set_verbose = True 👈 See RAW REQUEST/RESPONSE 

litellm.vertex_ai_safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
response = completion(
    model="vertex_ai/gemini-pro", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

```yaml
model_list:
  - model_name: gemini-experimental
    litellm_params:
      model: vertex_ai/gemini-experimental
      vertex_project: litellm-epic
      vertex_location: us-central1

litellm_settings:
    vertex_ai_safety_settings:
      - category: HARM_CATEGORY_HARASSMENT
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_HATE_SPEECH
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_SEXUALLY_EXPLICIT
        threshold: BLOCK_NONE
      - category: HARM_CATEGORY_DANGEROUS_CONTENT
        threshold: BLOCK_NONE
```
</TabItem>
</Tabs>

## Set Vertex Project & Vertex Location
All calls using Vertex AI require the following parameters:
* Your Project ID
```python
import os, litellm 

# set via env var
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811" # Your Project ID`

### OR ###

# set directly on module 
litellm.vertex_project = "hardy-device-38811" # Your Project ID`
```
* Your Project Location
```python
import os, litellm 

# set via env var
os.environ["VERTEXAI_LOCATION"] = "us-central1 # Your Location

### OR ###

# set directly on module 
litellm.vertex_location = "us-central1 # Your Location
```
## Anthropic 
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| claude-3-opus@20240229   | `completion('vertex_ai/claude-3-opus@20240229', messages)` |
| claude-3-5-sonnet@20240620  | `completion('vertex_ai/claude-3-5-sonnet@20240620', messages)` |
| claude-3-sonnet@20240229   | `completion('vertex_ai/claude-3-sonnet@20240229', messages)` |
| claude-3-haiku@20240307   | `completion('vertex_ai/claude-3-haiku@20240307', messages)` |
| claude-3-7-sonnet@20250219   | `completion('vertex_ai/claude-3-7-sonnet@20250219', messages)` |

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

model = "claude-3-sonnet@20240229"

vertex_ai_project = "your-vertex-project" # can also set this as os.environ["VERTEXAI_PROJECT"]
vertex_ai_location = "your-vertex-location" # can also set this as os.environ["VERTEXAI_LOCATION"]

response = completion(
    model="vertex_ai/" + model,
    messages=[{"role": "user", "content": "hi"}],
    temperature=0.7,
    vertex_ai_project=vertex_ai_project,
    vertex_ai_location=vertex_ai_location,
)
print("\nModel Response", response)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: anthropic-vertex
      litellm_params:
        model: vertex_ai/claude-3-sonnet@20240229
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-east-1"
    - model_name: anthropic-vertex
      litellm_params:
        model: vertex_ai/claude-3-sonnet@20240229
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-west-1"
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "anthropic-vertex", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>



### Usage - `thinking` / `reasoning_content`


<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

resp = completion(
    model="vertex_ai/claude-3-7-sonnet-20250219",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    thinking={"type": "enabled", "budget_tokens": 1024},
)

```

</TabItem>

<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
- model_name: claude-3-7-sonnet-20250219
  litellm_params:
    model: vertex_ai/claude-3-7-sonnet-20250219
    vertex_ai_project: "my-test-project"
    vertex_ai_location: "us-west-1"
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "claude-3-7-sonnet-20250219",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "thinking": {"type": "enabled", "budget_tokens": 1024}
  }'
```

</TabItem>
</Tabs>


**Expected Response**

```python
ModelResponse(
    id='chatcmpl-c542d76d-f675-4e87-8e5f-05855f5d0f5e',
    created=1740470510,
    model='claude-3-7-sonnet-20250219',
    object='chat.completion',
    system_fingerprint=None,
    choices=[
        Choices(
            finish_reason='stop',
            index=0,
            message=Message(
                content="The capital of France is Paris.",
                role='assistant',
                tool_calls=None,
                function_call=None,
                provider_specific_fields={
                    'citations': None,
                    'thinking_blocks': [
                        {
                            'type': 'thinking',
                            'thinking': 'The capital of France is Paris. This is a very straightforward factual question.',
                            'signature': 'EuYBCkQYAiJAy6...'
                        }
                    ]
                }
            ),
            thinking_blocks=[
                {
                    'type': 'thinking',
                    'thinking': 'The capital of France is Paris. This is a very straightforward factual question.',
                    'signature': 'EuYBCkQYAiJAy6AGB...'
                }
            ],
            reasoning_content='The capital of France is Paris. This is a very straightforward factual question.'
        )
    ],
    usage=Usage(
        completion_tokens=68,
        prompt_tokens=42,
        total_tokens=110,
        completion_tokens_details=None,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=0,
            text_tokens=None,
            image_tokens=None
        ),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0
    )
)
```



## Meta/Llama API
 
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| meta/llama-3.2-90b-vision-instruct-maas | `completion('vertex_ai/meta/llama-3.2-90b-vision-instruct-maas', messages)` |
| meta/llama3-8b-instruct-maas | `completion('vertex_ai/meta/llama3-8b-instruct-maas', messages)` |
| meta/llama3-70b-instruct-maas | `completion('vertex_ai/meta/llama3-70b-instruct-maas', messages)` |
| meta/llama3-405b-instruct-maas   | `completion('vertex_ai/meta/llama3-405b-instruct-maas', messages)` |
| meta/llama-4-scout-17b-16e-instruct-maas | `completion('vertex_ai/meta/llama-4-scout-17b-16e-instruct-maas', messages)` |
| meta/llama-4-scout-17-128e-instruct-maas | `completion('vertex_ai/meta/llama-4-scout-128b-16e-instruct-maas', messages)` |
| meta/llama-4-maverick-17b-128e-instruct-maas | `completion('vertex_ai/meta/llama-4-maverick-17b-128e-instruct-maas',messages)` |
| meta/llama-4-maverick-17b-16e-instruct-maas | `completion('vertex_ai/meta/llama-4-maverick-17b-16e-instruct-maas',messages)` |

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

model = "meta/llama3-405b-instruct-maas"

vertex_ai_project = "your-vertex-project" # can also set this as os.environ["VERTEXAI_PROJECT"]
vertex_ai_location = "your-vertex-location" # can also set this as os.environ["VERTEXAI_LOCATION"]

response = completion(
    model="vertex_ai/" + model,
    messages=[{"role": "user", "content": "hi"}],
    vertex_ai_project=vertex_ai_project,
    vertex_ai_location=vertex_ai_location,
)
print("\nModel Response", response)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: anthropic-llama
      litellm_params:
        model: vertex_ai/meta/llama3-405b-instruct-maas
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-east-1"
    - model_name: anthropic-llama
      litellm_params:
        model: vertex_ai/meta/llama3-405b-instruct-maas
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-west-1"
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "anthropic-llama", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>

## Mistral API

[**Supported OpenAI Params**](https://github.com/BerriAI/litellm/blob/e0f3cd580cb85066f7d36241a03c30aa50a8a31d/litellm/llms/openai.py#L137)
 
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| mistral-large@latest   | `completion('vertex_ai/mistral-large@latest', messages)` |
| mistral-large@2407   | `completion('vertex_ai/mistral-large@2407', messages)` |
| mistral-nemo@latest   | `completion('vertex_ai/mistral-nemo@latest', messages)` |
| codestral@latest   | `completion('vertex_ai/codestral@latest', messages)` |
| codestral@@2405   | `completion('vertex_ai/codestral@2405', messages)` |

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

model = "mistral-large@2407"

vertex_ai_project = "your-vertex-project" # can also set this as os.environ["VERTEXAI_PROJECT"]
vertex_ai_location = "your-vertex-location" # can also set this as os.environ["VERTEXAI_LOCATION"]

response = completion(
    model="vertex_ai/" + model,
    messages=[{"role": "user", "content": "hi"}],
    vertex_ai_project=vertex_ai_project,
    vertex_ai_location=vertex_ai_location,
)
print("\nModel Response", response)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: vertex-mistral
      litellm_params:
        model: vertex_ai/mistral-large@2407
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-east-1"
    - model_name: vertex-mistral
      litellm_params:
        model: vertex_ai/mistral-large@2407
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-west-1"
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "vertex-mistral", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>


### Usage - Codestral FIM

Call Codestral on VertexAI via the OpenAI [`/v1/completion`](https://platform.openai.com/docs/api-reference/completions/create) endpoint for FIM tasks. 

Note: You can also call Codestral via `/chat/completion`.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
# OR run `!gcloud auth print-access-token` in your terminal

model = "codestral@2405"

vertex_ai_project = "your-vertex-project" # can also set this as os.environ["VERTEXAI_PROJECT"]
vertex_ai_location = "your-vertex-location" # can also set this as os.environ["VERTEXAI_LOCATION"]

response = text_completion(
    model="vertex_ai/" + model,
    vertex_ai_project=vertex_ai_project,
    vertex_ai_location=vertex_ai_location,
    prompt="def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():", 
    suffix="return True",                                              # optional
    temperature=0,                                                     # optional
    top_p=1,                                                           # optional
    max_tokens=10,                                                     # optional
    min_tokens=10,                                                     # optional
    seed=10,                                                           # optional
    stop=["return"],                                                   # optional
)

print("\nModel Response", response)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: vertex-codestral
      litellm_params:
        model: vertex_ai/codestral@2405
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-east-1"
    - model_name: vertex-codestral
      litellm_params:
        model: vertex_ai/codestral@2405
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-west-1"
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl -X POST 'http://0.0.0.0:4000/completions' \
      -H 'Authorization: Bearer sk-1234' \
      -H 'Content-Type: application/json' \
      -d '{
            "model": "vertex-codestral", # 👈 the 'model_name' in config
            "prompt": "def is_odd(n): \n return n % 2 == 1 \ndef test_is_odd():", 
            "suffix":"return True",                                              # optional
            "temperature":0,                                                     # optional
            "top_p":1,                                                           # optional
            "max_tokens":10,                                                     # optional
            "min_tokens":10,                                                     # optional
            "seed":10,                                                           # optional
            "stop":["return"],                                                   # optional
        }'
```

</TabItem>
</Tabs>


## AI21 Models
 
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| jamba-1.5-mini@001   | `completion(model='vertex_ai/jamba-1.5-mini@001', messages)` |
| jamba-1.5-large@001   | `completion(model='vertex_ai/jamba-1.5-large@001', messages)` |

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

model = "meta/jamba-1.5-mini@001"

vertex_ai_project = "your-vertex-project" # can also set this as os.environ["VERTEXAI_PROJECT"]
vertex_ai_location = "your-vertex-location" # can also set this as os.environ["VERTEXAI_LOCATION"]

response = completion(
    model="vertex_ai/" + model,
    messages=[{"role": "user", "content": "hi"}],
    vertex_ai_project=vertex_ai_project,
    vertex_ai_location=vertex_ai_location,
)
print("\nModel Response", response)
```
</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: jamba-1.5-mini
      litellm_params:
        model: vertex_ai/jamba-1.5-mini@001
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-east-1"
    - model_name: jamba-1.5-large
      litellm_params:
        model: vertex_ai/jamba-1.5-large@001
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-west-1"
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "jamba-1.5-large",
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>


## Gemini Pro
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| gemini-pro   | `completion('gemini-pro', messages)`, `completion('vertex_ai/gemini-pro', messages)` |

## Fine-tuned Models

You can call fine-tuned Vertex AI Gemini models through LiteLLM

| Property | Details |
|----------|---------|
| Provider Route | `vertex_ai/gemini/{MODEL_ID}` |
| Vertex Documentation | [Vertex AI - Fine-tuned Gemini Models](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning#test_the_tuned_model_with_a_prompt)|
| Supported Operations | `/chat/completions`, `/completions`, `/embeddings`, `/images` |

To use a model that follows the `/gemini` request/response format, simply set the model parameter as 

```python title="Model parameter for calling fine-tuned gemini models"
model="vertex_ai/gemini/<your-finetuned-model>"
```

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python showLineNumbers title="Example"
import litellm
import os

## set ENV variables
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = litellm.completion(
  model="vertex_ai/gemini/<your-finetuned-model>",  # e.g. vertex_ai/gemini/4965075652664360960
  messages=[{ "content": "Hello, how are you?","role": "user"}],
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Add Vertex Credentials to your env 

```bash title="Authenticate to Vertex AI"
!gcloud auth application-default login
```

2. Setup config.yaml 

```yaml showLineNumbers title="Add to litellm config"
- model_name: finetuned-gemini
  litellm_params:
    model: vertex_ai/gemini/<ENDPOINT_ID>
    vertex_project: <PROJECT_ID>
    vertex_location: <LOCATION>
```

3. Test it! 

<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

```python showLineNumbers title="Example request"
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-key",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="finetuned-gemini",
    messages=[
        {"role": "user", "content": "hi"}
    ]
)
print(response)
```

</TabItem>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="Example request"
curl --location 'https://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: <LITELLM_KEY>' \
--data '{"model": "finetuned-gemini" ,"messages":[{"role": "user", "content":[{"type": "text", "text": "hi"}]}]}'
```

</TabItem>
</Tabs>

</TabItem>
</Tabs>



## Model Garden

:::tip

All OpenAI compatible models from Vertex Model Garden are supported. 

:::

#### Using Model Garden

**Almost all Vertex Model Garden models are OpenAI compatible.**

<Tabs>

<TabItem value="openai" label="OpenAI Compatible Models">

| Property | Details |
|----------|---------|
| Provider Route | `vertex_ai/openai/{MODEL_ID}` |
| Vertex Documentation | [Vertex Model Garden - OpenAI Chat Completions](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gradio_streaming_chat_completions.ipynb), [Vertex Model Garden](https://cloud.google.com/model-garden?hl=en) |
| Supported Operations | `/chat/completions`, `/embeddings` |

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

## set ENV variables
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = completion(
  model="vertex_ai/openai/<your-endpoint-id>", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="proxy" label="Proxy">


**1. Add to config**

```yaml
model_list:
    - model_name: llama3-1-8b-instruct
      litellm_params:
        model: vertex_ai/openai/5464397967697903616
        vertex_ai_project: "my-test-project"
        vertex_ai_location: "us-east-1"
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "llama3-1-8b-instruct", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```




</TabItem>

</Tabs>

</TabItem>

<TabItem value="non-openai" label="Non-OpenAI Compatible Models">

```python
from litellm import completion
import os

## set ENV variables
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = completion(
  model="vertex_ai/<your-endpoint-id>", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

</Tabs>



## Gemini Pro Vision
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| gemini-pro-vision   | `completion('gemini-pro-vision', messages)`, `completion('vertex_ai/gemini-pro-vision', messages)`|

## Gemini 1.5 Pro (and Vision)
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| gemini-1.5-pro   | `completion('gemini-1.5-pro', messages)`, `completion('vertex_ai/gemini-1.5-pro', messages)` |
| gemini-1.5-flash-preview-0514   | `completion('gemini-1.5-flash-preview-0514', messages)`, `completion('vertex_ai/gemini-1.5-flash-preview-0514', messages)` |
| gemini-1.5-pro-preview-0514   | `completion('gemini-1.5-pro-preview-0514', messages)`, `completion('vertex_ai/gemini-1.5-pro-preview-0514', messages)` |




#### Using Gemini Pro Vision

Call `gemini-pro-vision` in the same input/output format as OpenAI [`gpt-4-vision`](https://docs.litellm.ai/docs/providers/openai#openai-vision-models)

LiteLLM Supports the following image types passed in `url`
- Images with Cloud Storage URIs - gs://cloud-samples-data/generative-ai/image/boats.jpeg
- Images with direct links - https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg
- Videos with Cloud Storage URIs - https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4
- Base64 Encoded Local Images

**Example Request - image url**

<Tabs>

<TabItem value="direct" label="Images with direct links">

```python
import litellm

response = litellm.completion(
  model = "vertex_ai/gemini-pro-vision",
  messages=[
      {
          "role": "user",
          "content": [
                          {
                              "type": "text",
                              "text": "Whats in this image?"
                          },
                          {
                              "type": "image_url",
                              "image_url": {
                              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                              }
                          }
                      ]
      }
  ],
)
print(response)
```
</TabItem>

<TabItem value="base" label="Local Base64 Images">

```python
import litellm

def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "cached_logo.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)
response = litellm.completion(
    model="vertex_ai/gemini-pro-vision",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Whats in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64_image
                    },
                },
            ],
        }
    ],
)
print(response)
```
</TabItem>
</Tabs>

## Usage - Function Calling 

LiteLLM supports Function Calling for Vertex AI gemini models. 

```python
from litellm import completion
import os
# set env
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ".."
os.environ["VERTEX_AI_PROJECT"] = ".."
os.environ["VERTEX_AI_LOCATION"] = ".."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

response = completion(
    model="vertex_ai/gemini-pro-vision",
    messages=messages,
    tools=tools,
)
# Add any assertions, here to check response args
print(response)
assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
assert isinstance(
    response.choices[0].message.tool_calls[0].function.arguments, str
)

```


## Usage - PDF / Videos / Audio etc. Files 

Pass any file supported by Vertex AI, through LiteLLM. 

LiteLLM Supports the following file types passed in url. 

Using `file` message type for VertexAI is live from v1.65.1+ 

```
Files with Cloud Storage URIs - gs://cloud-samples-data/generative-ai/image/boats.jpeg
Files with direct links - https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg
Videos with Cloud Storage URIs - https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4
Base64 Encoded Local Files
```

<Tabs>
<TabItem value="sdk" label="SDK">

### **Using `gs://` or any URL**
```python
from litellm import completion

response = completion(
    model="vertex_ai/gemini-1.5-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a very professional document summarization specialist. Please summarize the given document."},
                {
                    "type": "file",
                    "file": {
                        "file_id": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
                        "format": "application/pdf" # OPTIONAL - specify mime-type
                    }
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```

### **using base64**
```python
from litellm import completion
import base64
import requests

# URL of the file
url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"

# Download the file
response = requests.get(url)
file_data = response.content

encoded_file = base64.b64encode(file_data).decode("utf-8")

response = completion(
    model="vertex_ai/gemini-1.5-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a very professional document summarization specialist. Please summarize the given document."},
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:application/pdf;base64,{encoded_file}", # 👈 PDF
                    }  
                },
                {
                    "type": "audio_input",
                    "audio_input {
                        "audio_input": f"data:audio/mp3;base64,{encoded_file}", # 👈 AUDIO File ('file' message works as too)
                    }  
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add model to config 

```yaml
- model_name: gemini-1.5-flash
  litellm_params:
    model: vertex_ai/gemini-1.5-flash
    vertex_credentials: "/path/to/service_account.json"
```

2. Start Proxy

```
litellm --config /path/to/config.yaml
```

3. Test it! 

**Using `gs://`**
```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-1.5-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "You are a very professional document summarization specialist. Please summarize the given document"
          },
          {
                "type": "file",
                "file": {
                    "file_id": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
                    "format": "application/pdf" # OPTIONAL
                }
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'

```


```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-1.5-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "You are a very professional document summarization specialist. Please summarize the given document"
          },
          {
                "type": "file",
                "file": {
                    "file_data": f"data:application/pdf;base64,{encoded_file}", # 👈 PDF
                },
            },
            {
                "type": "audio_input",
                "audio_input {
                    "audio_input": f"data:audio/mp3;base64,{encoded_file}", # 👈 AUDIO File ('file' message works as too)
                }  
            },
    ]
      }
    ],
    "max_tokens": 300
  }'

```
</TabItem>
</Tabs>


## Chat Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| chat-bison-32k   | `completion('chat-bison-32k', messages)` |
| chat-bison       | `completion('chat-bison', messages)`     |
| chat-bison@001   | `completion('chat-bison@001', messages)` |

## Code Chat Models
| Model Name           | Function Call                              |
|----------------------|--------------------------------------------|
| codechat-bison       | `completion('codechat-bison', messages)`     |
| codechat-bison-32k   | `completion('codechat-bison-32k', messages)` |
| codechat-bison@001   | `completion('codechat-bison@001', messages)` |

## Text Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| text-bison       | `completion('text-bison', messages)` |
| text-bison@001   | `completion('text-bison@001', messages)` |

## Code Text Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| code-bison       | `completion('code-bison', messages)` |
| code-bison@001   | `completion('code-bison@001', messages)` |
| code-gecko@001   | `completion('code-gecko@001', messages)` |
| code-gecko@latest| `completion('code-gecko@latest', messages)` |


## **Embedding Models**

#### Usage - Embedding

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm
from litellm import embedding
litellm.vertex_project = "hardy-device-38811" # Your Project ID
litellm.vertex_location = "us-central1"  # proj location

response = embedding(
    model="vertex_ai/textembedding-gecko",
    input=["good morning from litellm"],
)
print(response)
```
</TabItem>

<TabItem value="proxy" label="LiteLLM PROXY">


1. Add model to config.yaml
```yaml
model_list:
  - model_name: snowflake-arctic-embed-m-long-1731622468876
    litellm_params:
      model: vertex_ai/<your-model-id>
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request using OpenAI Python SDK, Langchain Python SDK

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    model="snowflake-arctic-embed-m-long-1731622468876", 
    input = ["good morning from litellm", "this is another item"],
)

print(response)
```


</TabItem>
</Tabs>

#### Supported Embedding Models
All models listed [here](https://github.com/BerriAI/litellm/blob/57f37f743886a0249f630a6792d49dffc2c5d9b7/model_prices_and_context_window.json#L835) are supported

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text-embedding-004 | `embedding(model="vertex_ai/text-embedding-004", input)` | 
| text-multilingual-embedding-002 | `embedding(model="vertex_ai/text-multilingual-embedding-002", input)` | 
| textembedding-gecko | `embedding(model="vertex_ai/textembedding-gecko", input)` | 
| textembedding-gecko-multilingual | `embedding(model="vertex_ai/textembedding-gecko-multilingual", input)` | 
| textembedding-gecko-multilingual@001 | `embedding(model="vertex_ai/textembedding-gecko-multilingual@001", input)` | 
| textembedding-gecko@001 | `embedding(model="vertex_ai/textembedding-gecko@001", input)` | 
| textembedding-gecko@003 | `embedding(model="vertex_ai/textembedding-gecko@003", input)` | 
| text-embedding-preview-0409 | `embedding(model="vertex_ai/text-embedding-preview-0409", input)` |
| text-multilingual-embedding-preview-0409 | `embedding(model="vertex_ai/text-multilingual-embedding-preview-0409", input)` | 
| Fine-tuned OR Custom Embedding models | `embedding(model="vertex_ai/<your-model-id>", input)` | 

### Supported OpenAI (Unified) Params

| [param](../embedding/supported_embedding.md#input-params-for-litellmembedding) | type | [vertex equivalent](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api) |
|-------|-------------|--------------------|
| `input` | **string or List[string]** | `instances` |
| `dimensions` | **int** | `output_dimensionality` |
| `input_type` | **Literal["RETRIEVAL_QUERY","RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION"]** | `task_type` |

#### Usage with OpenAI (Unified) Params


<Tabs>
<TabItem value="sdk" label="SDK">

```python
response = litellm.embedding(
    model="vertex_ai/text-embedding-004",
    input=["good morning from litellm", "gm"]
    input_type = "RETRIEVAL_DOCUMENT",
    dimensions=1,
)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY">


```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    model="text-embedding-004", 
    input = ["good morning from litellm", "gm"],
    dimensions=1,
    extra_body = {
        "input_type": "RETRIEVAL_QUERY",
    }
)

print(response)
```
</TabItem>
</Tabs>


### Supported Vertex Specific Params

| param | type |
|-------|-------------|
| `auto_truncate` | **bool** |
| `task_type` | **Literal["RETRIEVAL_QUERY","RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION"]** |
| `title` | **str** |

#### Usage with Vertex Specific Params  (Use `task_type` and `title`)

You can pass any vertex specific params to the embedding model. Just pass them to the embedding function like this: 

[Relevant Vertex AI doc with all embedding params](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#request_body)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
response = litellm.embedding(
    model="vertex_ai/text-embedding-004",
    input=["good morning from litellm", "gm"]
    task_type = "RETRIEVAL_DOCUMENT",
    title = "test",
    dimensions=1,
    auto_truncate=True,
)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY">


```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.embeddings.create(
    model="text-embedding-004", 
    input = ["good morning from litellm", "gm"],
    dimensions=1,
    extra_body = {
        "task_type": "RETRIEVAL_QUERY",
        "auto_truncate": True,
        "title": "test",
    }
)

print(response)
```
</TabItem>
</Tabs>

## **Multi-Modal Embeddings**


Known Limitations:
- Only supports 1 image / video / image per request
- Only supports GCS or base64 encoded images / videos

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

Using GCS Images

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input="gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png" # will be sent as a gcs image
)
```

Using base 64 encoded images

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input="data:image/jpeg;base64,..." # will be sent as a base64 encoded image
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: multimodalembedding@001
    litellm_params:
      model: vertex_ai/multimodalembedding@001
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK, Langchain Python SDK


<Tabs>

<TabItem value="OpenAI SDK" label="OpenAI SDK">

Requests with GCS Image / Video URI

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png",
)

print(response)
```

Requests with base64 encoded images

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = "data:image/jpeg;base64,...",
)

print(response)
```

</TabItem>

<TabItem value="langchain" label="Langchain">

Requests with GCS Image / Video URI
```python
from langchain_openai import OpenAIEmbeddings

embeddings_models = "multimodalembedding@001"

embeddings = OpenAIEmbeddings(
    model="multimodalembedding@001",
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",  # type: ignore
)


query_result = embeddings.embed_query(
    "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
)
print(query_result)

```

Requests with base64 encoded images

```python
from langchain_openai import OpenAIEmbeddings

embeddings_models = "multimodalembedding@001"

embeddings = OpenAIEmbeddings(
    model="multimodalembedding@001",
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",  # type: ignore
)


query_result = embeddings.embed_query(
    "data:image/jpeg;base64,..."
)
print(query_result)

```

</TabItem>

</Tabs>
</TabItem>


<TabItem value="proxy-vtx" label="LiteLLM PROXY (Vertex SDK)">

1. Add model to config.yaml
```yaml
default_vertex_config:
  vertex_project: "adroit-crow-413218"
  vertex_location: "us-central1"
  vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK

```python
import vertexai

from vertexai.vision_models import Image, MultiModalEmbeddingModel, Video
from vertexai.vision_models import VideoSegmentConfig
from google.auth.credentials import Credentials


LITELLM_PROXY_API_KEY = "sk-1234"
LITELLM_PROXY_BASE = "http://0.0.0.0:4000/vertex-ai"

import datetime

class CredentialsWrapper(Credentials):
    def __init__(self, token=None):
        super().__init__()
        self.token = token
        self.expiry = None  # or set to a future date if needed
        
    def refresh(self, request):
        pass
    
    def apply(self, headers, token=None):
        headers['Authorization'] = f'Bearer {self.token}'

    @property
    def expired(self):
        return False  # Always consider the token as non-expired

    @property
    def valid(self):
        return True  # Always consider the credentials as valid

credentials = CredentialsWrapper(token=LITELLM_PROXY_API_KEY)

vertexai.init(
    project="adroit-crow-413218",
    location="us-central1",
    api_endpoint=LITELLM_PROXY_BASE,
    credentials = credentials,
    api_transport="rest",
   
)

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
image = Image.load_from_file(
    "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"
)

embeddings = model.get_embeddings(
    image=image,
    contextual_text="Colosseum",
    dimension=1408,
)
print(f"Image Embedding: {embeddings.image_embedding}")
print(f"Text Embedding: {embeddings.text_embedding}")
```

</TabItem>
</Tabs>


### Text + Image + Video Embeddings

<Tabs>
<TabItem value="sdk" label="SDK">

Text + Image 

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input=["hey", "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"] # will be sent as a gcs image
)
```

Text + Video 

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input=["hey", "gs://my-bucket/embeddings/supermarket-video.mp4"] # will be sent as a gcs image
)
```

Image + Video 

```python
response = await litellm.aembedding(
    model="vertex_ai/multimodalembedding@001",
    input=["gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png", "gs://my-bucket/embeddings/supermarket-video.mp4"] # will be sent as a gcs image
)
```


</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: multimodalembedding@001
    litellm_params:
      model: vertex_ai/multimodalembedding@001
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK, Langchain Python SDK


Text + Image 

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = ["hey", "gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png"],
)

print(response)
```

Text + Video 
```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = ["hey", "gs://my-bucket/embeddings/supermarket-video.mp4"],
)

print(response)
```

Image + Video 
```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# # request sent to model set on litellm proxy, `litellm --model`
response = client.embeddings.create(
    model="multimodalembedding@001", 
    input = ["gs://cloud-samples-data/vertex-ai/llm/prompts/landmark1.png", "gs://my-bucket/embeddings/supermarket-video.mp4"],
)

print(response)
```

</TabItem>
</Tabs>


## **Image Generation Models**

Usage 

```python
response = await litellm.aimage_generation(
    prompt="An olympic size swimming pool",
    model="vertex_ai/imagegeneration@006",
    vertex_ai_project="adroit-crow-413218",
    vertex_ai_location="us-central1",
)
```

**Generating multiple images**

Use the `n` parameter to pass how many images you want generated
```python
response = await litellm.aimage_generation(
    prompt="An olympic size swimming pool",
    model="vertex_ai/imagegeneration@006",
    vertex_ai_project="adroit-crow-413218",
    vertex_ai_location="us-central1",
    n=1,
)
```

### Supported Image Generation Models

| Model Name                   | FUsage                                            |
|------------------------------|--------------------------------------------------------------|
| `imagen-3.0-generate-001`      | `litellm.image_generation('vertex_ai/imagen-3.0-generate-001', prompt)` |
| `imagen-3.0-fast-generate-001` | `litellm.image_generation('vertex_ai/imagen-3.0-fast-generate-001', prompt)` |
| `imagegeneration@006`          | `litellm.image_generation('vertex_ai/imagegeneration@006', prompt)`  |
| `imagegeneration@005`          | `litellm.image_generation('vertex_ai/imagegeneration@005', prompt)`  |
| `imagegeneration@002`          | `litellm.image_generation('vertex_ai/imagegeneration@002', prompt)`  |




## **Gemini TTS (Text-to-Speech) Audio Output**

:::info

LiteLLM supports Gemini TTS models on Vertex AI that can generate audio responses using the OpenAI-compatible `audio` parameter format.

:::

### Supported Models

LiteLLM supports Gemini TTS models with audio capabilities on Vertex AI (e.g. `vertex_ai/gemini-2.5-flash-preview-tts` and `vertex_ai/gemini-2.5-pro-preview-tts`). For the complete list of available TTS models and voices, see the [official Gemini TTS documentation](https://ai.google.dev/gemini-api/docs/speech-generation).

### Limitations

:::warning

**Important Limitations**:
- Gemini TTS models only support the `pcm16` audio format
- **Streaming support has not been added** to TTS models yet
- The `modalities` parameter must be set to `['audio']` for TTS requests

:::

### Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import json

## GET CREDENTIALS
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

response = completion(
    model="vertex_ai/gemini-2.5-flash-preview-tts",
    messages=[{"role": "user", "content": "Say hello in a friendly voice"}],
    modalities=["audio"],  # Required for TTS models
    audio={
        "voice": "Kore",
        "format": "pcm16"  # Required: must be "pcm16"
    },
    vertex_credentials=vertex_credentials_json
)

print(response)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: gemini-tts-flash
    litellm_params:
      model: vertex_ai/gemini-2.5-flash-preview-tts
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
  - model_name: gemini-tts-pro
    litellm_params:
      model: vertex_ai/gemini-2.5-pro-preview-tts
      vertex_project: "your-project-id"
      vertex_location: "us-central1"
      vertex_credentials: "/path/to/service_account.json"
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Make TTS request

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "gemini-tts-flash",
    "messages": [{"role": "user", "content": "Say hello in a friendly voice"}],
    "modalities": ["audio"],
    "audio": {
      "voice": "Kore",
      "format": "pcm16"
    }
  }'
```

</TabItem>
</Tabs>

### Advanced Usage

You can combine TTS with other Gemini features:

```python
response = completion(
    model="vertex_ai/gemini-2.5-pro-preview-tts",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that speaks clearly."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    modalities=["audio"],
    audio={
        "voice": "Charon",
        "format": "pcm16"
    },
    temperature=0.7,
    max_tokens=150,
    vertex_credentials=vertex_credentials_json
)
```

For more information about Gemini's TTS capabilities and available voices, see the [official Gemini TTS documentation](https://ai.google.dev/gemini-api/docs/speech-generation).

## **Text to Speech APIs**

:::info

LiteLLM supports calling [Vertex AI Text to Speech API](https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech) in the OpenAI text to speech API format

:::



### Usage - Basic

<Tabs>
<TabItem value="sdk" label="SDK">

Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param

**Sync Usage**

```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"
response = litellm.speech(
    model="vertex_ai/",
    input="hello what llm guardrail do you have",
)
response.stream_to_file(speech_file_path)
```

**Async Usage**
```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"
response = litellm.aspeech(
    model="vertex_ai/",
    input="hello what llm guardrail do you have",
)
response.stream_to_file(speech_file_path)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

1. Add model to config.yaml
```yaml
model_list:
  - model_name: vertex-tts
    litellm_params:
      model: vertex_ai/ # Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param
      vertex_project: "adroit-crow-413218"
      vertex_location: "us-central1"
      vertex_credentials: adroit-crow-413218-a956eef1a2a8.json 

litellm_settings:
  drop_params: True
```

2. Start Proxy 

```
$ litellm --config /path/to/config.yaml
```

3. Make Request use OpenAI Python SDK


```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

# see supported values for "voice" on vertex here: 
# https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech
response = client.audio.speech.create(
    model = "vertex-tts",
    input="the quick brown fox jumped over the lazy dogs",
    voice={'languageCode': 'en-US', 'name': 'en-US-Studio-O'}
)
print("response from proxy", response)
```

</TabItem>
</Tabs>


### Usage - `ssml` as input

Pass your `ssml` as input to the `input` param, if it contains `<speak>`, it will be automatically detected and passed as `ssml` to the Vertex AI API

If you need to force your `input` to be passed as `ssml`, set `use_ssml=True`

<Tabs>
<TabItem value="sdk" label="SDK">

Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param


```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"


ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

response = litellm.speech(
    input=ssml,
    model="vertex_ai/test",
    voice={
        "languageCode": "en-UK",
        "name": "en-UK-Studio-O",
    },
    audioConfig={
        "audioEncoding": "LINEAR22",
        "speakingRate": "10",
    },
)
response.stream_to_file(speech_file_path)
```

</TabItem>

<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

# see supported values for "voice" on vertex here: 
# https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech
response = client.audio.speech.create(
    model = "vertex-tts",
    input=ssml,
    voice={'languageCode': 'en-US', 'name': 'en-US-Studio-O'},
)
print("response from proxy", response)
```

</TabItem>
</Tabs>


### Forcing SSML Usage

You can force the use of SSML by setting the `use_ssml` parameter to `True`. This is useful when you want to ensure that your input is treated as SSML, even if it doesn't contain the `<speak>` tags.

Here are examples of how to force SSML usage:


<Tabs>
<TabItem value="sdk" label="SDK">

Vertex AI does not support passing a `model` param - so passing `model=vertex_ai/` is the only required param


```python
speech_file_path = Path(__file__).parent / "speech_vertex.mp3"


ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

response = litellm.speech(
    input=ssml,
    use_ssml=True,
    model="vertex_ai/test",
    voice={
        "languageCode": "en-UK",
        "name": "en-UK-Studio-O",
    },
    audioConfig={
        "audioEncoding": "LINEAR22",
        "speakingRate": "10",
    },
)
response.stream_to_file(speech_file_path)
```

</TabItem>

<TabItem value="proxy" label="LiteLLM PROXY (Unified Endpoint)">

```python
import openai

client = openai.OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

ssml = """
<speak>
    <p>Hello, world!</p>
    <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
</speak>
"""

# see supported values for "voice" on vertex here: 
# https://console.cloud.google.com/vertex-ai/generative/speech/text-to-speech
response = client.audio.speech.create(
    model = "vertex-tts",
    input=ssml, # pass as None since OpenAI SDK requires this param
    voice={'languageCode': 'en-US', 'name': 'en-US-Studio-O'},
    extra_body={"use_ssml": True},
)
print("response from proxy", response)
```

</TabItem>
</Tabs>

## **Batch APIs**

Just add the following Vertex env vars to your environment. 

```bash
# GCS Bucket settings, used to store batch prediction files in
export GCS_BUCKET_NAME = "litellm-testing-bucket" # the bucket you want to store batch prediction files in
export GCS_PATH_SERVICE_ACCOUNT="/path/to/service_account.json" # path to your service account json file

# Vertex /batch endpoint settings, used for LLM API requests
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json" # path to your service account json file
export VERTEXAI_LOCATION="us-central1" # can be any vertex location
export VERTEXAI_PROJECT="my-test-project" 
```

### Usage


#### 1. Create a file of batch requests for vertex

LiteLLM expects the file to follow the **[OpenAI batches files format](https://platform.openai.com/docs/guides/batch)**

Each `body` in the file should be an **OpenAI API request**

Create a file called `vertex_batch_completions.jsonl` in the current working directory, the `model` should be the Vertex AI model name
```
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gemini-1.5-flash-001", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 10}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gemini-1.5-flash-001", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 10}}
```


#### 2. Upload a File of batch requests

For `vertex_ai` litellm will upload the file to the provided `GCS_BUCKET_NAME`

```python
import os
oai_client = OpenAI(
    api_key="sk-1234",               # litellm proxy API key
    base_url="http://localhost:4000" # litellm proxy base url
)
file_name = "vertex_batch_completions.jsonl" # 
_current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_current_dir, file_name)
file_obj = oai_client.files.create(
    file=open(file_path, "rb"),
    purpose="batch",
    extra_body={"custom_llm_provider": "vertex_ai"}, # tell litellm to use vertex_ai for this file upload
)
```

**Expected Response**

```json
{
    "id": "gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/d3f198cd-c0d1-436d-9b1e-28e3f282997a",
    "bytes": 416,
    "created_at": 1733392026,
    "filename": "litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/d3f198cd-c0d1-436d-9b1e-28e3f282997a",
    "object": "file",
    "purpose": "batch",
    "status": "uploaded",
    "status_details": null
}
```



#### 3. Create a batch 

```python
batch_input_file_id = file_obj.id # use `file_obj` from step 2
create_batch_response = oai_client.batches.create(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id=batch_input_file_id, # example input_file_id = "gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/c2b1b785-252b-448c-b180-033c4c63b3ce"
    extra_body={"custom_llm_provider": "vertex_ai"}, # tell litellm to use `vertex_ai` for this batch request
)
```

**Expected Response**

```json
{
    "id": "3814889423749775360",
    "completion_window": "24hrs",
    "created_at": 1733392026,
    "endpoint": "",
    "input_file_id": "gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/d3f198cd-c0d1-436d-9b1e-28e3f282997a",
    "object": "batch",
    "status": "validating",
    "cancelled_at": null,
    "cancelling_at": null,
    "completed_at": null,
    "error_file_id": null,
    "errors": null,
    "expired_at": null,
    "expires_at": null,
    "failed_at": null,
    "finalizing_at": null,
    "in_progress_at": null,
    "metadata": null,
    "output_file_id": "gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001",
    "request_counts": null
}
```

#### 4. Retrieve a batch

```python
retrieved_batch = oai_client.batches.retrieve(
    batch_id=create_batch_response.id,
    extra_body={"custom_llm_provider": "vertex_ai"}, # tell litellm to use `vertex_ai` for this batch request
)
```

**Expected Response**

```json
{
    "id": "3814889423749775360",
    "completion_window": "24hrs",
    "created_at": 1736500100,
    "endpoint": "",
    "input_file_id": "gs://example-bucket-1-litellm/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/7b2e47f5-3dd4-436d-920f-f9155bbdc952",
    "object": "batch",
    "status": "completed",
    "cancelled_at": null,
    "cancelling_at": null,
    "completed_at": null,
    "error_file_id": null,
    "errors": null,
    "expired_at": null,
    "expires_at": null,
    "failed_at": null,
    "finalizing_at": null,
    "in_progress_at": null,
    "metadata": null,
    "output_file_id": "gs://example-bucket-1-litellm/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001",
    "request_counts": null
}
```


## **Fine Tuning APIs**


| Property | Details |
|----------|---------|
| Description | Create Fine Tuning Jobs in Vertex AI (`/tuningJobs`) using OpenAI Python SDK |
| Vertex Fine Tuning Documentation | [Vertex Fine Tuning](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning#create-tuning) |

### Usage

#### 1. Add `finetune_settings` to your config.yaml
```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

# 👇 Key change: For /fine_tuning/jobs endpoints
finetune_settings:
  - custom_llm_provider: "vertex_ai"
    vertex_project: "adroit-crow-413218"
    vertex_location: "us-central1"
    vertex_credentials: "/Users/ishaanjaffer/Downloads/adroit-crow-413218-a956eef1a2a8.json"
```

#### 2. Create a Fine Tuning Job

<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

```python
ft_job = await client.fine_tuning.jobs.create(
    model="gemini-1.0-pro-002",                  # Vertex model you want to fine-tune
    training_file="gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",                 # file_id from create file response
    extra_body={"custom_llm_provider": "vertex_ai"}, # tell litellm proxy which provider to use
)
```
</TabItem>

<TabItem value="curl" label="curl">

```shell
curl http://localhost:4000/v1/fine_tuning/jobs \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
    "custom_llm_provider": "vertex_ai",
    "model": "gemini-1.0-pro-002",
    "training_file": "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl"
    }'
```
</TabItem>

</Tabs>


**Advanced use case - Passing `adapter_size` to the Vertex AI API**

Set hyper_parameters, such as `n_epochs`, `learning_rate_multiplier` and `adapter_size`. [See Vertex Advanced Hyperparameters](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning#advanced_use_case)


<Tabs>
<TabItem value="openai" label="OpenAI Python SDK">

```python

ft_job = client.fine_tuning.jobs.create(
    model="gemini-1.0-pro-002",                  # Vertex model you want to fine-tune
    training_file="gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",                 # file_id from create file response
    hyperparameters={
        "n_epochs": 3,                      # epoch_count on Vertex
        "learning_rate_multiplier": 0.1,    # learning_rate_multiplier on Vertex
        "adapter_size": "ADAPTER_SIZE_ONE"  # type: ignore, vertex specific hyperparameter
    },
    extra_body={
        "custom_llm_provider": "vertex_ai",
    },
)
```
</TabItem>

<TabItem value="curl" label="curl">

```shell
curl http://localhost:4000/v1/fine_tuning/jobs \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-1234" \
    -d '{
    "custom_llm_provider": "vertex_ai",
    "model": "gemini-1.0-pro-002",
    "training_file": "gs://cloud-samples-data/ai-platform/generative_ai/sft_train_data.jsonl",
    "hyperparameters": {
        "n_epochs": 3,
        "learning_rate_multiplier": 0.1,
        "adapter_size": "ADAPTER_SIZE_ONE"
    }
    }'
```
</TabItem>

</Tabs>


## Extra

### Using `GOOGLE_APPLICATION_CREDENTIALS`
Here's the code for storing your service account credentials as `GOOGLE_APPLICATION_CREDENTIALS` environment variable:


```python
import os 
import tempfile

def load_vertex_ai_credentials():
  # Define the path to the vertex_key.json file
  print("loading vertex ai credentials")
  filepath = os.path.dirname(os.path.abspath(__file__))
  vertex_key_path = filepath + "/vertex_key.json"

  # Read the existing content of the file or create an empty dictionary
  try:
      with open(vertex_key_path, "r") as file:
          # Read the file content
          print("Read vertexai file path")
          content = file.read()

          # If the file is empty or not valid JSON, create an empty dictionary
          if not content or not content.strip():
              service_account_key_data = {}
          else:
              # Attempt to load the existing JSON content
              file.seek(0)
              service_account_key_data = json.load(file)
  except FileNotFoundError:
      # If the file doesn't exist, create an empty dictionary
      service_account_key_data = {}

  # Create a temporary file
  with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
      # Write the updated content to the temporary file
      json.dump(service_account_key_data, temp_file, indent=2)

  # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)
```


### Using GCP Service Account 

:::info

Trying to deploy LiteLLM on Google Cloud Run? Tutorial [here](https://docs.litellm.ai/docs/proxy/deploy#deploy-on-google-cloud-run)

:::

1. Figure out the Service Account bound to the Google Cloud Run service

<Image img={require('../../img/gcp_acc_1.png')} />

2. Get the FULL EMAIL address of the corresponding Service Account

3. Next, go to IAM & Admin > Manage Resources , select your top-level project that houses your Google Cloud Run Service

Click `Add Principal`

<Image img={require('../../img/gcp_acc_2.png')}/>

4. Specify the Service Account as the principal and Vertex AI User as the role

<Image img={require('../../img/gcp_acc_3.png')}/>

Once that's done, when you deploy the new container in the Google Cloud Run service, LiteLLM will have automatic access to all Vertex AI endpoints.


s/o @[Darien Kindlund](https://www.linkedin.com/in/kindlund/) for this tutorial




