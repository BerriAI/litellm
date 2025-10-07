import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI - Response API

## Usage

### LiteLLM Python SDK


#### Non-streaming
```python showLineNumbers title="OpenAI Non-streaming Response"
import litellm

# Non-streaming response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

print(response)
```

#### Streaming
```python showLineNumbers title="OpenAI Streaming Response"
import litellm

# Streaming response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

#### Image Generation with Streaming
```python showLineNumbers title="OpenAI Streaming Image Generation"
import litellm
import base64

# Streaming image generation with partial images
stream = litellm.responses(
    model="gpt-4.1",  # Use an actual image generation model
    input="Generate a gorgeous image of a river made of white owl feathers",
    stream=True,
    tools=[{"type": "image_generation", "partial_images": 2}],

)

for event in stream:
    if event.type == "response.image_generation_call.partial_image":
        idx = event.partial_image_index
        image_base64 = event.partial_image_b64
        image_bytes = base64.b64decode(image_base64)
        with open(f"river{idx}.png", "wb") as f:
            f.write(image_bytes)
```

#### GET a Response
```python showLineNumbers title="Get Response by ID"
import litellm

# First, create a response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

# Get the response ID
response_id = response.id

# Retrieve the response by ID
retrieved_response = litellm.get_responses(
    response_id=response_id
)

print(retrieved_response)

# For async usage
# retrieved_response = await litellm.aget_responses(response_id=response_id)
```

#### DELETE a Response
```python showLineNumbers title="Delete Response by ID"
import litellm

# First, create a response
response = litellm.responses(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    max_output_tokens=100
)

# Get the response ID
response_id = response.id

# Delete the response by ID
delete_response = litellm.delete_responses(
    response_id=response_id
)

print(delete_response)

# For async usage
# delete_response = await litellm.adelete_responses(response_id=response_id)
```


### LiteLLM Proxy with OpenAI SDK

1. Set up config.yaml

```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: openai/o1-pro
    litellm_params:
      model: openai/o1-pro
      api_key: os.environ/OPENAI_API_KEY
```

2. Start LiteLLM Proxy Server

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Use OpenAI SDK with LiteLLM Proxy

#### Non-streaming
```python showLineNumbers title="OpenAI Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
```

#### Streaming
```python showLineNumbers title="OpenAI Proxy Streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Streaming response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)

for event in response:
    print(event)
```

#### Image Generation with Streaming
```python showLineNumbers title="OpenAI Proxy Streaming Image Generation"
from openai import OpenAI
import base64

# Initialize client with your proxy URL
client = OpenAI(api_key="sk-1234", base_url="http://localhost:4000")

stream = client.responses.create(
    model="gpt-4.1",
    input="Draw a gorgeous image of a river made of white owl feathers, snaking its way through a serene winter landscape",
    stream=True,
    tools=[{"type": "image_generation", "partial_images": 2}],
)


for event in stream:
    print(f"event: {event}")
    if event.type == "response.image_generation_call.partial_image":
        idx = event.partial_image_index
        image_base64 = event.partial_image_b64
        image_bytes = base64.b64decode(image_base64)
        with open(f"river{idx}.png", "wb") as f:
            f.write(image_bytes)

```

#### GET a Response
```python showLineNumbers title="Get Response by ID with OpenAI SDK"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# First, create a response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

# Get the response ID
response_id = response.id

# Retrieve the response by ID
retrieved_response = client.responses.retrieve(response_id)

print(retrieved_response)
```

#### DELETE a Response
```python showLineNumbers title="Delete Response by ID with OpenAI SDK"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# First, create a response
response = client.responses.create(
    model="openai/o1-pro",
    input="Tell me a three sentence bedtime story about a unicorn."
)

# Get the response ID
response_id = response.id

# Delete the response by ID
delete_response = client.responses.delete(response_id)

print(delete_response)
```


## Supported Responses API Parameters

| Provider | Supported Parameters |
|----------|---------------------|
| `openai` | [All Responses API parameters are supported](https://github.com/BerriAI/litellm/blob/7c3df984da8e4dff9201e4c5353fdc7a2b441831/litellm/llms/openai/responses/transformation.py#L23) |

## Reusable Prompts

Use the `prompt` parameter to reference a stored prompt template and optionally supply variables.

```python showLineNumbers title="Stored Prompt"
import litellm

response = litellm.responses(
    model="openai/o1-pro",
    prompt={
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {
            "customer_name": "Jane Doe",
            "product": "40oz juice box",
        },
    },
)

print(response)
```

The same parameter is supported when calling the LiteLLM proxy with the OpenAI SDK:

```python showLineNumbers title="Stored Prompt via Proxy"
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000", api_key="your-api-key")

response = client.responses.create(
    model="openai/o1-pro",
    prompt={
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {
            "customer_name": "Jane Doe",
            "product": "40oz juice box",
        },
    },
)

print(response)
```

## Computer Use 

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import litellm

# Non-streaming response
response = litellm.responses(
    model="computer-use-preview",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser" # other possible values: "mac", "windows", "ubuntu"
    }],    
    input=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Check the latest OpenAI news on bing.com."
            }
            # Optional: include a screenshot of the initial state of the environment
            # {
            #     type: "input_image",
            #     image_url: f"data:image/png;base64,{screenshot_base64}"
            # }
          ]
        }
    ],
    reasoning={
        "summary": "concise",
    },
    truncation="auto"
)

print(response.output)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Set up config.yaml

```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: openai/o1-pro
    litellm_params:
      model: openai/o1-pro
      api_key: os.environ/OPENAI_API_KEY
```

2. Start LiteLLM Proxy Server

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```python showLineNumbers title="OpenAI Proxy Non-streaming Response"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Non-streaming response
response = client.responses.create(
    model="computer-use-preview",
    tools=[{
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser" # other possible values: "mac", "windows", "ubuntu"
    }],    
    input=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "Check the latest OpenAI news on bing.com."
            }
            # Optional: include a screenshot of the initial state of the environment
            # {
            #     type: "input_image",
            #     image_url: f"data:image/png;base64,{screenshot_base64}"
            # }
          ]
        }
    ],
    reasoning={
        "summary": "concise",
    },
    truncation="auto"
)

print(response)
```


</TabItem>
</Tabs>


## MCP Tools 

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python showLineNumbers title="MCP Tools with LiteLLM SDK"
import litellm
from typing import Optional

# Configure MCP Tools
MCP_TOOLS = [
    {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "allowed_tools": ["ask_question"]
    }
]

# Step 1: Make initial request - OpenAI will use MCP LIST and return MCP calls for approval
response = litellm.responses(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input="What transport protocols does the 2025-03-26 version of the MCP spec support?"
)

# Get the MCP approval ID
mcp_approval_id = None
for output in response.output:
    if output.type == "mcp_approval_request":
        mcp_approval_id = output.id
        break

# Step 2: Send followup with approval for the MCP call
response_with_mcp_call = litellm.responses(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input=[
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": mcp_approval_id
        }
    ],
    previous_response_id=response.id,
)

print(response_with_mcp_call)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Set up config.yaml

```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: openai/gpt-4.1
    litellm_params:
      model: openai/gpt-4.1
      api_key: os.environ/OPENAI_API_KEY
```

2. Start LiteLLM Proxy Server

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```python showLineNumbers title="MCP Tools with OpenAI SDK via LiteLLM Proxy"
from openai import OpenAI
from typing import Optional

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Configure MCP Tools
MCP_TOOLS = [
    {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "allowed_tools": ["ask_question"]
    }
]

# Step 1: Make initial request - OpenAI will use MCP LIST and return MCP calls for approval
response = client.responses.create(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input="What transport protocols does the 2025-03-26 version of the MCP spec support?"
)

# Get the MCP approval ID
mcp_approval_id = None
for output in response.output:
    if output.type == "mcp_approval_request":
        mcp_approval_id = output.id
        break

# Step 2: Send followup with approval for the MCP call
response_with_mcp_call = client.responses.create(
    model="openai/gpt-4.1",
    tools=MCP_TOOLS,
    input=[
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": mcp_approval_id
        }
    ],
    previous_response_id=response.id,
)

print(response_with_mcp_call)
```

</TabItem>
</Tabs>


## Verbosity Parameter

The `verbosity` parameter is supported for the `responses` API.

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python showLineNumbers title="Verbosity Parameter"
from litellm import responses

question = "Write a poem about a boy and his first pet dog."

for verbosity in ["low", "medium", "high"]:
    response = responses(
        model="gpt-5-mini",
        input=question,
        text={"verbosity": verbosity}
    )

    print(response)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```python
from openai import OpenAI
import pandas as pd
from IPython.display import display

client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

question = "Write a poem about a boy and his first pet dog."

data = []

for verbosity in ["low", "medium", "high"]:
    response = client.responses.create(
        model="gpt-5-mini",
        input=question,
        text={"verbosity": verbosity}
    )

    # Extract text
    output_text = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content in item.content:
                if hasattr(content, "text"):
                    output_text += content.text

    usage = response.usage
    data.append({
        "Verbosity": verbosity,
        "Sample Output": output_text,
        "Output Tokens": usage.output_tokens
    })

# Create DataFrame
df = pd.DataFrame(data)

# Display nicely with centered headers
pd.set_option('display.max_colwidth', None)
styled_df = df.style.set_table_styles(
    [
        {'selector': 'th', 'props': [('text-align', 'center')]},  # Center column headers
        {'selector': 'td', 'props': [('text-align', 'left')]}     # Left-align table cells
    ]
)

display(styled_df)

```


</TabItem>
</Tabs>

## Free-form Function Calling

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">


```python showLineNumbers title="Free-form Function Calling"
import litellm

response = litellm.responses(
    response = client.responses.create(
    model="gpt-5-mini",
    input="Please use the code_exec tool to calculate the area of a circle with radius equal to the number of 'r's in strawberry",
    text={"format": {"type": "text"}},
    tools=[
        {
            "type": "custom",
            "name": "code_exec",
            "description": "Executes arbitrary python code",
        }
    ]
)
print(response.output)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```python showLineNumbers title="Free-form Function Calling"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

response = client.responses.create(
    model="gpt-5-mini",
    input="Please use the code_exec tool to calculate the area of a circle with radius equal to the number of 'r's in strawberry",
    text={"format": {"type": "text"}},
    tools=[
        {
            "type": "custom",
            "name": "code_exec",
            "description": "Executes arbitrary python code",
        }
    ]
)
print(response.output)
```


</TabItem>
</Tabs>

## Context-Free Grammar 

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python showLineNumbers title="Context-Free Grammar"
import litellm

import textwrap

# ----------------- grammars for MS SQL dialect -----------------
mssql_grammar = textwrap.dedent(r"""
            // ---------- Punctuation & operators ----------
            SP: " "
            COMMA: ","
            GT: ">"
            EQ: "="
            SEMI: ";"

            // ---------- Start ----------
            start: "SELECT" SP "TOP" SP NUMBER SP select_list SP "FROM" SP table SP "WHERE" SP amount_filter SP "AND" SP date_filter SP "ORDER" SP "BY" SP sort_cols SEMI

            // ---------- Projections ----------
            select_list: column (COMMA SP column)*
            column: IDENTIFIER

            // ---------- Tables ----------
            table: IDENTIFIER

            // ---------- Filters ----------
            amount_filter: "total_amount" SP GT SP NUMBER
            date_filter: "order_date" SP GT SP DATE

            // ---------- Sorting ----------
            sort_cols: "order_date" SP "DESC"

            // ---------- Terminals ----------
            IDENTIFIER: /[A-Za-z_][A-Za-z0-9_]*/
            NUMBER: /[0-9]+/
            DATE: /'[0-9]{4}-[0-9]{2}-[0-9]{2}'/
    """)

sql_prompt_mssql = (
    "Call the mssql_grammar to generate a query for Microsoft SQL Server that retrieve the "
    "five most recent orders per customer, showing customer_id, order_id, order_date, and total_amount, "
    "where total_amount > 500 and order_date is after '2025-01-01'. "
)


response = litellm.responses(
    model="gpt-5",
    input=sql_prompt_mssql,
    text={"format": {"type": "text"}},
    tools=[
        {
            "type": "custom",
            "name": "mssql_grammar",
            "description": "Executes read-only Microsoft SQL Server queries limited to SELECT statements with TOP and basic WHERE/ORDER BY. YOU MUST REASON HEAVILY ABOUT THE QUERY AND MAKE SURE IT OBEYS THE GRAMMAR.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": mssql_grammar
            }
        },
    ],
    parallel_tool_calls=False
)

print("--- MS SQL Query ---")
print(response_mssql.output[1].input)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```python showLineNumbers title="Context-Free Grammar"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

import textwrap

# ----------------- grammars for MS SQL dialect -----------------
mssql_grammar = textwrap.dedent(r"""
            // ---------- Punctuation & operators ----------
            SP: " "
            COMMA: ","
            GT: ">"
            EQ: "="
            SEMI: ";"

            // ---------- Start ----------
            start: "SELECT" SP "TOP" SP NUMBER SP select_list SP "FROM" SP table SP "WHERE" SP amount_filter SP "AND" SP date_filter SP "ORDER" SP "BY" SP sort_cols SEMI

            // ---------- Projections ----------
            select_list: column (COMMA SP column)*
            column: IDENTIFIER

            // ---------- Tables ----------
            table: IDENTIFIER

            // ---------- Filters ----------
            amount_filter: "total_amount" SP GT SP NUMBER
            date_filter: "order_date" SP GT SP DATE

            // ---------- Sorting ----------
            sort_cols: "order_date" SP "DESC"

            // ---------- Terminals ----------
            IDENTIFIER: /[A-Za-z_][A-Za-z0-9_]*/
            NUMBER: /[0-9]+/
            DATE: /'[0-9]{4}-[0-9]{2}-[0-9]{2}'/
    """)

sql_prompt_mssql = (
    "Call the mssql_grammar to generate a query for Microsoft SQL Server that retrieve the "
    "five most recent orders per customer, showing customer_id, order_id, order_date, and total_amount, "
    "where total_amount > 500 and order_date is after '2025-01-01'. "
)


response = client.responses.create(
    model="gpt-5",
    input=sql_prompt_mssql,
    text={"format": {"type": "text"}},
    tools=[
        {
            "type": "custom",
            "name": "mssql_grammar",
            "description": "Executes read-only Microsoft SQL Server queries limited to SELECT statements with TOP and basic WHERE/ORDER BY. YOU MUST REASON HEAVILY ABOUT THE QUERY AND MAKE SURE IT OBEYS THE GRAMMAR.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": mssql_grammar
            }
        },
    ],
    parallel_tool_calls=False
)

print("--- MS SQL Query ---")
print(response_mssql.output[1].input)
```

</TabItem>
</Tabs>

## Minimal Reasoning

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">


```python showLineNumbers title="Minimal Reasoning"
import litellm

response = litellm.responses(
    model="gpt-5",
    input= [{ 'role': 'developer', 'content': prompt }, 
            { 'role': 'user', 'content': 'The food that the restaurant was great! I recommend it to everyone.' }],
    reasoning = {
        "effort": "minimal"
    },
)

print(response)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```python showLineNumbers title="Minimal Reasoning"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)


prompt = "Classify sentiment of the review as positive|neutral|negative. Return one word only." 


response = client.responses.create(
    model="gpt-5",
    input= [{ 'role': 'developer', 'content': prompt }, 
            { 'role': 'user', 'content': 'The food that the restaurant was great! I recommend it to everyone.' }],
    reasoning = {
        "effort": "minimal"
    },
)

# Extract model's text output
output_text = ""
for item in response.output:
    if hasattr(item, "content"):
        for content in item.content:
            if hasattr(content, "text"):
                output_text += content.text

# Token usage details
usage = response.usage

print("--------------------------------")
print("Output:")
print(output_text)



```


</TabItem>
</Tabs>
