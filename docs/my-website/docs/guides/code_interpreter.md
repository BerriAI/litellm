import Image from '@theme/IdealImage';

# Code Interpreter

Use OpenAI's Code Interpreter tool to execute Python code in a secure, sandboxed environment.

| Feature | Supported |
|---------|-----------|
| LiteLLM Python SDK | ✅ |
| LiteLLM AI Gateway | ✅ |
| Supported Providers | `openai` |

## LiteLLM AI Gateway

### API (OpenAI SDK)

Use the OpenAI SDK pointed at your LiteLLM Gateway:

```python showLineNumbers title="code_interpreter_gateway.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",  # Your LiteLLM API key
    base_url="http://localhost:4000"
)

response = client.responses.create(
    model="openai/gpt-4o",
    tools=[{"type": "code_interpreter"}],
    input="Calculate the first 20 fibonacci numbers and plot them"
)

print(response)
```

#### Streaming

```python showLineNumbers title="code_interpreter_streaming.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

stream = client.responses.create(
    model="openai/gpt-4o",
    tools=[{"type": "code_interpreter"}],
    input="Generate sample sales data CSV and create a visualization",
    stream=True
)

for event in stream:
    print(event)
```

#### Get Generated File Content

```python showLineNumbers title="get_file_content_gateway.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# 1. Run code interpreter
response = client.responses.create(
    model="openai/gpt-4o",
    tools=[{"type": "code_interpreter"}],
    input="Create a scatter plot and save as PNG"
)

# 2. Get container_id from response
container_id = response.output[0].container_id

# 3. List files
files = client.containers.files.list(container_id=container_id)

# 4. Download file content
for file in files.data:
    content = client.containers.files.content(
        container_id=container_id,
        file_id=file.id
    )
    
    with open(file.filename, "wb") as f:
        f.write(content.read())
    print(f"Downloaded: {file.filename}")
```

### AI Gateway UI

The LiteLLM Admin UI includes built-in Code Interpreter support.

<Image img={require('../../img/code_interp.png')} />

**Steps:**

1. Go to **Playground** in the LiteLLM UI
2. Select an **OpenAI model** (e.g., `openai/gpt-4o`)
3. Select `/v1/responses` as the endpoint under **Endpoint Type**
4. Toggle **Code Interpreter** in the left panel
5. Send a prompt requesting code execution or file generation

The UI will display:
- Executed Python code (collapsible)
- Generated images inline
- Download links for files (CSVs, etc.)

## LiteLLM Python SDK

### Run Code Interpreter

```python showLineNumbers title="code_interpreter.py"
import litellm

response = litellm.responses(
    model="openai/gpt-4o",
    input="Generate a bar chart of quarterly sales and save as PNG",
    tools=[{"type": "code_interpreter"}]
)

print(response)
```

### Get Generated File Content

After Code Interpreter runs, retrieve the generated files:

```python showLineNumbers title="get_file_content.py"
import litellm

# 1. Run code interpreter
response = litellm.responses(
    model="openai/gpt-4o",
    input="Create a pie chart of market share and save as PNG",
    tools=[{"type": "code_interpreter"}]
)

# 2. Extract container_id from response
container_id = response.output[0].container_id  # e.g. "cntr_abc123..."

# 3. List files in container
files = litellm.list_container_files(
    container_id=container_id,
    custom_llm_provider="openai"
)

# 4. Download each file
for file in files.data:
    content = litellm.retrieve_container_file_content(
        container_id=container_id,
        file_id=file.id,
        custom_llm_provider="openai"
    )
    
    with open(file.filename, "wb") as f:
        f.write(content)
    print(f"Downloaded: {file.filename}")
```


## Related

- [Containers API](/docs/containers) - Manage containers
- [Container Files API](/docs/container_files) - Manage files within containers
- [OpenAI Code Interpreter Docs](https://platform.openai.com/docs/guides/tools-code-interpreter) - Official OpenAI documentation
