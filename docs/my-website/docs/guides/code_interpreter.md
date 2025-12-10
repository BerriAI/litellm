# Code Interpreter

Use OpenAI's Code Interpreter tool through LiteLLM to execute Python code in a secure, sandboxed environment.

| Feature | Support |
|---------|---------|
| Provider | OpenAI only |
| Endpoint | `/v1/responses` |
| File Generation | ✅ Charts, CSVs, images |
| Container Management | ✅ Via `/v1/containers` API |

:::tip
Code Interpreter automatically creates containers and generates files. Use the [Containers API](/docs/containers) to manage files after execution.
:::

## Using on LiteLLM UI

The LiteLLM Admin UI includes built-in Code Interpreter support.

**Steps:**
1. Go to **Playground** in the LiteLLM UI
2. Select an **OpenAI model** (e.g., `openai/gpt-4o`)
3. Toggle **Code Interpreter** in the left panel
4. Send a prompt requesting code execution or file generation

The UI will display:
- Executed Python code (collapsible)
- Generated images inline
- Download links for other files (CSVs, etc.)

## API Usage

### Basic Request

```bash showLineNumbers title="code_interpreter_request.sh"
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "openai/gpt-4o",
    "tools": [{"type": "code_interpreter"}],
    "input": "Generate a bar chart of sales by quarter and save as PNG"
  }'
```

### Python SDK

```python showLineNumbers title="code_interpreter.py"
import litellm

response = litellm.responses(
    model="openai/gpt-4o",
    input="Calculate the first 20 fibonacci numbers and plot them",
    tools=[{"type": "code_interpreter"}]
)

print(response)
```

### With OpenAI Client

```python showLineNumbers title="code_interpreter_openai_client.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

response = client.responses.create(
    model="openai/gpt-4o",
    tools=[{"type": "code_interpreter"}],
    input="Create a CSV with sample sales data"
)

print(response)
```

### Streaming

```python showLineNumbers title="code_interpreter_streaming.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

stream = client.responses.create(
    model="openai/gpt-4o",
    tools=[{"type": "code_interpreter"}],
    input="Generate a pie chart of market share",
    stream=True
)

for event in stream:
    print(event)
```

## Retrieving Generated Files

Code Interpreter stores generated files in containers. Use the [Containers API](/docs/containers) to retrieve them.

```python showLineNumbers title="retrieve_files.py"
from litellm import list_container_files, retrieve_container_file_content

# List files in container (container_id from response)
files = list_container_files(
    container_id="cntr_abc123...",
    custom_llm_provider="openai"
)

# Download file content
for file in files.data:
    content = retrieve_container_file_content(
        container_id="cntr_abc123...",
        file_id=file.id,
        custom_llm_provider="openai"
    )
    
    with open(file.filename, "wb") as f:
        f.write(content)
```

## Limitations

- **OpenAI only** - Code Interpreter is currently only available on OpenAI models
- **Container expiry** - Containers expire after 20 minutes of inactivity
- **File size limits** - Subject to OpenAI's file size restrictions

## Related

- [Containers API](/docs/containers) - Manage containers and files
- [Responses API](/docs/response_api) - Full responses endpoint documentation
- [OpenAI Code Interpreter Docs](https://platform.openai.com/docs/guides/tools-code-interpreter) - Official OpenAI documentation

