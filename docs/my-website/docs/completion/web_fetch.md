import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Web Fetch

The web fetch tool allows LLMs to retrieve full content from specified web pages and PDF documents. This enables AI models to access real-time information from the internet and incorporate web content into their responses. 

## Web Fetch vs Web Search

**Web Fetch** retrieves the full content from specific web pages that you provide URLs for, while **Web Search** performs internet searches to find relevant information based on your queries.

| Feature | Web Fetch | Web Search |
|---------|-----------|------------|
| **Purpose** | Retrieve content from specific URLs | Search the internet for information |
| **Input** | You provide exact URLs to fetch | You provide search queries/questions |
| **Output** | Full page content from specified URLs | Search results with relevant information |
| **Use Cases** | - Analyzing specific articles<br/>- Comparing content from known websites<br/>- Extracting data from particular pages | - Finding current news/events<br/>- Researching topics<br/>- Getting real-time information |


**Example Web Fetch**: "Fetch the content from https://example.com/pricing and summarize it"  
**Example Web Search**: "What are the latest AI developments this week?"

**Supported Providers:**
- Anthropic API (`anthropic/`)

**Supported Tool Types:**
- `web_fetch_20250910` - Web content retrieval tool with usage limits, domain filtering, and citation support


## Quick Start

### LiteLLM Python SDK

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Web fetch tool
tools = [
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 5,
    }
]

messages = [
    {
        "role": "user", 
        "content": "Please analyze the content at https://example.com/article and summarize the main points"
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

### LiteLLM Proxy

1. Define web fetch models on config.yaml

```yaml
model_list:
  - model_name: claude-3-5-sonnet-latest # Anthropic claude-3-5-sonnet-latest
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY
```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Test it using the OpenAI Python SDK

```python
import os 
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # your litellm proxy api key
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-latest",
    messages=[
        {
            "role": "user", 
            "content": "Please fetch and analyze the content from https://news.ycombinator.com and tell me about the top stories"
        }
    ],
    tools=[
        {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_uses": 5,
        }
    ]
)

print(response)
```

## Supported Models

Web fetch is available on the following Anthropic API models:

- `claude-opus-4-1-20250805` (Claude Opus 4.1)
- `claude-opus-4-20250514` (Claude Opus 4)
- `claude-sonnet-4-20250514` (Claude Sonnet 4)
- `claude-3-7-sonnet-20250219` (Claude Sonnet 3.7)
- `claude-3-5-sonnet-latest` (Claude Sonnet 3.5 v2 - deprecated)
- `claude-3-5-haiku-latest` (Claude Haiku 3.5)

:::note
The web fetch tool currently does not support websites dynamically rendered via JavaScript.
:::

## Usage Examples

### Basic Web Content Retrieval

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 3,
    }
]

messages = [
    {
        "role": "user",
        "content": "Fetch the latest news from https://techcrunch.com and summarize the top 3 articles"
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

### Research and Analysis

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch", 
        "max_uses": 10,
    }
]

messages = [
    {
        "role": "user",
        "content": "Research the latest developments in AI by fetching content from multiple tech news websites and provide a comprehensive analysis"
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

### Content Comparison

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 5,
    }
]

messages = [
    {
        "role": "user",
        "content": "Compare the pricing information from https://openai.com/pricing and https://anthropic.com/pricing and create a comparison table"
    }
]

response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

## Advanced Usage with Multiple Tools

You can combine web fetch with other tools like computer use or text editor:

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

tools = [
    {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 5,
    },
    {
        "type": "text_editor_20250124", 
        "name": "str_replace_editor"
    }
]

messages = [
    {
        "role": "user",
        "content": "Fetch the latest AI research papers from arXiv, analyze them, and create a detailed report file with your findings"
    }
]
    
response = completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=messages,
    tools=tools,
)

print(response)
```

## Spec

### Web Fetch Tool (`web_fetch_20250910`)

The web fetch tool supports the following parameters:

```json
{
  "type": "web_fetch_20250910",
  "name": "web_fetch",

  // Optional: Limit the number of fetches per request
  "max_uses": 10,

  // Optional: Only fetch from these domains
  "allowed_domains": ["example.com", "docs.example.com"],

  // Optional: Never fetch from these domains
  "blocked_domains": ["private.example.com"],

  // Optional: Enable citations for fetched content
  "citations": {
    "enabled": true
  },

  // Optional: Maximum content length in tokens
  "max_content_tokens": 100000
}
```

