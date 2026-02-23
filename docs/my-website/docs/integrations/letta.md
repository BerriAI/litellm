import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Letta Integration

[Letta](https://github.com/letta-ai/letta) (formerly MemGPT) is a framework for building stateful LLM agents with persistent memory. This guide shows how to integrate both LiteLLM SDK and LiteLLM Proxy with Letta to leverage multiple LLM providers while building memory-enabled agents.

## What is Letta?

Letta allows you to build LLM agents that can:
- Maintain long-term memory across conversations
- Use function calling for tool interactions
- Handle large context windows efficiently
- Persist agent state and memory

## Prerequisites

```bash
pip install letta litellm
```

## Quick Start

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

### 1. Start LiteLLM Proxy

First, create a configuration file for your LiteLLM proxy:

```yaml
# config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-35-turbo
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: "2023-07-01-preview"
```

Start the proxy:

```bash
litellm --config config.yaml --port 4000
```

### 2. Configure Letta with LiteLLM Proxy

Configure Letta to use your LiteLLM proxy endpoint:

```python
import letta
from letta import create_client

# Configure Letta to use LiteLLM proxy
client = create_client()

# Configure the LLM endpoint
client.set_default_llm_config(
    model="gpt-4",  # This should match a model from your LiteLLM config
    model_endpoint_type="openai",
    model_endpoint="http://localhost:4000",  # Your LiteLLM proxy URL
    context_window=8192
)

# Configure embedding endpoint (optional)
client.set_default_embedding_config(
    embedding_endpoint_type="openai",
    embedding_endpoint="http://localhost:4000",
    embedding_model="text-embedding-ada-002"
)
```

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK">

### 1. Configure LiteLLM SDK

Set up your API keys and configure LiteLLM:

```python
import os
import litellm

# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Optional: Configure default settings
litellm.set_verbose = True  # For debugging
```

### 2. Create Custom LLM Wrapper for Letta

Create a custom LLM wrapper that uses LiteLLM SDK:

```python
import letta
from letta import create_client
from letta.llm_api.llm_api_base import LLMConfig
import litellm
from typing import List, Dict, Any

class LiteLLMWrapper:
    def __init__(self, model: str):
        self.model = model
    
    def chat_completions_create(self, messages: List[Dict], **kwargs):
        # Use LiteLLM SDK for completion
        response = litellm.completion(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response

# Configure Letta with custom LiteLLM wrapper
client = create_client()

# Set up LLM configuration using direct SDK integration
llm_config = LLMConfig(
    model="gpt-4",  # or "claude-3-sonnet", "azure/gpt-35-turbo", etc.
    model_endpoint_type="openai",
    context_window=8192
)

client.set_default_llm_config(llm_config)
```

</TabItem>
</Tabs>

### 3. Create and Use a Letta Agent

<Tabs>
<TabItem value="proxy" label="Using LiteLLM Proxy">

```python
import letta
from letta import create_client

# Create Letta client
client = create_client()

# Create a new agent
agent_state = client.create_agent(
    name="my-assistant",
    system="You are a helpful assistant with persistent memory.",
    llm_config=client.get_default_llm_config(),
    embedding_config=client.get_default_embedding_config()
)

# Send a message to the agent
response = client.user_message(
    agent_id=agent_state.id,
    message="Hi! My name is Alice and I love reading science fiction books."
)

print(f"Agent response: {response.messages[-1].text}")

# Send another message - the agent will remember previous context
response = client.user_message(
    agent_id=agent_state.id,
    message="What did I tell you about my interests?"
)

print(f"Agent response: {response.messages[-1].text}")
```

</TabItem>
<TabItem value="sdk" label="Using LiteLLM SDK">

```python
import letta
from letta import create_client
import litellm
import os

# Set up environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Create Letta client with LiteLLM integration
client = create_client()

# Create a new agent
agent_state = client.create_agent(
    name="my-assistant",
    system="You are a helpful assistant with persistent memory.",
    llm_config=client.get_default_llm_config(),
    embedding_config=client.get_default_embedding_config()
)

# Send a message to the agent
response = client.user_message(
    agent_id=agent_state.id,
    message="Hi! My name is Alice and I love reading science fiction books."
)

print(f"Agent response: {response.messages[-1].text}")

# Send another message - the agent will remember previous context
response = client.user_message(
    agent_id=agent_state.id,
    message="What did I tell you about my interests?"
)

print(f"Agent response: {response.messages[-1].text}")
```

</TabItem>
</Tabs>

## Advanced Configuration

### Using Different Models for Different Agents

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

```python
from letta import LLMConfig, EmbeddingConfig

# Create different LLM configurations pointing to your proxy
gpt4_config = LLMConfig(
    model="gpt-4",
    model_endpoint_type="openai",
    model_endpoint="http://localhost:4000",
    context_window=8192
)

claude_config = LLMConfig(
    model="claude-3-sonnet",
    model_endpoint_type="openai",  # Using OpenAI-compatible endpoint
    model_endpoint="http://localhost:4000",
    context_window=200000
)

# Create agents with different configurations
research_agent = client.create_agent(
    name="research-agent",
    system="You are a research assistant specialized in analysis.",
    llm_config=claude_config  # Use Claude for research tasks
)

creative_agent = client.create_agent(
    name="creative-agent", 
    system="You are a creative writing assistant.",
    llm_config=gpt4_config  # Use GPT-4 for creative tasks
)
```

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK">

```python
import os
import litellm
from letta import LLMConfig, EmbeddingConfig

# Set up API keys for different providers
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# Create different LLM configurations for direct SDK usage
gpt4_config = LLMConfig(
    model="openai/gpt-4",  # Using LiteLLM model format
    model_endpoint_type="openai",
    context_window=8192
)

claude_config = LLMConfig(
    model="anthropic/claude-3-sonnet-20240229",  # Using LiteLLM model format
    model_endpoint_type="openai",
    context_window=200000
)

# Create agents with different configurations
research_agent = client.create_agent(
    name="research-agent",
    system="You are a research assistant specialized in analysis.",
    llm_config=claude_config  # Use Claude for research tasks
)

creative_agent = client.create_agent(
    name="creative-agent", 
    system="You are a creative writing assistant.",
    llm_config=gpt4_config  # Use GPT-4 for creative tasks
)
```

</TabItem>
</Tabs>

### Function Calling with Tools

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy">

```python
# Define custom tools for your agent
def search_web(query: str) -> str:
    """Search the web for information"""
    # Your web search implementation
    return f"Search results for: {query}"

def save_note(content: str) -> str:
    """Save a note to persistent storage"""
    # Your note saving implementation
    return f"Note saved: {content}"

# Create agent with tools (using proxy endpoint)
agent_state = client.create_agent(
    name="research-assistant",
    system="You are a research assistant that can search the web and save notes.",
    llm_config=client.get_default_llm_config(),
    embedding_config=client.get_default_embedding_config(),
    tools=[search_web, save_note]
)

# The agent can now use these tools
response = client.user_message(
    agent_id=agent_state.id,
    message="Search for recent developments in AI and save important findings."
)
```

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK">

```python
import litellm
import os

# Set up API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Define custom tools for your agent
def search_web(query: str) -> str:
    """Search the web for information"""
    # Your web search implementation
    return f"Search results for: {query}"

def save_note(content: str) -> str:
    """Save a note to persistent storage"""
    # Your note saving implementation
    return f"Note saved: {content}"

# Create agent with tools (using LiteLLM SDK directly)
agent_state = client.create_agent(
    name="research-assistant",
    system="You are a research assistant that can search the web and save notes.",
    llm_config=LLMConfig(
        model="openai/gpt-4",  # Direct model specification
        model_endpoint_type="openai",
        context_window=8192
    ),
    embedding_config=client.get_default_embedding_config(),
    tools=[search_web, save_note]
)

# The agent can now use these tools
response = client.user_message(
    agent_id=agent_state.id,
    message="Search for recent developments in AI and save important findings."
)
```

</TabItem>
</Tabs>

## Authentication

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy Authentication">

If your LiteLLM proxy requires authentication:

```python
import os
from letta import LLMConfig

# Set up authenticated configuration
llm_config = LLMConfig(
    model="gpt-4",
    model_endpoint_type="openai",
    model_endpoint="http://localhost:4000",
    model_wrapper="openai",
    context_window=8192
)

# If using API keys with your proxy
os.environ["OPENAI_API_KEY"] = "your-litellm-proxy-api-key"

client = create_client()
client.set_default_llm_config(llm_config)
```

For proxy with authentication enabled:

```yaml
# config.yaml with auth
general_settings:
  master_key: "your-master-key"

model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
```

```python
# Configure Letta with authenticated proxy
llm_config = LLMConfig(
    model="gpt-4",
    model_endpoint_type="openai",
    model_endpoint="http://localhost:4000",
    context_window=8192,
    api_key="your-master-key"  # Proxy master key
)
```

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK Authentication">

With LiteLLM SDK, set up your provider API keys directly:

```python
import os
import litellm

# Set up API keys for different providers
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key" 
os.environ["AZURE_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_API_BASE"] = "https://your-resource.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"

# Optional: Configure default settings
litellm.api_key = os.environ.get("OPENAI_API_KEY")  # Default key
litellm.set_verbose = True  # For debugging

# Use in Letta configuration
from letta import LLMConfig

llm_config = LLMConfig(
    model="openai/gpt-4",  # Will use OPENAI_API_KEY automatically
    model_endpoint_type="openai",
    context_window=8192
)

# Or for Azure
azure_config = LLMConfig(
    model="azure/gpt-35-turbo", 
    model_endpoint_type="openai",
    context_window=4096
)
```

</TabItem>
</Tabs>

## Load Balancing and Fallbacks

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy Features">

LiteLLM proxy's load balancing and fallback features work seamlessly with Letta:

```yaml
# config.yaml with fallbacks
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
    tpm: 40000
    rpm: 500

  - model_name: gpt-4  # Same model name for fallback
    litellm_params:
      model: azure/gpt-4
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: "2023-07-01-preview"
    tpm: 80000
    rpm: 800

router_settings:
  routing_strategy: "usage-based-routing"
  fallbacks: [{"gpt-4": ["azure/gpt-4"]}]
```

The proxy handles all routing, load balancing, and fallbacks transparently for Letta.

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK Router">

With LiteLLM SDK, you can set up routing and fallbacks programmatically:

```python
import litellm
from litellm import Router

# Configure router with multiple models
router = Router(
    model_list=[
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": os.environ["OPENAI_API_KEY"]
            },
            "tpm": 40000,
            "rpm": 500
        },
        {
            "model_name": "gpt-4",  # Same name for fallback
            "litellm_params": {
                "model": "azure/gpt-4", 
                "api_key": os.environ["AZURE_API_KEY"],
                "api_base": os.environ["AZURE_API_BASE"],
                "api_version": "2023-07-01-preview"
            },
            "tpm": 80000,
            "rpm": 800
        }
    ],
    fallbacks=[{"gpt-4": ["azure/gpt-4"]}],
    routing_strategy="usage-based-routing"
)

# Create custom completion function for Letta
def custom_completion(messages, model="gpt-4", **kwargs):
    return router.completion(
        model=model,
        messages=messages,
        **kwargs
    )

# Use with Letta by monkey-patching or custom wrapper
litellm.completion = custom_completion
```

</TabItem>
</Tabs>

## Monitoring and Observability

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy Monitoring">

Enable logging to track your Letta agents' LLM usage through the proxy:

```yaml
# config.yaml with logging
model_list:
  # ... your models

litellm_settings:
  success_callback: ["langfuse"]  # or other observability tools
  
environment_variables:
  LANGFUSE_PUBLIC_KEY: "your-key"
  LANGFUSE_SECRET_KEY: "your-secret"
```

View metrics in the proxy dashboard:
```bash
# Start proxy with UI
litellm --config config.yaml --port 4000 --detailed_debug
```

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK Monitoring">

Set up observability directly in your SDK integration:

```python
import litellm
import os

# Configure observability callbacks
os.environ["LANGFUSE_PUBLIC_KEY"] = "your-key"
os.environ["LANGFUSE_SECRET_KEY"] = "your-secret"

# Set global callbacks
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# Optional: Set up custom logging
litellm.set_verbose = True

# Create custom completion wrapper with logging
def logged_completion(messages, model="gpt-4", **kwargs):
    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            **kwargs
        )
        # Custom logging logic here if needed
        return response
    except Exception as e:
        # Custom error handling
        print(f"LLM call failed: {e}")
        raise

# Use in Letta configuration
litellm.completion = logged_completion
```

</TabItem>
</Tabs>

## Example: Multi-Agent System

<Tabs>
<TabItem value="proxy" label="Using LiteLLM Proxy">

```python
import letta
from letta import create_client, LLMConfig

client = create_client()

# Create specialized agents using proxy endpoints
agents = {}

# Research agent using Claude for analysis
agents['researcher'] = client.create_agent(
    name="researcher",
    system="You are a research specialist. Analyze information thoroughly.",
    llm_config=LLMConfig(
        model="claude-3-sonnet",
        model_endpoint="http://localhost:4000",
        model_endpoint_type="openai"
    )
)

# Writer agent using GPT-4 for content creation
agents['writer'] = client.create_agent(
    name="writer",
    system="You are a content writer. Create engaging, well-structured content.",
    llm_config=LLMConfig(
        model="gpt-4",
        model_endpoint="http://localhost:4000", 
        model_endpoint_type="openai"
    )
)

# Coordinator workflow
def research_and_write_workflow(topic: str):
    # Research phase
    research_response = client.user_message(
        agent_id=agents['researcher'].id,
        message=f"Research the topic: {topic}. Provide key insights and data."
    )
    
    research_results = research_response.messages[-1].text
    
    # Writing phase
    write_response = client.user_message(
        agent_id=agents['writer'].id,
        message=f"Based on this research: {research_results}\n\nWrite an article about {topic}."
    )
    
    return write_response.messages[-1].text

# Execute workflow
article = research_and_write_workflow("The future of AI in healthcare")
print(article)
```

</TabItem>
<TabItem value="sdk" label="Using LiteLLM SDK">

```python
import letta
from letta import create_client, LLMConfig
import litellm
import os

# Set up environment
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

client = create_client()

# Create specialized agents using direct SDK models
agents = {}

# Research agent using Claude for analysis
agents['researcher'] = client.create_agent(
    name="researcher",
    system="You are a research specialist. Analyze information thoroughly.",
    llm_config=LLMConfig(
        model="anthropic/claude-3-sonnet-20240229",
        model_endpoint_type="openai"
    )
)

# Writer agent using GPT-4 for content creation
agents['writer'] = client.create_agent(
    name="writer",
    system="You are a content writer. Create engaging, well-structured content.",
    llm_config=LLMConfig(
        model="openai/gpt-4",
        model_endpoint_type="openai"
    )
)

# Cost-conscious agent using GPT-3.5
agents['reviewer'] = client.create_agent(
    name="reviewer",
    system="You are an editor. Review and improve content quality.",
    llm_config=LLMConfig(
        model="openai/gpt-3.5-turbo",
        model_endpoint_type="openai"
    )
)

# Enhanced workflow with multiple agents
def enhanced_workflow(topic: str):
    # Research phase
    research_response = client.user_message(
        agent_id=agents['researcher'].id,
        message=f"Research the topic: {topic}. Provide key insights and data."
    )
    
    research_results = research_response.messages[-1].text
    
    # Writing phase
    write_response = client.user_message(
        agent_id=agents['writer'].id,
        message=f"Based on this research: {research_results}\n\nWrite an article about {topic}."
    )
    
    draft_article = write_response.messages[-1].text
    
    # Review phase
    review_response = client.user_message(
        agent_id=agents['reviewer'].id,
        message=f"Please review and improve this article:\n\n{draft_article}"
    )
    
    return review_response.messages[-1].text

# Execute enhanced workflow
article = enhanced_workflow("The future of AI in healthcare")
print(article)
```

</TabItem>
</Tabs>

## Best Practices

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy Best Practices">

1. **Model Selection**: Use appropriate models for different tasks:
   - Claude for analysis and reasoning
   - GPT-4 for creative tasks
   - GPT-3.5-turbo for simple interactions

2. **Proxy Configuration**:
   - Set appropriate rate limits and timeouts
   - Use fallbacks for reliability
   - Enable authentication for production

3. **Memory Management**: Letta handles memory automatically, but monitor usage with large contexts

4. **Cost Optimization**: 
   - Use the proxy's budgeting features to control costs
   - Set up rate limiting per user/team
   - Monitor token usage through proxy dashboard

5. **Monitoring**: Enable observability to track agent performance and token usage

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK Best Practices">

1. **Model Selection**: Choose models based on task requirements:
   - Use `openai/gpt-4` for complex reasoning
   - Use `anthropic/claude-3-sonnet-20240229` for analysis
   - Use `openai/gpt-3.5-turbo` for cost-effective simple tasks

2. **Error Handling**: Implement robust error handling with retries:
   ```python
   import litellm
   from litellm import completion
   
   # Set up retry logic
   litellm.num_retries = 3
   litellm.request_timeout = 60
   
   # Custom error handling
   def safe_completion(**kwargs):
       try:
           return completion(**kwargs)
       except Exception as e:
           print(f"LLM call failed: {e}")
           # Implement fallback logic
           return completion(model="openai/gpt-3.5-turbo", **kwargs)
   ```

3. **Cost Management**:
   - Use cheaper models for non-critical tasks
   - Implement token counting and budgets
   - Cache responses when appropriate

4. **Performance**:
   - Use async operations for concurrent requests
   - Implement connection pooling
   - Monitor response times

5. **Security**:
   - Store API keys securely (environment variables)
   - Rotate keys regularly
   - Implement rate limiting

</TabItem>
</Tabs>

## Troubleshooting

<Tabs>
<TabItem value="proxy" label="LiteLLM Proxy Issues">

### Connection Issues
```bash
# Test your LiteLLM proxy
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Configuration Debugging
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test Letta configuration
client = create_client()
print(client.get_default_llm_config())
```

### Common Proxy Issues
- **Port conflicts**: Make sure port 4000 isn't in use
- **Model not found**: Verify model names match your config.yaml
- **Authentication errors**: Check master key configuration
- **Rate limiting**: Monitor proxy logs for rate limit hits

</TabItem>
<TabItem value="sdk" label="LiteLLM SDK Issues">

### API Key Issues
```python
import os
import litellm

# Check if API keys are set
print("OpenAI Key:", os.environ.get("OPENAI_API_KEY", "Not set"))
print("Anthropic Key:", os.environ.get("ANTHROPIC_API_KEY", "Not set"))

# Test direct LiteLLM call
try:
    response = litellm.completion(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print("LiteLLM working:", response.choices[0].message.content)
except Exception as e:
    print("LiteLLM error:", e)
```

### Configuration Debugging
```python
# Enable verbose logging
litellm.set_verbose = True

# Test model availability
models = ["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"]
for model in models:
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        print(f"✓ {model} working")
    except Exception as e:
        print(f"✗ {model} failed: {e}")
```

### Common SDK Issues
- **Import errors**: Ensure `pip install litellm letta` is run
- **Model format**: Use `provider/model` format (e.g., `openai/gpt-4`)
- **API key format**: Different providers have different key formats
- **Rate limits**: Implement exponential backoff for retries

</TabItem>
</Tabs>

## Resources

- [Letta Documentation](https://docs.letta.ai/)
- [LiteLLM Proxy Documentation](../proxy/quick_start.md)
- [LiteLLM SDK Documentation](../completion/input.md)
- [Function Calling Guide](../completion/function_call.md)
- [Observability Setup](../observability/langfuse_integration.md)
- [Router Configuration](../routing.md)