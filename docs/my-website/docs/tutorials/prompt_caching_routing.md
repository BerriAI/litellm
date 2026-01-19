# Claude Code - Prompt Caching Based Routing

Maximize cache hit rates by routing requests with the same cacheable content to the same deployment. This reduces costs by up to 90% and improves latency for Claude Code usage.

## How It Works

When you have multiple deployments of the same model, LiteLLM's **Prompt Caching Based Routing** ensures that requests with identical cacheable prompts are routed to the same deployment. This maximizes cache hit rates.

**Without Prompt Caching Routing:**
- Request 1 with cacheable content → Deployment A (cache write)
- Request 2 with same content → Deployment B (cache miss, full cost)
- Request 3 with same content → Deployment C (cache miss, full cost)

**With Prompt Caching Routing:**
- Request 1 with cacheable content → Deployment A (cache write)
- Request 2 with same content → Deployment A (cache hit, 90% cost reduction)
- Request 3 with same content → Deployment A (cache hit, 90% cost reduction)

## Prerequisites

- LiteLLM Proxy Server
- Multiple Anthropic API keys (for multiple deployments)
- Claude Code installed

## Quick Start

### 1. Setup config.yaml

Create a configuration file with multiple Claude deployments:

```yaml
model_list:
  - model_name: claude-code
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY_1
  
  - model_name: claude-code
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY_2
  
  - model_name: claude-code
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY_3

# Enable prompt caching based routing
router_settings:
  enable_pre_call_checks: true

# Add the prompt caching deployment check
callbacks:
  - prompt_caching_deployment_check
```

Set your environment variables:

```bash
export ANTHROPIC_API_KEY_1="sk-ant-..."
export ANTHROPIC_API_KEY_2="sk-ant-..."
export ANTHROPIC_API_KEY_3="sk-ant-..."
```

### 2. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Configure Claude Code

Point Claude Code to your LiteLLM proxy:

```bash
export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_AUTH_TOKEN="sk-1234"  # Your LiteLLM virtual key

# Start Claude Code
claude
```

### 4. Test Prompt Caching Routing

Send a query with cacheable content (>1024 tokens):

```bash
# In Claude Code
> Analyze the main.py file and suggest improvements

# This creates a cache on deployment A
```

Send another query with the same codebase context:

```bash
# In Claude Code  
> What are the key functions in main.py?

# This routes to deployment A and hits the cache!
```

### 5. Verify Cache Hits

Check the LiteLLM logs to confirm routing:

```bash
# Look for these log messages:
# First request:
litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: Adding model_id to cache: deployment-abc-123

# Second request:
litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: Found cached model_id: deployment-abc-123
litellm.router: Routing to deployment: deployment-abc-123
```

Or check the LiteLLM UI:

1. Navigate to `http://localhost:4000/ui`
2. Go to "Logs" page
3. Filter by model name: `claude-code`
4. Look for consecutive requests
5. Verify they used the same `model_id` (deployment)

## Configuration Options

### Supported Call Types

Prompt caching routing works with:
- `completion` - Standard OpenAI-compatible completions
- `acompletion` - Async completions
- `anthropic_messages` - Anthropic's native messages API (used by Claude Code)

### Cache Requirements

For a prompt to be cached:
1. **Minimum token count**: >1024 tokens (Anthropic requirement)
2. **Cache control directive**: Must include `cache_control` in the message
3. **Supported models**: Anthropic Claude models (3.5 Sonnet, 3 Opus, etc.)

### Optional: Auto-Inject Cache Control

You can configure LiteLLM to automatically add cache control directives:

```yaml
model_list:
  - model_name: claude-code
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20240620
      api_key: os.environ/ANTHROPIC_API_KEY_1
      cache_control_injection_points:
        - location: message
          role: system
```

This automatically marks system messages for caching. [Learn more about auto-inject prompt caching](./prompt_caching.md).

## Example: Claude Code Prompt Caching

Here's how Claude Code uses prompt caching with LiteLLM routing:

### First Request (Cache Write)

```json
{
  "model": "claude-code",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Here is the codebase context:\n\n[... 5000 lines of code ...]",
          "cache_control": {"type": "ephemeral"}
        },
        {
          "type": "text",
          "text": "Analyze the main.py file"
        }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "usage": {
    "input_tokens": 5234,
    "cache_creation_input_tokens": 5000,
    "cache_read_input_tokens": 0,
    "output_tokens": 150
  }
}
```

**LiteLLM Action:**
- Routes to deployment A (e.g., `model_id: deployment-abc-123`)
- Caches the model_id for this prompt content

### Second Request (Cache Hit)

```json
{
  "model": "claude-code",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Here is the codebase context:\n\n[... same 5000 lines of code ...]",
          "cache_control": {"type": "ephemeral"}
        },
        {
          "type": "text",
          "text": "What are the key functions?"
        }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "usage": {
    "input_tokens": 5234,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 5000,
    "output_tokens": 120
  }
}
```

**LiteLLM Action:**
- Detects same cacheable content
- Routes to deployment A (`model_id: deployment-abc-123`)
- Cache hit! 90% cost reduction on input tokens

## Performance Benefits

### Cost Reduction

Example savings with Claude Code:
- 10 queries per day with same codebase context
- 5000 tokens of cacheable content per query
- **Without caching**: 10 × 5000 = 50,000 input tokens
- **With caching**: 5000 (write) + 9 × 500 (reads) = 9,500 input tokens
- **Savings**: 81% reduction in input token costs

### Latency Reduction

Cache hits are typically 2-3x faster:
- **Cache write**: ~2-3 seconds
- **Cache hit**: ~0.8-1.2 seconds

## Troubleshooting

### Cache Misses Despite Same Content

**Problem**: Requests with the same content are not hitting the cache.

**Solutions**:
1. Verify `enable_pre_call_checks: true` in router_settings
2. Ensure `prompt_caching_deployment_check` is in callbacks
3. Check that prompts have >1024 tokens
4. Verify `cache_control` directive is present

### Routing to Different Deployments

**Problem**: Consecutive requests are going to different deployments.

**Solutions**:
1. Check LiteLLM logs for cache hits:
   ```bash
   grep "prompt_caching_deployment_check" litellm.log
   ```
2. Verify the prompt content is identical (whitespace matters!)
3. Ensure you're using the same model name

### Not Working with Claude Code

**Problem**: Claude Code requests are not being cached.

**Solutions**:
1. Verify Claude Code is using `anthropic_messages` call type
2. Check LiteLLM version (requires v1.80.12+)
3. Enable debug logging:
   ```yaml
   litellm_settings:
     set_verbose: true
   ```

## FAQ

### Q: Does this work with other providers besides Anthropic?

**A**: Currently, prompt caching routing is optimized for Anthropic Claude models. Other providers may be supported in the future.

### Q: How long are prompts cached?

**A**: Anthropic's ephemeral cache lasts for 5 minutes. After 5 minutes of inactivity, the cache expires.

### Q: Can I use this with streaming?

**A**: Yes! Prompt caching routing works with both streaming and non-streaming requests.

### Q: What happens if a deployment goes down?

**A**: LiteLLM will automatically fallback to another healthy deployment. The cache will be rebuilt on the new deployment.

### Q: How many deployments should I configure?

**A**: For optimal cache hit rates, use 2-3 deployments. More deployments may dilute cache effectiveness unless you have very high traffic.

## Related

- [Auto-Inject Prompt Caching](./prompt_caching.md)
- [Claude Code Quickstart](./claude_responses_api.md)
- [Claude Code Cost Tracking](./claude_code_customer_tracking.md)
- [Router Configuration](../routing.md)
