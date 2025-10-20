# AWS Bedrock Agent Configuration Guide for LiteLLM Proxy

## Overview

This guide explains how to configure AWS Bedrock agent-specific parameters in LiteLLM proxy, including enabling/disabling tracing for the InvokeAgent API. 

**Key Finding**: LiteLLM already supports passing agent parameters through the existing configuration system - no source code modifications are needed!

## Problem Statement

The AWS Bedrock InvokeAgent API supports a `enableTrace` parameter to control whether detailed tracing is enabled for debugging and monitoring. Users want to:

1. Control tracing behavior through LiteLLM proxy configuration
2. Have different tracing settings for development vs production
3. Pass other agent-specific parameters like `sessionID`, `memoryId`, etc.

## Solution

The current LiteLLM implementation in `litellm/llms/bedrock/chat/invoke_agent/transformation.py` uses `**optional_params` which allows configuration parameters to override default values.

```python
# Current implementation (lines 151-155)
return {
    "inputText": query,
    "enableTrace": True,  # Default value
    **optional_params,    # These override the defaults!
}
```

This means agent parameters can be configured in the `litellm_params` section of your proxy configuration.

## Configuration Examples

### Basic Configuration

```yaml
model_list:
  # Agent with tracing disabled (recommended for production)
  - model_name: "my-bedrock-agent"
    litellm_params:
      model: "bedrock/agent/YOUR_AGENT_ID/YOUR_ALIAS_ID"
      aws_access_key_id: "os.environ/AWS_ACCESS_KEY_ID"
      aws_secret_access_key: "os.environ/AWS_SECRET_ACCESS_KEY"
      aws_region_name: "us-west-2"
      enableTrace: false  # Controls tracing behavior
```

### Advanced Configuration

```yaml
model_list:
  # Development agent with full tracing and debugging
  - model_name: "bedrock-agent-debug"
    litellm_params:
      model: "bedrock/agent/DEV123/DEBUGALIAS"
      aws_access_key_id: "os.environ/AWS_ACCESS_KEY_ID"
      aws_secret_access_key: "os.environ/AWS_SECRET_ACCESS_KEY"
      aws_region_name: "us-west-2"
      # Agent-specific parameters
      enableTrace: true
      sessionID: "debug-session-123"
      memoryId: "debug-memory-456"
      
  # Production agent optimized for performance
  - model_name: "bedrock-agent-prod"
    litellm_params:
      model: "bedrock/agent/PROD456/PRODALIAS"
      aws_access_key_id: "os.environ/AWS_ACCESS_KEY_ID"
      aws_secret_access_key: "os.environ/AWS_SECRET_ACCESS_KEY"
      aws_region_name: "us-east-1"
      # Agent-specific parameters
      enableTrace: false  # Disabled for better performance
```

## Supported Agent Parameters

Based on the AWS Bedrock InvokeAgent API documentation and current LiteLLM implementation, these parameters are supported:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `enableTrace` | boolean | Enable/disable detailed tracing | `true` or `false` |
| `sessionID` | string | Custom session identifier | `"custom-session-123"` |
| `memoryId` | string | Memory ID for persistent conversations | `"memory-456"` |
| `endSession` | boolean | Whether to end the current session | `true` or `false` |

Additional parameters can be passed through and will be included in the InvokeAgent request payload.

## Request-Time Parameter Override

You can also override configuration parameters at request time:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-master-key" \
  -d '{
    "model": "bedrock-agent-prod",
    "messages": [{"role": "user", "content": "Hello"}],
    "enableTrace": true  // Override config setting
  }'
```

## Load Balancing with Different Tracing Settings

You can configure load balancing between agents with different tracing settings:

```yaml
router_settings:
  model_group_alias:
    "bedrock-prod": [
      {"model_name": "bedrock-agent-prod-1"},  # enableTrace: false
      {"model_name": "bedrock-agent-prod-2"}   # enableTrace: false
    ]
    "bedrock-debug": [
      {"model_name": "bedrock-agent-debug-1"}, # enableTrace: true
      {"model_name": "bedrock-agent-debug-2"}  # enableTrace: true
    ]
```

## Testing Your Configuration

Use the provided test script to verify parameter behavior:

```bash
python test_bedrock_agent_tracing.py
```

This will confirm that:
- `enableTrace: false` properly disables tracing
- `enableTrace: true` enables tracing
- Additional parameters are passed through correctly
- Parameter precedence works as expected

## Tracing Best Practices

### Production Deployments
```yaml
enableTrace: false  # Disable for better performance and lower costs
```

### Development/Testing
```yaml
enableTrace: true   # Enable for debugging and monitoring
```

### Debugging Issues
```yaml
enableTrace: true
# Also enable LiteLLM verbose logging
general_settings:
  set_verbose: true
```

## Cost and Performance Considerations

- **Tracing Enabled**: Higher latency, more detailed logs, additional AWS charges for trace data
- **Tracing Disabled**: Better performance, lower costs, minimal logging

Choose based on your use case:
- Production workloads: `enableTrace: false`
- Development/debugging: `enableTrace: true`
- Monitoring/analytics: `enableTrace: true` with log aggregation

## Troubleshooting

### Common Issues

1. **Parameters not taking effect**: 
   - Verify the parameter is in `litellm_params` section
   - Check for typos in parameter names
   - Enable verbose logging to see request payloads

2. **Tracing still enabled despite `enableTrace: false`**:
   - Confirm the parameter is properly formatted as a boolean
   - Check if request-time parameters are overriding config

3. **Agent not found errors**:
   - Verify your agent ID and alias ID are correct
   - Ensure the agent is deployed in the specified AWS region
   - Check AWS credentials and permissions

### Debug Configuration

```yaml
model_list:
  - model_name: "debug-agent"
    litellm_params:
      model: "bedrock/agent/YOUR_AGENT/YOUR_ALIAS"
      # ... other config ...
      enableTrace: true

general_settings:
  set_verbose: true  # Enable detailed LiteLLM logging
  master_key: "os.environ/LITELLM_MASTER_KEY"
```

## Example Files

This guide includes several example configuration files:

- `bedrock_agent_quick_start.yaml` - Minimal configuration to get started
- `bedrock_agent_tracing_examples.yaml` - Comprehensive examples with different scenarios
- `bedrock_agent_load_balancer.yaml` - Load balancing configuration with different tracing settings
- `test_bedrock_agent_tracing.py` - Test script to verify parameter behavior

## Summary

✅ **No source code modifications needed** - LiteLLM already supports agent parameter configuration  
✅ **Configure `enableTrace` through `litellm_params`** - Works with existing configuration system  
✅ **Request-time overrides supported** - Parameters can be overridden per request  
✅ **Load balancing compatible** - Different agents can have different tracing settings  
✅ **Production ready** - Tested and validated approach  

The key insight is that LiteLLM's existing `**optional_params` mechanism in the Bedrock agent transformation already supports passing custom parameters, including `enableTrace`, without any source code changes.