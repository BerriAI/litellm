
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# MCP Cost Tracking

LiteLLM provides two ways to track costs for MCP tool calls:

| Method | When to Use | What It Does |
|--------|-------------|--------------|
| **Config-based Cost Tracking** | Simple cost tracking with fixed costs per tool/server | Automatically tracks costs based on configuration |
| **Custom Post-MCP Hook** | Dynamic cost tracking with custom logic | Allows custom cost calculations and response modifications |

### Config-based Cost Tracking

Configure fixed costs for MCP servers directly in your config.yaml:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

mcp_servers:
  zapier_server:
    url: "https://actions.zapier.com/mcp/sk-xxxxx/sse"
    mcp_info:
      mcp_server_cost_info:
        # Default cost for all tools in this server
        default_cost_per_query: 0.01
        # Custom cost for specific tools
        tool_name_to_cost_per_query:
          send_email: 0.05
          create_document: 0.03
          
  expensive_api_server:
    url: "https://api.expensive-service.com/mcp"
    mcp_info:
      mcp_server_cost_info:
        default_cost_per_query: 1.50
```

### Custom Post-MCP Hook

Use this when you need dynamic cost calculation or want to modify the MCP response before it's returned to the user.

#### 1. Create a custom MCP hook file

```python title="custom_mcp_hook.py" showLineNumbers
from typing import Optional
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.mcp import MCPPostCallResponseObject


class CustomMCPCostTracker(CustomLogger):
    """
    Custom handler for MCP cost tracking and response modification
    """
    
    async def async_post_mcp_tool_call_hook(
        self, 
        kwargs, 
        response_obj: MCPPostCallResponseObject, 
        start_time, 
        end_time
    ) -> Optional[MCPPostCallResponseObject]:
        """
        Called after each MCP tool call. 
        Modify costs and response before returning to user.
        """
        
        # Extract tool information from kwargs
        tool_name = kwargs.get("name", "")
        server_name = kwargs.get("server_name", "")
        
        # Calculate custom cost based on your logic
        custom_cost = 42.00
        
        # Set the response cost
        response_obj.hidden_params.response_cost = custom_cost
        
  
      
        return response_obj
    

# Create instance for LiteLLM to use
custom_mcp_cost_tracker = CustomMCPCostTracker()
```

#### 2. Configure in config.yaml

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

# Add your custom MCP hook
callbacks:
  - custom_mcp_hook.custom_mcp_cost_tracker

mcp_servers:
  zapier_server:
    url: "https://actions.zapier.com/mcp/sk-xxxxx/sse"
```

#### 3. Start the proxy

```shell
$ litellm --config /path/to/config.yaml 
```

When MCP tools are called, your custom hook will:
1. Calculate costs based on your custom logic
2. Modify the response if needed
3. Track costs in LiteLLM's logging system

