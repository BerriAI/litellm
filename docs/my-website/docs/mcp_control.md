import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# MCP Permission Management

Control which MCP servers and tools can be accessed by specific keys, teams, or organizations in LiteLLM. When a client attempts to list or call tools, LiteLLM enforces access controls based on configured permissions.

## Overview

LiteLLM provides fine-grained permission management for MCP servers, allowing you to:

- **Restrict MCP access by entity**: Control which keys, teams, or organizations can access specific MCP servers
- **Tool-level filtering**: Automatically filter available tools based on entity permissions
- **Centralized control**: Manage all MCP permissions from the LiteLLM Admin UI or API

This ensures that only authorized entities can discover and use MCP tools, providing an additional security layer for your MCP infrastructure.

:::info Related Documentation
- [MCP Overview](./mcp.md) - Learn about MCP in LiteLLM
- [MCP Cost Tracking](./mcp_cost.md) - Track costs for MCP tool calls
- [MCP Guardrails](./mcp_guardrail.md) - Apply security guardrails to MCP calls
- [Using MCP](./mcp_usage.md) - How to use MCP with LiteLLM
:::

## How It Works

LiteLLM supports managing permissions for MCP Servers by Keys, Teams, Organizations (entities) on LiteLLM. When a MCP client attempts to list tools, LiteLLM will only return the tools the entity has permissions to access.

When Creating a Key, Team, or Organization, you can select the allowed MCP Servers that the entity has access to.

<Image 
  img={require('../img/mcp_key.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>


## Set Allowed Tools for a Key, Team, or Organization

Control which tools different teams can access from the same MCP server. For example, give your Engineering team access to `list_repositories`, `create_issue`, and `search_code`, while Sales only gets `search_code` and `close_issue`.


This video shows how to set allowed tools for a Key, Team, or Organization.

<iframe width="840" height="500" src="https://www.loom.com/embed/7464d444c3324078892367272fe50745" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
