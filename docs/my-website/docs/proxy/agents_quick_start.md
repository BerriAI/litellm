# Agents Quick Start Guide

This is a condensed guide to get you started with the Agents feature in LiteLLM UI quickly.

## 🚀 5-Minute Setup

### Prerequisites
- LiteLLM Proxy server running
- Admin access to the UI
- An agent endpoint ready (or use a test endpoint)

## Step-by-Step Agent Creation

### 1. Navigate to Agents Tab
```
Dashboard → Agents (left sidebar)
```

### 2. Click "Add New Agent"
Location: Top-right corner of the Agents page

### 3. Fill Required Fields

#### Minimum Required Information:
```yaml
Agent Name: my-first-agent
Display Name: My First Agent
Description: A simple test agent
URL: http://localhost:9999/
```

#### Add at least one skill:
```yaml
Skill ID: greet
Skill Name: Greeting Skill
Description: Greets users
Tags: greeting, hello
Examples: hi, hello
```

### 4. Click "Create Agent"

Done! Your agent is now registered.

## UI Structure Overview

```
┌─────────────────────────────────────────────────────────┐
│                    LiteLLM Agents UI                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Agents                          [+ Add New Agent]       │
│  ─────────────────────────────────────────────────────  │
│                                                           │
│  List of A2A-spec agents that are available to be used   │
│  in your organization. Go to AI Hub, to make agents      │
│  public.                                                  │
│                                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Agent Name    │ Display Name  │ Skills │ Actions  │  │
│  ├───────────────────────────────────────────────────┤  │
│  │ support-agent │ Support Bot   │   3    │ Edit Del │  │
│  │ data-agent    │ Data Analyzer │   5    │ Edit Del │  │
│  │ code-agent    │ Code Helper   │   2    │ Edit Del │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Add Agent Modal Structure

```
┌────────────────────────────────────────────────────────┐
│  🤖 Add New Agent                                 [X]   │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Agent Type: [A2A Agent ▼]                             │
│                                                         │
│  ▼ Basic Information (Required)                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Agent Name: [_________________]                   │ │
│  │ Display Name: [_________________]                 │ │
│  │ Description: [_________________]                  │ │
│  │              [_________________]                  │ │
│  │ URL: [_________________]                          │ │
│  │ Version: [1.0.0]                                  │ │
│  │ Protocol Version: [1.0]                           │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
│  ▼ Skills (Required)                                   │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Skill 1:                                          │ │
│  │   ID: [_________________]                         │ │
│  │   Name: [_________________]                       │ │
│  │   Description: [_________]                        │ │
│  │   Tags: [_________________]                       │ │
│  │   Examples: [_____________]                       │ │
│  │                                  [Remove Skill]   │ │
│  └──────────────────────────────────────────────────┘ │
│  [+ Add Skill]                                         │
│                                                         │
│  ▶ Capabilities (Optional)                             │
│  ▶ Optional Settings                                   │
│  ▶ LiteLLM Parameters                                  │
│                                                         │
│                          [Cancel]  [Create Agent]      │
└────────────────────────────────────────────────────────┘
```

## Form Sections at a Glance

| Section | Required | Key Fields |
|---------|----------|------------|
| **Basic Information** | ✅ Yes | agent_name, name, description, url |
| **Skills** | ✅ Yes | At least one skill with id, name, description, tags |
| **Capabilities** | ❌ No | streaming, pushNotifications, stateTransitionHistory |
| **Optional Settings** | ❌ No | iconUrl, documentationUrl |
| **LiteLLM Parameters** | ❌ No | model, make_public |

## Example: Complete Agent Configuration

### Minimal Configuration
```json
{
  "agent_name": "simple-agent",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Simple Agent",
    "description": "A basic agent for testing",
    "url": "http://localhost:9999/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": false
    },
    "skills": [
      {
        "id": "hello",
        "name": "Hello Skill",
        "description": "Says hello",
        "tags": ["greeting"]
      }
    ]
  }
}
```

### Full Configuration
```json
{
  "agent_name": "advanced-support-agent",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Advanced Support Agent",
    "description": "Comprehensive customer support with multiple skills",
    "url": "https://support-agent.example.com/",
    "version": "2.1.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": true,
      "pushNotifications": true,
      "stateTransitionHistory": true
    },
    "skills": [
      {
        "id": "ticket_creation",
        "name": "Create Support Ticket",
        "description": "Creates a new support ticket with customer information",
        "tags": ["support", "ticket", "create"],
        "examples": ["create a ticket", "open new ticket", "file a complaint"]
      },
      {
        "id": "knowledge_search",
        "name": "Search Knowledge Base",
        "description": "Searches internal knowledge base for solutions",
        "tags": ["search", "knowledge", "documentation"],
        "examples": ["how to reset password", "search docs", "find solution"]
      },
      {
        "id": "escalate",
        "name": "Escalate to Human",
        "description": "Escalates complex issues to human support agents",
        "tags": ["escalate", "human", "transfer"],
        "examples": ["talk to a person", "escalate", "speak to agent"]
      }
    ],
    "iconUrl": "https://example.com/agent-icon.png",
    "documentationUrl": "https://docs.example.com/support-agent"
  },
  "litellm_params": {
    "model": "gpt-4",
    "make_public": false
  }
}
```

## Common Workflows

### Workflow 1: Register Internal Agent
```
1. Click "Add New Agent"
2. Set agent_name: "internal-helper"
3. Fill basic info and skills
4. Leave "Make Public" OFF
5. Create → Agent available to your team only
```

### Workflow 2: Share Agent Publicly
```
1. Create agent (or edit existing)
2. Toggle "Make Public" ON in LiteLLM Parameters
3. Create/Update agent
4. Agent appears in AI Hub for discovery
```

### Workflow 3: Update Agent Skills
```
1. Click on agent in list
2. Click "Edit" button
3. Scroll to Skills section
4. Add/remove/modify skills
5. Click "Update Agent"
```

### Workflow 4: Test Agent
```
1. Register agent
2. Go to Playground tab
3. Select your agent
4. Send test messages
5. Verify responses
```

## Field Reference Quick Lookup

### Agent Name vs Display Name
- **agent_name**: Technical ID, lowercase-with-hyphens, unique, unchangeable
- **name**: Display name shown in UI, user-friendly, can have spaces

### Skills Components
- **ID**: Programmatic identifier (e.g., `search_db`)
- **Name**: Human name (e.g., "Search Database")
- **Description**: What it does
- **Tags**: Search keywords (comma-separated)
- **Examples**: Sample queries (comma-separated)

### URLs
- **url**: Where your agent lives (must be accessible)
- **iconUrl**: Optional image for agent avatar
- **documentationUrl**: Optional link to agent docs

## Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| "Agent name already exists" | Choose a different agent_name |
| Agent not in list | Refresh page, check you're logged in as admin |
| Can't create agent | Verify all required fields (red asterisks) |
| Agent URL unreachable | Test URL in browser first, ensure agent is running |
| Can't edit agent | Check you have admin permissions |

## API Quick Reference

### Create Agent
```bash
POST /v1/agents
Authorization: Bearer YOUR_KEY
Content-Type: application/json

{
  "agent_name": "...",
  "agent_card_params": { ... }
}
```

### List All Agents
```bash
GET /v1/agents
Authorization: Bearer YOUR_KEY
```

### Get Specific Agent
```bash
GET /v1/agents/{agent_id}
Authorization: Bearer YOUR_KEY
```

### Update Agent
```bash
PUT /v1/agents/{agent_id}
Authorization: Bearer YOUR_KEY
Content-Type: application/json

{
  "agent_card_params": { ... }
}
```

### Delete Agent
```bash
DELETE /v1/agents/{agent_id}
Authorization: Bearer YOUR_KEY
```

### Make Agent Public
```bash
POST /v1/agents/{agent_id}/make_public
Authorization: Bearer YOUR_KEY
```

## Next Steps

After creating your first agent:

1. **Test It**: Use the Playground to test agent responses
2. **Monitor It**: Check Analytics tab for usage metrics
3. **Secure It**: Set up proper authentication on your agent endpoint
4. **Document It**: Add documentation URL for team reference
5. **Share It**: Make public when ready for broader use

## Learn More

- [Complete Agents UI Guide](./agents_ui_guide.md) - Detailed documentation
- [Agent API Reference](../api-reference/agents.md) - Full API docs
- [A2A Specification](https://a2a.ai/spec) - Protocol details
- [Building Custom Agents](./custom-agents.md) - Development guide

## Support

Need help? 
- 📖 [LiteLLM Docs](https://docs.litellm.ai)
- 💬 [Discord Community](https://discord.com/invite/wuPM9dRgDw)
- 🐛 [GitHub Issues](https://github.com/BerriAI/litellm/issues)
