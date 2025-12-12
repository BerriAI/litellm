# LiteLLM Agents UI Guide

## Overview

The Agents feature in LiteLLM allows you to register, manage, and discover A2A (Agent-to-Agent) specification-compliant agents through both the Admin UI and REST API. This guide covers how to use the Agents tab in the LiteLLM UI.

## What are Agents?

Agents in LiteLLM follow the [A2A (Agent-to-Agent) protocol specification](https://a2a.ai/spec), which provides a standardized way for AI agents to:
- Discover other agents' capabilities
- Communicate with each other
- Execute tasks through well-defined skills
- Support various interaction modes (text, audio, video)

## Accessing the Agents Tab

1. Start your LiteLLM Proxy server with the Admin UI enabled
2. Navigate to your proxy URL (e.g., `http://localhost:4000`)
3. Log in with your admin credentials
4. Click on the **"Agents"** tab in the left sidebar

The Agents page displays:
- A list of all registered agents in your organization
- Each agent's name, description, capabilities, and skills
- Options to add, edit, delete, and make agents public

## Adding a New Agent

### Step 1: Click "Add New Agent"

On the Agents page, click the **"+ Add New Agent"** button in the top-right corner. This opens the agent creation modal.

### Step 2: Select Agent Type

The form begins with an **Agent Type** dropdown that allows you to choose between different agent implementations:

- **A2A Agent** (default) - Standard A2A-compliant agent
- **Custom Agent Types** - May include integrations like Bedrock AgentCore or other providers

Each agent type shows:
- A logo/icon
- Display name
- Brief description

### Step 3: Fill in Basic Information (Required)

The form is organized into collapsible sections. The **Basic Information** section includes required fields:

#### **Agent Name** (Required)
- **Field:** `agent_name`
- **Purpose:** Unique identifier for the agent in LiteLLM
- **Example:** `customer-support-agent`, `data-analysis-agent`
- **Note:** Must be unique across your organization

#### **Display Name** (Required)
- **Field:** `name`
- **Purpose:** Human-readable name shown in the UI and agent card
- **Example:** "Customer Support Agent", "Data Analysis Assistant"

#### **Description** (Required)
- **Field:** `description`
- **Purpose:** Detailed description of what the agent does
- **Type:** Multi-line text area (3 rows)
- **Example:** "An AI agent specialized in handling customer support inquiries, routing tickets, and providing instant responses to common questions."

#### **URL** (Required)
- **Field:** `url`
- **Purpose:** Base URL where the agent is hosted
- **Example:** `http://localhost:9999/`, `https://my-agent.example.com/`
- **Note:** This is where LiteLLM will send requests to communicate with your agent

#### **Version** (Optional)
- **Field:** `version`
- **Default:** `1.0.0`
- **Purpose:** Semantic version of your agent

#### **Protocol Version** (Optional)
- **Field:** `protocolVersion`
- **Default:** `1.0`
- **Purpose:** A2A protocol version the agent supports

### Step 4: Define Skills (Required)

The **Skills** section defines the specific capabilities your agent can perform. Click **"+ Add Skill"** to add one or more skills.

For each skill, you must provide:

#### **Skill ID** (Required)
- **Field:** `id`
- **Purpose:** Unique identifier for this skill
- **Example:** `hello_world`, `search_database`, `generate_report`

#### **Skill Name** (Required)
- **Field:** `name`
- **Purpose:** Human-readable name for the skill
- **Example:** "Returns hello world", "Search Customer Database", "Generate Analytics Report"

#### **Description** (Required)
- **Field:** `description`
- **Purpose:** Detailed description of what this skill does
- **Type:** Multi-line text area (2 rows)
- **Example:** "Searches the customer database using natural language queries and returns relevant records"

#### **Tags** (Required)
- **Field:** `tags`
- **Format:** Comma-separated values
- **Purpose:** Keywords for discovering this skill
- **Example:** `hello world, greeting`, `search, database, query`

#### **Examples** (Optional)
- **Field:** `examples`
- **Format:** Comma-separated values
- **Purpose:** Example queries or commands that trigger this skill
- **Example:** `hi, hello world`, `find customer John Doe, search for orders`

You can add multiple skills by clicking **"+ Add Skill"** again, and remove skills using the **"Remove Skill"** button.

### Step 5: Configure Capabilities (Optional)

The **Capabilities** section defines optional features your agent supports:

#### **Streaming**
- **Field:** `streaming`
- **Type:** Toggle switch
- **Default:** Off
- **Purpose:** Whether the agent can stream responses in real-time

#### **Push Notifications**
- **Field:** `pushNotifications`
- **Type:** Toggle switch
- **Purpose:** Whether the agent can send push notifications

#### **State Transition History**
- **Field:** `stateTransitionHistory`
- **Type:** Toggle switch
- **Purpose:** Whether the agent maintains state transition history

### Step 6: Optional Settings

The **Optional Settings** section includes additional metadata:

#### **Icon URL**
- **Field:** `iconUrl`
- **Type:** URL input
- **Example:** `https://example.com/icon.png`
- **Purpose:** URL to an icon representing your agent

#### **Documentation URL**
- **Field:** `documentationUrl`
- **Type:** URL input
- **Example:** `https://docs.example.com`
- **Purpose:** Link to detailed documentation for your agent

#### **Supports Authenticated Extended Card**
- **Field:** `supportsAuthenticatedExtendedCard`
- **Type:** Toggle switch
- **Purpose:** Whether the agent supports authenticated extended agent cards

### Step 7: LiteLLM Parameters (Optional)

The **LiteLLM Parameters** section contains LiteLLM-specific settings:

#### **Model (Optional)**
- **Field:** `model`
- **Purpose:** Specify a LiteLLM model to use for this agent
- **Example:** `gpt-4`, `claude-3-opus`, `gemini-pro`

#### **Make Public**
- **Field:** `make_public`
- **Type:** Toggle switch
- **Default:** Off
- **Purpose:** If enabled, this agent will be discoverable in the public Agent Hub
- **Note:** Public agents can be discovered by other organizations

### Step 8: Create the Agent

1. Review all the information you've entered
2. Click the **"Create Agent"** button at the bottom of the form
3. If successful, you'll see a success notification
4. The agent will appear in the agents list

If there are validation errors, they will be highlighted in red with helpful error messages.

## Managing Existing Agents

### Viewing Agent Details

Click on any agent in the list to view its full details, including:
- Complete agent card information
- All configured skills
- Capabilities and settings
- Creation and update timestamps
- Creator information

### Editing an Agent

1. Click on an agent to view its details
2. Click the **"Edit"** button
3. Modify any fields (except the agent name, which is immutable)
4. Click **"Update Agent"** to save changes

### Making an Agent Public

To share your agent with the Agent Hub:

1. Find the agent in the list
2. Toggle the **"Make Public"** switch, or
3. Use the **"Make Public"** action button
4. Public agents are discoverable by other LiteLLM instances

### Deleting an Agent

1. Click the **"Delete"** button next to an agent
2. Confirm the deletion in the modal dialog
3. **Warning:** This action cannot be undone

## Agent List View

The agents table displays:

### Columns
- **Agent Name** - The unique identifier
- **Display Name** - Human-readable name
- **Description** - Brief description of capabilities
- **Skills** - Number of skills the agent has
- **Public** - Whether the agent is publicly discoverable
- **Created** - When the agent was registered
- **Actions** - Edit, Delete, Make Public buttons

### Filtering and Search
- Use the search bar to filter agents by name or description
- Filter by public/private status
- Sort by creation date, name, or other columns

## Agent Hub

The Agent Hub (accessible via "Go to AI Hub" link) shows:
- All public agents from your organization
- Agents shared by other organizations (if configured)
- Ability to discover and use agents across teams

## API Integration

All UI operations have corresponding API endpoints. See the [Agent API Reference](../api-reference/agents.md) for details.

### Example: Creating an Agent via API

```bash
curl -X POST "http://localhost:4000/v1/agents" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "my-custom-agent",
    "agent_card_params": {
      "protocolVersion": "1.0",
      "name": "Hello World Agent",
      "description": "Just a hello world agent",
      "url": "http://localhost:9999/",
      "version": "1.0.0",
      "defaultInputModes": ["text"],
      "defaultOutputModes": ["text"],
      "capabilities": {
        "streaming": true
      },
      "skills": [
        {
          "id": "hello_world",
          "name": "Returns hello world",
          "description": "just returns hello world",
          "tags": ["hello world"],
          "examples": ["hi", "hello world"]
        }
      ]
    },
    "litellm_params": {
      "model": "gpt-4",
      "make_public": false
    }
  }'
```

## Use Cases

### 1. Customer Support Agent
Register an agent that handles customer inquiries with skills for:
- Searching knowledge base
- Creating support tickets
- Escalating to human agents

### 2. Data Analysis Agent
Create an agent with skills for:
- Querying databases
- Generating reports
- Creating visualizations

### 3. Code Assistant Agent
Build an agent that can:
- Review code
- Suggest improvements
- Generate documentation

### 4. Research Agent
Deploy an agent with abilities to:
- Search academic papers
- Summarize findings
- Track citations

## Best Practices

### Agent Design
1. **Clear Naming** - Use descriptive, unique agent names
2. **Detailed Descriptions** - Help users understand what your agent does
3. **Well-Defined Skills** - Each skill should have a single, clear purpose
4. **Good Examples** - Provide realistic example queries for each skill
5. **Appropriate Tags** - Use relevant tags for discoverability

### Security
1. **Authentication** - Ensure your agent endpoint requires authentication
2. **Private by Default** - Only make agents public when necessary
3. **Access Control** - Use LiteLLM's team and key permissions to control access
4. **Validate Inputs** - Your agent should validate all incoming requests

### Performance
1. **Enable Streaming** - For better user experience with long responses
2. **Optimize URLs** - Use low-latency endpoints for agent URLs
3. **Monitor Usage** - Track agent usage in the Analytics tab

### Maintenance
1. **Version Your Agents** - Use semantic versioning for changes
2. **Update Documentation** - Keep documentation URLs current
3. **Test Before Public** - Thoroughly test agents before making them public
4. **Monitor Errors** - Check logs for agent communication issues

## Troubleshooting

### Agent Not Appearing in List
- Check that you're logged in as an admin user
- Verify the agent was successfully created (check for success notification)
- Refresh the page

### Cannot Create Agent
- Ensure the agent name is unique
- Verify all required fields are filled
- Check that the URL is valid and accessible
- Confirm you have admin permissions

### Agent Communication Errors
- Verify the agent URL is correct and reachable
- Check that your agent is running and responding
- Review the agent's authentication requirements
- Check LiteLLM proxy logs for detailed error messages

### Cannot Make Agent Public
- Ensure you have admin permissions
- Check that the agent has all required fields filled
- Verify the agent is properly validated

## Related Documentation

- [Agent API Reference](../api-reference/agents.md)
- [A2A Protocol Specification](https://a2a.ai/spec)
- [Agent Permissions and Access Control](./agent-permissions.md)
- [Building Custom Agents](./custom-agents.md)

## Support

For questions or issues:
- Check the [LiteLLM Documentation](https://docs.litellm.ai)
- Open an issue on [GitHub](https://github.com/BerriAI/litellm)
- Join the [Discord community](https://discord.com/invite/wuPM9dRgDw)
