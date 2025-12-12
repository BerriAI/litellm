# LiteLLM Agents - Complete Examples and Tutorials

This directory contains comprehensive examples and tutorials for working with LiteLLM Agents in the Admin UI.

## 📁 Files in This Directory

- **example_agent_config.yaml** - Sample agent configurations for the proxy config file
- **tutorial_simple_agent.md** - Step-by-step tutorial for creating your first agent
- **tutorial_advanced_agent.md** - Advanced agent with multiple skills and capabilities
- **best_practices.md** - Best practices for agent development and deployment

## 🎯 Quick Start

### Option 1: Using the UI (Recommended for beginners)

1. Start your LiteLLM proxy with UI enabled:
   ```bash
   litellm --config proxy_config.yaml
   ```

2. Open the UI at `http://localhost:4000`

3. Navigate to the "Agents" tab (🤖 icon in left sidebar)

4. Click "+ Add New Agent" button

5. Fill in the form and click "Create Agent"

### Option 2: Using the API

```bash
curl -X POST "http://localhost:4000/v1/agents" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d @example_agent.json
```

### Option 3: Using Config File

Add agents to your `proxy_config.yaml`:
```yaml
agents:
  - agent_name: my-agent
    agent_card_params:
      # ... agent configuration
```

## 📚 Documentation

### Complete Documentation Set
1. **[Agents UI Visual Guide](../../../docs/my-website/docs/proxy/agents_ui_visual_guide.md)** - Detailed visual walkthrough with UI mockups
2. **[Agents UI Guide](../../../docs/my-website/docs/proxy/agents_ui_guide.md)** - Complete reference for all features
3. **[Agents Quick Start](../../../docs/my-website/docs/proxy/agents_quick_start.md)** - Get started in 5 minutes

## 🎓 Tutorials

### Tutorial 1: Simple Hello World Agent (5 minutes)

Create a basic agent with one skill:

```json
{
  "agent_name": "hello-world",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Hello World Agent",
    "description": "A simple greeting agent",
    "url": "http://localhost:9999/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": false
    },
    "skills": [
      {
        "id": "greet",
        "name": "Greeting",
        "description": "Greets users warmly",
        "tags": ["greeting", "hello"],
        "examples": ["hi", "hello", "hey"]
      }
    ]
  }
}
```

**In the UI:**
1. Click "Add New Agent"
2. Agent Name: `hello-world`
3. Display Name: `Hello World Agent`
4. Description: `A simple greeting agent`
5. URL: `http://localhost:9999/`
6. Click "+ Add Skill"
   - ID: `greet`
   - Name: `Greeting`
   - Description: `Greets users warmly`
   - Tags: `greeting, hello`
   - Examples: `hi, hello, hey`
7. Click "Create Agent"

### Tutorial 2: Customer Support Agent (15 minutes)

Build a multi-skill agent for customer support:

```json
{
  "agent_name": "support-bot",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Support Bot",
    "description": "24/7 customer support assistant",
    "url": "https://support.example.com/agent",
    "version": "2.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": true,
      "pushNotifications": true
    },
    "skills": [
      {
        "id": "answer_faq",
        "name": "Answer FAQs",
        "description": "Answers frequently asked questions",
        "tags": ["faq", "questions", "help"],
        "examples": [
          "how do I reset my password",
          "what are your hours",
          "how do I contact support"
        ]
      },
      {
        "id": "create_ticket",
        "name": "Create Support Ticket",
        "description": "Creates a support ticket for complex issues",
        "tags": ["ticket", "issue", "problem"],
        "examples": [
          "create a ticket",
          "report an issue",
          "file a complaint"
        ]
      },
      {
        "id": "track_order",
        "name": "Track Order",
        "description": "Tracks customer orders",
        "tags": ["order", "tracking", "shipping"],
        "examples": [
          "where is my order",
          "track order #12345",
          "shipping status"
        ]
      }
    ],
    "iconUrl": "https://example.com/support-icon.png",
    "documentationUrl": "https://docs.example.com/support-bot"
  },
  "litellm_params": {
    "model": "gpt-4",
    "make_public": false
  }
}
```

**UI Steps:**
1. Open "Add New Agent" modal
2. Fill Basic Information section
3. Add three skills (use "+ Add Skill" button)
4. Enable "Streaming" and "Push Notifications" in Capabilities
5. Add Icon URL and Documentation URL in Optional Settings
6. Select model "gpt-4" in LiteLLM Parameters
7. Create the agent

### Tutorial 3: Data Analysis Agent (20 minutes)

Advanced agent with data processing capabilities:

```json
{
  "agent_name": "data-analyzer",
  "agent_card_params": {
    "protocolVersion": "1.0",
    "name": "Data Analysis Agent",
    "description": "Analyzes data, generates insights, and creates visualizations",
    "url": "http://data-agent.internal:8080/",
    "version": "3.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {
      "streaming": true,
      "stateTransitionHistory": true
    },
    "skills": [
      {
        "id": "query_data",
        "name": "Query Database",
        "description": "Executes SQL queries on the data warehouse",
        "tags": ["database", "sql", "query", "data"],
        "examples": [
          "show sales by region",
          "query customer data",
          "get revenue report"
        ]
      },
      {
        "id": "analyze_trends",
        "name": "Analyze Trends",
        "description": "Identifies patterns and trends in data",
        "tags": ["analysis", "trends", "patterns", "insights"],
        "examples": [
          "analyze sales trends",
          "find patterns in data",
          "identify anomalies"
        ]
      },
      {
        "id": "create_visualization",
        "name": "Create Visualization",
        "description": "Generates charts and graphs",
        "tags": ["visualization", "chart", "graph", "dashboard"],
        "examples": [
          "create a bar chart",
          "visualize trends",
          "make a dashboard"
        ]
      },
      {
        "id": "export_report",
        "name": "Export Report",
        "description": "Exports analysis as PDF/Excel",
        "tags": ["export", "report", "download", "pdf"],
        "examples": [
          "export to pdf",
          "download report",
          "save as excel"
        ]
      },
      {
        "id": "schedule_report",
        "name": "Schedule Reports",
        "description": "Schedules automated report generation",
        "tags": ["schedule", "automation", "recurring"],
        "examples": [
          "schedule daily report",
          "send weekly summary",
          "automate monthly report"
        ]
      }
    ],
    "iconUrl": "https://example.com/data-icon.png",
    "documentationUrl": "https://docs.example.com/data-agent"
  },
  "litellm_params": {
    "model": "claude-3-opus",
    "make_public": false
  }
}
```

## 🔧 Real-World Use Cases

### Use Case 1: Enterprise IT Help Desk

**Scenario**: Automate common IT support tasks

**Agent Configuration**:
- Skills: Password reset, Software installation, VPN setup, Hardware request
- Capabilities: Streaming (for real-time responses)
- Model: GPT-4 (for better understanding of technical issues)

**Benefits**:
- 24/7 availability
- Instant responses to common issues
- Reduces IT team workload by 40%

### Use Case 2: E-commerce Assistant

**Scenario**: Help customers browse and purchase products

**Agent Configuration**:
- Skills: Product search, Recommendations, Order tracking, Returns
- Capabilities: Streaming, Push Notifications (order updates)
- Model: Claude-3 (for natural conversation)

**Benefits**:
- Increases conversion rates
- Reduces cart abandonment
- Handles multiple customers simultaneously

### Use Case 3: Research Assistant

**Scenario**: Help researchers find and analyze academic papers

**Agent Configuration**:
- Skills: Paper search, Citation management, Summarization, Literature review
- Capabilities: State Transition History (track research progress)
- Model: GPT-4 (for accurate academic understanding)

**Benefits**:
- Saves research time
- Comprehensive literature coverage
- Organized research workflow

## 🎨 UI Workflow Examples

### Workflow 1: Create and Test Agent

```
1. Navigate to Agents tab
2. Click "+ Add New Agent"
3. Fill in agent details
4. Add skills
5. Click "Create Agent"
6. Go to Playground tab
7. Select your agent
8. Send test message
9. Verify response
10. Iterate as needed
```

### Workflow 2: Make Agent Public

```
1. Find your agent in the list
2. Click Actions → "Make Public"
3. Confirm in modal dialog
4. Agent now appears in AI Hub
5. Other users can discover it
6. Monitor usage in Analytics
```

### Workflow 3: Update Agent Skills

```
1. Click on agent name
2. View agent details
3. Click "Edit" button
4. Scroll to Skills section
5. Modify existing skills or add new ones
6. Click "Update Agent"
7. Test changes in Playground
```

## 📊 Monitoring and Analytics

### Track Agent Usage

1. Go to Analytics tab
2. Filter by agent name
3. View metrics:
   - Total requests
   - Success rate
   - Average response time
   - Error patterns
   - Most used skills

### Debug Agent Issues

1. Check Logs tab
2. Filter by agent_id
3. Review error messages
4. Check agent endpoint health
5. Verify agent URL accessibility

## 🔐 Security Best Practices

### 1. Authentication
```yaml
agent_card_params:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
  security:
    - bearerAuth: []
```

### 2. Rate Limiting
Configure in LiteLLM proxy:
```yaml
litellm_settings:
  rate_limit: 100  # requests per minute
```

### 3. Private Agents
Keep sensitive agents private:
```yaml
litellm_params:
  make_public: false
```

### 4. Team Permissions
Restrict agent access by team:
```bash
# Assign agent to specific team
POST /v1/agents/{agent_id}/teams
{
  "team_ids": ["team-1", "team-2"]
}
```

## 🐛 Troubleshooting

### Problem: Agent not appearing in list

**Solutions:**
1. Refresh the page
2. Check you're logged in as admin
3. Verify agent was created (check for success message)
4. Check browser console for errors

### Problem: Cannot create agent

**Solutions:**
1. Verify all required fields are filled
2. Check agent name is unique
3. Ensure URL is valid and reachable
4. Confirm you have admin permissions
5. Check for validation errors in red

### Problem: Agent communication errors

**Solutions:**
1. Verify agent URL is correct
2. Test agent endpoint directly
3. Check agent is running
4. Review agent logs
5. Check network/firewall settings

### Problem: Skills not working

**Solutions:**
1. Verify skill IDs are unique
2. Check skill descriptions are clear
3. Add more example queries
4. Test with exact example phrases
5. Review agent's skill implementation

## 💡 Tips and Tricks

### Naming Conventions
```
✅ Good:
- agent_name: customer-support-v2
- skill_id: create_ticket

❌ Bad:
- agent_name: Agent 1 (spaces)
- skill_id: CreateTicket (camelCase)
```

### Skill Descriptions
```
✅ Good: "Creates a new support ticket with customer information and routes to appropriate team"

❌ Bad: "Creates ticket"
```

### Tags Best Practices
```
✅ Good: ["support", "ticket", "create", "help"]

❌ Bad: ["support ticket creation"] (too specific)
```

### Testing Workflow
1. Create agent with one skill first
2. Test thoroughly
3. Add more skills incrementally
4. Test after each addition
5. Monitor error rates

## 📦 Example Files

See the following example files in this directory:

1. **example_agent_config.yaml** - YAML configuration for multiple agents
2. **simple_agent.json** - JSON for a basic agent
3. **advanced_agent.json** - JSON for a complex multi-skill agent
4. **enterprise_agent.json** - JSON for an enterprise-grade agent

## 🚀 Next Steps

1. ✅ Create your first agent using the simple example
2. 📖 Read the [Complete UI Guide](../../../docs/my-website/docs/proxy/agents_ui_guide.md)
3. 🎨 Study the [Visual Walkthrough](../../../docs/my-website/docs/proxy/agents_ui_visual_guide.md)
4. 🔧 Build a custom agent for your use case
5. 📊 Monitor usage and iterate
6. 🌐 Share successful agents in AI Hub

## 📞 Get Help

- **Documentation**: https://docs.litellm.ai
- **Discord Community**: https://discord.com/invite/wuPM9dRgDw
- **GitHub Issues**: https://github.com/BerriAI/litellm/issues
- **Email Support**: support@litellm.ai

## 🤝 Contributing

Have a great agent example? Contribute it!

1. Fork the repository
2. Add your example to this directory
3. Update this README
4. Submit a pull request

---

**Happy Agent Building! 🤖**
