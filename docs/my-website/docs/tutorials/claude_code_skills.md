# Manage LiteLLM with AI Agents

[litellm-skills](https://github.com/BerriAI/litellm-skills) is a collection of [Agent Skills](https://agentskills.io) for managing a live LiteLLM proxy deployment. Install them once and any agent that supports the Agent Skills standard (Claude Code, OpenCode, OpenClaw, etc.) can create users, teams, keys, models, MCP servers, agents, and query usage — all by running `curl` commands against your proxy.

## Install

```bash
git clone https://github.com/BerriAI/litellm-skills.git ~/.claude/skills/litellm
cd ~/.claude/skills
ln -s litellm/add-user add-user
ln -s litellm/update-user update-user
ln -s litellm/delete-user delete-user
ln -s litellm/add-team add-team
ln -s litellm/update-team update-team
ln -s litellm/delete-team delete-team
ln -s litellm/add-key add-key
ln -s litellm/update-key update-key
ln -s litellm/delete-key delete-key
ln -s litellm/add-org add-org
ln -s litellm/delete-org delete-org
ln -s litellm/add-model add-model
ln -s litellm/update-model update-model
ln -s litellm/delete-model delete-model
ln -s litellm/add-mcp add-mcp
ln -s litellm/update-mcp update-mcp
ln -s litellm/delete-mcp delete-mcp
ln -s litellm/add-agent add-agent
ln -s litellm/update-agent update-agent
ln -s litellm/delete-agent delete-agent
ln -s litellm/view-usage view-usage
```

Each symlink exposes a `SKILL.md` at the path the agent's skill discovery mechanism expects.

## Requirements

- `curl` installed
- A running LiteLLM proxy (local or remote)
- A proxy admin key — not a virtual key scoped to `llm_api_routes`

## Available Skills

### Users

| Skill | What it does |
|-------|-------------|
| `/add-user` | Create a user — email, role, budget, model access |
| `/update-user` | Update budget, role, or models for an existing user |
| `/delete-user` | Delete one or more users |

### Teams

| Skill | What it does |
|-------|-------------|
| `/add-team` | Create a team with budget and model limits |
| `/update-team` | Update budget, models, or rate limits |
| `/delete-team` | Delete one or more teams |

### API Keys

| Skill | What it does |
|-------|-------------|
| `/add-key` | Generate a key scoped to a user, team, budget, and expiry |
| `/update-key` | Update budget, models, or expiry |
| `/delete-key` | Delete by key value or alias |

### Organizations

| Skill | What it does |
|-------|-------------|
| `/add-org` | Create an org with budget and model access |
| `/delete-org` | Delete one or more orgs |

### Models

| Skill | What it does |
|-------|-------------|
| `/add-model` | Add any provider (OpenAI, Azure, Anthropic, Bedrock, Ollama…) and test it |
| `/update-model` | Rotate credentials or swap the underlying deployment |
| `/delete-model` | Remove a model |

### MCP Servers

| Skill | What it does |
|-------|-------------|
| `/add-mcp` | Register an MCP server (SSE, HTTP, or stdio) |
| `/update-mcp` | Update URL, credentials, or allowed tools |
| `/delete-mcp` | Remove an MCP server |

### Agents

| Skill | What it does |
|-------|-------------|
| `/add-agent` | Create an agent backed by a model and optional MCP servers |
| `/update-agent` | Swap the model or update description and limits |
| `/delete-agent` | Remove an agent |

### Usage

| Skill | What it does |
|-------|-------------|
| `/view-usage` | Daily spend and token activity — by user, team, org, or model |

## How it works

When you invoke a skill, the agent asks for your `LITELLM_BASE_URL` and admin key, collects the fields needed for that operation, runs the `curl`, and shows the result. For example:

```
/add-model
```
→ Agent asks: provider, public name, credentials. Adds the model, runs a test completion, reports pass/fail.

```
/view-usage
```
→ Agent asks: date range (defaults to current month), optional team/model filter. Prints a table of daily requests, tokens, and spend.

## Related

- [litellm-skills on GitHub](https://github.com/BerriAI/litellm-skills)
- [Virtual Keys](../proxy/virtual_keys.md) — managing API keys on the proxy
- [Team-based routing](../proxy/team_based_routing.md) — setting up teams
- [Model Management](../proxy/model_management.md) — adding models via config or API
