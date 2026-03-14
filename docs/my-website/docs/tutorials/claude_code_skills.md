# Claude Code Skills for LiteLLM

[Agent Skills](https://agentskills.io) are reusable prompt templates that give Claude Code new capabilities via slash commands. The [litellm-skills](https://github.com/BerriAI/litellm-skills) repo provides ready-made skills for managing a live LiteLLM proxy — creating users, teams, keys, models, MCP servers, agents, and viewing usage — all via `curl` from inside any Claude Code session.

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

Claude Code discovers skills by looking for `SKILL.md` files at `~/.claude/skills/<skill-name>/SKILL.md`. The symlinks expose each sub-skill at the expected depth.

## Requirements

- `curl` installed
- A running LiteLLM proxy (local or remote)
- A proxy admin key (not a virtual key scoped to `llm_api_routes`)

## Available Skills

### Users

| Command | Description |
|---------|-------------|
| `/add-user` | Create a user with email, role, budget, and model access |
| `/update-user` | Update budget, role, or model access for an existing user |
| `/delete-user` | Delete one or more users |

### Teams

| Command | Description |
|---------|-------------|
| `/add-team` | Create a team with budget and model limits |
| `/update-team` | Update budget, models, or rate limits for an existing team |
| `/delete-team` | Delete one or more teams |

### API Keys

| Command | Description |
|---------|-------------|
| `/add-key` | Generate an API key scoped to a user, team, budget, and expiry |
| `/update-key` | Update budget, models, or expiry for an existing key |
| `/delete-key` | Delete one or more keys by value or alias |

### Organizations

| Command | Description |
|---------|-------------|
| `/add-org` | Create an organization with budget and model access |
| `/delete-org` | Delete one or more organizations |

### Models

| Command | Description |
|---------|-------------|
| `/add-model` | Add any LLM provider (OpenAI, Azure, Anthropic, Bedrock, Ollama…) and test it |
| `/update-model` | Rotate credentials or change the underlying deployment |
| `/delete-model` | Remove a model from the proxy |

### MCP Servers

| Command | Description |
|---------|-------------|
| `/add-mcp` | Register an MCP server (SSE, HTTP, or stdio) |
| `/update-mcp` | Update URL, credentials, or allowed tools |
| `/delete-mcp` | Remove an MCP server registration |

### Agents

| Command | Description |
|---------|-------------|
| `/add-agent` | Create an AI agent backed by a model and optional MCP servers |
| `/update-agent` | Swap the model or update description and limits |
| `/delete-agent` | Remove an agent |

### Usage

| Command | Description |
|---------|-------------|
| `/view-usage` | Query daily spend and token activity by user, team, org, or model |

## Usage

When you invoke a skill, Claude will ask for your `LITELLM_BASE_URL` and admin key, then collect the minimum required fields for the operation and run the `curl`. For example:

```
/add-model
```

Claude will ask: which provider, what to call it, and any credentials — then add it to your proxy and run a test call to confirm it routes correctly.

```
/view-usage
```

Claude will ask for a date range (defaults to the current month) and print a table of daily requests, tokens, and spend.

## Related

- [litellm-skills on GitHub](https://github.com/BerriAI/litellm-skills)
- [Virtual Keys](../proxy/virtual_keys.md) — Managing API keys on the proxy
- [Team-based routing](../proxy/team_based_routing.md) — Setting up teams
- [Model Management](../proxy/model_management.md) — Adding models via config or API
- [MCP Servers](./claude_mcp.md) — Connecting MCP servers to Claude Code via LiteLLM
