---
title: LiteLLM as Agent & MCP Gateway
sidebar_label: Agent & MCP Gateway
---

# LiteLLM as an Agent & MCP Gateway

**LiteLLM is a unified gateway for LLMs, agents, and MCP.** You do not need a separate agent gateway or MCP gateway — LiteLLM provides both natively.

If you're considering adopting [A2A (Agent-to-Agent)](./a2a) or [multi-MCP](./mcp) in your stack, this page explains what you need to know about LiteLLM.

---

## Do I Need a Separate Agent or MCP Gateway?

**No.** LiteLLM is both:

| Capability | What LiteLLM Provides |
|------------|------------------------|
| **LLM Gateway** | Single endpoint for 100+ models (OpenAI, Anthropic, Vertex, Bedrock, etc.) |
| **Agent Gateway (A2A)** | Invoke A2A agents, Vertex AI Agent Engine, LangGraph, Bedrock AgentCore, Azure AI Foundry, Pydantic AI |
| **MCP Gateway** | Central MCP endpoint with per-key/team access control, cost tracking, guardrails |

One gateway. One config. One place to manage keys, budgets, and access.

---

## Key Concepts

### A2A (Agent-to-Agent) Protocol

LiteLLM implements the [A2A protocol](https://github.com/google/A2A) for invoking agents. You can:

- Add A2A-compatible agents via the Admin UI or config
- Invoke agents via the A2A SDK or OpenAI-compatible `/chat/completions` with `a2a/` model prefix
- Track agent usage, costs, and logs in LiteLLM
- Control which keys/teams can access which agents

**Supported agent providers:** A2A, Vertex AI Agent Engine, LangGraph, Azure AI Foundry, Bedrock AgentCore, Pydantic AI

👉 [A2A Agent Gateway →](./a2a)

### MCP (Model Context Protocol)

LiteLLM provides an MCP Gateway so you can:

- Use a single endpoint for all MCP tools (Streamable HTTP, SSE, stdio)
- Control MCP access by key, team, or organization
- Apply guardrails and cost tracking to MCP calls
- Use MCP tools with any LiteLLM-supported model (not just one provider)

👉 [MCP Gateway →](./mcp)

---

## Multi-MCP & Multi-Agent: What to Know

### One Gateway for Multiple MCP Servers

You can add many MCP servers to LiteLLM. Each is namespaced by server name, so tool names stay unique. Access is controlled per key/team — no need to run separate MCP gateways per environment.

### One Gateway for Multiple Agents

Similarly, you can onboard multiple A2A agents. LiteLLM handles routing, load balancing, and logging. Clients call a single endpoint; LiteLLM routes to the right agent.

### Same Auth, Budgets, and Observability

Whether you're calling an LLM, an agent, or MCP tools, you use the same:

- Virtual keys and team-based access
- Budget and rate limit controls
- Logging and spend tracking
- Guardrails (where applicable)

---

## When to Use LiteLLM vs. a Provider-Specific Gateway

| Scenario | Recommendation |
|---------|----------------|
| You use multiple LLM providers (OpenAI + Anthropic + Vertex, etc.) | **LiteLLM** — single interface, one place for keys and budgets |
| You want MCP tools with any model | **LiteLLM** — MCP Gateway works with all supported models |
| You're building agents and need A2A | **LiteLLM** — native A2A support, no extra gateway |
| You only use one provider and want minimal setup | Provider SDK or gateway may be simpler |
| You need provider-specific features not yet in LiteLLM | Check [pass-through endpoints](./pass_through/intro) or provider docs |

---

## Quick Links

- [A2A Agent Gateway](./a2a) — Add and invoke agents
- [MCP Gateway](./mcp) — Add MCP servers and control access
- [AI Gateway (Proxy) Overview](./simple_proxy) — Full proxy capabilities
- [Virtual Keys](./proxy/virtual_keys) — Manage access and budgets
