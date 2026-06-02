---
slug: litellm-rust-launch
title: "Launching LiteLLM-Rust: A Minimal Rust AI Gateway for Coding Agents"
date: 2026-06-01T22:34:00
authors:
  - krrish
  - ishaan
description: "Launching LiteLLM-Rust — a minimal, MIT-licensed Rust AI Gateway for coding agents. Drop-in compatible with your LiteLLM config.yaml + DB. Early, experimental, feedback welcome."
tags: [launch, rust, coding-agents, ai-gateway]
hide_table_of_contents: true
---

import { LatencyCompare, DropInMigration, RemoteAgentsFlow } from './diagrams';

*Last Updated: June 2026*

Today we're launching **LiteLLM-Rust** — a minimal Rust-based AI Gateway for coding agents.

Repo: [github.com/LiteLLM-Labs/litellm-rust](https://github.com/LiteLLM-Labs/litellm-rust)

It does three things:

1. **LiteLLM-compatible** — your existing `config.yaml` and database work out of the box.
2. **Fast** — targeting **`<1ms`** overhead latency on Claude Code calls.
3. **Built for autonomous agents** — sandboxing via E2B and Daytona today, with durable sessions, memory, artifacts, and vault on the roadmap.

{/* truncate */}

## LiteLLM-compatible AI Gateway

LiteLLM-Rust reads the same `config.yaml` format and the same database schema as the Python LiteLLM AI Gateway. Keys, virtual keys, teams, budgets, routing rules, and fallbacks carry over without changes. Client SDKs and admin workflows stay the same.

<DropInMigration />

```bash
litellm-rust --config /etc/litellm/config.yaml --port 4000
```

Same interface contract — only the runtime changes.

## Fast: `<1ms` overhead on Claude Code calls

Coding agents like Claude Code fan out many LLM calls per task. Every millisecond of gateway overhead compounds across tool calls. Our goal with LiteLLM-Rust is **sub-millisecond overhead** on the hot path, by removing Python from request forwarding entirely.

<LatencyCompare />

## Built for autonomous agents

A reliable AI Gateway for coding agents needs more than fast forwarding. It needs to schedule, sandbox, and persist state for long-running agent runs.

LiteLLM-Rust ships today with:

- **Sandboxing** via [E2B](https://e2b.dev/) and [Daytona](https://www.daytona.io/) — agents run in isolated environments, no host access
- **Claude Code scheduling** — kick off agent runs on cron, webhook, or API trigger

On the roadmap:

- **Durable sessions** — resume long-running agents across restarts
- **Memory** — persistent context across runs
- **Artifacts** — store and retrieve agent outputs
- **Vault** — secrets management for agent execution

<RemoteAgentsFlow />

## Key Takeaways

- LiteLLM-Rust is a minimal, MIT-licensed Rust AI Gateway built for coding agents
- Drop-in compatible with your existing LiteLLM `config.yaml` and database
- Sub-millisecond overhead is the performance target on Claude Code calls
- Sandboxing (E2B + Daytona) ships today; durable sessions, memory, artifacts, and vault on the roadmap
- Early and experimental — feedback welcome on [Discord](https://discord.gg/wuPM9dRgDw)

---

## Frequently Asked Questions

### Is it open-source?

Yes. 100% open-source under the MIT license. Repo: [github.com/LiteLLM-Labs/litellm-rust](https://github.com/LiteLLM-Labs/litellm-rust).

### Is it part of my existing LiteLLM deployment?

No. LiteLLM-Rust is a separate repo. Goal is to explore the design space safely and bring the learnings back to the core LiteLLM project over time.

### How mature is it?

Early and experimental. We're shipping it to gather feedback from coding-agent teams running it against real workloads. Please join the [Discord](https://discord.gg/wuPM9dRgDw) and tell us what's working and what's missing.

### How is this different from the existing Python LiteLLM AI Gateway?

The Python LiteLLM AI Gateway is the production-grade, feature-complete AI Gateway used by enterprise deployments today — and remains the recommended choice. LiteLLM-Rust is a minimal, performance-focused exploration aimed at coding-agent workloads. For teams with strict uptime and compliance requirements, [LiteLLM Enterprise](https://litellm.ai/enterprise) on the Python AI Gateway provides SSO/SCIM, air-gapped deployment, 24/7 SLA support, and advanced guardrails.

---

## Recommended Reading

- [LiteLLM AI Gateway — full feature overview](https://docs.litellm.ai/docs/simple_proxy)
- [Achieving sub-millisecond proxy overhead](https://docs.litellm.ai/blog/sub-millisecond-proxy-overhead)
- [Load balancing and routing across 100+ LLM providers](https://docs.litellm.ai/docs/routing)
