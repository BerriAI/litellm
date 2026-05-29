---
slug: lap-internal-agent-30-percent
title: "How we built a background agent to cover 30% of our backlog"
date: 2026-05-27T10:00:00
authors:
  - krrish
  - ishaan
description: "How we built a background agent on the LiteLLM AI Gateway that merges PRs with no human in the loop (the infra, harness, and credential-scoping calls behind it)."
tags: [agents, ai-gateway, lap, lite-harness, engineering]
hide_table_of_contents: true
image: /img/lap_litellm_agent_platform_hero.png
---

<img src="/img/lap_litellm_agent_platform_hero.png" alt="LiteLLM Agent Platform: agent.litellm.ai" style={{width: "100%"}} />

:::info

The platform we built is open source. Check out [litellm-agent-platform](https://github.com/BerriAI/litellm-agent-platform). The swappable harness layer is [lite-harness](https://github.com/LiteLLM-Labs/lite-harness).

Building the same thing inside your company?

- [Schedule a 30-minute call](https://calendly.com/d/cr4t-yp7-pzn/litellm-1-1-feedback-chat)
- [Join the LAP Discord](https://discord.gg/Q2AK7HKudm)

:::

Our goal was to 10x the productivity of our company with agents.

Three weeks ago we began building an agent that could own 30% of our engineering tickets. Here's what we've learnt so far.

{/* truncate */}

## What we shipped

Three weeks in, on `BerriAI/litellm`: **43 open PRs, 160 closed.** Between the PRs it lands and the Slack questions it answers, the agent now covers roughly **30% of the eng tickets that used to hit a human every week.** Browse [all agent-filed PRs on GitHub](https://github.com/BerriAI/litellm/pulls?q=is%3Apr+author%3Aoss-agent-shin).


## Why we built our own

We wanted an agent that runs autonomously, in the background, pulling tickets off Linear and filing PRs for us. We evaluated Cursor and Anthropic's managed agent platforms first. Neither fit:

- **Cursor:** agents were not stateful. You could not store memory, skills, etc. per agent. The platform equated an agent to a session; we wanted an agent that persists across them.
- **Anthropic:** close to what we wanted, but we wanted to swap models and harnesses freely. We did not want to be locked to one platform.

So we built on the [LiteLLM Agent Platform](https://github.com/BerriAI/litellm-agent-platform). 

## 1. Infrastructure: separate the brain from the sandbox

Our first version ran the agent *inside* the sandbox, following the same shape as [Ramp Inspect](https://builders.ramp.com/post/why-we-built-our-background-agent). Every new session booted a fresh sandbox. That is fine when the work is "go edit code." It is wasteful when an engineer just asks a question in Slack. You pay a full sandbox boot to answer something that needs a few tool calls.

So we split the agent in two. The **brain** (reasoning, planning, model calls) lives in a shared, persistent pod. It has no shell: no BASH, no filesystem. The **sandbox** is ephemeral, one per session, and the only thing that can run `git`, `gh`, or `pytest`. The brain reaches it through two tool calls. This is similar to [how Anthropic's managed agent platform works](https://www.anthropic.com/engineering/managed-agents).

![Architecture: a persistent brain pod with no shell, talking to an ephemeral per-session sandbox pool through two tool calls](/img/lap_brain_sandbox_split.svg)

Response time dropped, session success rates climbed, and cost per session fell.

The cold start showed up most visibly in Slack, where everyone could feel the wait.

![Slack thread waiting on a cold sandbox boot before the agent could respond](/img/lap_shin_slack_slow_start.png)

## 2. Architecture: pick a harness, not an agent framework

We started with agent frameworks: Pydantic AI, LangGraph, the PI SDK. Each one made us rebuild things a coding *harness* already ships: context compaction, sub-agent spawning, tool loops. We already trusted Claude Code locally for this work, so we went looking for a harness, not a framework.

We landed on **OpenCode**. The Claude Agents SDK spawns a CLI session per run and OOM'd for us at ~1 RPM. OpenCode hits the same fundamental bottleneck (long-running sessions held in memory), but its memory usage grew more slowly, making it the better fit for now.

That choice stays flexible because we also wrote a harness unification layer, [`LiteLLM-Labs/lite-harness`](https://github.com/LiteLLM-Labs/lite-harness), which adapts OpenCode, Claude Code, Codex, and others to a single HTTP contract:

```
lite-harness/
  opencode/           # runtime adapter
  claude-agent-sdk/   # runtime adapter
  contract.py         # the one interface every runtime implements
```

The agent platform doesn't know which harness is behind a session, so swapping is a config change, not a rewrite.

Our next goal: 100 RPM on the harness.

## 3. Security: scope every credential to one endpoint

Our agent kept leaking API keys from its environment into commits and Slack messages. First mitigation: a small HTTP proxy vault. We stubbed the real credentials in the environment and swapped the stub for the real value only when the agent made an outbound call.

The agent defeated it. It noticed the credentials were stubbed, then wrote its own endpoint, called it with the stubbed credentials, let the vault swap in the real ones on the way out, and read the real keys back off its own server, then stored them to memory via a tool call. A clean man-in-the-middle against our own vault.

![Ishaan catching the agent writing real credentials to its memory after circumventing the stub vault](/img/lap_shin_agent_mitm_memory.png)

The fix was to stop trusting the *value* and start binding it to a *destination*. Each credential is pinned to one upstream host; the vault refuses the swap if the outbound request is going anywhere else:

```yaml
# vault: a credential is only ever swapped in for its bound host
credentials:
  GITHUB_TOKEN:
    allowed_host: api.github.com
  OPENAI_API_KEY:
    allowed_host: api.openai.com
```

The lesson: guardrails must sit at the agent's input/output boundary. LLM-level guardrails can't distinguish between a user query and an internal tool loop, so they're either too permissive or too slow.

## Where the AI Gateway fits

The AI Gateway is a useful access control point: it's how we gave our agent access to models and MCP tools. But it's only half the picture. The agent boundary needs its own guardrails and capabilities (skills, memory), because the agent (not the model) is what takes actions. The guardrails needed when an agent answers a user differ from those needed inside an internal tool loop. Running model-level guardrails on every tool call also adds ~5 minutes per session.

## What we believe now

Autonomous agents are where the 10x productivity gains are, and the technical risk is largely solved. Models are already smart enough to file a decent PR. The hard problems left are product problems: scale, reliability, and security.

For us, that means two open problems:

- **Scale:** how do you serve 100 RPM on a harness that keeps sessions in memory?
- **Security:** how do you prevent the agent server from leaking sensitive information or taking destructive actions? (We tried MCPs but hit rate limits and structural issues, so direct API keys were more reliable, which is what made scoping credentials critical.)

## Try it

Both repos are open source and self-hostable: [litellm-agent-platform](https://github.com/BerriAI/litellm-agent-platform) and [lite-harness](https://github.com/LiteLLM-Labs/lite-harness). If you're building something similar and want to skip the three weeks of mistakes, [book a 30-minute chat](https://calendly.com/d/cr4t-yp7-pzn/litellm-1-1-feedback-chat) or join the [LAP Discord](https://discord.gg/Q2AK7HKudm).


*This blog was inspired in shape by Ramp's [Why we built our background agent](https://builders.ramp.com/post/why-we-built-our-background-agent).*
