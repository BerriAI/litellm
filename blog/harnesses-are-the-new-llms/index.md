---
slug: agents-are-the-new-llms
title: "A Unified Agent Control Plane"
date: 2026-06-10T09:00:00
authors:
- krrish
description: "The AI Gateway is moving up the stack: from routing model calls to routing agent work."
tags: [ideas, harnesses, ai-gateway, agents]
hide_table_of_contents: true
---

import { StackComparison, ConvergenceHero } from './diagrams';

<ConvergenceHero />

*Last updated: June 2026*

Agent infrastructure is already separating into three layers: models, harnesses, and runtimes. We believe a fourth layer will emerge: the unified agent control plane. This will allow calling agents living in different agent runtimes, all from 1 place. 

The reason is that companies will not run every agent on one runtime. Coding agents may run on Bedrock AgentCore or Claude Managed Agents. Data agents may run inside Elastic, Databricks, or Snowflake. Internal workflow agents may run on custom infrastructure. The control plane emerges because companies want one place where all of these agents can be used, regardless of where they were built or run.

But a registry alone is not enough. Anyone can build a list of agents.

The harder problem is invocation. Agent runtimes expose similar primitives — agents, sessions, events, tools — but they do not expose them through the same APIs. So if you want one place to actually use these agents, not just list them, the control plane has to manage agent runtimes, schedules, memory, and sessions.

This is the same pattern LiteLLM saw with models. Companies did not just need a catalog of models. They needed one interface to call them. The only change, is that the primitive is now the agent session, not the model call.

## The Stack of the Future

<StackComparison />

The important shift is that the gateway is no longer just routing model calls. It is routing agent work.

With LLMs, the stack became:

* **Models:** GPT, Claude, Gemini, Llama
* **Inference providers:** OpenAI, Anthropic, Bedrock, Vertex, Azure, vLLM
* **Gateway:** routing, fallbacks, logging, spend tracking, auth, billing
* **Applications:** copilots, workflows, internal tools, products

With agents, we think the stack becomes:

* **Models:** Claude, GPT, Gemini, open-source models
* **Harnesses:** Claude Code, Codex, OpenCode, Hermes, DeepAgents
* **Agent runtimes:** Claude Managed Agents, Bedrock AgentCore, Gemini Enterprise Agent Platform, self-hosted runtimes
* **Agent control plane:** multi-runtime platform where teams manage agent runtimes, schedules, memory, and sessions.
* **Applications:** coding agents, support agents, data agents, security agents

## Why companies will need this

At LiteLLM, we are already seeing our team work across multiple agent runtimes. Some people are building on Claude Managed Agents, others are on N8N or Cursor. 

This fragmentation makes it hard for agents built on these platforms to be shareable, and everyone to benefit from the work done so far. 

By having the agents live in 1 place, everyone can leverage these agents - even if the PR Babysitter Agent was written in Claude Managed Agents, which not everyone has direct access to. 

That is the control plane problem.

This is also why we think the AI Gateway moves up the stack. The gateway starts by managing model calls. But as agents become the dominant use-case for AI, the gateway has to manage agent sessions too.

## What we are building

[LiteLLM Agent Platform](https://github.com/LiteLLM-Labs/litellm-agent-platform) is our experiment in this direction.

LiteLLM Agent Platform is a Rust-based AI Gateway and Agent Control Plane. The goal is to let teams register, invoke, observe, and govern agents across multiple runtimes.

We are starting with coding agents because the need is obvious. They are long-running, stateful, tool-heavy, and expensive enough to require real infrastructure.

We are already seeing early users resonate with this pattern. Some companies want LAP to act as a central control plane for agents built by different teams on different runtimes. For example, one team might build an agent on Elastic’s runtime to analyze Kibana logs, but the company may want to expose that agent internally through a common gateway.

This is the architecture we believe is coming: models become interchangeable, harnesses become specialized, runtimes become managed, and the gateway becomes the control plane for agent work.

If this matches what you are seeing, we would love feedback on LiteLLM Agent Platform:

https://github.com/LiteLLM-Labs/litellm-agent-platform

## Frequently Asked Questions

### Is LiteLLM building a second product?

No. LAP is an experimental project. The goal is to learn quickly and bring the right pieces into LiteLLM over time.

### Is LAP production-ready?

No. LAP is pre-v0. APIs may change as we work with early users and contributors.

If you want to contribute, file an issue or join our Discord:

https://discord.gg/Nkxw3rm3EE
