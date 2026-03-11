---
title: Tutorials
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Tutorials** are end-to-end walkthroughs for integrating LiteLLM with popular AI tools, agent frameworks, and real-world workflows. Each tutorial is focused on a specific use case you can follow from start to finish.

---

## Browse Tutorials

<NavigationCards
columns={2}
items={[
  {
    icon: "🛠️",
    title: "AI Coding Tools",
    description: "Integrate LiteLLM with OpenWebUI, Claude Code, Gemini CLI, OpenAI Codex, Cursor, GitHub Copilot, and more.",
    to: "/docs/ai_tools",
  },
  {
    icon: "🤖",
    title: "Agent SDKs",
    description: "Use LiteLLM with OpenAI Agents SDK, Claude Agent SDK, CopilotKit, Google ADK, and LiveKit.",
    to: "/docs/agent_sdks",
  },
  {
    icon: "🔑",
    title: "Authentication & Access",
    description: "Set up Microsoft SSO, SCIM provisioning, default teams, and tag-based access management.",
    to: "/docs/tutorials/msft_sso",
  },
  {
    icon: "💸",
    title: "Prompt Caching & Cost Tracking",
    description: "Learn to enable prompt caching, track costs for coding tools, and manage spend across teams.",
    to: "/docs/tutorials/prompt_caching",
  },
  {
    icon: "🛡️",
    title: "Guardrails & PII Masking",
    description: "Integrate Aporia guardrails, Presidio PII masking, and content safety filters.",
    to: "/docs/tutorials/litellm_proxy_aporia",
  },
  {
    icon: "📊",
    title: "Logging & Observability",
    description: "Send logs to Elasticsearch, set up Langfuse tracing, and monitor your LLM usage.",
    to: "/docs/tutorials/elasticsearch_logging",
  },
  {
    icon: "🐍",
    title: "LiteLLM Python SDK",
    description: "Step-by-step tutorials for Azure OpenAI, HuggingFace, TogetherAI, model fallbacks, and more.",
    to: "/docs/tutorials/azure_openai",
  },
  {
    icon: "🎙️",
    title: "Realtime & Audio",
    description: "Build real-time audio apps with Gemini and other providers via the LiteLLM proxy.",
    to: "/docs/tutorials/gemini_realtime_with_audio",
  },
]}
/>

---

## What You'll Learn

LiteLLM tutorials are practical and hands-on — each one walks through a real integration or workflow you can apply directly to your project.

### Integrations
- Connect LiteLLM to coding tools like Claude Code, Cursor, and GitHub Copilot
- Use LiteLLM as a gateway for agent frameworks like OpenAI Agents SDK and Google ADK

### Operations
- Set up SSO, SCIM, and team-based access control
- Track spend, enable prompt caching, and manage budgets per key or team

### Safety & Observability
- Add guardrails and PII masking to your LLM proxy
- Route logs to Elasticsearch, Langfuse, and other observability platforms

:::tip
New to LiteLLM? Start with the [Getting Started](/) guide, then come back here for the integration tutorials that fit your stack.
:::
