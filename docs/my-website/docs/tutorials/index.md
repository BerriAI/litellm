---
title: Tutorials
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Tutorials** are step-by-step walkthroughs for integrating LiteLLM with external tools, frameworks, and services — or building complete end-to-end workflows.

> Need help choosing the right path before you start? See [Learn →](/docs/learn)

---

## Start Here

<NavigationCards
columns={3}
items={[
  {
    icon: "🐍",
    title: "SDK Quickstart",
    description: "Use this if you want to make your first LiteLLM call before following integration tutorials.",
    to: "/docs/learn/sdk_quickstart",
  },
  {
    icon: "🖥️",
    title: "Gateway Quickstart",
    description: "Use this if your tutorial depends on the LiteLLM proxy or shared infrastructure.",
    to: "/docs/learn/gateway_quickstart",
  },
  {
    icon: "📚",
    title: "Need Feature References?",
    description: "Use Guides when you know the capability you need and just want the doc for it.",
    to: "/docs/guides",
  },
]}
/>

---

## Getting Started

<NavigationCards
columns={2}
items={[
  {
    icon: "⚡",
    title: "Getting Started",
    description: "Installation, playground, text completion, and mock completions.",
    to: "/docs/tutorials/getting_started",
  },
]}
/>

---

## Integrations

<NavigationCards
columns={2}
items={[
  {
    icon: "🤖",
    title: "Agent SDKs & Frameworks",
    description: "OpenAI Agents SDK, Claude Agent SDK, Google ADK, CopilotKit, Letta, LiveKit, Instructor.",
    to: "/docs/agent_sdks",
  },
  {
    icon: "🛠️",
    title: "AI Coding Tools",
    description: "Claude Code, Cursor, GitHub Copilot, Gemini CLI, OpenCode, Qwen Code, OpenAI Codex.",
    to: "/docs/ai_tools",
  },
  {
    icon: "🐍",
    title: "Provider Tutorials",
    description: "Set up Azure OpenAI, HuggingFace, TogetherAI, local models, Gradio, and more.",
    to: "/docs/tutorials/provider_tutorials",
  },
]}
/>

---

## Production

<NavigationCards
columns={2}
items={[
  {
    icon: "🖥️",
    title: "Proxy & Gateway",
    description: "Access control, SSO, SCIM, tag management, and prompt caching.",
    to: "/docs/tutorials/proxy_gateway",
  },
  {
    icon: "🔍",
    title: "Observability & Safety",
    description: "Logging to Elasticsearch, Aporia guardrails, Presidio PII masking, and evaluation suites.",
    to: "/docs/tutorials/observability_safety",
  },
]}
/>

---

## Advanced Features

<NavigationCards
columns={2}
items={[
  {
    icon: "⚙️",
    title: "Advanced Features",
    description: "Model fallbacks, provider-specific parameters, and realtime audio.",
    to: "/docs/tutorials/advanced_features",
  },
]}
/>
