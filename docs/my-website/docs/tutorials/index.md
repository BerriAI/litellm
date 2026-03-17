---
title: Tutorials
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Tutorials** are step-by-step walkthroughs for integrating LiteLLM with external tools, frameworks, and services — or building complete end-to-end workflows.

> Need help choosing the right path before you start? See [Learn →](/docs/learn)

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
    title: "Python SDK",
    description: "Gradio, fallbacks, provider-specific params — no proxy required.",
    to: "/docs/tutorials/python_sdk",
  },
  {
    icon: "🔌",
    title: "Provider Setup",
    description: "Azure OpenAI, HuggingFace, TogetherAI, local models, and more.",
    to: "/docs/tutorials/provider_tutorials",
  },
]}
/>

---

## Proxy

<NavigationCards
columns={2}
items={[
  {
    icon: "👥",
    title: "Proxy: Admin & Access",
    description: "User and team management, SSO, SCIM, and routing rules.",
    to: "/docs/tutorials/proxy_admin_access",
  },
  {
    icon: "🛡️",
    title: "Proxy: Features & Safety",
    description: "Prompt caching, passthrough APIs, realtime, guardrails, and PII masking.",
    to: "/docs/tutorials/proxy_features_safety",
  },
]}
/>

---

## Observability & Evaluation

<NavigationCards
columns={2}
items={[
  {
    icon: "🔍",
    title: "Observability & Evaluation",
    description: "Logging to Elasticsearch, benchmarking, and evaluation suites.",
    to: "/docs/tutorials/observability_evaluation",
  },
]}
/>
