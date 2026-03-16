---
title: Tutorials
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Tutorials** are step-by-step walkthroughs for integrating LiteLLM with external tools, frameworks, and services — or building complete end-to-end workflows.

> Need help with a specific LiteLLM SDK feature or proxy config? See [Guides →](/docs/guides)

---

## Getting Started

<NavigationCards
columns={2}
items={[
  {
    icon: "⚡",
    title: "Set Up Your Environment",
    description: "Get your API keys for OpenAI, Cohere, and AI21 — no waitlist required — and configure your environment before making your first LLM call.",
    to: "/docs/tutorials/installation",
  },
  {
    icon: "🧪",
    title: "Build an LLM Playground",
    description: "Create a Streamlit playground to evaluate and compare multiple LLM providers side by side in under 10 minutes.",
    to: "/docs/tutorials/first_playground",
  },
]}
/>

---

## Browse All Tutorials

<NavigationCards
columns={2}
items={[
  {
    icon: "🤖",
    title: "Agent Frameworks",
    description: "OpenAI Agents SDK, Claude Agent SDK, Google ADK, CopilotKit, Letta, LiveKit, Instructor.",
    to: "/docs/tutorials/openai_agents_sdk",
  },
  {
    icon: "🛠️",
    title: "AI Coding Tools",
    description: "Claude Code, Cursor, GitHub Copilot, Gemini CLI, OpenCode, Qwen Code, OpenAI Codex.",
    to: "/docs/tutorials/cursor_integration",
  },
  {
    icon: "🐍",
    title: "Provider Setup",
    description: "Step-by-step setup for Azure OpenAI, HuggingFace, TogetherAI, local models, and more.",
    to: "/docs/tutorials/azure_openai",
  },
  {
    icon: "📊",
    title: "Evaluation & Benchmarking",
    description: "Compare LLMs side by side, run evaluation suites, and benchmark with LM Evaluation Harness.",
    to: "/docs/tutorials/compare_llms",
  },
  {
    icon: "🔍",
    title: "Observability & Safety",
    description: "Send logs to Elasticsearch and other platforms. Add Aporia guardrails and Presidio PII masking.",
    to: "/docs/tutorials/elasticsearch_logging",
  },
  {
    icon: "👥",
    title: "Access Control & Administration",
    description: "Onboard users to default teams, configure Microsoft SSO, set up SCIM, and track spend.",
    to: "/docs/tutorials/default_team_self_serve",
  },
  {
    icon: "🎙️",
    title: "Realtime & Audio",
    description: "Build real-time audio apps with Gemini and xAI via the LiteLLM proxy.",
    to: "/docs/tutorials/gemini_realtime_with_audio",
  },
  {
    icon: "⚙️",
    title: "Advanced Techniques",
    description: "Prompt caching, model fallbacks, tag management, provider-specific parameters, and more.",
    to: "/docs/tutorials/prompt_caching",
  },
]}
/>
