---
title: Tutorials
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**New to LiteLLM? Start here.** Tutorials are end-to-end walkthroughs you can follow from zero to working integration — no prior LiteLLM knowledge required.

---

## Start Here

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
  {
    icon: "🤖",
    title: "Use with an Agent SDK",
    description: "Connect LiteLLM to OpenAI Agents SDK, Claude Agent SDK, Google ADK, or CopilotKit.",
    to: "/docs/tutorials/openai_agents_sdk",
  },
  {
    icon: "🛠️",
    title: "Use with a Coding Tool",
    description: "Point Cursor, GitHub Copilot, Gemini CLI, or Claude Code at LiteLLM to use any model behind your favorite coding tool.",
    to: "/docs/tutorials/cursor_integration",
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
    title: "Agent SDKs & Frameworks",
    description: "OpenAI Agents SDK, Claude Agent SDK, CopilotKit, Google ADK, LiveKit, Instructor.",
    to: "/docs/tutorials/openai_agents_sdk",
  },
  {
    icon: "🛠️",
    title: "AI Coding Tools",
    description: "Claude Code, Cursor, GitHub Copilot, Gemini CLI, OpenCode, Qwen Code, OpenAI Codex.",
    to: "/docs/tutorials/cursor_integration",
  },
  {
    icon: "👥",
    title: "User & Team Onboarding",
    description: "Onboard users to default teams, enable self-serve AI exploration, and manage access at scale.",
    to: "/docs/tutorials/default_team_self_serve",
  },
  {
    icon: "💸",
    title: "Cost & Spend Tracking",
    description: "Track usage and costs for coding tools like Claude Code, Roo Code, Gemini CLI, and OpenAI Codex.",
    to: "/docs/tutorials/cost_tracking_coding",
  },
  {
    icon: "🛡️",
    title: "Guardrails & PII Masking",
    description: "Aporia guardrails, Presidio PII masking, and content safety — step by step.",
    to: "/docs/tutorials/litellm_proxy_aporia",
  },
  {
    icon: "📊",
    title: "Logging & Observability",
    description: "Send logs to Elasticsearch, Langfuse, and other observability platforms.",
    to: "/docs/tutorials/elasticsearch_logging",
  },
  {
    icon: "🐍",
    title: "Provider Walkthroughs",
    description: "Step-by-step guides for Azure OpenAI, HuggingFace, TogetherAI, model fallbacks, and more.",
    to: "/docs/tutorials/azure_openai",
  },
  {
    icon: "🎙️",
    title: "Realtime & Audio",
    description: "Build real-time audio apps with Gemini and xAI via the LiteLLM proxy.",
    to: "/docs/tutorials/gemini_realtime_with_audio",
  },
]}
/>
