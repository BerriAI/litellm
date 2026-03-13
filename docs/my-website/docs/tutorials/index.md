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
    title: "First LLM Call",
    description: "Install LiteLLM and make your first call to OpenAI, Anthropic, Bedrock, or any provider in under 5 minutes.",
    to: "/docs/tutorials/installation",
  },
  {
    icon: "🖥️",
    title: "Run the Proxy (LLM Gateway)",
    description: "Start a local OpenAI-compatible gateway, add a model, and call it — the fastest way to understand what the proxy does.",
    to: "/docs/tutorials/model_config_proxy",
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
    description: "Point Claude Code, Cursor, GitHub Copilot, or Gemini CLI at LiteLLM to use any model behind your favorite coding tool.",
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
    to: "/ai_tools",
  },
  {
    icon: "🔑",
    title: "Proxy & Gateway",
    description: "Config setup, SSO, SCIM provisioning, default teams, tag-based access, prompt caching.",
    to: "/docs/tutorials/model_config_proxy",
  },
  {
    icon: "💸",
    title: "Cost & Spend Tracking",
    description: "Track costs per key, team, and coding tool. Enable prompt caching to cut spend.",
    to: "/docs/tutorials/prompt_caching",
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
