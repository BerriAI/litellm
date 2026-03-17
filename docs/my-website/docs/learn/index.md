---
title: Learn LiteLLM
sidebar_label: Learn
slug: /learn
---

import NavigationCards from '@site/src/components/NavigationCards';

LiteLLM gives you a single OpenAI-compatible interface to 100+ LLM providers. Use this page to find the right starting point for where you are.

---

## Get Started

Set up your environment and make your first LLM call.

<NavigationCards
columns={3}
items={[
  {
    icon: "⚡",
    title: "Set Up Your Environment",
    description: "Get API keys and configure your environment before making your first LLM call.",
    to: "/docs/tutorials/installation",
  },
  {
    icon: "🧪",
    title: "Build an LLM Playground",
    description: "Compare multiple LLM providers side by side in under 10 minutes with Streamlit.",
    to: "/docs/tutorials/first_playground",
  },
  {
    icon: "📞",
    title: "Make Your First Call",
    description: "Use the completion() function to call OpenAI, Anthropic, Vertex, or 100+ providers — same interface, one line of code.",
    to: "/docs/#quick-start",
  },
]}
/>

[View all getting started tutorials →](/docs/tutorials/getting_started)

---

## Guides

Focused references for SDK features and proxy configuration. Each guide is self-contained.

<NavigationCards
columns={3}
items={[
  {
    icon: "⚡",
    title: "Completion Basics",
    description: "Streaming, function calling, JSON mode, vision, audio, and more.",
    to: "/docs/guides/completion_basics",
  },
  {
    icon: "🎛️",
    title: "Models & Configuration",
    description: "Fine-tuned models, security settings, and adapters.",
    to: "/docs/guides/models_configuration",
  },
  {
    icon: "💰",
    title: "Budgets & Cost",
    description: "Set spend limits and track costs across teams and deployments.",
    to: "/docs/guides/budgets_cost",
  },
]}
/>

[View all guides →](/docs/guides)

---

## Tutorials

Step-by-step walkthroughs for integrating LiteLLM with external tools and services.

<NavigationCards
columns={3}
items={[
  {
    icon: "🤖",
    title: "Agent SDKs & Frameworks",
    description: "OpenAI Agents SDK, Claude Agent SDK, Google ADK, CopilotKit, and more.",
    to: "/docs/agent_sdks",
  },
  {
    icon: "🛠️",
    title: "AI Coding Tools",
    description: "Claude Code, Cursor, GitHub Copilot, Gemini CLI, OpenCode, and more.",
    to: "/docs/ai_tools",
  },
  {
    icon: "🐍",
    title: "Provider Tutorials",
    description: "Set up Azure OpenAI, HuggingFace, TogetherAI, local models, and more.",
    to: "/docs/tutorials/provider_tutorials",
  },
]}
/>

[View all tutorials →](/docs/tutorials)

---

## Production

Add observability, access control, and safety to your deployment. See the [Production section](/docs/tutorials#production) in Tutorials for Observability & Safety, Proxy & Gateway, and more.
