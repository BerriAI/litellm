---
title: Learn LiteLLM
sidebar_label: Learn
slug: /learn
---

import NavigationCards from '@site/src/components/NavigationCards';

Learn how to use LiteLLM — from hands-on tutorials that walk you through real integrations to reference guides covering every SDK feature and configuration option.

---

## Tutorials

End-to-end walkthroughs you can follow step-by-step — no prior LiteLLM knowledge required.

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

## Guides

Deep dives into specific LiteLLM features and configurations — pick a topic and go.

<NavigationCards
columns={2}
items={[
  {
    icon: "💬",
    title: "Completion & Messaging",
    description: "Streaming, function calling, JSON mode, vision, audio, batching, and provider-specific params.",
    to: "/docs/completion/stream",
  },
  {
    icon: "✍️",
    title: "Prompt Engineering",
    description: "Prompt caching, formatting, message trimming, model aliases, mock requests, and reliable completions.",
    to: "/docs/completion/prompt_caching",
  },
  {
    icon: "🤖",
    title: "AI Capabilities",
    description: "Web search, web fetch, computer use, knowledge bases, code interpreter, and video generation.",
    to: "/docs/completion/web_search",
  },
  {
    icon: "⚙️",
    title: "Models & Customization",
    description: "Fine-tuned models, security settings, budget management, and custom adapters.",
    to: "/docs/guides/finetuned_models",
  },
]}
/>
