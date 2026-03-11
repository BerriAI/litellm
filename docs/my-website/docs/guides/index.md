---
title: Guides
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Guides** walk you through specific LiteLLM features and configurations — from streaming and function calling to prompt engineering and model customization. Pick a topic below to get started.

---

## Browse by Topic

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

---

## What You'll Learn

LiteLLM guides focus on practical, real-world use of the SDK and proxy — covering the features you reach for most.

### SDK Features
- How to stream responses and handle partial output
- How to use function calling and structured JSON output
- How to apply prompt caching, trimming, and formatting
- How to leverage vision, audio, and multi-modal inputs

### AI Capabilities
- Web search and web fetch integrations
- Computer use and agentic control
- Code interpreter and Veo video generation

### Configuration & Customization
- Fine-tuning support and adapter creation
- Security settings and budget controls

:::tip
Each guide is self-contained — jump to any topic directly without needing to read the others first.
:::
