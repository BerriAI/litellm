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
    icon: "🎛️",
    title: "Fine-tuned Models",
    description: "Call fine-tuned OpenAI, Azure, and Vertex AI models using LiteLLM with custom model names.",
    to: "/docs/guides/finetuned_models",
  },
  {
    icon: "🔒",
    title: "Security Settings",
    description: "Configure SSL certificates and HTTP proxy settings for secure LiteLLM deployments.",
    to: "/docs/guides/security_settings",
  },
  {
    icon: "🖥️",
    title: "Code Interpreter",
    description: "Use OpenAI's Code Interpreter tool to execute Python code in a secure, sandboxed environment.",
    to: "/docs/guides/code_interpreter",
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
