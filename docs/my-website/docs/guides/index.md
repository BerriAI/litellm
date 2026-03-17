---
title: Guides
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Guides** are focused references for specific LiteLLM SDK features and proxy configuration options. Each guide is self-contained — jump to any topic without reading the others first.

> Looking for step-by-step integration walkthroughs? See [Tutorials →](/docs/tutorials)

---

## Core Features

<NavigationCards
columns={2}
items={[
  {
    icon: "⚡",
    title: "Completion Basics",
    description: "Streaming, function calling, JSON mode, vision, audio, and more.",
    to: "/docs/guides/completion_basics",
  },
  {
    icon: "📥",
    title: "Documents, Images & Messages",
    description: "Document understanding, image generation, message trimming, and sanitization.",
    to: "/docs/guides/input_output_handling",
  },
  {
    icon: "🔄",
    title: "Prompt Optimization",
    description: "Prompt caching and prompt formatting for better performance.",
    to: "/docs/guides/prompt_optimization",
  },
  {
    icon: "🤖",
    title: "AI Capabilities",
    description: "Web search, code interpreter, knowledge base, and more.",
    to: "/docs/guides/ai_capabilities",
  },
]}
/>

---

## Configuration & Cost

<NavigationCards
columns={2}
items={[
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