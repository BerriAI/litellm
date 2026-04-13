---
title: Guides
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

**Guides** are focused references organized by the job you are trying to do with LiteLLM: make requests, use tools, handle media, manage context, or operate the gateway safely.

> New to LiteLLM or not sure whether you need the SDK or Gateway path first? Start at [Learn →](/docs/learn)

---

## Build With LiteLLM

<NavigationCards
columns={3}
items={[
  {
    icon: "⚡",
    title: "Core Requests",
    description: "Streaming, batching, structured outputs, and reasoning behavior.",
    to: "/docs/guides/core_request_response_patterns",
  },
  {
    icon: "🛠️",
    title: "Tool Calling",
    description: "Function calling, web tools, interception patterns, computer use, code interpreter, and tool-call hygiene.",
    to: "/docs/guides/tools_integrations",
  },
  {
    icon: "🖼️",
    title: "Multimodal I/O",
    description: "Vision, audio, PDFs, image generation, and video generation.",
    to: "/docs/guides/multimodal_io",
  },
  {
    icon: "📚",
    title: "Retrieval & Knowledge",
    description: "Vector stores, file search, citations, and knowledge-base routing.",
    to: "/docs/guides/retrieval_knowledge",
  },
  {
    icon: "🧠",
    title: "Prompts & Context",
    description: "Prompt caching, trimming, formatting, assistant prefill, and predicted outputs.",
    to: "/docs/guides/prompts_context",
  },
]}
/>

---

## Operate & Extend

<NavigationCards
columns={3}
items={[
  {
    icon: "🎛️",
    title: "Compatibility & Extensibility",
    description: "Provider-specific params, model aliases, fine-tuned models, and adapters.",
    to: "/docs/guides/compatibility_extensibility",
  },
  {
    icon: "🧪",
    title: "Reliability, Testing & Spend",
    description: "Retries, fallbacks, mock responses, and budget controls.",
    to: "/docs/guides/reliability_testing_spend",
  },
  {
    icon: "🔒",
    title: "Security & Network",
    description: "SSL, custom CA bundles, HTTP proxy settings, and per-service verification.",
    to: "/docs/guides/security_network",
  },
]}
/>
