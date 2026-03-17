---
title: Learn LiteLLM
sidebar_label: Learn
slug: /learn
---

import NavigationCards from '@site/src/components/NavigationCards';

LiteLLM gives you one OpenAI-compatible interface for 100+ LLM providers. Start with the path that matches your setup.

---

## Start Here

Pick one path first.

<NavigationCards
columns={2}
items={[
  {
    icon: "🐍",
    title: "SDK Quickstart",
    description: "Use LiteLLM directly in application code.",
    listDescription: [
      "Install",
      "First request",
      "Next SDK features",
    ],
    to: "/docs/learn/sdk_quickstart",
  },
  {
    icon: "🖥️",
    title: "Gateway Quickstart",
    description: "Run LiteLLM as a shared gateway.",
    listDescription: [
      "Start proxy",
      "Add models and keys",
      "Connect clients",
    ],
    to: "/docs/learn/gateway_quickstart",
  },
]}
/>

---

## Common Tasks

Jump to a specific task.

<NavigationCards
columns={3}
items={[
  {
    icon: "⚡",
    title: "Stream Responses",
    description: "Return tokens as they are generated.",
    to: "/docs/completion/stream",
  },
  {
    icon: "🧰",
    title: "Use Tools",
    description: "Add function calling to your app.",
    to: "/docs/completion/function_call",
  },
  {
    icon: "🔀",
    title: "Add Routing",
    description: "Retries, fallbacks, and load balancing.",
    to: "/docs/routing",
  },
  {
    icon: "🔑",
    title: "Set Up Keys",
    description: "Gateway auth, virtual keys, and access control.",
    to: "/docs/proxy/virtual_keys",
  },
  {
    icon: "📈",
    title: "Add Logging",
    description: "Capture request logs and spend data.",
    to: "/docs/proxy/logging",
  },
  {
    icon: "🌐",
    title: "Choose A Provider",
    description: "Find provider-specific auth and params.",
    to: "/docs/providers",
  },
]}
/>

---

## Explore

Use these when you want examples or tool-specific walkthroughs.

<NavigationCards
columns={3}
items={[
  {
    icon: "🧪",
    title: "Playground",
    description: "Compare providers side by side.",
    to: "/docs/tutorials/first_playground",
  },
  {
    icon: "🤖",
    title: "Agent SDKs",
    description: "OpenAI Agents SDK, Claude Agent SDK, ADK, and more.",
    to: "/docs/agent_sdks",
  },
  {
    icon: "🛠️",
    title: "AI Coding Tools",
    description: "Claude Code, Cursor, Copilot, Gemini CLI, and more.",
    to: "/docs/ai_tools",
  },
]}
/>

---

## Docs Map

Use these when you already know the type of doc you want.

<NavigationCards
columns={3}
items={[
  {
    icon: "📚",
    title: "Guides",
    description: "Feature reference.",
    to: "/docs/guides",
  },
  {
    icon: "🛠️",
    title: "Tutorials",
    description: "Step-by-step integrations.",
    to: "/docs/tutorials",
  },
  {
    icon: "🌐",
    title: "Providers",
    description: "Provider-specific auth and params.",
    to: "/docs/providers",
  },
]}
/>

Not sure where to start? Use [SDK Quickstart](/docs/learn/sdk_quickstart) for app code or [Gateway Quickstart](/docs/learn/gateway_quickstart) for shared infrastructure.
