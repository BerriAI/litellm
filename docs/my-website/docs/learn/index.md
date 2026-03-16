---
title: Learn LiteLLM
sidebar_label: Learn
slug: /learn
---

import NavigationCards from '@site/src/components/NavigationCards';

Learn how to use LiteLLM — from reference guides covering specific SDK features and configurations to hands-on tutorials that walk you through real integrations end to end.

---

## Guides

Deep dives into specific LiteLLM features and configurations — pick a topic and go.

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

## Tutorials

End-to-end walkthroughs you can follow step-by-step — no prior LiteLLM knowledge required.

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
