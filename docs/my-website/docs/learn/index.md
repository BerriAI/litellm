---
title: Learn LiteLLM
sidebar_label: Learn
slug: /learn
---

import NavigationCards from '@site/src/components/NavigationCards';

LiteLLM gives you a single OpenAI-compatible interface to 100+ LLM providers. Use this page to find the right starting point for where you are.

---

## 1. Make Your First Call

Set up your environment and call your first model.

<NavigationCards
columns={2}
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
]}
/>

---

## 2. Connect Your Tools

Point your existing tools and frameworks at LiteLLM.

<NavigationCards
columns={2}
items={[
  {
    icon: "🛠️",
    title: "AI Coding Tools",
    description: "Route Claude Code, Cursor, GitHub Copilot, Gemini CLI, and OpenAI Codex through LiteLLM.",
    to: "/docs/tutorials/cursor_integration",
  },
  {
    icon: "🤖",
    title: "Agent Frameworks",
    description: "Use LiteLLM with OpenAI Agents SDK, Claude Agent SDK, Google ADK, CopilotKit, and more.",
    to: "/docs/tutorials/openai_agents_sdk",
  },
  {
    icon: "🐍",
    title: "Provider Setup",
    description: "Configure Azure OpenAI, HuggingFace, TogetherAI, local models, and 100+ other providers.",
    to: "/docs/tutorials/azure_openai",
  },
  {
    icon: "💬",
    title: "Chat Interfaces",
    description: "Connect LiteLLM to OpenWebUI, LangChain, and other chat frontends.",
    to: "/docs/tutorials/openweb_ui",
  },
]}
/>

---

## 3. Configure LiteLLM

Learn how specific SDK features and proxy options work.

<NavigationCards
columns={2}
items={[
  {
    icon: "🎛️",
    title: "Fine-tuned Models",
    description: "Call fine-tuned OpenAI, Azure, and Vertex AI models with custom model names.",
    to: "/docs/guides/finetuned_models",
  },
  {
    icon: "🔒",
    title: "Security Settings",
    description: "Configure SSL certificates and HTTP proxy settings for secure deployments.",
    to: "/docs/guides/security_settings",
  },
  {
    icon: "🖥️",
    title: "Code Interpreter",
    description: "Execute Python code in a secure sandbox via OpenAI's Code Interpreter tool.",
    to: "/docs/guides/code_interpreter",
  },
  {
    icon: "⚙️",
    title: "Advanced Techniques",
    description: "Prompt caching, model fallbacks, tag management, and provider-specific parameters.",
    to: "/docs/tutorials/prompt_caching",
  },
]}
/>

---

## 4. Run in Production

Add observability, access control, and safety to your deployment.

<NavigationCards
columns={2}
items={[
  {
    icon: "🔍",
    title: "Observability & Safety",
    description: "Logging to Elasticsearch, Aporia guardrails, and Presidio PII masking.",
    to: "/docs/tutorials/elasticsearch_logging",
  },
  {
    icon: "👥",
    title: "Access Control & Administration",
    description: "SSO, SCIM, team onboarding, and spend tracking.",
    to: "/docs/tutorials/default_team_self_serve",
  },
  {
    icon: "📊",
    title: "Evaluation & Benchmarking",
    description: "Compare models, run evaluation suites, and benchmark with LM Evaluation Harness.",
    to: "/docs/tutorials/compare_llms",
  },
  {
    icon: "📈",
    title: "Integrations Overview",
    description: "Langfuse, Datadog, Prometheus, Guardrails, Prompt Management, and more.",
    to: "/docs/integrations",
  },
]}
/>
