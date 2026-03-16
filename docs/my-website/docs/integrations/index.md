---
title: Integrations
sidebar_label: Overview
---

import NavigationCards from '@site/src/components/NavigationCards';

This section covers integrations with various tools and services that can be used with LiteLLM (either Proxy or SDK).

---

## Observability

Track, debug, and analyze LLM calls with observability platforms.

<NavigationCards
columns={3}
items={[
  {
    icon: "🪢",
    title: "Langfuse",
    description: "LLM observability and analytics.",
    to: "../observability/langfuse_integration",
  },
  {
    icon: "🐶",
    title: "Datadog",
    description: "Metrics, traces, and dashboards.",
    to: "../observability/datadog",
  },
  {
    icon: "📡",
    title: "OpenTelemetry",
    description: "Vendor-neutral tracing.",
    to: "../observability/opentelemetry_integration",
  },
  {
    icon: "🔗",
    title: "LangSmith",
    description: "LLM debugging and evaluation.",
    to: "../observability/langsmith_integration",
  },
  {
    icon: "🔥",
    title: "Arize / Phoenix",
    description: "ML observability and evaluation.",
    to: "../observability/arize_integration",
  },
  {
    icon: "🌀",
    title: "Helicone",
    description: "LLM request logging and analytics.",
    to: "../observability/helicone_integration",
  },
  {
    icon: "📊",
    title: "MLflow",
    description: "Experiment tracking.",
    to: "../observability/mlflow",
  },
  {
    icon: "🏋️",
    title: "Weights & Biases",
    description: "ML experiment tracking.",
    to: "../observability/wandb_integration",
  },
  {
    icon: "📉",
    title: "PostHog",
    description: "Product analytics.",
    to: "../observability/posthog_integration",
  },
]}
/>

[View all observability integrations →](../observability/callbacks.md)

---

## Alerting & Monitoring

Set up alerts, metrics collection, and infrastructure monitoring.

<NavigationCards
columns={2}
items={[
  {
    icon: "📈",
    title: "Prometheus",
    description: "Metrics collection and monitoring.",
    to: "../proxy/prometheus",
  },
  {
    icon: "🚨",
    title: "PagerDuty",
    description: "Incident response and alerting.",
    to: "../proxy/pagerduty",
  },
  {
    icon: "🔔",
    title: "Alerting",
    description: "Slack, Teams, and webhook alerts.",
    to: "../proxy/alerting",
  },
  {
    icon: "🔍",
    title: "Pyroscope",
    description: "Continuous profiling.",
    to: "../proxy/pyroscope_profiling",
  },
]}
/>

---

## Guardrails

Add safety and content filtering to LLM calls.

<NavigationCards
columns={3}
items={[
  {
    icon: "⚡",
    title: "Quick Start",
    description: "Get started with guardrails.",
    to: "../proxy/guardrails/quick_start",
  },
  {
    icon: "🛡️",
    title: "Lakera AI",
    description: "Prompt injection detection.",
    to: "../proxy/guardrails/lakera_ai",
  },
  {
    icon: "☁️",
    title: "Azure Content Safety",
    description: "Content moderation.",
    to: "../proxy/guardrails/azure_content_guardrail",
  },
  {
    icon: "🛏️",
    title: "Bedrock Guardrails",
    description: "AWS Bedrock safety.",
    to: "../proxy/guardrails/bedrock",
  },
  {
    icon: "🤖",
    title: "OpenAI Moderation",
    description: "OpenAI content policy.",
    to: "../proxy/guardrails/openai_moderation",
  },
  {
    icon: "🔐",
    title: "Secret Detection",
    description: "Prevent credential leaks.",
    to: "../proxy/guardrails/secret_detection",
  },
  {
    icon: "🕵️",
    title: "PII Masking",
    description: "Mask sensitive data.",
    to: "../proxy/guardrails/pii_masking_v2",
  },
]}
/>

[View all guardrail providers →](../proxy/guardrails/quick_start.md)

---

## Policies

Define and enforce usage policies across your LLM deployment.

<NavigationCards
columns={3}
items={[
  {
    icon: "📋",
    title: "Guardrail Policies",
    description: "Policy-based guardrail rules.",
    to: "../proxy/guardrails/guardrail_policies",
  },
  {
    icon: "🔀",
    title: "Policy Flow Builder",
    description: "Visual policy configuration.",
    to: "../proxy/guardrails/policy_flow_builder",
  },
  {
    icon: "📄",
    title: "Policy Templates",
    description: "Pre-built policy templates.",
    to: "../proxy/guardrails/policy_templates",
  },
]}
/>

---

## AI Tools

Connect LiteLLM to AI-powered coding and productivity tools.

<NavigationCards
columns={3}
items={[
  {
    icon: "💬",
    title: "OpenWebUI",
    description: "Self-hosted ChatGPT-style interface.",
    to: "../tutorials/openweb_ui",
  },
  {
    icon: "🤖",
    title: "Claude Code",
    description: "Use LiteLLM with Claude Code.",
    to: "../tutorials/claude_responses_api",
  },
  {
    icon: "🖱️",
    title: "Cursor",
    description: "AI code editor integration.",
    to: "../tutorials/cursor_integration",
  },
  {
    icon: "🐙",
    title: "GitHub Copilot",
    description: "GitHub Copilot integration.",
    to: "../tutorials/github_copilot_integration",
  },
  {
    icon: "💻",
    title: "OpenCode",
    description: "Open source coding assistant.",
    to: "../tutorials/opencode_integration",
  },
  {
    icon: "🔧",
    title: "Retool Assist",
    description: "Retool AI assistant.",
    to: "../tutorials/retool_assist",
  },
]}
/>

---

## Agent SDKs

Use LiteLLM with agent frameworks and SDKs.

<NavigationCards
columns={3}
items={[
  {
    icon: "🤖",
    title: "OpenAI Agents SDK",
    description: "Build agents with OpenAI's SDK.",
    to: "../tutorials/openai_agents_sdk",
  },
  {
    icon: "🧠",
    title: "Claude Agent SDK",
    description: "Build agents with Anthropic's SDK.",
    to: "../tutorials/claude_agent_sdk",
  },
  {
    icon: "🌐",
    title: "Google ADK",
    description: "Google Agent Development Kit.",
    to: "../tutorials/google_adk",
  },
  {
    icon: "🚀",
    title: "CopilotKit",
    description: "In-app AI copilots.",
    to: "../tutorials/copilotkit_sdk",
  },
  {
    icon: "🧬",
    title: "Letta",
    description: "Build stateful LLM agents with persistent memory.",
    to: "./letta",
  },
  {
    icon: "🎙️",
    title: "LiveKit",
    description: "Real-time voice and video AI agents.",
    to: "../tutorials/livekit_xai_realtime",
  },
]}
/>

---

## Prompt Management

Manage, version, and deploy prompts.

<NavigationCards
columns={3}
items={[
  {
    icon: "📝",
    title: "LiteLLM Prompt Management",
    description: "Built-in prompt management.",
    to: "../proxy/litellm_prompt_management",
  },
  {
    icon: "🔌",
    title: "Custom Prompt Management",
    description: "Bring your own prompt store.",
    to: "../proxy/custom_prompt_management",
  },
  {
    icon: "🔥",
    title: "Arize Phoenix Prompts",
    description: "Prompt management with Phoenix.",
    to: "../proxy/arize_phoenix_prompts",
  },
]}
/>

---

## Manage with AI Agents

Use AI agents to manage your LiteLLM deployment — create users, teams, keys, models, and more via natural language.

<NavigationCards
columns={1}
items={[
  {
    icon: "🤖",
    title: "LiteLLM Skills",
    description: "Manage LiteLLM via Claude Code — create keys, teams, models, and more using natural language commands.",
    to: "../tutorials/claude_code_skills",
  },
]}
/>
