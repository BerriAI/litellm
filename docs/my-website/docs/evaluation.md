---
id: evaluation
title: Evaluation Overview
sidebar_label: Evaluation
description: Comprehensive evaluation requirements mapped to LiteLLM features, documentation, and capabilities.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

This page consolidates evaluation requirements and maps them to corresponding LiteLLM capabilities, documentation links, and examples.

<Tabs>
<TabItem value="core-platform" label="Core Platform" default>

## Core Platform

### Caching

| Title | Description | Documentation |
|-------|-------------|---------------|
| Prompt Caching | Cache repeated prompts to reduce cost and latency across providers. | [View Docs](https://docs.litellm.ai/docs/proxy/caching) |
| Response Caching | Cache model responses with configurable TTL and cache-bypass options. | [View Docs](https://docs.litellm.ai/docs/proxy/caching) |
| Per-Route Cache TTLs | Define different TTLs per route or model for prompt/response cache entries. | [View Docs](https://docs.litellm.ai/docs/proxy/caching) |
| Cache Bypass Controls | Allow clients or rules to skip cache reads/writes for sensitive calls. | [View Docs](https://docs.litellm.ai/docs/proxy/caching) |
| Semantic/Content-Aware Caching | Reduce re-computation by caching semantically-similar requests. | [View Docs](https://docs.litellm.ai/docs/proxy/caching) |
| Cache Invalidation Controls | Clear stale cache entries during rollouts or policy changes. | [View Docs](https://docs.litellm.ai/docs/proxy/caching) |

### Routing

| Title | Description | Documentation |
|-------|-------------|---------------|
| Unified API Gateway for Multiple LLM Providers | Single endpoint to access local-hosted and multi-cloud LLMs across providers. | [View Docs](https://docs.litellm.ai/docs/providers) |
| Supported Endpoints Catalog | **Core:** [`/chat/completions`](https://docs.litellm.ai/docs/proxy/user_keys#chatcompletions) • [`/completions`](https://docs.litellm.ai/docs/text_completion) • [`/v1/messages`](https://docs.litellm.ai/docs/providers/anthropic)<br/>**Audio:** [`/audio/transcriptions`](https://docs.litellm.ai/docs/audio_transcription) • [`/audio/speech`](https://docs.litellm.ai/docs/text_to_speech)<br/>**Images:** [`/images/generations`](https://docs.litellm.ai/docs/image_generation) • [`/images/edits`](https://docs.litellm.ai/docs/image_edits)<br/>**Embeddings & Search:** [`/embeddings`](https://docs.litellm.ai/docs/proxy/embedding) • [`/rerank`](https://docs.litellm.ai/docs/rerank) • [`/vector_stores`](https://docs.litellm.ai/docs/vector_stores/create) • [`/search`](https://docs.litellm.ai/docs/search)<br/>**Assistants & Batch:** [`/assistants`](https://docs.litellm.ai/docs/assistants) • [`/threads`](https://docs.litellm.ai/docs/assistants) • [`/batches`](https://docs.litellm.ai/docs/batches) • [`/responses`](https://docs.litellm.ai/docs/response_api)<br/>**Other:** [`/fine_tuning/jobs`](https://docs.litellm.ai/docs/fine_tuning) • [`/moderations`](https://docs.litellm.ai/docs/moderation) • [`/ocr`](https://docs.litellm.ai/docs/ocr) • [`/mcp/tools`](https://docs.litellm.ai/docs/mcp) • [`/realtime`](https://docs.litellm.ai/docs/realtime) • [`/files`](https://docs.litellm.ai/docs/files_endpoints) | [**View Entire List**](https://docs.litellm.ai/docs/supported_endpoints) |
| Advanced Routing Strategies | Route based on budget, use-case, availability, rate limits, and lowest cost. | [View Docs](https://docs.litellm.ai/docs/routing#advanced---routing-strategies-%EF%B8%8F) |
| Reliable Completions | Provider retries and fallbacks for resilient completions with exponential backoff and jitter. | [View Docs](https://docs.litellm.ai/docs/completion/reliable_completions) |
| Cost-Based Routing | Automatically select lowest-cost viable provider/model. | [View Docs](https://docs.litellm.ai/docs/routing#advanced---routing-strategies-%EF%B8%8F) |
| Rate-Limit-Aware Routing | Choose providers based on available request/token headroom, with fallback to alternate models when nearing RPM/TPM caps. | [View Docs](https://docs.litellm.ai/docs/routing#advanced---routing-strategies-%EF%B8%8F) |
| Availability-Based Routing | Reroute during provider outages to sustain uptime. | [View Docs](https://docs.litellm.ai/docs/completion/reliable_completions) |
| Budget-Aware Routing | Select route based on remaining budget headroom, with fallback based on per-team/key remaining budget. | [View Docs](https://docs.litellm.ai/docs/routing#advanced---routing-strategies-%EF%B8%8F) |
| Latency-Aware Routing | Prefer providers with lower observed latency. | [View Docs](https://docs.litellm.ai/docs/routing#advanced---routing-strategies-%EF%B8%8F) |
| Error-Rate-Aware Routing | Avoid providers showing elevated error rates. | [View Docs](https://docs.litellm.ai/docs/routing#advanced---routing-strategies-%EF%B8%8F) |

</TabItem>

<TabItem value="cost-efficiency" label="Cost & Efficiency">

## Cost & Efficiency Optimization

### Budgets

| Title | Description | Documentation |
|-------|-------------|---------------|
| Usage & Cost Tracking | Track spend and tokens per model, key, user, team, and environment. | [View Docs](https://docs.litellm.ai/docs/proxy/cost_tracking) |
| Budget Enforcement Policies | Set and enforce budgets for teams, users, and API keys. | [View Docs](https://docs.litellm.ai/docs/proxy/billing) |
| Budget Refresh Schedules | Support monthly/daily automatic budget refresh windows with configurable duration (seconds, minutes, hours, days). | [View Docs](https://docs.litellm.ai/docs/proxy/users#reset-budgets) |
| Per-Key Budgets | Budget caps for individual API keys. | [View Docs](https://docs.litellm.ai/docs/proxy/users#virtual-key) |
| Per-User Budgets | Limit spend at the user account level. | [View Docs](https://docs.litellm.ai/docs/proxy/users#internal-user) |
| Per-Model Budgets | Assign budgets by model family or provider. | [View Docs](https://docs.litellm.ai/docs/proxy/billing) |
| Team Budgets | Assign budgets and quotas scoped to a team. | [View Docs](https://docs.litellm.ai/docs/proxy/team_budgets) |

</TabItem>

<TabItem value="enterprise" label="Enterprise">

## Enterprise

### Alerting

| Title | Description | Documentation |
|-------|-------------|---------------|
| LLM Performance Alerts | Detect model/provider outages, slow API calls (exceeding `alerting_threshold`), hanging requests, failed API calls, and sudden error spikes. | [View Docs](https://docs.litellm.ai/docs/proxy/alerting) |
| Budget & Spend Alerts | Daily/weekly spend summaries per team or tag, soft budget threshold notifications at X% consumption, and budget limit alerts. | [View Docs](https://docs.litellm.ai/docs/proxy/alerting) |
| Daily Health Reports | Automated daily status summaries including top 5 slowest deployments, top 5 deployments with most failed requests, and system health metrics. | [View Docs](https://docs.litellm.ai/docs/proxy/alerting) |

### Deployment

| Title | Description | Documentation |
|-------|-------------|---------------|
| Deployment Options | Deploy via Docker, Kubernetes, Helm, Terraform, AWS CloudFormation, Google Cloud Run, Render, Railway, or Docker Compose with support for database, Redis, and production-ready configurations. | [View Docs](https://docs.litellm.ai/docs/proxy/deploy) |
| Control Plane & Data Plane | Separate planes for global management and regional execution with multi-region/multi-cloud failover for high availability. | [View Docs](https://docs.litellm.ai/docs/proxy/control_plane_and_data_plane) |
| Timeout Configuration | Global and per-model/provider timeouts to avoid hung requests. | [View Docs](https://docs.litellm.ai/docs/proxy/timeout) |
| Concurrent Usage Testing | Simulate load to validate throughput targets. | [View Docs](https://docs.litellm.ai/docs/load_test_advanced) |

</TabItem>

<TabItem value="observability" label="Observability">

## Monitoring, Logging & Observability

### Integrations

| Title | Description | Documentation |
|-------|-------------|---------------|
| Datadog Integration | Publish metrics and traces to Datadog for dashboards and alerts, including pre-built panels for latency, errors, and usage. | [View Docs](https://docs.litellm.ai/docs/observability/datadog) |
| Prometheus Metrics | Expose proxy metrics for scrape and alert rules. | [View Docs](https://docs.litellm.ai/docs/proxy/prometheus) |
| SIEM & Tooling Integrations | Forward logs and events to external observability stacks. | [View Docs](https://docs.litellm.ai/docs/integrations) |

### Logging

| Title | Description | Documentation |
|-------|-------------|---------------|
| Request/Response Logging | Enable or disable logging to capture payloads, identifiers, and outcomes for auditing with structured logging fields (user_id, call_id, model, tokens, latency). | [View Docs](https://docs.litellm.ai/docs/proxy/logging) |
| Logging Payload Specification | Reference documentation for all available fields and data captured in LiteLLM logging payloads. | [View Docs](https://docs.litellm.ai/docs/proxy/logging_spec) |
| Custom Callbacks | Integrate custom logging hooks to process and forward structured logs to external systems (SIEM, observability platforms, databases) with real-time token usage, cost tracking, and event handling. | [View Docs](https://docs.litellm.ai/docs/observability/callbacks) |
| PII-Safe Logging Practices | Use Presidio guardrails to mask or block PII, PHI, and sensitive data before logging. | [View Docs](https://docs.litellm.ai/docs/proxy/guardrails/pii_masking_v2) |

### Metrics & Dashboards

| Title | Description | Documentation |
|-------|-------------|---------------|
| Latency, Error Rate, Token Usage | Track p50/p95 latency, error counts, and token consumption with dashboards for latency percentiles over time. | [View Docs](https://docs.litellm.ai/docs/proxy/prometheus) |
| Request Throughput Metrics | Dashboard visualizations showing the number of requests processed over time, broken down by API route, provider, or model. | [View Docs](https://docs.litellm.ai/docs/proxy/prometheus) |
| Error Rate Panels | Track HTTP status codes and failures. | [View Docs](https://docs.litellm.ai/docs/proxy/prometheus) |
| Budget & Spend Metrics | Monitor spend/budget usage per team/key with metrics and visualize budget burn-down per team/key. | [View Docs](https://docs.litellm.ai/docs/proxy/prometheus) |
| Daily Summary Reports | Automated daily summaries of usage and health. | [View Docs](https://docs.litellm.ai/docs/proxy/prometheus) |
| Cache Metrics | Export cache hit/miss metrics for dashboards. | [View Docs](https://docs.litellm.ai/docs/proxy/caching) |

</TabItem>

<TabItem value="performance" label="Performance">

## Performance

### Reliability

| Title | Description | Documentation |
|-------|-------------|---------------|
| Production Best Practices | Production deployment recommendations including configuration, machine specifications, Redis optimization, worker management, and database connection pooling. | [View Docs](https://docs.litellm.ai/docs/proxy/prod) |
| Gateway Overhead P50/P90/P99 | LiteLLM proxy adds minimal latency overhead compared to direct provider API calls. | [View Docs](https://docs.litellm.ai/docs/benchmarks) |
| Provider Latency Comparison | Compare observed latencies across providers. | [View Docs](https://docs.litellm.ai/docs/benchmarks) |
| Load Test Toolkit | Use mock requests and scenarios to validate SLOs. | [View Docs](https://docs.litellm.ai/docs/load_test_advanced) |

</TabItem>

<TabItem value="security" label="Security & Compliance">

## Security & Compliance

### Identity

| Title | Description | Documentation |
|-------|-------------|---------------|
| RBAC & Team Segmentation | Enforce permissions by roles; segment teams and models. | [View Docs](https://docs.litellm.ai/docs/proxy/access_control) |
| User/Team Rate Limits | Set RPM/TPM per user/team/model/key. | [View Docs](https://docs.litellm.ai/docs/proxy/users#set-rate-limits) |
| SSO & OAuth | Integrate identity providers via SSO/OAuth. | [View Docs](https://docs.litellm.ai/docs/proxy/custom_sso) |
| MCP Permission Management | Constrain model control permissions by user/team. | [View Docs](https://docs.litellm.ai/docs/mcp#mcp-permission-management) |
| Virtual Keys & Rotation | Create, rotate, and revoke virtual keys at scale with configurable rotation strategy (schedule/events). | [View Docs](https://docs.litellm.ai/docs/proxy/virtual_keys) |
| Team-Scoped Keys | Create keys scoped to specific teams for isolation. | [View Docs](https://docs.litellm.ai/docs/proxy/team_budgets) |
| TLS Encryption Policy | TLS 1.2+ for secure transport between clients and gateway for all inbound connections. | [View Docs](https://docs.litellm.ai/docs/data_security) |
| Self-Hosted Data Policy | Ensure no persistent storage of prompts/responses when self-hosted. | [View Docs](https://docs.litellm.ai/docs/data_security) |
| IP Allow/Deny Lists | Enforce network-level access using IP-based policies and prevent lateral movement between teams and models. | [View Docs](https://docs.litellm.ai/docs/proxy/ip_address) |
| AWS Secrets Manager | Store and rotate provider secrets via AWS Secrets Manager with automation. | [View Docs](https://docs.litellm.ai/docs/secret#aws-secret-manager) |

### Guardrails

| Title | Description | Documentation |
|-------|-------------|---------------|
| Guardrails Suite | Configure content filtering, prompt injection detection, PII masking, and security guardrails with support for multiple providers (Presidio, Lakera, Aporia, Bedrock, Pangea, and more). | [View Docs](https://docs.litellm.ai/docs/proxy/guardrails/quick_start) |
| PII/PHI Masking | Mask or block personally identifiable information and protected health information using Presidio with configurable entity types and actions. | [View Docs](https://docs.litellm.ai/docs/proxy/guardrails/pii_masking_v2) |
| Prompt Injection Detection | Detect and block prompt injection attacks and jailbreak attempts using similarity checks, LLM API calls, or third-party services. | [View Docs](https://docs.litellm.ai/docs/proxy/guardrails/prompt_injection) |
| Secret Detection | Detect and mask secrets, API keys, and sensitive credentials in prompts and responses. | [View Docs](https://docs.litellm.ai/docs/proxy/guardrails/secret_detection) |

</TabItem>
</Tabs>
