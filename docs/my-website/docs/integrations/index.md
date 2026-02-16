# Integrations

LiteLLM has a broad integration ecosystem across SDK and Proxy workflows. This page gives a complete taxonomy and quick entry points for each category.

## Compatibility Legend

- `SDK`: Works directly with LiteLLM Python SDK (for example `litellm.completion()`).
- `Proxy`: Requires LiteLLM Proxy.
- `SDK + Proxy`: Available in both SDK and Proxy setups.

## Integration Categories

| Category | Approx. Count | Primary Docs | Compatibility |
| --- | --- | --- | --- |
| AI Tools and Frameworks | 20+ | [AI Tools tutorials](../tutorials/openweb_ui.md), [Agent SDKs](../tutorials/claude_agent_sdk.md), [Letta](./letta.md) | Proxy |
| Observability and Logging | 40+ | [Observability index](../observability/callbacks.md) | SDK + Proxy |
| Guardrails and Safety | 30+ | [Guardrails quick start](../proxy/guardrails/quick_start.md) | Proxy |
| Alerting and Monitoring | 4+ | [Alerting](../proxy/alerting.md), [Prometheus](../proxy/prometheus.md), [PagerDuty](../proxy/pagerduty.md) | Proxy |
| Billing and Cost Tracking | 2+ | [Lago](../observability/lago.md), [OpenMeter](../observability/openmeter.md) | SDK + Proxy |
| Prompt Management (Beta) | 5+ | [Prompt management](../proxy/prompt_management.md) | Proxy |
| Infrastructure and Storage | 3+ | [GCS Bucket logging](../observability/gcs_bucket_integration.md), [FOCUS exports](../observability/focus.md) | SDK + Proxy |

## Popular Integrations by Category

### AI Tools and Frameworks (Proxy)

- IDEs and coding tools: [Cursor](../tutorials/cursor_integration.md), [OpenCode](../tutorials/opencode_integration.md), [GitHub Copilot](../tutorials/github_copilot_integration.md), [OpenAI Codex](../tutorials/openai_codex.md)
- AI assistants and CLIs: [Gemini CLI](../tutorials/litellm_gemini_cli.md), [Qwen Code CLI](../tutorials/litellm_qwen_code_cli.md), [Claude Code](../tutorials/claude_responses_api.md)
- UIs and agent frameworks: [OpenWebUI](../tutorials/openweb_ui.md), [Letta](./letta.md), [Claude Agent SDK](../tutorials/claude_agent_sdk.md)

### Observability and Logging (SDK + Proxy)

- Analytics and tracing: [Langfuse](../observability/langfuse_integration.md), [LangSmith](../observability/langsmith_integration.md), [Arize](../observability/arize_integration.md), [Braintrust](../observability/braintrust.md), [Opik](../observability/opik_integration.md), [MLflow](../observability/mlflow.md), [PostHog](../observability/posthog_integration.md)
- Monitoring and telemetry: [Datadog](../observability/datadog.md), [OpenTelemetry](../observability/opentelemetry_integration.md), [AgentOps](../observability/agentops_integration.md)
- Full list: use the `Observability` section in the integrations sidebar.

### Guardrails and Safety (Proxy)

- Prompt and content security: [Prompt Security](../proxy/guardrails/prompt_security.md), [Prompt Injection Detection](../proxy/guardrails/prompt_injection.md), [Secret Detection](../proxy/guardrails/secret_detection.md), [PII Masking](../proxy/guardrails/pii_masking_v2.md)
- Guardrail providers: [Lakera AI](../proxy/guardrails/lakera_ai.md), [Guardrails AI](../proxy/guardrails/guardrails_ai.md), [Bedrock Guardrails](../proxy/guardrails/bedrock.md), [Azure Content Safety](../proxy/guardrails/azure_content_guardrail.md), [HiddenLayer](../proxy/guardrails/hiddenlayer.md), [PANW Prisma AIRS](../proxy/guardrails/panw_prisma_airs.md)
- Full list: use the `Guardrails` section in the integrations sidebar.

### Prompt Management (Beta, Proxy)

- [LiteLLM Prompt Management](../proxy/litellm_prompt_management.md)
- [Custom Prompt Management](../proxy/custom_prompt_management.md)
- [Native LiteLLM Prompt Registry](../proxy/native_litellm_prompt.md)
- [Arize Phoenix Prompts](../proxy/arize_phoenix_prompts.md)

## SDK vs Proxy Quick Guide

- Start with SDK integrations if you are embedding LiteLLM directly into an app.
- Use Proxy integrations for centralized policy enforcement, guardrails, team governance, and operational controls.
- Many observability integrations can be used in both modes.
