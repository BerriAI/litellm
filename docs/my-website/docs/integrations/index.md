# Integrations

LiteLLM’s integration surface is split between SDK-only capabilities and proxy-centric capabilities. This page is the index for the full integration ecosystem.

## What is this page for?

Use this page to quickly identify integrations by intent, then open the integration’s dedicated doc for configuration and examples.

## How to start fast

- For app-level or single-service integrations (direct SDK usage), use `SDK` entries.
- For enterprise control planes, policy, or redaction/governance, start with `Proxy` entries.
- For observability and usage tracing, prefer entries marked `SDK + Proxy` to keep parity across direct calls and routed calls.

## Compatibility Legend

- `SDK`: Works with LiteLLM Python SDK (`litellm.completion()`, `chat.completions`, etc.)
- `Proxy`: Requires LiteLLM Proxy
- `SDK + Proxy`: Available in both SDK and Proxy modes

## Integration Categories

| Category | Approx. Count | Compatibility |
| --- | --- | --- |
| AI Tools and Frameworks | 20+ | Proxy |
| Observability and Logging | 40+ | SDK + Proxy |
| Guardrails and Safety | 30+ | Proxy |
| Alerting and Monitoring | 4+ | Proxy |
| Billing and Cost Tracking | 2+ | SDK + Proxy |
| Prompt Management (Beta) | 5+ | Proxy |
| Infrastructure and Storage | 3+ | SDK + Proxy |

## AI Tools and Frameworks (Proxy)

- `openweb_ui` — [OpenWebUI](../tutorials/openweb_ui.md)
- `agent_sdk` — [Claude Agent SDK](../tutorials/claude_agent_sdk.md)
- `assistant_clients` — [Cursor](../tutorials/cursor_integration.md), [OpenCode](../tutorials/opencode_integration.md), [OpenAI Codex](../tutorials/openai_codex.md)
- `assistant_code_tools` — [GitHub Copilot](../tutorials/github_copilot_integration.md), [Gemini CLI](../tutorials/litellm_gemini_cli.md), [Qwen Code CLI](../tutorials/litellm_qwen_code_cli.md)
- `assistant_platforms` — [Claude Code](../tutorials/claude_responses_api.md), [Claude Code MCP](../tutorials/claude_mcp.md), [Claude Non-Anthropic Models](../tutorials/claude_non_anthropic_models.md)
- `agent_integrations` — [LiveKit Realtime](../tutorials/livekit_xai_realtime.md), [Google ADK](../tutorials/google_adk.md), [Claude Agent SDK](../tutorials/claude_agent_sdk.md)
- `letta` — [Letta](./letta.md)

## Observability and Logging (SDK + Proxy)

- `logs_analytics` — [Langfuse](../observability/langfuse_integration.md), [LangSmith](../observability/langsmith_integration.md), [Arize](../observability/arize_integration.md), [Braintrust](../observability/braintrust.md), [Opik](../observability/opik_integration.md), [PostHog](../observability/posthog_integration.md), [MLflow](../observability/mlflow.md), [Argilla](../observability/argilla.md), [Langtrace](../observability/langtrace_integration.md), [PromptLayer](../observability/promptlayer_integration.md)
- `telemetry` — [Telemetry](../observability/telemetry.md), [Datadog](../observability/datadog.md), [Sentry](../observability/sentry.md), [SigNoz](../observability/signoz.md), [Logfire](../observability/logfire_integration.md), [Azure Sentinel](../observability/azure_sentinel.md), [Helicone](../observability/helicone_integration.md), [CloudZero](../observability/cloudzero.md), [Slack](../observability/slack_integration.md), [Galileo](../observability/galileo_integration.md)
- `opentelemetry` — [OpenTelemetry](../observability/opentelemetry_integration.md), [Langfuse OpenTelemetry](../observability/langfuse_otel_integration.md), [Levo AI](../observability/levo_integration.md), [Traceloop](../observability/traceloop_integration.md), [Weave](../observability/weave_integration.md)
- `callback_integrations` — [Callbacks](../observability/callbacks.md), [Custom API Callback](../observability/custom_callback.md), [Generic API Callback](../observability/generic_api.md), [AgentOps](../observability/agentops_integration.md), [Humanloop](../observability/humanloop.md), [Literal AI](../observability/literalai_integration.md)
- `evals_observability` — [OpenAI Function and output logging docs](../observability/raw_request_response.md), [Deepeval](../observability/deepeval_integration.md), [Phoenix](../observability/phoenix_integration.md), [W&B](../observability/wandb_integration.md), [OpenMeter](../observability/openmeter.md), [Lago](../observability/lago.md), [CloudZero](../observability/cloudzero.md), [Focus Export](../observability/focus.md)
- `platform_observability` — [Qualifire](../observability/qualifire_integration.md), [Lunary](../observability/lunary_integration.md), [Greenscale](../observability/greenscale_integration.md), [Athina](../observability/athina_integration.md), [Sumo Logic](../observability/sumologic_integration.md), [Supabase Logging](../observability/supabase_integration.md), [PagerDuty + Logging](../proxy/pagerduty.md), [Scrub Logged Data](../observability/scrub_data.md)

For the complete Observability & Logging inventory, use:
- [Integrations sidebar category: Observability](/docs/integrations/observability)

## Guardrails and Safety (Proxy)

- `content_protection` — [Prompt Security](../proxy/guardrails/prompt_security.md), [Prompt Injection Detection](../proxy/guardrails/prompt_injection.md), [Tool Permission](../proxy/guardrails/tool_permission.md), [Secret Detection](../proxy/guardrails/secret_detection.md)
- `detection` — [PII Masking](../proxy/guardrails/pii_masking_v2.md), [Secret Detection](../proxy/guardrails/secret_detection.md), [Content Moderation](../proxy/guardrails/openai_moderation.md), [Model Armor](../proxy/guardrails/model_armor.md)
- `providers` — [Lakera AI](../proxy/guardrails/lakera_ai.md), [Guardrails AI](../proxy/guardrails/guardrails_ai.md), [Aporia](../proxy/guardrails/aporia_api.md), [Bedrock Guardrails](../proxy/guardrails/bedrock.md), [Azure Content Safety](../proxy/guardrails/azure_content_guardrail.md), [HiddenLayer](../proxy/guardrails/hiddenlayer.md), [Qualifire](../proxy/guardrails/qualifire.md), [PII + PHI Presidio](../tutorials/presidio_pii_masking.md)
- `specialized` — [PANW Prisma AIRS](../proxy/guardrails/panw_prisma_airs.md), [Zscaler AI Guard](../proxy/guardrails/zscaler_ai_guard.md), [IBM Guardrails](../proxy/guardrails/ibm_guardrails.md), [Noma Security](../proxy/guardrails/noma_security.md), [Onyx Security](../proxy/guardrails/onyx_security.md), [Javelin](../proxy/guardrails/javelin.md)
- `advanced` — [Lasso Security](../proxy/guardrails/lasso_security.md), [Grayswan](../proxy/guardrails/grayswan.md), [EnkryptAI](../proxy/guardrails/enkryptai.md), [Pangea](../proxy/guardrails/pangea.md), [DynamoAI](../proxy/guardrails/dynamoai.md), [Pillar Security](../proxy/guardrails/pillar_security.md), [AIM Security](../proxy/guardrails/aim_security.md)
- `custom` — [Quick Start](../proxy/guardrails/quick_start.md), [Custom Guardrail API](../proxy/guardrails/custom_guardrail.md), [Custom Code Guardrail](../proxy/guardrails/custom_code_guardrail.md), [LiteLLM Content Filter](../proxy/guardrails/litellm_content_filter.md), [Load Balancing](../proxy/guardrails/guardrail_load_balancing.md), [Testing Playground](../proxy/guardrails/test_playground.md)
- `policies` — [Guardrail Policies](../proxy/guardrails/guardrail_policies.md), [Policy Templates](../proxy/guardrails/policy_templates.md), [Policy Tags](../proxy/guardrails/policy_tags.md)

## Alerting and Monitoring (Proxy)

- [Prometheus](../proxy/prometheus.md)
- [PagerDuty](../proxy/pagerduty.md)
- [Proxy Alerting](../proxy/alerting.md)
- [Proxy health checks and monitoring](../proxy/health.md)

## Billing and Cost Tracking (SDK + Proxy)

- [Lago](../observability/lago.md)
- [OpenMeter](../observability/openmeter.md)
- [Proxy cost controls via team/team-budget flows](../proxy/cost_tracking.md)
- [Soft and hard budget controls](../proxy/team_budgets.md)

## Prompt Management (Beta, Proxy)

- [LiteLLM Prompt Management](../proxy/litellm_prompt_management.md)
- [Native LiteLLM Prompts](../proxy/native_litellm_prompt.md)
- [Custom Prompt Management](../proxy/custom_prompt_management.md)
- [Prompt Management in Dashboard](../proxy/prompt_management.md)
- [Arize Phoenix Prompts](../proxy/arize_phoenix_prompts.md)

## Infrastructure and Storage (SDK + Proxy)

- [Google Cloud Storage Logging](../observability/gcs_bucket_integration.md)
- [FOCUS Export to S3](../observability/focus.md)
- [S3 Logging and Caching Callbacks](../proxy/logging.md#s3-buckets)
- [Azure Blob Storage Logging](../proxy/logging.md#azure-blob-storage)
- [DynamoDB Logging](../proxy/logging.md#dynamodb)
- [Supabase Output](../observability/supabase_integration.md)

## Next steps

- Click a category above and then open the full destination list in the sidebar.
- If you are implementing a deployment, start with:
  - `Observability and Logging` for telemetry
  - `Guardrails and Safety` for policy controls
  - `AI Tools and Frameworks` for client apps and assistants
