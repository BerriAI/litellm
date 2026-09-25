# Appendix: integrations

Generated from `litellm/integrations/`, `litellm/proxy/guardrails/guardrail_hooks/`, `litellm/secret_managers/`, `litellm/proxy/pass_through_endpoints/llm_provider_handlers/`, and the callback registry in `litellm/litellm_core_utils/custom_logger_registry.py`. `callback` rows are the string name that enables them in `litellm_settings.callbacks` / `success_callback`; `callback_module` rows are integration modules not wired into that registry directly

## Logging callbacks

| name | code | docs |
|---|---|---|
| `agentops` | `custom_logger_registry:AgentOps` | [docs](https://docs.litellm.ai/docs/observability/agentops_integration) |
| `anthropic_cache_control_hook` | `custom_logger_registry:AnthropicCacheControlHook` |  |
| `argilla` | `custom_logger_registry:ArgillaLogger` | [docs](https://docs.litellm.ai/docs/observability/argilla) |
| `arize` | `custom_logger_registry:OpenTelemetry` | [docs](https://docs.litellm.ai/docs/observability/arize_integration) |
| `arize_phoenix` | `custom_logger_registry:OpenTelemetry` |  |
| `aws_sqs` | `custom_logger_registry:SQSLogger` |  |
| `azure_sentinel` | `custom_logger_registry:AzureSentinelLogger` | [docs](https://docs.litellm.ai/docs/observability/azure_sentinel) |
| `azure_storage` | `custom_logger_registry:AzureBlobStorageLogger` |  |
| `bitbucket` | `custom_logger_registry:BitBucketPromptManager` |  |
| `braintrust` | `custom_logger_registry:BraintrustLogger` | [docs](https://docs.litellm.ai/docs/observability/braintrust) |
| `cloudzero` | `custom_logger_registry:CloudZeroLogger` | [docs](https://docs.litellm.ai/docs/observability/cloudzero) |
| `datadog` | `custom_logger_registry:DataDogLogger` | [docs](https://docs.litellm.ai/docs/observability/datadog) |
| `datadog_llm_observability` | `custom_logger_registry:DataDogLLMObsLogger` |  |
| `datadog_metrics` | `custom_logger_registry:DatadogMetricsLogger` |  |
| `deepeval` | `custom_logger_registry:DeepEvalLogger` | [docs](https://docs.litellm.ai/docs/observability/deepeval_integration) |
| `dotprompt` | `custom_logger_registry:DotpromptManager` |  |
| `dynamic_rate_limiter` | `custom_logger_registry:_PROXY_DynamicRateLimitHandler` |  |
| `dynamic_rate_limiter_v3` | `custom_logger_registry:_PROXY_DynamicRateLimitHandlerV3` |  |
| `focus` | `custom_logger_registry:FocusLogger` | [docs](https://docs.litellm.ai/docs/observability/focus) |
| `galileo` | `custom_logger_registry:GalileoObserve` |  |
| `gcs_bucket` | `custom_logger_registry:GCSBucketLogger` | [docs](https://docs.litellm.ai/docs/observability/gcs_bucket_integration) |
| `gcs_pubsub` | `custom_logger_registry:GcsPubSubLogger` |  |
| `generic_api` | `custom_logger_registry:GenericAPILogger` | [docs](https://docs.litellm.ai/docs/observability/generic_api) |
| `gitlab` | `custom_logger_registry:GitLabPromptManager` |  |
| `humanloop` | `custom_logger_registry:HumanloopLogger` | [docs](https://docs.litellm.ai/docs/observability/humanloop) |
| `lago` | `custom_logger_registry:LagoLogger` | [docs](https://docs.litellm.ai/docs/observability/lago) |
| `langfuse` | `custom_logger_registry:LangfusePromptManagement` | [docs](https://docs.litellm.ai/docs/observability/langfuse_integration) |
| `langfuse_otel` | `custom_logger_registry:OpenTelemetry` |  |
| `langsmith` | `custom_logger_registry:LangsmithLogger` | [docs](https://docs.litellm.ai/docs/observability/langsmith_integration) |
| `langtrace` | `custom_logger_registry:OpenTelemetry` | [docs](https://docs.litellm.ai/docs/observability/langtrace_integration) |
| `levo` | `custom_logger_registry:OpenTelemetry` | [docs](https://docs.litellm.ai/docs/observability/levo_integration) |
| `litellm_agent` | `custom_logger_registry:LiteLLMAgentModelResolver` |  |
| `literalai` | `custom_logger_registry:LiteralAILogger` | [docs](https://docs.litellm.ai/docs/observability/literalai_integration) |
| `logfire` | `custom_logger_registry:OpenTelemetry` | [docs](https://docs.litellm.ai/docs/observability/logfire_integration) |
| `mavvrik` | `custom_logger_registry:MavvrikFocusLogger` | [docs](https://docs.litellm.ai/docs/observability/mavvrik) |
| `mlflow` | `custom_logger_registry:MlflowLogger` | [docs](https://docs.litellm.ai/docs/observability/mlflow) |
| `newrelic` | `custom_logger_registry:NewRelicLogger` | [docs](https://docs.litellm.ai/docs/observability/newrelic) |
| `openmeter` | `custom_logger_registry:OpenMeterLogger` | [docs](https://docs.litellm.ai/docs/observability/openmeter) |
| `opentelemetry` | `custom_logger_registry:OpenTelemetry` | [docs](https://docs.litellm.ai/docs/observability/opentelemetry_integration) |
| `opik` | `custom_logger_registry:OpikLogger` | [docs](https://docs.litellm.ai/docs/observability/opik_integration) |
| `otel` | `custom_logger_registry:OpenTelemetry` |  |
| `pagerduty` | `custom_logger_registry:PagerDutyAlerting` |  |
| `pointfive` | `custom_logger_registry:PointFiveLogger` | [docs](https://docs.litellm.ai/docs/observability/pointfive) |
| `posthog` | `custom_logger_registry:PostHogLogger` | [docs](https://docs.litellm.ai/docs/observability/posthog_integration) |
| `prometheus` | `custom_logger_registry:PrometheusLogger` |  |
| `resend_email` | `custom_logger_registry:ResendEmailLogger` |  |
| `s3_v2` | `custom_logger_registry:S3Logger` |  |
| `sendgrid_email` | `custom_logger_registry:SendGridEmailLogger` |  |
| `smtp_email` | `custom_logger_registry:SMTPEmailLogger` |  |
| `vantage` | `custom_logger_registry:VantageLogger` | [docs](https://docs.litellm.ai/docs/observability/vantage) |
| `vector_store_pre_call_hook` | `custom_logger_registry:VectorStorePreCallHook` |  |
| `weave_otel` | `custom_logger_registry:OpenTelemetry` |  |

## Guardrail providers

| name | code | docs |
|---|---|---|
| `agent_365` | `litellm/proxy/guardrails/guardrail_hooks/agent_365` |  |
| `aim` | `litellm/proxy/guardrails/guardrail_hooks/aim` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/aim_security) |
| `akto` | `litellm/proxy/guardrails/guardrail_hooks/akto` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/akto) |
| `alice` | `litellm/proxy/guardrails/guardrail_hooks/alice` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/alice) |
| `aporia_ai` | `litellm/proxy/guardrails/guardrail_hooks/aporia_ai` |  |
| `azure` | `litellm/proxy/guardrails/guardrail_hooks/azure` |  |
| `bedrock_guardrails` | `litellm/proxy/guardrails/guardrail_hooks/bedrock_guardrails.py` |  |
| `block_code_execution` | `litellm/proxy/guardrails/guardrail_hooks/block_code_execution` |  |
| `cato_networks` | `litellm/proxy/guardrails/guardrail_hooks/cato_networks` |  |
| `cisco_ai_defense` | `litellm/proxy/guardrails/guardrail_hooks/cisco_ai_defense` |  |
| `compresr` | `litellm/proxy/guardrails/guardrail_hooks/compresr` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/compresr) |
| `conduct` | `litellm/proxy/guardrails/guardrail_hooks/conduct` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/conduct) |
| `content_text` | `litellm/proxy/guardrails/guardrail_hooks/content_text.py` |  |
| `crowdstrike_aidr` | `litellm/proxy/guardrails/guardrail_hooks/crowdstrike_aidr` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/crowdstrike_aidr) |
| `custom_code` | `litellm/proxy/guardrails/guardrail_hooks/custom_code` |  |
| `custom_guardrail` | `litellm/proxy/guardrails/guardrail_hooks/custom_guardrail.py` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/custom_guardrail) |
| `deepkeep` | `litellm/proxy/guardrails/guardrail_hooks/deepkeep` |  |
| `dynamoai` | `litellm/proxy/guardrails/guardrail_hooks/dynamoai` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/dynamoai) |
| `enkryptai` | `litellm/proxy/guardrails/guardrail_hooks/enkryptai` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/enkryptai) |
| `generic_guardrail_api` | `litellm/proxy/guardrails/guardrail_hooks/generic_guardrail_api` |  |
| `grayswan` | `litellm/proxy/guardrails/guardrail_hooks/grayswan` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/grayswan) |
| `guardrails_ai` | `litellm/proxy/guardrails/guardrail_hooks/guardrails_ai` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/guardrails_ai) |
| `headroom` | `litellm/proxy/guardrails/guardrail_hooks/headroom` |  |
| `hiddenlayer` | `litellm/proxy/guardrails/guardrail_hooks/hiddenlayer` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/hiddenlayer) |
| `ibm_guardrails` | `litellm/proxy/guardrails/guardrail_hooks/ibm_guardrails` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/ibm_guardrails) |
| `javelin` | `litellm/proxy/guardrails/guardrail_hooks/javelin` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/javelin) |
| `lakera_ai` | `litellm/proxy/guardrails/guardrail_hooks/lakera_ai.py` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/lakera_ai) |
| `lakera_ai_v2` | `litellm/proxy/guardrails/guardrail_hooks/lakera_ai_v2.py` |  |
| `lasso` | `litellm/proxy/guardrails/guardrail_hooks/lasso` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/lasso_security) |
| `litellm_content_filter` | `litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/litellm_content_filter) |
| `llm_as_a_judge` | `litellm/proxy/guardrails/guardrail_hooks/llm_as_a_judge` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/llm_as_a_judge) |
| `mcp_end_user_permission` | `litellm/proxy/guardrails/guardrail_hooks/mcp_end_user_permission` |  |
| `mcp_jwt_signer` | `litellm/proxy/guardrails/guardrail_hooks/mcp_jwt_signer` |  |
| `mcp_security` | `litellm/proxy/guardrails/guardrail_hooks/mcp_security` |  |
| `microsoft_purview` | `litellm/proxy/guardrails/guardrail_hooks/microsoft_purview` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/microsoft_purview) |
| `model_armor` | `litellm/proxy/guardrails/guardrail_hooks/model_armor` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/model_armor) |
| `noma` | `litellm/proxy/guardrails/guardrail_hooks/noma` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/noma_security) |
| `onyx` | `litellm/proxy/guardrails/guardrail_hooks/onyx` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/onyx_security) |
| `openai` | `litellm/proxy/guardrails/guardrail_hooks/openai` |  |
| `ovalix` | `litellm/proxy/guardrails/guardrail_hooks/ovalix` |  |
| `pangea` | `litellm/proxy/guardrails/guardrail_hooks/pangea` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/pangea) |
| `panw_prisma_airs` | `litellm/proxy/guardrails/guardrail_hooks/panw_prisma_airs` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/panw_prisma_airs) |
| `pillar` | `litellm/proxy/guardrails/guardrail_hooks/pillar` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/pillar_security) |
| `presidio` | `litellm/proxy/guardrails/guardrail_hooks/presidio.py` |  |
| `prompt_security` | `litellm/proxy/guardrails/guardrail_hooks/prompt_security` |  |
| `promptguard` | `litellm/proxy/guardrails/guardrail_hooks/promptguard` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/promptguard) |
| `qohash` | `litellm/proxy/guardrails/guardrail_hooks/qohash` |  |
| `qualifire` | `litellm/proxy/guardrails/guardrail_hooks/qualifire` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/qualifire) |
| `repelloai` | `litellm/proxy/guardrails/guardrail_hooks/repelloai` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/repelloai) |
| `rubrik` | `litellm/proxy/guardrails/guardrail_hooks/rubrik` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/rubrik) |
| `semantic_guard` | `litellm/proxy/guardrails/guardrail_hooks/semantic_guard` |  |
| `singulr` | `litellm/proxy/guardrails/guardrail_hooks/singulr` |  |
| `straiker` | `litellm/proxy/guardrails/guardrail_hooks/straiker` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/straiker) |
| `tool_permission` | `litellm/proxy/guardrails/guardrail_hooks/tool_permission.py` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/tool_permission) |
| `tool_policy` | `litellm/proxy/guardrails/guardrail_hooks/tool_policy` |  |
| `typesafe` | `litellm/proxy/guardrails/guardrail_hooks/typesafe` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/typesafe) |
| `unified_guardrail` | `litellm/proxy/guardrails/guardrail_hooks/unified_guardrail` |  |
| `vigil_guard` | `litellm/proxy/guardrails/guardrail_hooks/vigil_guard` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/vigil_guard) |
| `xecguard` | `litellm/proxy/guardrails/guardrail_hooks/xecguard` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/xecguard) |
| `zscaler_ai_guard` | `litellm/proxy/guardrails/guardrail_hooks/zscaler_ai_guard` | [docs](https://docs.litellm.ai/docs/proxy/guardrails/zscaler_ai_guard) |

## Secret managers

| name | code | docs |
|---|---|---|
| `aws_secret_manager` | `litellm/secret_managers/aws_secret_manager.py` | [docs](https://docs.litellm.ai/docs/secret_managers/aws_secret_manager) |
| `aws_secret_manager_v2` | `litellm/secret_managers/aws_secret_manager_v2.py` |  |
| `base_secret_manager` | `litellm/secret_managers/base_secret_manager.py` |  |
| `custom_secret_manager_loader` | `litellm/secret_managers/custom_secret_manager_loader.py` |  |
| `cyberark_secret_manager` | `litellm/secret_managers/cyberark_secret_manager.py` | [docs](https://docs.litellm.ai/docs/secret_managers/cyberark) |
| `dispatch` | `litellm/secret_managers/dispatch.py` |  |
| `get_azure_ad_token_provider` | `litellm/secret_managers/get_azure_ad_token_provider.py` |  |
| `google_kms` | `litellm/secret_managers/google_kms.py` | [docs](https://docs.litellm.ai/docs/secret_managers/google_kms) |
| `google_secret_manager` | `litellm/secret_managers/google_secret_manager.py` | [docs](https://docs.litellm.ai/docs/secret_managers/google_secret_manager) |
| `hashicorp_secret_manager` | `litellm/secret_managers/hashicorp_secret_manager.py` |  |
| `main` | `litellm/secret_managers/main.py` |  |
| `secret_manager_handler` | `litellm/secret_managers/secret_manager_handler.py` |  |

## Pass-through providers

| name | code | docs |
|---|---|---|
| `anthropic_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/anthropic_passthrough_logging_handler.py` |  |
| `assembly_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/assembly_passthrough_logging_handler.py` |  |
| `azure_speech_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/azure_speech_passthrough_logging_handler.py` | [docs](https://docs.litellm.ai/docs/pass_through/azure_speech) |
| `base_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/base_passthrough_logging_handler.py` |  |
| `batch_attribution` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/batch_attribution.py` |  |
| `cohere_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/cohere_passthrough_logging_handler.py` | [docs](https://docs.litellm.ai/docs/pass_through/cohere) |
| `comprehend_medical_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/comprehend_medical_passthrough_logging_handler.py` | [docs](https://docs.litellm.ai/docs/pass_through/comprehend_medical) |
| `cursor_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/cursor_passthrough_logging_handler.py` | [docs](https://docs.litellm.ai/docs/pass_through/cursor) |
| `deepgram_listen_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/deepgram_listen_passthrough_logging_handler.py` |  |
| `fal_ai_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/fal_ai_passthrough_logging_handler.py` |  |
| `gemini_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/gemini_passthrough_logging_handler.py` |  |
| `openai_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/openai_passthrough_logging_handler.py` |  |
| `tinyfish_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/tinyfish_passthrough_logging_handler.py` | [docs](https://docs.litellm.ai/docs/pass_through/tinyfish) |
| `transcribe_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/transcribe_passthrough_logging_handler.py` | [docs](https://docs.litellm.ai/docs/pass_through/transcribe) |
| `typesafe_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/typesafe_passthrough_logging_handler.py` | [docs](https://docs.litellm.ai/docs/pass_through/typesafe) |
| `vertex_ai_live_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/vertex_ai_live_passthrough_logging_handler.py` |  |
| `vertex_passthrough_logging_handler` | `litellm/proxy/pass_through_endpoints/llm_provider_handlers/vertex_passthrough_logging_handler.py` |  |

## Callback modules not in the registry

| name | code | docs |
|---|---|---|
| `SlackAlerting` | `litellm/integrations/SlackAlerting` |  |
| `_types` | `litellm/integrations/_types` |  |
| `additional_logging_utils` | `litellm/integrations/additional_logging_utils.py` |  |
| `athina` | `litellm/integrations/athina.py` | [docs](https://docs.litellm.ai/docs/observability/athina_integration) |
| `batch_utils` | `litellm/integrations/batch_utils.py` |  |
| `braintrust_logging` | `litellm/integrations/braintrust_logging.py` |  |
| `braintrust_mock_client` | `litellm/integrations/braintrust_mock_client.py` |  |
| `code_interpreter_interception` | `litellm/integrations/code_interpreter_interception` |  |
| `compression_interception` | `litellm/integrations/compression_interception` |  |
| `custom_batch_logger` | `litellm/integrations/custom_batch_logger.py` |  |
| `custom_logger` | `litellm/integrations/custom_logger.py` |  |
| `custom_prompt_management` | `litellm/integrations/custom_prompt_management.py` |  |
| `custom_secret_manager` | `litellm/integrations/custom_secret_manager.py` |  |
| `custom_sso_handler` | `litellm/integrations/custom_sso_handler.py` |  |
| `dynamodb` | `litellm/integrations/dynamodb.py` |  |
| `email_alerting` | `litellm/integrations/email_alerting.py` |  |
| `email_templates` | `litellm/integrations/email_templates` |  |
| `generic_prompt_management` | `litellm/integrations/generic_prompt_management` |  |
| `greenscale` | `litellm/integrations/greenscale.py` | [docs](https://docs.litellm.ai/docs/observability/greenscale_integration) |
| `helicone` | `litellm/integrations/helicone.py` | [docs](https://docs.litellm.ai/docs/observability/helicone_integration) |
| `helicone_mock_client` | `litellm/integrations/helicone_mock_client.py` |  |
| `langsmith_mock_client` | `litellm/integrations/langsmith_mock_client.py` |  |
| `literal_ai` | `litellm/integrations/literal_ai.py` |  |
| `logfire_logger` | `litellm/integrations/logfire_logger.py` |  |
| `lunary` | `litellm/integrations/lunary.py` | [docs](https://docs.litellm.ai/docs/observability/lunary_integration) |
| `mavvrik_focus` | `litellm/integrations/mavvrik_focus` |  |
| `mock_client_factory` | `litellm/integrations/mock_client_factory.py` |  |
| `opentelemetry_utils` | `litellm/integrations/opentelemetry_utils` |  |
| `posthog_mock_client` | `litellm/integrations/posthog_mock_client.py` |  |
| `prometheus_helpers` | `litellm/integrations/prometheus_helpers` |  |
| `prometheus_metrics_endpoint` | `litellm/integrations/prometheus_metrics_endpoint.py` |  |
| `prometheus_services` | `litellm/integrations/prometheus_services.py` |  |
| `prompt_layer` | `litellm/integrations/prompt_layer.py` |  |
| `prompt_management_base` | `litellm/integrations/prompt_management_base.py` |  |
| `s3` | `litellm/integrations/s3.py` |  |
| `shadow_eval_logger` | `litellm/integrations/shadow_eval_logger.py` |  |
| `sqs` | `litellm/integrations/sqs.py` |  |
| `supabase` | `litellm/integrations/supabase.py` | [docs](https://docs.litellm.ai/docs/observability/supabase_integration) |
| `test_httpx` | `litellm/integrations/test_httpx.py` |  |
| `traceloop` | `litellm/integrations/traceloop.py` |  |
| `vector_store_integrations` | `litellm/integrations/vector_store_integrations` |  |
| `weave` | `litellm/integrations/weave` |  |
| `websearch_interception` | `litellm/integrations/websearch_interception` | [docs](https://docs.litellm.ai/docs/integrations/websearch_interception) |
| `weights_biases` | `litellm/integrations/weights_biases.py` |  |
