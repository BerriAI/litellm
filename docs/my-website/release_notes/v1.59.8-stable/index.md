---
title: v1.59.8-stable
slug: v1.59.8-stable
date: 2025-01-31T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1743638400&v=beta&t=39KOXMUFedvukiWWVPHf3qI45fuQD7lNglICwN31DrI
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [admin ui, logging, db schema]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

# v1.59.8-stable



:::info

Get a 7 day free trial for LiteLLM Enterprise [here](https://litellm.ai/#trial).

**no call needed**

:::


## New Models / Updated Models 

1. New OpenAI `/image/variations` endpoint support 
2. Topaz API support on OpenAI `/image/variations` endpoint 
3. Deepseek - r1 support w/ reasoning_content (Deepseek API, Vertex AI, Bedrock) 
4. Azure - Add azure o1 pricing 
5. Anthropic - handle `-latest` tag in model for cost calculation
6. Gemini-2.0-flash-thinking - add model pricing (it’s 0.0)
7. Bedrock - add stability model pricing 
8. Bedrock - add us.amazon.nova-lite-v1:0 to model cost map
9. TogetherAI - add new together_ai llama3.3 models 

## LLM Translation

1. LM Studio -> fix async embedding call 
2. Gpt 4o models - fix response_format translation 
3. Bedrock nova - expand supported document types to include .md, .csv, etc. 
4. Bedrock - docs on IAM role based access for bedrock - https://docs.litellm.ai/docs/providers/bedrock#sts-role-based-auth
5. Bedrock - cache IAM role credentials when used 
6. Google AI Studio (`gemini/`) - support gemini 'frequency_penalty' and 'presence_penalty'
7. Azure O1 - fix model name check 
8. Anthropic - new `provider_headers` field to pass only Anthropic API `anthropic-beta` header. Useful when switching between anthropic/bedrock/vertex. 
9. WatsonX - ZenAPIKey support for WatsonX 
10. Ollama Chat - support json schema response format 
11. Bedrock - return correct bedrock status code and error message if error during streaming
12. Anthropic - Supported nested json schema on anthropic calls
13. OpenAI - `metadata` param preview support 
    1. SDK - enable via `litellm.enable_preview_features = True` 
    2. PROXY - enable via `litellm_settings::enable_preview_features: true` 
14. Replicate - retry completion response on status=processing 

## Spend Tracking Improvements

1. Bedrock - QA asserts all bedrock regional models have same `supported_` as base model 
2. Bedrock - fix bedrock converse cost tracking w/ region name specified
3. Spend Logs reliability fix - when `user` passed in request body is int instead of string 
4. Ensure ‘base_model’ cost tracking works across all endpoints 
5. Fixes for Image generation cost tracking 
6. Anthropic - fix anthropic end user cost tracking
7. JWT / OIDC Auth - add end user id tracking from jwt auth

## Management Endpoints / UI

1. allows team member to become admin post-add (ui + endpoints) 
2. New edit/delete button for updating team membership on UI 
3. If team admin - show all team keys 
4. Model Hub - clarify cost of models is per 1m tokens 
5. Invitation Links - fix invalid url generated
6. New - SpendLogs Table Viewer - allows proxy admin to view spend logs on UI 
    1. New spend logs - allow proxy admin to ‘opt in’ to logging request/response in spend logs table - enables easier abuse detection 
    2. Show country of origin in spend logs 
    3. Add pagination + filtering by key name/team name 
7. `/key/delete` - allow team admin to delete team keys 
8. Internal User ‘view’ - fix spend calculation when team selected
9. Model Analytics is now on Free  
10. Usage page - shows days when spend = 0, and round spend on charts to 2 sig figs 
11. Public Teams - allow admins to expose teams for new users to ‘join’ on UI - https://docs.litellm.ai/docs/proxy/public_teams
12. Guardrails
    1. set/edit guardrails on a virtual key 
    2. Allow setting guardrails on a team 
    3. Set guardrails on team create + edit page
13. Support temporary budget increases on `/key/update` - new `temp_budget_increase` and `temp_budget_expiry` fields - https://litellm-api.up.railway.app/#/key%20management/update_key_fn_key_update_post
14. support writing new key alias to aws secret manager - on key rotation

## Helm

1. add securityContext and pull policy values to migration job (s/o https://github.com/Hexoplon) 
2. allow specifying envVars on values.yaml
3. new helm lint test

## Logging / Guardrail Integrations

1. log the used prompt when prompt management used
2. support s3 logging with team alias prefixes - https://docs.litellm.ai/docs/proxy/logging#team-alias-prefix-in-object-key
3. Prometheus
    1. fix litellm_llm_api_time_to_first_token_metric not populating for bedrock models
    2. emit remaining team budget metric on regular basis (even when call isn’t made) - allows for more stable metrics on Grafana/etc. 
    3. add key and team level budget metrics
    4. emit `litellm_overhead_latency_metric` 
    5. Emit `litellm_team_budget_reset_at_metric` and `litellm_api_key_budget_remaining_hours_metric` 
4. Datadog - support logging spend tags to Datadog. 
5. Langfuse - fix logging request tags, read from standard logging payload 
6. GCS - don’t truncate payload on logging 
7. New GCS Pub/Sub logging support https://docs.litellm.ai/docs/proxy/logging#google-cloud-storage---pubsub-topic
8. Add AIM Guardrails support

## Security

1. New Enterprise SLA for patching security vulnerabilities 
2. Hashicorp - support using vault namespace for TLS auth 
3. Azure - DefaultAzureCredential support 

## Health Checks

1. Cleanup pricing-only model names from wildcard route list - prevent bad health checks 
2. Allow specifying a health check model for wildcard routes - https://docs.litellm.ai/docs/proxy/health#wildcard-routes
3. New ‘health_check_timeout ‘ param with default 1min upperbound to prevent bad model from health check to hang and cause pod restarts. 
4. Datadog - add data dog service health check + expose new `/health/services` endpoint

## Performance / Reliability improvements

1. 3x increase in RPS - moving to orjson for reading request body 
2. LLM Routing speedup - using cached get model group info 
3. SDK speedup - using cached get model info helper - reduces CPU work to get model info 
4. Proxy speedup - only read request body 1 time per request 
5. Infinite loop detection scripts added to codebase 
6. Bedrock - pure async image transformation requests 
7. Cooldowns - single deployment model group if 100% calls fail in high traffic - prevents an o1 outage from impacting other calls 
8. Response Headers - return 
    1. `x-litellm-timeout` 
    2. `x-litellm-attempted-retries`
    3. `x-litellm-overhead-duration-ms` 
    4. `x-litellm-response-duration-ms` 
9. ensure duplicate callbacks are not added to proxy
10. Requirements.txt - bump certifi version

## General Proxy Improvements

1. JWT / OIDC Auth - new `enforce_rbac` param,allows proxy admin to prevent any unmapped yet authenticated jwt tokens from calling proxy
2. fix custom openapi schema generation for customized swagger’s 
3. Request Headers - support reading `x-litellm-timeout` param from request headers. Enables model timeout control when using Vercel’s AI SDK + LiteLLM Proxy. 

## Complete Git Diff

This is the diff between v1.57.8-stable and v1.59.8-stable.

Use this to see the changes in the codebase.

[**Git Diff**](https://github.com/BerriAI/litellm/compare/v1.57.8-stable...v1.59.8-stable)
