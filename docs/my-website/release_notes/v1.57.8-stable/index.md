---
title: v1.57.8-stable
slug: v1.57.8-stable
date: 2025-01-11T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1743638400&v=beta&t=39KOXMUFedvukiWWVPHf3qI45fuQD7lNglICwN31DrI
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc
tags: [langfuse, humanloop, alerting, prometheus, secret management, management endpoints, ui, prompt management, finetuning, batch]
hide_table_of_contents: false
---

`alerting`, `prometheus`, `secret management`, `management endpoints`, `ui`, `prompt management`, `finetuning`, `batch`


:::note

v1.57.8-stable, is currently being tested. It will be released on 2025-01-12. 

:::


## New / Updated Models

1. Mistral large pricing - https://github.com/BerriAI/litellm/pull/7452
2. Cohere command-r7b-12-2024 pricing - https://github.com/BerriAI/litellm/pull/7553/files
3. Voyage - new models, prices and context window information - https://github.com/BerriAI/litellm/pull/7472
4. Anthropic - bump Bedrock claude-3-5-haiku max_output_tokens to 8192

## General Proxy Improvements

1. Health check support for realtime models 
2. Support calling Azure realtime routes via virtual keys 
3. Support custom tokenizer on `/utils/token_counter` - useful when checking token count for self-hosted models 
4. Request Prioritization - support on `/v1/completion` endpoint as well 

## LLM Translation Improvements

1. Deepgram STT support. [Start Here](https://docs.litellm.ai/docs/providers/deepgram)
2. OpenAI Moderations - `omni-moderation-latest` support. [Start Here](https://docs.litellm.ai/docs/moderation)
3. Azure O1 - fake streaming support. This ensures if a `stream=true` is passed, the response is streamed. [Start Here](https://docs.litellm.ai/docs/providers/azure)
4. Anthropic - non-whitespace char stop sequence handling - [PR](https://github.com/BerriAI/litellm/pull/7484)
5. Azure OpenAI - support entrata id username + password based auth. [Start Here](https://docs.litellm.ai/docs/providers/azure#entrata-id---use-tenant_id-client_id-client_secret)
6. LM Studio - embedding route support. [Start Here](https://docs.litellm.ai/docs/providers/lm-studio)
7. WatsonX - ZenAPIKeyAuth support. [Start Here](https://docs.litellm.ai/docs/providers/watsonx)
    
## Prompt Management Improvements

1. Langfuse integration
2. HumanLoop integration 
3. Support for using load balanced models 
4. Support for loading optional params from prompt manager 

[Start Here](https://docs.litellm.ai/docs/proxy/prompt_management)

## Finetuning + Batch APIs Improvements

1. Improved unified endpoint support for Vertex AI finetuning - [PR](https://github.com/BerriAI/litellm/pull/7487)
2. Add support for retrieving vertex api batch jobs - [PR](https://github.com/BerriAI/litellm/commit/13f364682d28a5beb1eb1b57f07d83d5ef50cbdc)

## *NEW* Alerting Integration

PagerDuty Alerting Integration. 

Handles two types of alerts:

- High LLM API Failure Rate. Configure X fails in Y seconds to trigger an alert.
- High Number of Hanging LLM Requests. Configure X hangs in Y seconds to trigger an alert.


[Start Here](https://docs.litellm.ai/docs/proxy/pagerduty)

## Prometheus Improvements

Added support for tracking latency/spend/tokens based on custom metrics. [Start Here](https://docs.litellm.ai/docs/proxy/prometheus#beta-custom-metrics)

## *NEW* Hashicorp Secret Manager Support 

Support for reading credentials + writing LLM API keys. [Start Here](https://docs.litellm.ai/docs/secret#hashicorp-vault)

## Management Endpoints / UI Improvements

1. Create and view organizations + assign org admins on the Proxy UI
2. Support deleting keys by key_alias
3. Allow assigning teams to org on UI
4. Disable using ui session token for 'test key' pane
5. Show model used in 'test key' pane 
6. Support markdown output in 'test key' pane

## Helm Improvements

1. Prevent istio injection for db migrations cron job
2. allow using migrationJob.enabled variable within job

## Logging Improvements

1. braintrust logging: respect project_id, add more metrics  - https://github.com/BerriAI/litellm/pull/7613
2. Athina - support base url - `ATHINA_BASE_URL`
3. Lunary - Allow passing custom parent run id to LLM Calls 



## Git Diff 

This is the diff between v1.56.3-stable and v1.57.8-stable. 

Use this to see the changes in the codebase. 

[Git Diff](https://github.com/BerriAI/litellm/compare/v1.56.3-stable...189b67760011ea313ca58b1f8bd43aa74fbd7f55)