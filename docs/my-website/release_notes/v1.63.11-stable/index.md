---
title: v1.63.11-stable
slug: v1.63.11-stable
date: 2025-03-15T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1743638400&v=beta&t=39KOXMUFedvukiWWVPHf3qI45fuQD7lNglICwN31DrI
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

tags: [credential management, thinking content, responses api, snowflake]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

These are the changes since `v1.63.2-stable`.

This release is primarily focused on:
- [Beta] Responses API Support
- Snowflake Cortex Support
- UI - Credential Management, re-use credentials when adding new models
- UI - Test Connection to LLM Provider before adding a model

:::info

This release will be live on 03/16/2025

:::

<!-- <Image img={require('../../img/release_notes/v16311_release.jpg')} /> -->

## Demo Instance

Here's a Demo Instance to test changes:
- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234


## LLM Translation

<Image img={require('../../img/release_notes/responses_api.png')} />

1. **New Endpoints**
- [Beta] POST `/responses` API. [Get Started](https://docs.litellm.ai/docs/response_api)

2. **New LLM Providers**
- Snowflake Cortex [Get Started](https://docs.litellm.ai/docs/providers/snowflake)

3. **New models**

- Support OpenRouter `reasoning_content` on streaming [Get Started](https://docs.litellm.ai/docs/reasoning_content)
- Support Bedrock converse cache token tracking [Get Started](https://docs.litellm.ai/docs/completion/prompt_caching)

4. **Bug Fixes**

- Return `code`, `param` and `type` on OpenAI bad request error [More information on litellm exceptions](https://docs.litellm.ai/docs/exception_mapping)
- Fix Bedrock converse chunk parsing to only return empty dict on tool use [PR](https://github.com/BerriAI/litellm/pull/9166)
- Fix Azure Function Calling Bug & Update Default API Version to `2025-02-01-preview` [PR](https://github.com/BerriAI/litellm/pull/9191)
- Fix Perplexity incorrect streaming response [PR](https://github.com/BerriAI/litellm/pull/9081)
- Fix Triton streaming completions bug [PR](https://github.com/BerriAI/litellm/pull/8386)
- Fix: String `data:` stripped from entire content in streamed Gemini responses [PR](https://github.com/BerriAI/litellm/pull/9070)
- Fix: Support bytes.IO when handling audio files for transcription [PR](https://github.com/BerriAI/litellm/pull/9071)
- Fix: "system" role has become unacceptable in Ollama [PR](https://github.com/BerriAI/litellm/pull/9261)
- Handle HTTP 201 status code in Vertex AI response [PR](https://github.com/BerriAI/litellm/pull/9193)


### New Models Added to Model Cost Map
- Add support for Amazon Nova Canvas model [PR](https://github.com/BerriAI/litellm/pull/7838)
- Add pricing for Jamba new models [PR](https://github.com/BerriAI/litellm/pull/9032)
- Add pricing for Amazon EU models [PR](https://github.com/BerriAI/litellm/pull/9056)
- Add Bedrock Deepseek R1 model pricing [PR](https://github.com/BerriAI/litellm/pull/9108)
- Update Gemini pricing: Gemma 3, Flash 2 thinking update, LearnLM [PR](https://github.com/BerriAI/litellm/pull/9190)
- Mark Cohere Embedding 3 models as Multimodal [PR](https://github.com/BerriAI/litellm/pull/9176)


## UI

### Re-Use Credentials on UI

You can now onboard LLM provider credentials on LiteLLM UI. Once these credentials are added you can re-use them when adding new models

<Image img={require('../../img/release_notes/credentials.jpg')} />


### Test Connections before adding models

Before adding a model you can test the connection to the LLM provider to verify you have setup your API Base + API Key correctly

<Image img={require('../../img/release_notes/litellm_test_connection.gif')} />

### General UI Improvements
1. Add Models Page
   - Allow adding Cerebras, Sambanova, Perplexity, Fireworks, Openrouter, TogetherAI Models, Text-Completion OpenAI on Admin UI
   - Allow adding EU OpenAI models
   - Fix: Instantly show edit + deletes to models
2. Keys Page
   - Fix: Instantly show newly created keys on Admin UI (don't require refresh)
   - Fix: Allow clicking into Top Keys when showing users Top API Key
   - Fix: Allow Filter Keys by Team Alias, Key Alias and Org
   - UI Improvements: Show 100 Keys Per Page, Use full height, increase width of key alias
3. Users Page
   - Fix: Show correct count of internal user keys on Users Page
   - Fix: Metadata not updating in Team UI
4. Logs Page
   - UI Improvements: Keep expanded log in focus on LiteLLM UI
   - UI Improvements: Minor improvements to logs page
   - Fix: Allow internal user to query their own logs


## Security

1. Support for Rotating Master Keys [Getting Started](https://docs.litellm.ai/docs/proxy/master_key_rotations)
2. Fix: Internal User Viewer Permissions, don't allow `internal_user_viewer` role to see `Test Key Page` or `Create Key Button` [Role based access controls](https://docs.litellm.ai/docs/proxy/access_control)
3. Emit audit logs on All user + model Create/Update/Delete endpoints [Get Started](https://docs.litellm.ai/docs/proxy/multiple_admins)
4. JWT
    - Support multiple JWT OIDC providers [Get Started](https://docs.litellm.ai/docs/proxy/token_auth)
    - Fix JWT access with Groups not working when team is assigned All Proxy Models access
5. Using K/V pairs in 1 AWS Secret [Get Started](https://docs.litellm.ai/docs/secret#using-kv-pairs-in-1-aws-secret)


## Logging Integrations

1. Prometheus: Track Azure LLM API latency metric [Get Started here](https://docs.litellm.ai/docs/proxy/prometheus#request-latency-metrics)
2. Allow switching off storing Error Logs in DB [Get Started here](https://docs.litellm.ai/docs/proxy/ui_logs)
3. Added tags, user_feedback and model_options to additional_keys which can be sent to Athina [Get Started here](https://docs.litellm.ai/docs/observability/athina_integration)

## OpenWebUI Integration - display `thinking` tokens
- Guide on getting started with LiteLLM x OpenWebUI. [Get Started](https://docs.litellm.ai/docs/tutorials/openweb_ui)
- Display `thinking` tokens on OpenWebUI (Bedrock, Anthropic, Deepseek) [Get Started](https://docs.litellm.ai/docs/tutorials/openweb_ui#render-thinking-content-on-openweb-ui)

<Image img={require('../../img/litellm_thinking_openweb.gif')} />


## Spend Tracking Improvements

1. Add Azure Data Zone pricing [PR](https://github.com/BerriAI/litellm/pull/9185)
   - LiteLLM Tracks cost for `azure/eu` and `azure/us` models
2. Cost Tracking for Responses API [Get Started](https://docs.litellm.ai/docs/response_api)
3. Fix Azure Whisper cost tracking [PR](https://github.com/BerriAI/litellm/pull/9166)

## Performance / Reliability improvements

1. Fix Redis cluster mode for routers [PR](https://github.com/BerriAI/litellm/pull/9010)
2. Delegate router Azure client init logic to Azure provider [PR](https://github.com/BerriAI/litellm/pull/9140)
3. Fix Azure AI services URL [PR](https://github.com/BerriAI/litellm/pull/9185)
4. Support extra_headers on Bedrock [PR](https://github.com/BerriAI/litellm/pull/9113)


## General Improvements
1. UI API Playground for testing LiteLLM translation [PR](https://github.com/BerriAI/litellm/pull/9073)
2. Fix: Correctly use `PROXY_LOGOUT_URL` when set [PR](https://github.com/BerriAI/litellm/pull/9117)
3. Bing Search Pass Through endpoint [PR](https://github.com/BerriAI/litellm/pull/8019)


## Complete Git Diff

[Here's the complete git diff](https://github.com/BerriAI/litellm/compare/v1.63.2-stable...v1.63.11-stable)