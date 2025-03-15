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
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGiM7ZrUwqu_Q/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1675971026692?e=1741824000&v=beta&t=eQnRdXPJo4eiINWTZARoYTfqh064pgZ-E21pQTSy8jc

tags: [credential management, thinking content, responses api, snowflake]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

These are the changes since `v1.63.2-stable`.

This release is primarily focused on:
- Credential Management (UI + API)
- Thinking Content Improvements (OpenWebUI, Bedrock, Anthropic, Deepseek)
- New Responses API
- New Provider Integrations (Snowflake Cortex)

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

- Support OpenRouter `reasoning_content` on streaming [PR](https://github.com/BerriAI/litellm/pull/9094)
- Support Bedrock converse cache token tracking [PR](https://github.com/BerriAI/litellm/pull/9221)

4. Bug Fixes
- Fix Bedrock chunk parsing [PR](https://github.com/BerriAI/litellm/pull/9166)
- Fix Azure Function Calling Bug & Update Default API Version to `2025-02-01-preview` [PR](https://github.com/BerriAI/litellm/pull/9191)
- Fix incorrect streaming response [PR](https://github.com/BerriAI/litellm/pull/9081)
- Fix Triton streaming completions bug [PR](https://github.com/BerriAI/litellm/pull/8386)
- Fix: String data stripped from entire content in streamed Gemini responses [PR](https://github.com/BerriAI/litellm/pull/9070)
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
- Add Azure Data Zone pricing [PR](https://github.com/BerriAI/litellm/pull/9185)


## Spend Tracking Improvements

1. Fix Batches API cost tracking + Log batch models in spend logs / standard logging payload [PR](https://github.com/BerriAI/litellm/pull/9077)
3. Fix Azure Whisper cost tracking [PR](https://github.com/BerriAI/litellm/pull/9166)

## Management Endpoints / UI
1. Add Models Page
   - Allow adding Text-Completion OpenAI models through UI [PR](https://github.com/BerriAI/litellm/pull/9102)
   - Allow adding EU OpenAI models [PR](https://github.com/BerriAI/litellm/pull/9042)
   - Allow adding Cerebras, Sambanova, Perplexity, Fireworks, Openrouter, TogetherAI Models on Admin UI [PR](https://github.com/BerriAI/litellm/pull/9069)
   - Fix: Instantly show edit + deletes to models [PR](https://github.com/BerriAI/litellm/pull/9258)
    - UI Test Connection feature [PR](https://github.com/BerriAI/litellm/pull/9272)
    - Support credential management on Proxy via CRUD endpoints - `credentials/*` [PR](https://github.com/BerriAI/litellm/pull/9124)
    - Add UI for credential management [PR](https://github.com/BerriAI/litellm/pull/9186)
    - Support reusing existing model credentials [PR](https://github.com/BerriAI/litellm/pull/9267)
2. Keys Page
   - Fix: Instantly show newly created keys on Admin UI (don't require refresh) [PR](https://github.com/BerriAI/litellm/pull/9257)
   - Fix: Allow clicking into Top Keys when showing users Top API Key [PR](https://github.com/BerriAI/litellm/pull/9225)
   - Fix: Allow Filter Keys by Team Alias, Key Alias and Org [PR](https://github.com/BerriAI/litellm/pull/9083)
   - UI Improvements: Show 100 Keys Per Page, Use full height, increase width of key alias [PR](https://github.com/BerriAI/litellm/pull/9064)
3. Users Page
   - Fix: Show correct count of internal user keys on Users Page [PR](https://github.com/BerriAI/litellm/pull/9082)
   - Fix: Metadata not updating in Team UI [PR](https://github.com/BerriAI/litellm/pull/9180)
4. Logs Page
   - UI Improvements: Keep expanded log in focus on LiteLLM UI [PR](https://github.com/BerriAI/litellm/pull/9061)
   - UI Improvements: Minor improvements to logs page [PR](https://github.com/BerriAI/litellm/pull/9076)
   - Fix: Allow internal user to query their own logs [PR](https://github.com/BerriAI/litellm/pull/9162)


## Security

6. Fix: Internal User Viewer Permissions [PR](https://github.com/BerriAI/litellm/pull/9148)
1. Support master key rotations [PR](https://github.com/BerriAI/litellm/pull/9041)
1. Emit audit logs on All user + model Create/Update/Delete endpoints [PR](https://github.com/BerriAI/litellm/pull/9223)
JWT
4. Using K/V pairs in 1 AWS Secret [PR](https://github.com/BerriAI/litellm/pull/9039)
5. Handle ManagedIdentityCredential in Azure AD token provider [PR](https://github.com/BerriAI/litellm/pull/9135)
6. Prioritize api_key over tenant_id for Azure AD token provider [PR](https://github.com/BerriAI/litellm/pull/8701)


- Support multiple JWT URLs [PR](https://github.com/BerriAI/litellm/pull/9047)
- Fix JWT access with Groups not working when team is assigned All Proxy Models access [PR](https://github.com/BerriAI/litellm/pull/8934)


## Logging / Guardrail Integrations

2. Track Azure LLM API latency metric [PR](https://github.com/BerriAI/litellm/pull/9217)
2. Allow switching off storing Error Logs in DB [PR](https://github.com/BerriAI/litellm/pull/9084)
3. Added tags, user_feedback and model_options to additional_keys which can be sent to Athina [PR](https://github.com/BerriAI/litellm/pull/8845)
4. Return `code`, `param` and `type` on OpenAI bad request error [PR](https://github.com/BerriAI/litellm/pull/9109)

## OpenWebUI Integration - display `thinking` tokens
- Guide on getting started with LiteLLM x OpenWebUI. [Get Started](https://docs.litellm.ai/docs/tutorials/openweb_ui)
- Display `thinking` tokens on OpenWebUI (Bedrock, Anthropic, Deepseek) [Get Started](https://docs.litellm.ai/docs/tutorials/openweb_ui#render-thinking-content-on-openweb-ui)

<Image img={require('../../img/litellm_thinking_openweb.gif')} />

## Performance / Reliability improvements

1. Fix Redis cluster mode for routers [PR](https://github.com/BerriAI/litellm/pull/9010)
2. Delegate router Azure client init logic to Azure provider [PR](https://github.com/BerriAI/litellm/pull/9140)
3. Fix Azure AI services URL [PR](https://github.com/BerriAI/litellm/pull/9185)
4. Support extra_headers on Bedrock [PR](https://github.com/BerriAI/litellm/pull/9113)


## General Improvements
5. UI API Playground for testing LiteLLM translation [PR](https://github.com/BerriAI/litellm/pull/9073)
4. Fix: Correctly use `PROXY_LOGOUT_URL` when set [PR](https://github.com/BerriAI/litellm/pull/9117)
Bing Search Pass Thru [PR](https://github.com/BerriAI/litellm/pull/8019)


## Complete Git Diff

[Here's the complete git diff](https://github.com/BerriAI/litellm/compare/v1.63.2-stable...v1.63.11-stable)