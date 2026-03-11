---
title: v1.70.1-stable - Gemini Realtime API Support
slug: v1.70.1-stable
date: 2025-05-17T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

hide_table_of_contents: false
---

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';



## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run
-e STORE_MODEL_IN_DB=True
-p 4000:4000
docker.litellm.ai/berriai/litellm:main-v1.70.1-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.70.1
```
</TabItem>
</Tabs>


## Key Highlights

LiteLLM v1.70.1-stable is live now. Here are the key highlights of this release:

- **Gemini Realtime API**: You can now call Gemini's Live API via the OpenAI /v1/realtime API
- **Spend Logs Retention Period**: Enable deleting spend logs older than a certain period.
- **PII Masking 2.0**: Easily configure masking or blocking specific PII/PHI entities on the UI

## Gemini Realtime API

<Image img={require('../../img/gemini_realtime.png')}/>


This release brings support for calling Gemini's realtime models (e.g. gemini-2.0-flash-live) via OpenAI's /v1/realtime API. This is great for developers as it lets them easily switch from OpenAI to Gemini by just changing the model name. 

Key Highlights: 
- Support for text + audio input/output
- Support for setting session configurations (modality, instructions, activity detection) in the OpenAI format
- Support for logging + usage tracking for realtime sessions

This is currently supported via Google AI Studio. We plan to release VertexAI support over the coming week.

[**Read more**](../../docs/providers/google_ai_studio/realtime)

## Spend Logs Retention Period

<Image img={require('../../img/delete_spend_logs.jpg')}/>



This release enables deleting LiteLLM Spend Logs older than a certain period. Since we now enable storing the raw request/response in the logs, deleting old logs ensures the database remains performant in production. 

[**Read more**](../../docs/proxy/spend_logs_deletion)

## PII Masking 2.0

<Image img={require('../../img/pii_masking_v2.png')}/>

This release brings improvements to our Presidio PII Integration. As a Proxy Admin, you now have the ability to:

- Mask or block specific entities (e.g., block medical licenses while masking other entities like emails).
- Monitor guardrails in production. LiteLLM Logs will now show you the guardrail run, the entities it detected, and its confidence score for each entity.

[**Read more**](../../docs/proxy/guardrails/pii_masking_v2)

## New Models / Updated Models

- **Gemini ([VertexAI](https://docs.litellm.ai/docs/providers/vertex#usage-with-litellm-proxy-server) + [Google AI Studio](https://docs.litellm.ai/docs/providers/gemini))**
    - `/chat/completion`
        - Handle audio input - [PR](https://github.com/BerriAI/litellm/pull/10739)
        - Fixes maximum recursion depth issue when using deeply nested response schemas with Vertex AI by Increasing DEFAULT_MAX_RECURSE_DEPTH from 10 to 100 in constants. [PR](https://github.com/BerriAI/litellm/pull/10798)
        - Capture reasoning tokens in streaming mode - [PR](https://github.com/BerriAI/litellm/pull/10789)
- **[Google AI Studio](../../docs/providers/google_ai_studio/realtime)**
    - `/realtime`
        - Gemini Multimodal Live API support
        - Audio input/output support, optional param mapping, accurate usage calculation - [PR](https://github.com/BerriAI/litellm/pull/10909)
- **[VertexAI](../../docs/providers/vertex#metallama-api)**
    - `/chat/completion`
        - Fix llama streaming error - where model response was nested in returned streaming chunk - [PR](https://github.com/BerriAI/litellm/pull/10878)
- **[Ollama](../../docs/providers/ollama)**
    - `/chat/completion`
        - structure responses fix - [PR](https://github.com/BerriAI/litellm/pull/10617)
- **[Bedrock](../../docs/providers/bedrock#litellm-proxy-usage)**
    - [`/chat/completion`](../../docs/providers/bedrock#litellm-proxy-usage)
        - Handle thinking_blocks when assistant.content is None - [PR](https://github.com/BerriAI/litellm/pull/10688)
        - Fixes to only allow accepted fields for tool json schema - [PR](https://github.com/BerriAI/litellm/pull/10062)
        - Add bedrock sonnet prompt caching cost information
        - Mistral Pixtral support - [PR](https://github.com/BerriAI/litellm/pull/10439)
        - Tool caching support - [PR](https://github.com/BerriAI/litellm/pull/10897)
    - [`/messages`](../../docs/anthropic_unified)
        - allow using dynamic AWS Params - [PR](https://github.com/BerriAI/litellm/pull/10769)
- **[Nvidia NIM](../../docs/providers/nvidia_nim)**
    - [`/chat/completion`](../../docs/providers/nvidia_nim#usage---litellm-proxy-server)
        - Add tools, tool_choice, parallel_tool_calls support - [PR](https://github.com/BerriAI/litellm/pull/10763)
- **[Novita AI](../../docs/providers/novita)**
    - New Provider added for `/chat/completion` routes - [PR](https://github.com/BerriAI/litellm/pull/9527)
- **[Azure](../../docs/providers/azure)**
    - [`/image/generation`](../../docs/providers/azure#image-generation)
        - Fix azure dall e 3 call with custom model name - [PR](https://github.com/BerriAI/litellm/pull/10776)
- **[Cohere](../../docs/providers/cohere)**
    - [`/embeddings`](../../docs/providers/cohere#embedding)
        - Migrate embedding to use `/v2/embed` - adds support for output_dimensions param - [PR](https://github.com/BerriAI/litellm/pull/10809)
- **[Anthropic](../../docs/providers/anthropic)**
    - [`/chat/completion`](../../docs/providers/anthropic#usage-with-litellm-proxy)
        - Web search tool support - native + openai format - [Get Started](../../docs/providers/anthropic#anthropic-hosted-tools-computer-text-editor-web-search)
- **[VLLM](../../docs/providers/vllm)**
    - [`/embeddings`](../../docs/providers/vllm#embeddings)
        - Support embedding input as list of integers
- **[OpenAI](../../docs/providers/openai)**
    - [`/chat/completion`](../../docs/providers/openai#usage---litellm-proxy-server)
        - Fix - b64 file data input handling - [Get Started](../../docs/providers/openai#pdf-file-parsing)
        - Add ‘supports_pdf_input’ to all vision models - [PR](https://github.com/BerriAI/litellm/pull/10897)

## LLM API Endpoints
- [**Responses API**](../../docs/response_api)
    - Fix delete API support - [PR](https://github.com/BerriAI/litellm/pull/10845)
- [**Rerank API**](../../docs/rerank)
    - `/v2/rerank` now registered as ‘llm_api_route’ - enabling non-admins to call it - [PR](https://github.com/BerriAI/litellm/pull/10861)

## Spend Tracking Improvements
- **`/chat/completion`, `/messages`**
    - Anthropic - web search tool cost tracking - [PR](https://github.com/BerriAI/litellm/pull/10846)
    - Groq - update model max tokens + cost information - [PR](https://github.com/BerriAI/litellm/pull/10077)
- **`/audio/transcription`**
    - Azure - Add gpt-4o-mini-tts pricing - [PR](https://github.com/BerriAI/litellm/pull/10807)
    - Proxy - Fix tracking spend by tag - [PR](https://github.com/BerriAI/litellm/pull/10832)
- **`/embeddings`**
    - Azure AI - Add cohere embed v4 pricing - [PR](https://github.com/BerriAI/litellm/pull/10806)

## Management Endpoints / UI
- **Models**
    - Ollama - adds api base param to UI 
- **Logs**
    - Add team id, key alias, key hash filter on logs - https://github.com/BerriAI/litellm/pull/10831
    - Guardrail tracing now in Logs UI - https://github.com/BerriAI/litellm/pull/10893
- **Teams**
    - Patch for updating team info when team in org and members not in org - https://github.com/BerriAI/litellm/pull/10835
- **Guardrails**
    - Add Bedrock, Presidio, Lakers guardrails on UI - https://github.com/BerriAI/litellm/pull/10874
    - See guardrail info page - https://github.com/BerriAI/litellm/pull/10904
    - Allow editing guardrails on UI - https://github.com/BerriAI/litellm/pull/10907
- **Test Key**
    - select guardrails to test on UI 



## Logging / Alerting Integrations
- **[StandardLoggingPayload](../../docs/proxy/logging_spec)**
    - Log any `x-` headers in requester metadata - [Get Started](../../docs/proxy/logging_spec#standardloggingmetadata)
    - Guardrail tracing now in standard logging payload - [Get Started](../../docs/proxy/logging_spec#standardloggingguardrailinformation)
- **[Generic API Logger](../../docs/proxy/logging#custom-callback-apis-async)**
    - Support passing application/json header 
- **[Arize Phoenix](../../docs/observability/phoenix_integration)**
    - fix: URL encode OTEL_EXPORTER_OTLP_TRACES_HEADERS for Phoenix Integration - [PR](https://github.com/BerriAI/litellm/pull/10654)
    - add guardrail tracing to OTEL, Arize phoenix - [PR](https://github.com/BerriAI/litellm/pull/10896)
- **[PagerDuty](../../docs/proxy/pagerduty)**
    - Pagerduty is now a free feature - [PR](https://github.com/BerriAI/litellm/pull/10857)
- **[Alerting](../../docs/proxy/alerting)**
    - Sending slack alerts on virtual key/user/team updates is now free - [PR](https://github.com/BerriAI/litellm/pull/10863)


## Guardrails
- **Guardrails**
    - New `/apply_guardrail` endpoint for directly testing a guardrail - [PR](https://github.com/BerriAI/litellm/pull/10867)
- **[Lakera](../../docs/proxy/guardrails/lakera_ai)**
    - `/v2` endpoints support - [PR](https://github.com/BerriAI/litellm/pull/10880)
- **[Presidio](../../docs/proxy/guardrails/pii_masking_v2)**
    - Fixes handling of message content on presidio guardrail integration - [PR](https://github.com/BerriAI/litellm/pull/10197)
    - Allow specifying PII Entities Config - [PR](https://github.com/BerriAI/litellm/pull/10810)
- **[Aim Security](../../docs/proxy/guardrails/aim_security)**
    - Support for anonymization in AIM Guardrails - [PR](https://github.com/BerriAI/litellm/pull/10757)



## Performance / Loadbalancing / Reliability improvements
- **Allow overriding all constants using a .env variable** - [PR](https://github.com/BerriAI/litellm/pull/10803)
- **[Maximum retention period for spend logs](../../docs/proxy/spend_logs_deletion)**
    - Add retention flag to config - [PR](https://github.com/BerriAI/litellm/pull/10815)
    - Support for cleaning up logs based on configured time period - [PR](https://github.com/BerriAI/litellm/pull/10872)

## General Proxy Improvements
- **Authentication**
    - Handle Bearer $LITELLM_API_KEY in x-litellm-api-key custom header [PR](https://github.com/BerriAI/litellm/pull/10776)
- **New Enterprise pip package** - `litellm-enterprise` - fixes issue where `enterprise` folder was not found when using pip package  
- **[Proxy CLI](../../docs/proxy/management_cli)**
    - Add `models import` command - [PR](https://github.com/BerriAI/litellm/pull/10581)
- **[OpenWebUI](../../docs/tutorials/openweb_ui#per-user-tracking)**
    - Configure LiteLLM to Parse User Headers from Open Web UI
- **[LiteLLM Proxy w/ LiteLLM SDK](../../docs/providers/litellm_proxy#send-all-sdk-requests-to-litellm-proxy)**
    - Option to force/always use the litellm proxy when calling via LiteLLM SDK


## New Contributors
* [@imdigitalashish](https://github.com/imdigitalashish) made their first contribution in PR [#10617](https://github.com/BerriAI/litellm/pull/10617)
* [@LouisShark](https://github.com/LouisShark) made their first contribution in PR [#10688](https://github.com/BerriAI/litellm/pull/10688)
* [@OscarSavNS](https://github.com/OscarSavNS) made their first contribution in PR [#10764](https://github.com/BerriAI/litellm/pull/10764)
* [@arizedatngo](https://github.com/arizedatngo) made their first contribution in PR [#10654](https://github.com/BerriAI/litellm/pull/10654)
* [@jugaldb](https://github.com/jugaldb) made their first contribution in PR [#10805](https://github.com/BerriAI/litellm/pull/10805)
* [@daikeren](https://github.com/daikeren) made their first contribution in PR [#10781](https://github.com/BerriAI/litellm/pull/10781)
* [@naliotopier](https://github.com/naliotopier) made their first contribution in PR [#10077](https://github.com/BerriAI/litellm/pull/10077)
* [@damienpontifex](https://github.com/damienpontifex) made their first contribution in PR [#10813](https://github.com/BerriAI/litellm/pull/10813)
* [@Dima-Mediator](https://github.com/Dima-Mediator) made their first contribution in PR [#10789](https://github.com/BerriAI/litellm/pull/10789)
* [@igtm](https://github.com/igtm) made their first contribution in PR [#10814](https://github.com/BerriAI/litellm/pull/10814)
* [@shibaboy](https://github.com/shibaboy) made their first contribution in PR [#10752](https://github.com/BerriAI/litellm/pull/10752)
* [@camfarineau](https://github.com/camfarineau) made their first contribution in PR [#10629](https://github.com/BerriAI/litellm/pull/10629)
* [@ajac-zero](https://github.com/ajac-zero) made their first contribution in PR [#10439](https://github.com/BerriAI/litellm/pull/10439)
* [@damgem](https://github.com/damgem) made their first contribution in PR [#9802](https://github.com/BerriAI/litellm/pull/9802)
* [@hxdror](https://github.com/hxdror) made their first contribution in PR [#10757](https://github.com/BerriAI/litellm/pull/10757)
* [@wwwillchen](https://github.com/wwwillchen) made their first contribution in PR [#10894](https://github.com/BerriAI/litellm/pull/10894)


## Demo Instance

Here's a Demo Instance to test changes:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234


## [Git Diff](https://github.com/BerriAI/litellm/releases)

