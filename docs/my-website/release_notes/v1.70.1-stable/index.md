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


## New Models / Updated Models

- **Gemini** (VertexAI + Google AI Studio)
    - /chat/completion - Handle audio input - https://github.com/BerriAI/litellm/pull/10739
    - Fixes maximum recursion depth issue when using deeply nested response schemas with Vertex AI by Increasing DEFAULT_MAX_RECURSE_DEPTH from 10 to 100 in constants. https://github.com/BerriAI/litellm/pull/10798
    - Capture reasoning tokens in streaming mode - https://github.com/BerriAI/litellm/pull/10789
- **VertexAI**
    - Fix llama streaming error - where model response was nested in returned streaming chunk - https://github.com/BerriAI/litellm/pull/10878
- **Ollama**
    - structure responses fix - https://github.com/BerriAI/litellm/pull/10617
- **Bedrock**
    - `/chat/completion` - Handle thinking_blocks when assistant.content is None - https://github.com/BerriAI/litellm/pull/10688
    - `/messages` - allow using dynamic AWS Params 
    - Fixes to only allow accepted fields for tool json schema - https://github.com/BerriAI/litellm/pull/10062
    - Add bedrock sonnet prompt caching cost information
    - Mistral Pixtral support - https://github.com/BerriAI/litellm/pull/10439
    - Tool caching support - https://github.com/BerriAI/litellm/pull/10897
- **Nvidia NIM**
    - Add tools, tool_choice, parallel_tool_calls support - https://github.com/BerriAI/litellm/pull/10763
- **LiteLLM Proxy (`litellm_proxy/`)**
    - Option to force/always use the litellm proxy when calling via LiteLLM SDK - https://github.com/BerriAI/litellm/pull/10773
- **Novita AI**
    - Support on `/chat/completion`, `/completions`, `/responses` API routes - https://github.com/BerriAI/litellm/pull/9527
- **Azure**
    - Fix azure dall e 3 call with custom model name - https://github.com/BerriAI/litellm/pull/10776
    2. Add gpt-4o-mini-tts pricing - https://github.com/BerriAI/litellm/pull/10807
    3. Add cohere embed v4 pricing - https://github.com/BerriAI/litellm/pull/10806
- **Cohere**
    - Migrate embedding to use `/v2/embed` - adds support for output_dimensions param - https://github.com/BerriAI/litellm/pull/10809
- **Groq**
    - Update model max tokens + cost information - https://github.com/BerriAI/litellm/pull/10077
- **Anthropic**
    - Web search tool support - native + openai format - https://github.com/BerriAI/litellm/pull/10846
- **VLLM**
    - Support embedding input as list of integers - https://github.com/BerriAI/litellm/pull/10629
- **OpenAI**
    - Fix - b64 file data input handling - https://github.com/BerriAI/litellm/pull/10897
    - Add ‘supports_pdf_input’ to all vision models - https://github.com/BerriAI/litellm/pull/10897



## LLM API Endpoints
- **Responses API**
    - Fix delete API support - https://github.com/BerriAI/litellm/pull/10845
- **Rerank API**
    - `/v2/rerank` now registered as ‘llm_api_route’ - enabling non-admins to call it - https://github.com/BerriAI/litellm/pull/10861
- **Realtime API**
    - Gemini Multimodal Live API support - https://github.com/BerriAI/litellm/pull/10841


## Spend Tracking Improvements
- Anthropic - web search tool cost tracking - https://github.com/BerriAI/litellm/pull/10846
- **`/audio/transcription`**
    - fix tracking spend by tag - https://github.com/BerriAI/litellm/pull/10832

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
- **StandardLoggingPayload**
    - Log any `x-` headers in requester metadata - https://github.com/BerriAI/litellm/pull/10818
    - Guardrail tracing now in standard logging payload - https://github.com/BerriAI/litellm/pull/10893
- **Generic API logger**
    - Support passing application/json header 
- **Arize Phoenix**
    - fix: URL encode OTEL_EXPORTER_OTLP_TRACES_HEADERS for Phoenix Integration - https://github.com/BerriAI/litellm/pull/10654
    - add guardrail tracing to OTEL, Arize phoenix - https://github.com/BerriAI/litellm/pull/10896
- **PagerDuty**
    - Pagerduty is now a free feature - https://github.com/BerriAI/litellm/pull/10857
- **Alerting**
    - Sending slack alerts on virtual key/user/team updates is now free - https://github.com/BerriAI/litellm/pull/10863


## Guardrails
- **Guardrails**
    - New `/apply_guardrail` endpoint for directly testing a guardrail - https://github.com/BerriAI/litellm/pull/10867
- **Lakera**
    - `/v2` endpoints support - https://github.com/BerriAI/litellm/pull/10880
- **Presidio**
    - Fixes handling of message content on presidio guardrail integration - https://github.com/BerriAI/litellm/pull/10197
    - Allow specifying PII Entities Config - https://github.com/BerriAI/litellm/pull/10810
- **AIM Guardrails**
    - Support for anonymization in AIM Guardrails - https://github.com/BerriAI/litellm/pull/10757



## Performance / Loadbalancing / Reliability improvements
- **Allow overriding all constants using a .env variable** - https://github.com/BerriAI/litellm/pull/10803
- **Maximum retention period for spend logs**
    - Add retention flag to config - https://github.com/BerriAI/litellm/pull/10815
    - Support for cleaning up logs based on configured time period - https://github.com/BerriAI/litellm/pull/10872
    - Support for specifying 

## General Proxy Improvements
- **Authentication**
    - Handle Bearer $LITELLM_API_KEY in x-litellm-api-key custom header - https://github.com/BerriAI/litellm/pull/10776
- **New Enterprise pip package** - `litellm-enterprise` - fixes issue where `enterprise` folder was not found when using pip package  
- **Proxy CLI**
    - Add `models import` command - https://github.com/BerriAI/litellm/pull/10581
- **Docs**
    - Document in-memory + disk caching - https://github.com/BerriAI/litellm/pull/10522
- **OpenWebUI**
    - Configure LiteLLM to Parse User Headers from Open Web UI - https://github.com/BerriAI/litellm/pull/9802


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
