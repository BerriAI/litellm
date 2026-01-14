---
title: v1.69.0-stable - Loadbalance Batch API Models
slug: v1.69.0-stable
date: 2025-05-10T10:00:00
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
docker.litellm.ai/berriai/litellm:main-v1.69.0-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.69.0.post1
```
</TabItem>
</Tabs>

## Key Highlights

LiteLLM v1.69.0-stable brings the following key improvements:

- **Loadbalance Batch API Models**: Easily loadbalance across multiple azure batch deployments using LiteLLM Managed Files
- **Email Invites 2.0**: Send new users onboarded to LiteLLM an email invite.
- **Nscale**: LLM API for compliance with European regulations.
- **Bedrock /v1/messages**: Use Bedrock Anthropic models with Anthropic's /v1/messages.

## Batch API Load Balancing

<Image 
img={require('../../img/release_notes/lb_batch.png')}
  style={{width: '100%', display: 'block', margin: '0 0 2rem 0'}}
/>


This release brings LiteLLM Managed File support to Batches. This is great for:

- Proxy Admins: You can now control which Batch models users can call.
- Developers: You no longer need to know the Azure deployment name when creating your batch .jsonl files - just specify the model your LiteLLM key has access to. 

Over time, we expect LiteLLM Managed Files to be the way most teams use Files across `/chat/completions`, `/batch`, `/fine_tuning` endpoints. 

[Read more here](https://docs.litellm.ai/docs/proxy/managed_batches)


## Email Invites

<Image 
  img={require('../../img/email_2_0.png')}
  style={{width: '100%', display: 'block', margin: '0 0 2rem 0'}}
/>

This release brings the following improvements to our email invite integration:
- New templates for user invited and key created events.
- Fixes for using SMTP email providers.
- Native support for Resend API.
- Ability for Proxy Admins to control email events. 

For LiteLLM Cloud Users, please reach out to us if you want this enabled for your instance. 

[Read more here](https://docs.litellm.ai/docs/proxy/email)


## New Models / Updated Models
- **Gemini ([VertexAI](https://docs.litellm.ai/docs/providers/vertex#usage-with-litellm-proxy-server) + [Google AI Studio](https://docs.litellm.ai/docs/providers/gemini))**
    - Added `gemini-2.5-pro-preview-05-06` models with pricing and context window info - [PR](https://github.com/BerriAI/litellm/pull/10597)
    - Set correct context window length for all Gemini 2.5 variants - [PR](https://github.com/BerriAI/litellm/pull/10690)
- **[Perplexity](../../docs/providers/perplexity)**: 
    - Added new Perplexity models - [PR](https://github.com/BerriAI/litellm/pull/10652) 
    - Added sonar-deep-research model pricing - [PR](https://github.com/BerriAI/litellm/pull/10537)
- **[Azure OpenAI](../../docs/providers/azure)**: 
  - Fixed passing through of azure_ad_token_provider parameter - [PR](https://github.com/BerriAI/litellm/pull/10694)
- **[OpenAI](../../docs/providers/openai)**:
    - Added support for pdf url's in 'file' parameter - [PR](https://github.com/BerriAI/litellm/pull/10640)
- **[Sagemaker](../../docs/providers/aws_sagemaker)**:
    - Fix content length for `sagemaker_chat` provider - [PR](https://github.com/BerriAI/litellm/pull/10607)
- **[Azure AI Foundry](../../docs/providers/azure_ai)**: 
    - Added cost tracking for the following models [PR](https://github.com/BerriAI/litellm/pull/9956)
        - DeepSeek V3 0324
        - Llama 4 Scout
        - Llama 4 Maverick
- **[Bedrock](../../docs/providers/bedrock)**: 
    - Added cost tracking for Bedrock Llama 4 models - [PR](https://github.com/BerriAI/litellm/pull/10582)
    - Fixed template conversion for Llama 4 models in Bedrock - [PR](https://github.com/BerriAI/litellm/pull/10582)
    - Added support for using Bedrock Anthropic models with /v1/messages format - [PR](https://github.com/BerriAI/litellm/pull/10681)
    - Added streaming support for Bedrock Anthropic models with /v1/messages format - [PR](https://github.com/BerriAI/litellm/pull/10710)
- **[OpenAI](../../docs/providers/openai)**: Added `reasoning_effort` support for `o3` models - [PR](https://github.com/BerriAI/litellm/pull/10591)
- **[Databricks](../../docs/providers/databricks)**:
    - Fixed issue when Databricks uses external model and delta could be empty - [PR](https://github.com/BerriAI/litellm/pull/10540)
- **[Cerebras](../../docs/providers/cerebras)**: Fixed Llama-3.1-70b model pricing and context window - [PR](https://github.com/BerriAI/litellm/pull/10648)
- **[Ollama](../../docs/providers/ollama)**: 
    - Fixed custom price cost tracking and added 'max_completion_token' support - [PR](https://github.com/BerriAI/litellm/pull/10636)
    - Fixed KeyError when using JSON response format - [PR](https://github.com/BerriAI/litellm/pull/10611)
- 🆕 **[Nscale](../../docs/providers/nscale)**: 
    - Added support for chat, image generation endpoints - [PR](https://github.com/BerriAI/litellm/pull/10638)

## LLM API Endpoints
- **[Messages API](../../docs/anthropic_unified)**: 
    - 🆕 Added support for using Bedrock Anthropic models with /v1/messages format - [PR](https://github.com/BerriAI/litellm/pull/10681) and streaming support - [PR](https://github.com/BerriAI/litellm/pull/10710)
- **[Moderations API](../../docs/moderations)**: 
    - Fixed bug to allow using LiteLLM UI credentials for /moderations API - [PR](https://github.com/BerriAI/litellm/pull/10723)  
- **[Realtime API](../../docs/realtime)**: 
    - Fixed setting 'headers' in scope for websocket auth requests and infinite loop issues - [PR](https://github.com/BerriAI/litellm/pull/10679)
- **[Files API](../../docs/proxy/litellm_managed_files)**:
    - Unified File ID output support - [PR](https://github.com/BerriAI/litellm/pull/10713)
    - Support for writing files to all deployments - [PR](https://github.com/BerriAI/litellm/pull/10708)
    - Added target model name validation - [PR](https://github.com/BerriAI/litellm/pull/10722)
- **[Batches API](../../docs/batches)**:
    - Complete unified batch ID support - replacing model in jsonl to be deployment model name - [PR](https://github.com/BerriAI/litellm/pull/10719)
  - Beta support for unified file ID (managed files) for batches - [PR](https://github.com/BerriAI/litellm/pull/10650)


## Spend Tracking / Budget Improvements
- Bug Fix - PostgreSQL Integer Overflow Error in DB Spend Tracking - [PR](https://github.com/BerriAI/litellm/pull/10697)

## Management Endpoints / UI
- **Models**
    - Fixed model info overwriting when editing a model on UI - [PR](https://github.com/BerriAI/litellm/pull/10726)
    - Fixed team admin model updates and organization creation with specific models - [PR](https://github.com/BerriAI/litellm/pull/10539)
- **Logs**:
  - Bug Fix -  copying Request/Response on Logs Page - [PR](https://github.com/BerriAI/litellm/pull/10720)
  - Bug Fix -  log did not remain in focus on QA Logs page + text overflow on error logs - [PR](https://github.com/BerriAI/litellm/pull/10725)
  - Added index for session_id on LiteLLM_SpendLogs for better query performance - [PR](https://github.com/BerriAI/litellm/pull/10727)
- **User Management**:
  - Added user management functionality to Python client library & CLI - [PR](https://github.com/BerriAI/litellm/pull/10627)
  - Bug Fix - Fixed SCIM token creation on Admin UI - [PR](https://github.com/BerriAI/litellm/pull/10628)
  - Bug Fix - Added 404 response when trying to delete verification tokens that don't exist - [PR](https://github.com/BerriAI/litellm/pull/10605)

## Logging / Guardrail Integrations
- **Custom Logger API**: v2 Custom Callback API (send llm logs to custom api) - [PR](https://github.com/BerriAI/litellm/pull/10575), [Get Started](https://docs.litellm.ai/docs/proxy/logging#custom-callback-apis-async)
- **OpenTelemetry**:
  - Fixed OpenTelemetry to follow genai semantic conventions + support for 'instructions' param for TTS - [PR](https://github.com/BerriAI/litellm/pull/10608)
- ** Bedrock PII**:
  - Add support for PII Masking with bedrock guardrails - [Get Started](https://docs.litellm.ai/docs/proxy/guardrails/bedrock#pii-masking-with-bedrock-guardrails), [PR](https://github.com/BerriAI/litellm/pull/10608)
- **Documentation**:
  - Added documentation for StandardLoggingVectorStoreRequest - [PR](https://github.com/BerriAI/litellm/pull/10535)

## Performance / Reliability Improvements
- **Python Compatibility**:
  - Added support for Python 3.11- (fixed datetime UTC handling) - [PR](https://github.com/BerriAI/litellm/pull/10701)
  - Fixed UnicodeDecodeError: 'charmap' on Windows during litellm import - [PR](https://github.com/BerriAI/litellm/pull/10542)
- **Caching**:
  - Fixed embedding string caching result - [PR](https://github.com/BerriAI/litellm/pull/10700)
  - Fixed cache miss for Gemini models with response_format - [PR](https://github.com/BerriAI/litellm/pull/10635)

## General Proxy Improvements
- **Proxy CLI**:
  - Added `--version` flag to `litellm-proxy` CLI - [PR](https://github.com/BerriAI/litellm/pull/10704)
  - Added dedicated `litellm-proxy` CLI - [PR](https://github.com/BerriAI/litellm/pull/10578)
- **Alerting**:
  - Fixed Slack alerting not working when using a DB - [PR](https://github.com/BerriAI/litellm/pull/10370)
- **Email Invites**:
  - Added V2 Emails with fixes for sending emails when creating keys + Resend API support - [PR](https://github.com/BerriAI/litellm/pull/10602)
  - Added user invitation emails - [PR](https://github.com/BerriAI/litellm/pull/10615)
  - Added endpoints to manage email settings - [PR](https://github.com/BerriAI/litellm/pull/10646)
- **General**:
  - Fixed bug where duplicate JSON logs were getting emitted - [PR](https://github.com/BerriAI/litellm/pull/10580)


## New Contributors
- [@zoltan-ongithub](https://github.com/zoltan-ongithub) made their first contribution in [PR #10568](https://github.com/BerriAI/litellm/pull/10568)
- [@mkavinkumar1](https://github.com/mkavinkumar1) made their first contribution in [PR #10548](https://github.com/BerriAI/litellm/pull/10548)
- [@thomelane](https://github.com/thomelane) made their first contribution in [PR #10549](https://github.com/BerriAI/litellm/pull/10549)
- [@frankzye](https://github.com/frankzye) made their first contribution in [PR #10540](https://github.com/BerriAI/litellm/pull/10540)
- [@aholmberg](https://github.com/aholmberg) made their first contribution in [PR #10591](https://github.com/BerriAI/litellm/pull/10591)
- [@aravindkarnam](https://github.com/aravindkarnam) made their first contribution in [PR #10611](https://github.com/BerriAI/litellm/pull/10611)
- [@xsg22](https://github.com/xsg22) made their first contribution in [PR #10648](https://github.com/BerriAI/litellm/pull/10648)
- [@casparhsws](https://github.com/casparhsws) made their first contribution in [PR #10635](https://github.com/BerriAI/litellm/pull/10635)
- [@hypermoose](https://github.com/hypermoose) made their first contribution in [PR #10370](https://github.com/BerriAI/litellm/pull/10370)
- [@tomukmatthews](https://github.com/tomukmatthews) made their first contribution in [PR #10638](https://github.com/BerriAI/litellm/pull/10638)
- [@keyute](https://github.com/keyute) made their first contribution in [PR #10652](https://github.com/BerriAI/litellm/pull/10652)
- [@GPTLocalhost](https://github.com/GPTLocalhost) made their first contribution in [PR #10687](https://github.com/BerriAI/litellm/pull/10687)
- [@husnain7766](https://github.com/husnain7766) made their first contribution in [PR #10697](https://github.com/BerriAI/litellm/pull/10697)
- [@claralp](https://github.com/claralp) made their first contribution in [PR #10694](https://github.com/BerriAI/litellm/pull/10694)
- [@mollux](https://github.com/mollux) made their first contribution in [PR #10690](https://github.com/BerriAI/litellm/pull/10690)
