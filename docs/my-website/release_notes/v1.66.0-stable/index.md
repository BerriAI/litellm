---
title: v1.66.0-stable - Realtime API Cost Tracking
slug: v1.66.0-stable
date: 2025-04-12T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

tags: ["sso", "unified_file_id", "cost_tracking", "security"]
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
docker.litellm.ai/berriai/litellm:main-v1.66.0-stable
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.66.0.post1
```
</TabItem>
</Tabs>

v1.66.0-stable is live now, here are the key highlights of this release

## Key Highlights
- **Realtime API Cost Tracking**: Track cost of realtime API calls
- **Microsoft SSO Auto-sync**: Auto-sync groups and group members from Azure Entra ID to LiteLLM
- **xAI grok-3**: Added support for `xai/grok-3` models
- **Security Fixes**: Fixed [CVE-2025-0330](https://www.cve.org/CVERecord?id=CVE-2025-0330) and [CVE-2024-6825](https://www.cve.org/CVERecord?id=CVE-2024-6825) vulnerabilities

Let's dive in.

## Realtime API Cost Tracking

<Image 
  img={require('../../img/realtime_api.png')}
  style={{width: '100%', display: 'block'}}
/>


This release adds Realtime API logging + cost tracking. 
- **Logging**: LiteLLM now logs the complete response from realtime calls to all logging integrations (DB, S3, Langfuse, etc.) 
- **Cost Tracking**: You can now set 'base_model' and custom pricing for realtime models. [Custom Pricing](../../docs/proxy/custom_pricing)
- **Budgets**: Your key/user/team budgets now work for realtime models as well.

Start [here](https://docs.litellm.ai/docs/realtime)



## Microsoft SSO Auto-sync

<Image 
  img={require('../../img/release_notes/sso_sync.png')}
  style={{width: '100%', display: 'block'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  Auto-sync groups and members from Azure Entra ID to LiteLLM
</p>

This release adds support for auto-syncing groups and members on Microsoft Entra ID with LiteLLM. This means that LiteLLM proxy administrators can spend less time managing teams and members and LiteLLM handles the following: 

- Auto-create teams that exist on Microsoft Entra ID 
- Sync team members on Microsoft Entra ID with LiteLLM teams

Get started with this [here](https://docs.litellm.ai/docs/tutorials/msft_sso)


## New Models / Updated Models

- **xAI**
    1. Added reasoning_effort support for `xai/grok-3-mini-beta` [Get Started](https://docs.litellm.ai/docs/providers/xai#reasoning-usage)
    2. Added cost tracking for `xai/grok-3` models [PR](https://github.com/BerriAI/litellm/pull/9920)

- **Hugging Face**
    1. Added inference providers support [Get Started](https://docs.litellm.ai/docs/providers/huggingface#serverless-inference-providers)

- **Azure**
    1. Added azure/gpt-4o-realtime-audio cost tracking [PR](https://github.com/BerriAI/litellm/pull/9893)

- **VertexAI**
    1. Added enterpriseWebSearch tool support [Get Started](https://docs.litellm.ai/docs/providers/vertex#grounding---web-search)
    2. Moved to only passing keys accepted by the Vertex AI response schema [PR](https://github.com/BerriAI/litellm/pull/8992)

- **Google AI Studio**
    1. Added cost tracking for `gemini-2.5-pro` [PR](https://github.com/BerriAI/litellm/pull/9837)
    2. Fixed pricing for 'gemini/gemini-2.5-pro-preview-03-25' [PR](https://github.com/BerriAI/litellm/pull/9896)
    3. Fixed handling file_data being passed in [PR](https://github.com/BerriAI/litellm/pull/9786)

- **Azure**
    1. Updated Azure Phi-4 pricing [PR](https://github.com/BerriAI/litellm/pull/9862)
    2. Added azure/gpt-4o-realtime-audio cost tracking [PR](https://github.com/BerriAI/litellm/pull/9893)

- **Databricks**
    1. Removed reasoning_effort from parameters [PR](https://github.com/BerriAI/litellm/pull/9811)
    2. Fixed custom endpoint check for Databricks [PR](https://github.com/BerriAI/litellm/pull/9925)

- **General**
    1. Added litellm.supports_reasoning() util to track if an llm supports reasoning [Get Started](https://docs.litellm.ai/docs/providers/anthropic#reasoning)
    2. Function Calling - Handle pydantic base model in message tool calls, handle tools = [], and support fake streaming on tool calls for meta.llama3-3-70b-instruct-v1:0 [PR](https://github.com/BerriAI/litellm/pull/9774)
    3. LiteLLM Proxy - Allow passing `thinking` param to litellm proxy via client sdk [PR](https://github.com/BerriAI/litellm/pull/9386)
    4. Fixed correctly translating 'thinking' param for litellm [PR](https://github.com/BerriAI/litellm/pull/9904)


## Spend Tracking Improvements
- **OpenAI, Azure**
    1. Realtime API Cost tracking with token usage metrics in spend logs [Get Started](https://docs.litellm.ai/docs/realtime)
- **Anthropic**
    1. Fixed Claude Haiku cache read pricing per token [PR](https://github.com/BerriAI/litellm/pull/9834)
    2. Added cost tracking for Claude responses with base_model [PR](https://github.com/BerriAI/litellm/pull/9897)
    3. Fixed Anthropic prompt caching cost calculation and trimmed logged message in db [PR](https://github.com/BerriAI/litellm/pull/9838)
- **General**
    1. Added token tracking and log usage object in spend logs [PR](https://github.com/BerriAI/litellm/pull/9843)
    2. Handle custom pricing at deployment level [PR](https://github.com/BerriAI/litellm/pull/9855)


## Management Endpoints / UI

- **Test Key Tab**
    1. Added rendering of Reasoning content, ttft, usage metrics on test key page [PR](https://github.com/BerriAI/litellm/pull/9931)

    <Image 
    img={require('../../img/release_notes/chat_metrics.png')}
    style={{width: '100%', display: 'block'}}
    />
    <p style={{textAlign: 'left', color: '#666'}}>
    View input, output, reasoning tokens, ttft metrics.
    </p>
- **Tag / Policy Management**
    1. Added Tag/Policy Management. Create routing rules based on request metadata. This allows you to enforce that requests with `tags="private"` only go to specific models. [Get Started](https://docs.litellm.ai/docs/tutorials/tag_management)

    <br />

    <Image 
    img={require('../../img/release_notes/tag_management.png')}
    style={{width: '100%', display: 'block'}}
    />
    <p style={{textAlign: 'left', color: '#666'}}>
    Create and manage tags.
    </p>
- **Redesigned Login Screen**
    1. Polished login screen [PR](https://github.com/BerriAI/litellm/pull/9778)
- **Microsoft SSO Auto-Sync**
    1. Added debug route to allow admins to debug SSO JWT fields [PR](https://github.com/BerriAI/litellm/pull/9835)
    2. Added ability to use MSFT Graph API to assign users to teams [PR](https://github.com/BerriAI/litellm/pull/9865)
    3. Connected litellm to Azure Entra ID Enterprise Application [PR](https://github.com/BerriAI/litellm/pull/9872)
    4. Added ability for admins to set `default_team_params` for when litellm SSO creates default teams [PR](https://github.com/BerriAI/litellm/pull/9895)
    5. Fixed MSFT SSO to use correct field for user email [PR](https://github.com/BerriAI/litellm/pull/9886)
    6. Added UI support for setting Default Team setting when litellm SSO auto creates teams [PR](https://github.com/BerriAI/litellm/pull/9918)
- **UI Bug Fixes**
    1. Prevented team, key, org, model numerical values changing on scrolling [PR](https://github.com/BerriAI/litellm/pull/9776)
    2. Instantly reflect key and team updates in UI [PR](https://github.com/BerriAI/litellm/pull/9825)

## Logging / Guardrail Improvements

- **Prometheus**
    1. Emit Key and Team Budget metrics on a cron job schedule [Get Started](https://docs.litellm.ai/docs/proxy/prometheus#initialize-budget-metrics-on-startup)

## Security Fixes

- Fixed [CVE-2025-0330](https://www.cve.org/CVERecord?id=CVE-2025-0330) - Leakage of Langfuse API keys in team exception handling [PR](https://github.com/BerriAI/litellm/pull/9830)
- Fixed [CVE-2024-6825](https://www.cve.org/CVERecord?id=CVE-2024-6825) - Remote code execution in post call rules [PR](https://github.com/BerriAI/litellm/pull/9826)

## Helm

- Added service annotations to litellm-helm chart [PR](https://github.com/BerriAI/litellm/pull/9840)
- Added extraEnvVars to the helm deployment [PR](https://github.com/BerriAI/litellm/pull/9292)

## Demo

Try this on the demo instance [today](https://docs.litellm.ai/docs/proxy/demo)

## Complete Git Diff

See the complete git diff since v1.65.4-stable, [here](https://github.com/BerriAI/litellm/releases/tag/v1.66.0-stable)


