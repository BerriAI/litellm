---
title: v1.66.0-stable
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

tags: []
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
ghcr.io/berriai/litellm:main-v1.66.0-stable
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
- **Microsoft SSO Auto-sync**: Auto-sync groups and group members from Azure Entra ID to LiteLLM
- **Unified File IDs**: Use the same file id across LLM API providers. 
- **Security Fixes**: Fixed [CVE-2025-0330](https://www.cve.org/CVERecord?id=CVE-2025-0330) and [CVE-2024-6825](https://www.cve.org/CVERecord?id=CVE-2024-6825) vulnerabilities

Let's dive in.

## New Models / Updated Models

- xAI
    1. Added cost tracking for `xai/grok-3` models [PR](https://github.com/BerriAI/litellm/pull/9920)
    2. Added reasoning_effort support for `xai/grok-3-mini-beta` model family [PR](https://github.com/BerriAI/litellm/pull/9932)

- Hugging Face
    1. Hugging Face - Added inference providers support [PR](https://github.com/BerriAI/litellm/pull/9773)

- Azure
    1. Azure - Added azure/gpt-4o-realtime-audio cost tracking [PR](https://github.com/BerriAI/litellm/pull/9893)

- VertexAI
    1. VertexAI - Added enterpriseWebSearch tool support [PR](https://github.com/BerriAI/litellm/pull/9856)
    2. VertexAI - Moved to only passing in accepted keys by vertex ai response schema [PR](https://github.com/BerriAI/litellm/pull/8992)

- Google AI Studio
    1. Google AI Studio - Added cost tracking for `gemini-2.5-pro` [PR](https://github.com/BerriAI/litellm/pull/9837)
    2. Google AI Studio - Fixed pricing for 'gemini/gemini-2.5-pro-preview-03-25' [PR](https://github.com/BerriAI/litellm/pull/9896)
    3. Google AI Studio - Fixed handling file_data being passed in [PR](https://github.com/BerriAI/litellm/pull/9786)

- Azure
    1. Azure - Updated Azure Phi-4 pricing [PR](https://github.com/BerriAI/litellm/pull/9862)
    2. Azure - Added azure/gpt-4o-realtime-audio cost tracking [PR](https://github.com/BerriAI/litellm/pull/9893)

- Databricks
    1. Databricks - Removed reasoning_effort from parameters [PR](https://github.com/BerriAI/litellm/pull/9811)
    2. Fixed custom endpoint check for Databricks [PR](https://github.com/BerriAI/litellm/pull/9925)

- General
    1. Function Calling - Handle pydantic base model in message tool calls, handle tools = [], and support fake streaming on tool calls for meta.llama3-3-70b-instruct-v1:0 [PR](https://github.com/BerriAI/litellm/pull/9774)
    2. LiteLLM Proxy - Allow passing `thinking` param to litellm proxy via client sdk [PR](https://github.com/BerriAI/litellm/pull/9386)
    3. Reasoning - Added litellm.supports_reasoning() util to track if an llm supports reasoning [PR](https://github.com/BerriAI/litellm/pull/9923)
    4. Fixed correctly translating 'thinking' param for litellm [PR](https://github.com/BerriAI/litellm/pull/9904)


## Spend Tracking Improvements

1. Realtime API Cost tracking with token usage metrics in spend logs [PR](https://github.com/BerriAI/litellm/pull/9795)
2. Fixed Claude Haiku cache read pricing per token [PR](https://github.com/BerriAI/litellm/pull/9834)
3. Added cost tracking for Claude responses with base_model [PR](https://github.com/BerriAI/litellm/pull/9897)
4. Fixed Anthropic prompt caching cost calculation and trimmed logged message in db [PR](https://github.com/BerriAI/litellm/pull/9838)
5. Added token tracking and log usage object in spend logs [PR](https://github.com/BerriAI/litellm/pull/9843)
6. Handle custom pricing at deployment level [PR](https://github.com/BerriAI/litellm/pull/9855)


## Management Endpoints / UI

1. Test Key Tab:
    1. Added rendering of Reasoning content, ttft, usage metrics on test key page [PR](https://github.com/BerriAI/litellm/pull/9931)

    <Image 
    img={require('../../img/release_notes/chat_metrics.png')}
    style={{width: '100%', display: 'block'}}
    />
    <p style={{textAlign: 'left', color: '#666'}}>
    View input, output, reasoning tokens, ttft metrics.
    </p>
2. Tag / Policy Management:
    1. Added Tag/Policy Management [PR](https://github.com/BerriAI/litellm/pull/9813)
3. Redesigned Login Screen:
    1. Polished login screen [PR](https://github.com/BerriAI/litellm/pull/9778)
4. Keys Page:
    1. Reflected key and team updates in UI [PR](https://github.com/BerriAI/litellm/pull/9825)
2. Microsoft SSO Auto-Sync:
    1. Added debug route to allow admins to debug SSO JWT fields [PR](https://github.com/BerriAI/litellm/pull/9835)
    2. Added ability to use MSFT Graph API to assign users to teams [PR](https://github.com/BerriAI/litellm/pull/9865)
    3. Connected LiteLLM to Azure Entra ID Enterprise Application [PR](https://github.com/BerriAI/litellm/pull/9872)
    4. Added ability for admins to set `default_team_params` for when litellm SSO creates default teams [PR](https://github.com/BerriAI/litellm/pull/9895)
    5. Fixed MSFT SSO to use correct field for user email [PR](https://github.com/BerriAI/litellm/pull/9886)
    6. Added UI support for setting Default Team setting when LiteLLM SSO auto creates teams [PR](https://github.com/BerriAI/litellm/pull/9918)
4. Experimental Features:
    1. Added Tag/Policy Management [PR](https://github.com/BerriAI/litellm/pull/9813)
5. UI Bug Fixes:
    1. Prevented team, key, org, model numerical values changing on scrolling [PR](https://github.com/BerriAI/litellm/pull/9776)

## Logging / Guardrail Improvements

1. Prometheus:
    - Emit Key and Team Budget metrics on a cron job schedule [PR](https://github.com/BerriAI/litellm/pull/9528)

## Security Fixes

1. Fixed [CVE-2025-0330](https://www.cve.org/CVERecord?id=CVE-2025-0330) - Leakage of Langfuse API keys in team exception handling [PR](https://github.com/BerriAI/litellm/pull/9830)
2. Fixed [CVE-2024-6825](https://www.cve.org/CVERecord?id=CVE-2024-6825) - Remote code execution in post call rules [PR](https://github.com/BerriAI/litellm/pull/9826)

## Helm

1. Added service annotations to litellm-helm chart [PR](https://github.com/BerriAI/litellm/pull/9840)
2. Added extraEnvVars to the helm deployment [PR](https://github.com/BerriAI/litellm/pull/9292)

## Demo

Try this on the demo instance [today](https://docs.litellm.ai/docs/proxy/demo)

## Complete Git Diff

See the complete git diff since v1.65.4-stable, [here](https://github.com/BerriAI/litellm/releases/tag/v1.66.0-stable)


