---
title: "[Preview] v1.80.7-stable - RAG API"
slug: "v1-80-7"
date: 2025-11-27T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
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

```showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-v1.80.7
```

</TabItem>

<TabItem value="pip" label="Pip">

```showLineNumbers title="pip install litellm"
pip install litellm==1.80.7
```

</TabItem>
</Tabs>

---

## Key Highlights

- **New RAG API** -[Unified RAG API with support for Vertex AI RAG engine and OpenAI Vector Stores](../../docs/rag_ingest)
- **Claude Skills API** - [Support for Anthropic's new Skills API with extended context and tool calling](../docs/skills)
- **Organization Usage** - Filter and track usage analytics at the organization level
- **Claude Opus 4.5** - Support for Anthropic's Claude Opus 4.5 via Anthropic, Bedrock, VertexAI
- **Guardrails for Passthrough** - Guardrails support for pass-through endpoints
- **Public AI Provider** - Support for publicai.co provider

---

### RAG API

<Image
img={require('../../img/release_notes/rag_api.png')}
style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

<br/>

Introducing a new RAG API on LiteLLM AI Gateway. You can provide documents (TXT, PDF, DOCX files) to LiteLLM's all-in-one document ingestion pipeline and it will handle OCR recognition, chunking, embedding, and storing data in your vector store of choice (OpenAI, Bedrock, Vertex AI, etc.).

Example usage for ingestion 

```showLineNumbers title="Ingest txt file Bedrock Knowledge Base"
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d "{
        \"file\": {
            \"filename\": \"document.txt\",
            \"content\": \"$(base64 -i document.txt)\",
            \"content_type\": \"text/plain\"
        },
        \"ingest_options\": {
            \"vector_store\": {
                \"custom_llm_provider\": \"bedrock\"
            }
        }
    }"
```

Example usage for querying the vector store

```showLineNumbers title="Search the Bedrock Knowledge Base"
curl -X POST "http://localhost:4000/v1/vector_stores/vs_692658d337c4819183f2ad8488d12fc9/search" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is LiteLLM?",
        "custom_llm_provider": "bedrock"
    }'
```

[Get Started](../../docs/rag_ingest)

---

### Organization Usage

<Image
img={require('../../img/release_notes/organization_usage.png')}
style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

Users can now filter usage statistics by organization, providing the same granular filtering capabilities available for teams.

**Details:**

- Filter usage analytics, spend logs, and activity metrics by organization ID
- View organization-level breakdowns alongside existing team and user-level filters
- Consistent filtering experience across all usage and analytics views

[PR #16560](https://github.com/BerriAI/litellm/pull/16560), [PR #17181](https://github.com/BerriAI/litellm/pull/17181)

---

## New Providers and Endpoints

### New Providers

| Provider | Supported Endpoints | Description |
| -------- | ------------------- | ----------- |
| [Public AI](../../docs/providers/publicai) | Chat completions | Support for publicai.co provider |
| [Eleven Labs](../../docs/providers/text_to_speech) | Text-to-speech | Text-to-speech provider integration |

### New LLM API Endpoints

| Endpoint | Method | Description | Documentation |
| -------- | ------ | ----------- | ------------- |
| `/v1/skills` | POST | Anthropic Skills API for extended context tool calling | [Skills API](../../docs/skills) |
| `/rag/ingest` | POST | Unified RAG API with Vertex AI RAG and Vector Stores | [RAG API](../../docs/rag_ingest) |

---

## New Models / Updated Models

#### New Model Support

| Provider | Model | Context Window | Input ($/1M tokens) | Output ($/1M tokens) | Features |
| -------- | ----- | -------------- | ------------------- | -------------------- | -------- |
| Anthropic | `claude-opus-4-5-20251101` | 200K | $5.00 | $25.00 | Chat, reasoning, vision, function calling, prompt caching |
| Bedrock | `anthropic.claude-opus-4-5-20251101-v1:0` | 200K | $5.00 | $25.00 | Chat, reasoning, vision, function calling, prompt caching |
| Bedrock | `us.anthropic.claude-opus-4-5-20251101-v1:0` | 200K | $5.00 | $25.00 | Chat, reasoning, vision, function calling, prompt caching |
| Bedrock | `amazon.nova-canvas-v1:0` | - | - | $0.06/image | Image generation |
| OpenRouter | `openrouter/anthropic/claude-opus-4.5` | 200K | $5.00 | $25.00 | Chat, reasoning, vision, function calling, prompt caching |
| Vertex AI | `vertex_ai/claude-opus-4-5` | 200K | $5.00 | $25.00 | Chat, reasoning, vision, function calling, prompt caching |
| Vertex AI | `vertex_ai/claude-opus-4-5@20251101` | 200K | $5.00 | $25.00 | Chat, reasoning, vision, function calling, prompt caching |
| Azure | `azure_ai/claude-opus-4-1` | 200K | $15.00 | $75.00 | Chat, reasoning, vision, function calling, prompt caching |
| Azure | `azure_ai/claude-sonnet-4-5` | 200K | $3.00 | $15.00 | Chat, reasoning, vision, function calling, prompt caching |
| Azure | `azure_ai/claude-haiku-4-5` | 200K | $1.00 | $5.00 | Chat, reasoning, vision, function calling, prompt caching |
| Fireworks AI | `fireworks_ai/accounts/fireworks/models/glm-4p6` | 202K | $0.55 | $2.19 | Chat, function calling |
| Public AI | `publicai/swiss-ai/apertus-8b-instruct` | 8K | Free | Free | Chat, function calling |
| Public AI | `publicai/swiss-ai/apertus-70b-instruct` | 8K | Free | Free | Chat, function calling |
| Public AI | `publicai/aisingapore/Gemma-SEA-LION-v4-27B-IT` | 8K | Free | Free | Chat, function calling |
| Public AI | `publicai/BSC-LT/salamandra-7b-instruct-tools-16k` | 16K | Free | Free | Chat, function calling |
| Public AI | `publicai/BSC-LT/ALIA-40b-instruct_Q8_0` | 8K | Free | Free | Chat, function calling |
| Public AI | `publicai/allenai/Olmo-3-7B-Instruct` | 32K | Free | Free | Chat, function calling |
| Public AI | `publicai/aisingapore/Qwen-SEA-LION-v4-32B-IT` | 32K | Free | Free | Chat, function calling |
| Public AI | `publicai/allenai/Olmo-3-7B-Think` | 32K | Free | Free | Chat, function calling, reasoning |
| Public AI | `publicai/allenai/Olmo-3-32B-Think` | 32K | Free | Free | Chat, function calling, reasoning |
| Cohere | `embed-multilingual-light-v3.0` | 1K | $0.10 | - | Embeddings, supports images |
| WatsonX | `watsonx/whisper-large-v3-turbo` | - | $0.0001/sec | - | Audio transcription |

#### Features

- **[Anthropic](../../docs/providers/anthropic)**
    - Add claude opus 4.5 model support - [PR #17043](https://github.com/BerriAI/litellm/pull/17043)
    - Add day 0 support for anthropic Tool Search, Programmatic Tool Calling, Input Examples, Effort Parameter - [PR #17091](https://github.com/BerriAI/litellm/pull/17091), [Docs](../../blog/anthropic_opus_4_5_and_advanced_features)
    - Add Anthropic Effort Parameter support - [PR #17091](https://github.com/BerriAI/litellm/pull/17091)

- **[Bedrock](../../docs/providers/bedrock)**
    - Fix bedrock claude opus 4.5 inference profile - only global currently - [PR #17101](https://github.com/BerriAI/litellm/pull/17101)
    - Add OpenAI compatible bedrock imported models (qwen etc) - [PR #17097](https://github.com/BerriAI/litellm/pull/17097)
    - Fix bedrock passthrough auth issue - [PR #16879](https://github.com/BerriAI/litellm/pull/16879)
    - Make Bedrock image generation more consistent - [PR #17021](https://github.com/BerriAI/litellm/pull/17021)

- **[Azure](../../docs/providers/azure)**
    - Add support for azure anthropic models via chat completion - [PR #16886](https://github.com/BerriAI/litellm/pull/16886)
    - Fix the azure auth format for videos - [PR #17009](https://github.com/BerriAI/litellm/pull/17009)
    - Fix `reasoning_effort="none"` not working on Azure for GPT-5.1 - [PR #17071](https://github.com/BerriAI/litellm/pull/17071)
    - Add GA protocol as configurable parameter for azure openai realtime api - [PR #17096](https://github.com/BerriAI/litellm/pull/17096)

- **[OpenRouter](../../docs/providers/openrouter)**
    - Add OpenRouter Opus 4.5 - [PR #17144](https://github.com/BerriAI/litellm/pull/17144)

- **[Fireworks AI](../../docs/providers/fireworksai)**
    - Add fireworks_ai/accounts/fireworks/models/glm-4p6 - [PR #17154](https://github.com/BerriAI/litellm/pull/17154)

- **[Vertex AI](../../docs/providers/vertex)**
    - Add vertex ai image gen support for both gemini and imagen models - [PR #17070](https://github.com/BerriAI/litellm/pull/17070)
    - Handle global location in context caching - [PR #16997](https://github.com/BerriAI/litellm/pull/16997)
    - Fix CreateCachedContentRequest enum error - [PR #16965](https://github.com/BerriAI/litellm/pull/16965)
    - Use the correct domain for the global location when counting tokens - [PR #17116](https://github.com/BerriAI/litellm/pull/17116)
    - Support Vertex AI batch listing in LiteLLM proxy - [PR #17079](https://github.com/BerriAI/litellm/pull/17079)
    - Fix default sample count for image generation - [PR #16403](https://github.com/BerriAI/litellm/pull/16403)

- **[Gemini](../../docs/providers/gemini)**
    - Add gemini file search support - [PR #17124](https://github.com/BerriAI/litellm/pull/17124)
    - Add gemini-3-pro-image-preview model support for imageSize parameter - [PR #17019](https://github.com/BerriAI/litellm/pull/17019)
    - Handle None or empty contents in Gemini token counter - [PR #17020](https://github.com/BerriAI/litellm/pull/17020)
    - Skip thinking config for image models - [PR #17027](https://github.com/BerriAI/litellm/pull/17027)

- **[WatsonX](../../docs/providers/watsonx)**
    - Add audio transcriptions for WatsonX - [PR #17160](https://github.com/BerriAI/litellm/pull/17160)

- **[OpenAI](../../docs/providers/openai)**
    - Fix gpt-5.1 temperature support when reasoning_effort is "none" or not specified - [PR #17011](https://github.com/BerriAI/litellm/pull/17011)

- **[Public AI](../../docs/providers/publicai)**
    - Add Provider publicai.co - [PR #17230](https://github.com/BerriAI/litellm/pull/17230)

- **[Cohere](../../docs/providers/cohere)**
    - Add cost tracking for cohere embed passthrough endpoint - [PR #17029](https://github.com/BerriAI/litellm/pull/17029)

- **[Eleven Labs](../../docs/providers/text_to_speech)**
    - Integrate eleven labs text-to-speech - [PR #16573](https://github.com/BerriAI/litellm/pull/16573)

### Bug Fixes

- **[OCI](../../docs/providers/oci)**
    - Fix pydantic validation errors during tool call with streaming - [PR #16899](https://github.com/BerriAI/litellm/pull/16899)

---

## LLM API Endpoints

#### Features

- **[Skills API (Anthropic)](../../docs/ski)**
    - New API - Claude Skills API. Create, List, Delete, Update Claude Skills - [PR #17042](https://github.com/BerriAI/litellm/pull/17042), [Docs](../../docs/skills)

- **[RAG API](../../docs/rag_ingest)**
    - New RAG API on LiteLLM AI Gateway (use with OpenAI Vector Store, Bedrock Knowledge Bases, Vertex AI RAG Engine) - [PR #17109](https://github.com/BerriAI/litellm/pull/17109)
    - Add support for Vertex RAG engine - [PR #17117](https://github.com/BerriAI/litellm/pull/17117)
    - Allow internal user keys to access api, allow using litellm credentials with API - [PR #17169](https://github.com/BerriAI/litellm/pull/17169)

- **[Search API](../../docs/search/index)**
    - Add search API logging and cost tracking in LiteLLM Proxy - [PR #17078](https://github.com/BerriAI/litellm/pull/17078)

- **[Responses API](../../docs/response_api)**
    - Fix prevent duplicate spend logs in Responses API for non-OpenAI providers - [PR #16992](https://github.com/BerriAI/litellm/pull/16992)
    - Support response_format parameter in completion -> responses bridge - [PR #16844](https://github.com/BerriAI/litellm/pull/16844)
    - Fix mcp tool call response logging + remove unmapped param error mid-stream - allows gpt-5 web search to work via responses api - [PR #16946](https://github.com/BerriAI/litellm/pull/16946)
    - Add header passing support for MCP tools in Responses API - [PR #16877](https://github.com/BerriAI/litellm/pull/16877)

- **[Image Edits API](../../docs/image_generation)**
    - Fix image edit endpoint - [PR #17046](https://github.com/BerriAI/litellm/pull/17046)

- **[Audio Transcription API](../../docs/audio_transcription)**
    - Add transcription exception handling for /audio/transcriptions - [PR #16791](https://github.com/BerriAI/litellm/pull/16791)
    - Fix 401 when audio/transcriptions - [PR #17023](https://github.com/BerriAI/litellm/pull/17023)

- **[Embeddings API](../../docs/embedding/supported_embedding)**
    - Add header forwarding in embeddings - [PR #16869](https://github.com/BerriAI/litellm/pull/16869)

- **[Passthrough Endpoints](../../docs/pass_through/vertex_ai)**
    - Add cost tracking for streaming in vertex ai passthrough - [PR #16874](https://github.com/BerriAI/litellm/pull/16874)
    - Add cost tracking for cohere embed passthrough endpoint - [PR #17029](https://github.com/BerriAI/litellm/pull/17029)

- **[Vector Stores](../../docs/vector_stores)**
    - Add method for extracting vector store ids from path params - [PR #16566](https://github.com/BerriAI/litellm/pull/16566)

- **General**
    - Fix propagate x-litellm-model-id in responses - [PR #16986](https://github.com/BerriAI/litellm/pull/16986)
    - Preserve content field even if null - [PR #16988](https://github.com/BerriAI/litellm/pull/16988)
    - Include server_tool_use in streaming usage - [PR #16826](https://github.com/BerriAI/litellm/pull/16826)
    - Fix Thinking may not be enabled when tool_choice forces tool use - [PR #17129](https://github.com/BerriAI/litellm/pull/17129)
    - Add missing standard logging object fields - [PR #17135](https://github.com/BerriAI/litellm/pull/17135)

#### Bugs

- **General**
    - Fix vector Store List Endpoint Returns 404 - [PR #17229](https://github.com/BerriAI/litellm/pull/17229)
    - Fix Videos lint errors - [PR #17125](https://github.com/BerriAI/litellm/pull/17125)
    - Do not include plaintext message in exception - [PR #17216](https://github.com/BerriAI/litellm/pull/17216)

---

## Management Endpoints / UI

#### Features

- **Virtual Keys**
    - Fix Create Key Duration - [PR #17170](https://github.com/BerriAI/litellm/pull/17170)

- **Models + Endpoints**
    - Allow adding Bedrock API Key when adding models - [PR #17153](https://github.com/BerriAI/litellm/pull/17153)
    - Add aws_bedrock_runtime_endpoint into Credential Types - [PR #17053](https://github.com/BerriAI/litellm/pull/17053)
    - Change provider create fields to JSON - [PR #16985](https://github.com/BerriAI/litellm/pull/16985)
    - Change model_hub_table to call getUiConfig before Fetching Public Data - [PR #17166](https://github.com/BerriAI/litellm/pull/17166)
    - Improve Wording for Config Models in Model Table - [PR #17100](https://github.com/BerriAI/litellm/pull/17100)

- **Teams & Users**
    - Deleting a User From Team Deletes key User Created for Team - [PR #17057](https://github.com/BerriAI/litellm/pull/17057)
    - Hide Default Team Settings From Proxy Admin Viewers - [PR #16900](https://github.com/BerriAI/litellm/pull/16900)
    - Add No Default Models for Team and User Settings - [PR #17037](https://github.com/BerriAI/litellm/pull/17037)
    - User Table Sort by All - [PR #17108](https://github.com/BerriAI/litellm/pull/17108)
    - Org Admin Team Permissions Fix - [PR #17110](https://github.com/BerriAI/litellm/pull/17110)
    - Better Loading State for Internal User Page - [PR #17168](https://github.com/BerriAI/litellm/pull/17168)

- **Permission Management**
    - Add `reject_metadata_tags` to prevent users from sending metadata.tags directly in requests - [PR #17088](https://github.com/BerriAI/litellm/pull/17088)
    - Disable global guardrails by key/team - [PR #16983](https://github.com/BerriAI/litellm/pull/16983)
    - Tool permission argument check - [PR #16982](https://github.com/BerriAI/litellm/pull/16982)
    - Add UI support for configuring tool permission guardrails - [PR #17050](https://github.com/BerriAI/litellm/pull/17050)

- **MCP Gateway**
    - Add backend support for OAuth2 auth_type registration via UI - [PR #17006](https://github.com/BerriAI/litellm/pull/17006)
    - Add UI support for registering MCP OAuth2 auth_type - [PR #17007](https://github.com/BerriAI/litellm/pull/17007)

- **General UI Improvements**
    - Ensure Unique Keys in Navbar Menu Items - [PR #16987](https://github.com/BerriAI/litellm/pull/16987)
    - Minor Cosmetic Changes for Buttons, Add Notification for Delete Team - [PR #16984](https://github.com/BerriAI/litellm/pull/16984)
    - Change Delete Modals to Common Component - [PR #17068](https://github.com/BerriAI/litellm/pull/17068)
    - Disable edit, delete, info for dynamically generated spend tags - [PR #17098](https://github.com/BerriAI/litellm/pull/17098)
    - Migrate modelInfoCall to ReactQuery - [PR #17123](https://github.com/BerriAI/litellm/pull/17123)
    - Migrate Provider Fields to React Query - [PR #17177](https://github.com/BerriAI/litellm/pull/17177)
    - Fix Flaky Test - [PR #17161](https://github.com/BerriAI/litellm/pull/17161)
    - Change Add Fallback Modal to use Antd Select - [PR #17223](https://github.com/BerriAI/litellm/pull/17223)

- **Infrastructure**
    - Non Root Docker Build - [PR #17060](https://github.com/BerriAI/litellm/pull/17060)
    - Add nodejs and npm to docker image for prisma generate - [PR #16903](https://github.com/BerriAI/litellm/pull/16903)
    - Bump: version 0.4.8 → 0.4.9 - [PR #17163](https://github.com/BerriAI/litellm/pull/17163)

- **Helm**
    - Enhancement: ServiceMonitor template rendering - [PR #17038](https://github.com/BerriAI/litellm/pull/17038)

#### Bugs

- **Database**
    - Distinguish permission errors from idempotent errors in Prisma migrations - [PR #17064](https://github.com/BerriAI/litellm/pull/17064)

---

## AI Integrations

### Logging

- **General**
    - Model Armor - Logging guardrail response on llm responses - [PR #16977](https://github.com/BerriAI/litellm/pull/16977)
    - Add missing standard logging object fields - [PR #17135](https://github.com/BerriAI/litellm/pull/17135)
    - Add cost tracking for cohere embed passthrough endpoint - [PR #17029](https://github.com/BerriAI/litellm/pull/17029)
    - Add cost tracking for streaming in vertex ai passthrough - [PR #16874](https://github.com/BerriAI/litellm/pull/16874)

### Guardrails

- **[Presidio](../../docs/proxy/guardrails)**
    - Add presidio pii masking tutorial with litellm - [PR #16969](https://github.com/BerriAI/litellm/pull/16969)

- **General**
    - Prompt security litellm - [PR #16365](https://github.com/BerriAI/litellm/pull/16365)
    - Add guardrails for pass through endpoints - [PR #17221](https://github.com/BerriAI/litellm/pull/17221)
    - Allow adding pass through guardrails through UI - [PR #17226](https://github.com/BerriAI/litellm/pull/17226)

### Prompt Management

- **General**
    - AI gateway prompt management documentation - [PR #16990](https://github.com/BerriAI/litellm/pull/16990)

---

## MCP Gateway

- **OAuth 2.0**
    - Add backend support for OAuth2 auth_type registration via UI - [PR #17006](https://github.com/BerriAI/litellm/pull/17006)
    - Add UI support for registering MCP OAuth2 auth_type - [PR #17007](https://github.com/BerriAI/litellm/pull/17007)

- **Tool Permissions**
    - Tool permission argument check - [PR #16982](https://github.com/BerriAI/litellm/pull/16982)
    - Add UI support for configuring tool permission guardrails - [PR #17050](https://github.com/BerriAI/litellm/pull/17050)

- **Configuration**
    - Remove unused MCP_PROTOCOL_VERSION_HEADER_NAME constant - [PR #17008](https://github.com/BerriAI/litellm/pull/17008)
    - Add header passing support for MCP tools in Responses API - [PR #16877](https://github.com/BerriAI/litellm/pull/16877)
    - Fix missing await - [PR #17103](https://github.com/BerriAI/litellm/pull/17103)

---

## Performance / Loadbalancing / Reliability improvements

- **Memory Optimization**
    - Lazy-load cost_calculator & logging to reduce memory + import time - [PR #17089](https://github.com/BerriAI/litellm/pull/17089)

- **Dependency Management**
    - Downgrade grpcio to < 1.68.0 - [PR #17090](https://github.com/BerriAI/litellm/pull/17090)
    - Upgrade websockets to v15 - [PR #16734](https://github.com/BerriAI/litellm/pull/16734)

- **Database Performance**
    - Optimize date filtering for spend logs queries - [PR #17073](https://github.com/BerriAI/litellm/pull/17073)

- **Request Handling**
    - Add automatic LiteLLM context headers (Pillar integration) - [PR #17076](https://github.com/BerriAI/litellm/pull/17076)

- **Generic API Support**
    - Make generic api OSS + support multiple generic API's - [PR #17152](https://github.com/BerriAI/litellm/pull/17152)

---

## Documentation Updates

- **Provider Documentation**
    - Model Compare UI - [PR #16979](https://github.com/BerriAI/litellm/pull/16979)
    - Perf release notes for v1.80.5-stable - [PR #16978](https://github.com/BerriAI/litellm/pull/16978)
    - Claude Skills API - [PR #17052](https://github.com/BerriAI/litellm/pull/17052)
    - Add strands tutorial - [PR #17039](https://github.com/BerriAI/litellm/pull/17039)

- **General Documentation**
    - AI gateway prompt management - [PR #16990](https://github.com/BerriAI/litellm/pull/16990)
    - Cleanup README and improve agent guides - [PR #17003](https://github.com/BerriAI/litellm/pull/17003)
    - Update broken documentation links in README - [PR #17002](https://github.com/BerriAI/litellm/pull/17002)
    - Update version and add preview tag - [PR #17032](https://github.com/BerriAI/litellm/pull/17032)
    - Document model pricing contribution process - [PR #17031](https://github.com/BerriAI/litellm/pull/17031)
    - Document event hook usage - [PR #17035](https://github.com/BerriAI/litellm/pull/17035)
    - Link to logging spec in callback docs - [PR #17049](https://github.com/BerriAI/litellm/pull/17049)
    - Add OpenAI Agents SDK to projects - [PR #17203](https://github.com/BerriAI/litellm/pull/17203)

---

## New Contributors

* @prawaan made their first contribution in [PR #16997](https://github.com/BerriAI/litellm/pull/16997)
* @lior-ps made their first contribution in [PR #16365](https://github.com/BerriAI/litellm/pull/16365)
* @HaiyiMei made their first contribution in [PR #17020](https://github.com/BerriAI/litellm/pull/17020)
* @yuya2017 made their first contribution in [PR #17064](https://github.com/BerriAI/litellm/pull/17064)
* @saar-win made their first contribution in [PR #17038](https://github.com/BerriAI/litellm/pull/17038)
* @sdip15fa made their first contribution in [PR #16965](https://github.com/BerriAI/litellm/pull/16965)
* @KeremTurgutlu made their first contribution in [PR #16826](https://github.com/BerriAI/litellm/pull/16826)
* @choigawoon made their first contribution in [PR #17019](https://github.com/BerriAI/litellm/pull/17019)
* @SamAcctX made their first contribution in [PR #17144](https://github.com/BerriAI/litellm/pull/17144)
* @naaa760 made their first contribution in [PR #17079](https://github.com/BerriAI/litellm/pull/17079)
* @abi-jey made their first contribution in [PR #17096](https://github.com/BerriAI/litellm/pull/17096)
* @hxyannay made their first contribution in [PR #16734](https://github.com/BerriAI/litellm/pull/16734)

---

## **[Full Changelog](https://github.com/BerriAI/litellm/compare/v1.80.5.rc.2...v1.80.7)**
