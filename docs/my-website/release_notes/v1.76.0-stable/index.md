---
title: "v1.76.0-stable - RPS Improvements"
slug: "v1-76-0"
date: 2025-08-23T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
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

:::info

This release is not live yet. 
:::


---
1. New Models / Updated Models
    1. Bugs
        1. OpenAI
            1. Gpt-5 chat: clarify does not support function calling https://github.com/BerriAI/litellm/pull/13612, s/o  @superpoussin22
        2. VertexAI
            1. fix vertexai batch file format by @thiagosalvatore in https://github.com/BerriAI/litellm/pull/13576
        3. LiteLLM Proxy (`litellm_proxy/`) 
            1. Add support for calling image_edits + image_generations via SDK to Proxy - https://github.com/BerriAI/litellm/pull/13735
        4. OpenRouter
            1. Fix max_output_tokens value for anthropic Claude 4 - https://github.com/BerriAI/litellm/pull/13526
        5. Gemini
            1. Fix prompt caching cost calculation - https://github.com/BerriAI/litellm/pull/13742
        6. Azure
            1. Support `../openai/v1/respones` api base - https://github.com/BerriAI/litellm/pull/13526
            2. Fix azure/gpt-5-chat max_input_tokens - https://github.com/BerriAI/litellm/pull/13660
        7. Groq
            1. streaming ASCII encoding issue - https://github.com/BerriAI/litellm/pull/13675
        8. Baseten
            1. Refactored integration to use new openai-compatible endpoints - https://github.com/BerriAI/litellm/pull/13783
        9. Bedrock
            1. fix application inference profile for pass-through endpoints for bedrock - https://github.com/BerriAI/litellm/pull/13881
        10. DataRobot
            1. Updated URL handling for DataRobot provider URL - https://github.com/BerriAI/litellm/pull/13880
    2. Features
        1. Together AI
            1. Added Qwen3, Deepseek R1 0528 Throughput, GLM 4.5 and GPT-OSS models cost tracking - https://github.com/BerriAI/litellm/pull/13637 s/o  Tasmay-Tibrewal
        2. Fireworks AI
            1. add fireworks_ai/accounts/fireworks/models/deepseek-v3-0324 - https://github.com/BerriAI/litellm/pull/13821
        3. VertexAI
            1. Add VertexAI qwen API Service - https://github.com/BerriAI/litellm/pull/13828
            2. Add new VertexAI image models vertex_ai/imagen-4.0-generate-001, vertex_ai/imagen-4.0-ultra-generate-001, vertex_ai/imagen-4.0-fast-generate-001  - https://github.com/BerriAI/litellm/pull/13874
        4. Anthropic
            1. Add long context support w/ cost tracking - https://github.com/BerriAI/litellm/pull/13759
        5. DeepInfra
            1. Add rerank endpoint support for deepinfra - https://github.com/BerriAI/litellm/pull/13820
            2. Add new models for cost tracking - https://github.com/BerriAI/litellm/pull/13883 s/o  @Toy-97
        6. Bedrock
            1. Add tool prompt caching on async calls - https://github.com/BerriAI/litellm/pull/13803 s/o  UlookEE
            2. role chaining and session name with webauthentication for aws bedrock - https://github.com/BerriAI/litellm/pull/13753 s/o RichardoC
        7. Ollama 
            1. Handle Ollama null response when using tool calling with non-tool trained models - https://github.com/BerriAI/litellm/pull/13902
        8. OpenRouter
            1. Add deepseek/deepseek-chat-v3.1 support - https://github.com/BerriAI/litellm/pull/13897
        9. Mistral
            1. Add support for calling mistral files via chat completions - https://github.com/BerriAI/litellm/pull/13866 s/o  @jinskjoy
            2. Handle empty assistant content - https://github.com/BerriAI/litellm/pull/13671
            3. Support new ‘thinking’ response block - https://github.com/BerriAI/litellm/pull/13671
        10. Databricks
            1. remove deprecated dbrx models (dbrx-instruct, llama 3.1) - https://github.com/BerriAI/litellm/pull/13843
        11. AI/ML API
            1. Image gen api support - https://github.com/BerriAI/litellm/pull/13893

Final Count = 27



1. LLM API Endpoints
    1. Bugs
        1. Responses API
            1. add default api version for openai responses api calls - https://github.com/BerriAI/litellm/pull/13526
            2. support allowed_openai_params - https://github.com/BerriAI/litellm/pull/13671
    2. Features
        1. 

Final Count = 2


1. MCP Gateway
    1. Bugs
        1. fix StreamableHTTPSessionManager .run() error - https://github.com/BerriAI/litellm/pull/13666
    2. Features
        1. 

Final count = 1


1. Vector Stores 
    1. Bugs
        1. Bedrock
            1. Using LiteLLM Managed Credentials for Query - https://github.com/BerriAI/litellm/pull/13787
            2. 
    2. Features 

Final Count = 1 

1. Management Endpoints / UI
    1. Bugs
        1. Passthrough
            1. Fix query passthrough deletion - https://github.com/BerriAI/litellm/pull/13622
    2. Features
        1. Models 
            1. Add Search Functionality for Public Model Names in Model Dashboard - https://github.com/BerriAI/litellm/pull/13687
            2. Auto-Add `azure/` to deployment Name in UI - https://github.com/BerriAI/litellm/pull/13685
            3. Models page row UI restructure - https://github.com/BerriAI/litellm/pull/13771
        2. Notifications
            1. Add new notifications toast UI everywhere - https://github.com/BerriAI/litellm/pull/13813
        3. Keys
            1. Fix key edit settings after regenerating a key - https://github.com/BerriAI/litellm/pull/13815
            2. Require team_id when creating service account keys - https://github.com/BerriAI/litellm/pull/13873 
            3. Filter - show all options on filter option click - https://github.com/BerriAI/litellm/pull/13858
        4. Usage
            1. Fix ‘Cannot read properties of undefined’ exception on user agent activity tab - https://github.com/BerriAI/litellm/pull/13892
        5. SSO
            1. Free SSO usage for up to 5 users - https://github.com/BerriAI/litellm/pull/13843
            2. 

Final Count = 10

1. Logging / Guardrail Integrations
    1. Bugs
        1. Bedrock Guardrails
            1. Add bedrock api key support - https://github.com/BerriAI/litellm/pull/13835
    2. Features
        1. Datadog LLM Observability
            1. Add support for Failure Logging https://github.com/BerriAI/litellm/pull/13726
            2. Add time to first token, litellm overhead, guardrail overhead latency metrics - https://github.com/BerriAI/litellm/pull/13734
            3. Add support for tracing guardrail input/output - https://github.com/BerriAI/litellm/pull/13767
        2. Langfuse OTEL
            1. Allow using Key/Team Based Logging - https://github.com/BerriAI/litellm/pull/13791
        3. AIM
            1. Migrate to new firewall API - https://github.com/BerriAI/litellm/pull/13748
        4. OTEL
            1. Add OTEL tracing for actual LLM API call - https://github.com/BerriAI/litellm/pull/13836
        5. MLFlow
            1. Include predicted output in MLflow tracing - https://github.com/BerriAI/litellm/pull/13795 s/o @TomeHirata  

Final Count = 8

1. Performance / Loadbalancing / Reliability improvements
    1. Bugs
        1. Cooldowns
            1. don't return raw Azure Exceptions to client (can contain prompt leakage) - https://github.com/BerriAI/litellm/pull/13529
        2. Auto-router
            1. Ensures the relevant dependencies for auto router existing on LiteLLM Docker - https://github.com/BerriAI/litellm/pull/13788
        3. Model alias
            1. Fix calling key with access to model alias - https://github.com/BerriAI/litellm/pull/13830
        4. 
    2. Features
        1. S3 Caching [doc link] 
            1. Use namespace as prefix for s3 cache - https://github.com/BerriAI/litellm/pull/13704
            2. Async S3 Caching support (4x RPS improvement) - https://github.com/BerriAI/litellm/pull/13852 s/o @michal-otmianowski 
        2. Model Group header forwarding [doc link] 
            1. reuse same logic as global header forwarding - https://github.com/BerriAI/litellm/pull/13741
            2. add support for hosted_vllm on UI - https://github.com/BerriAI/litellm/pull/13885
        3. Performance
            1. Improve LiteLLM Python SDK RPS by +200 RPS (braintrust import + aiohttp transport fixes) - https://github.com/BerriAI/litellm/pull/13839
            2. Use O(1) Set lookups for model routing - https://github.com/BerriAI/litellm/pull/13879
            3. Reduce Significant CPU overhead from litellm_logging.py - https://github.com/BerriAI/litellm/pull/13895
            4. Improvements for Async Success Handler (Logging Callbacks) - Approx +130 RPS - https://github.com/BerriAI/litellm/pull/13905


Final Count = 11

1. General Proxy Improvements
    1. Bugs
        1. Fix litellm compatibility with newest release of openAI (>v1.100.0) - https://github.com/BerriAI/litellm/pull/13728
        2. Helm
            1. Add possibility to configure resources for migrations-job - https://github.com/BerriAI/litellm/pull/13617
            2. Ensure Helm chart auto generated master keys follow sk-xxxx format - https://github.com/BerriAI/litellm/pull/13871
            3. Enhance database configuration: add support for optional endpointKey - https://github.com/BerriAI/litellm/pull/13763
        3. Rate Limits
            1. fixing descriptor/response size mismatch on parallel_request_limiter_v3 - https://github.com/BerriAI/litellm/pull/13863 s/o  luizrennocosta
        4. Non-root
            1. fix permission access on prisma migrate in non-root image - https://github.com/BerriAI/litellm/pull/13848 s/o @Ithanil
            2. 
    2. Features


Final Count = 6


Total = 66