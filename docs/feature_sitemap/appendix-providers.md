# Appendix: provider x endpoint support

Generated from `provider_endpoints_support.json` at the repo root. A `yes` cell means the provider supports that endpoint through LiteLLM; a blank cell means it does not. Each provider's code lives under `litellm/llms/<provider>/` (check the dir listing for the exact slug). Doc links resolve to docs.litellm.ai provider pages where they exist

| provider | docs | chat_completions | responses | messages | a2a | interactions | embeddings | search | image_generations | audio_transcriptions | audio_speech | batches | vector_stores_search | rerank | moderations | ocr | count_tokens | rag_ingest | vector_stores_create | realtime | rag_query | assistants | fine_tuning | text_completion | image_edits | video_generations | files | skills | sandbox | generateContent | image_variations | bedrock_invoke | bedrock_converse | container | compact | vector_store_files | container_files |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a2a` |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `abliteration` | [docs](https://docs.litellm.ai/docs/providers/abliteration) | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ai21` | [docs](https://docs.litellm.ai/docs/providers/ai21) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ai21_chat` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `aihubmix` |  | yes | yes | yes |  |  | yes |  | yes | yes | yes |  |  | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `aiml` | [docs](https://docs.litellm.ai/docs/providers/aiml) | yes | yes | yes | yes | yes | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `amazon_nova` | [docs](https://docs.litellm.ai/docs/providers/amazon_nova) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `anthropic` | [docs](https://docs.litellm.ai/docs/providers/anthropic) | yes | yes | yes | yes | yes |  |  |  |  |  | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |
| `anthropic_text` |  | yes | yes | yes | yes | yes |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |
| `apertis` | [docs](https://docs.litellm.ai/docs/providers/apertis) | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `apiserpent` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `assemblyai` |  | yes | yes | yes | yes | yes |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `auto_router` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `aws_polly` | [docs](https://docs.litellm.ai/docs/providers/aws_polly) |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `azure` |  | yes | yes | yes | yes | yes | yes |  | yes | yes | yes | yes | yes |  | yes |  |  |  |  |  |  | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `azure_ai` | [docs](https://docs.litellm.ai/docs/providers/azure_ai) | yes | yes | yes | yes | yes | yes |  | yes | yes | yes | yes | yes |  | yes | yes |  |  | yes |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |
| `azure_ai/agents` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `azure_ai/doc-intelligence` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `azure_text` |  | yes | yes | yes | yes | yes |  |  |  | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `baseten` | [docs](https://docs.litellm.ai/docs/providers/baseten) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `bedrock` | [docs](https://docs.litellm.ai/docs/providers/bedrock) | yes | yes | yes | yes | yes | yes |  |  |  |  |  | yes | yes |  |  | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  | yes | yes |  |  |  |  |
| `bedrock_mantle` | [docs](https://docs.litellm.ai/docs/providers/bedrock_mantle) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `black_forest_labs` | [docs](https://docs.litellm.ai/docs/providers/black_forest_labs) |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |
| `brave` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `bytez` | [docs](https://docs.litellm.ai/docs/providers/bytez) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `cerebras` | [docs](https://docs.litellm.ai/docs/providers/cerebras) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `charity_engine` |  | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `chatgpt` | [docs](https://docs.litellm.ai/docs/providers/chatgpt) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `chutes` | [docs](https://docs.litellm.ai/docs/providers/chutes) | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `clarifai` | [docs](https://docs.litellm.ai/docs/providers/clarifai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `cloudflare` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `codestral` | [docs](https://docs.litellm.ai/docs/providers/codestral) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `cognition` | [docs](https://docs.litellm.ai/docs/providers/cognition) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `cohere` | [docs](https://docs.litellm.ai/docs/providers/cohere) | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `cohere_chat` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `cometapi` | [docs](https://docs.litellm.ai/docs/providers/cometapi) | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `compactifai` | [docs](https://docs.litellm.ai/docs/providers/compactifai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `crusoe` | [docs](https://docs.litellm.ai/docs/providers/crusoe) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `cursor` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `custom` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `custom_openai` |  | yes | yes | yes | yes | yes |  |  |  | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `darkbloom` |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `dashscope` | [docs](https://docs.litellm.ai/docs/providers/dashscope) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `databricks` | [docs](https://docs.litellm.ai/docs/providers/databricks) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `dataforseo` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `datarobot` | [docs](https://docs.litellm.ai/docs/providers/datarobot) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `deepgram` | [docs](https://docs.litellm.ai/docs/providers/deepgram) | yes | yes | yes | yes | yes |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `deepinfra` | [docs](https://docs.litellm.ai/docs/providers/deepinfra) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `deepseek` | [docs](https://docs.litellm.ai/docs/providers/deepseek) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `docker_model_runner` | [docs](https://docs.litellm.ai/docs/providers/docker_model_runner) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `duckduckgo` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `e2b` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |
| `edenai` | [docs](https://docs.litellm.ai/docs/providers/edenai) | yes | yes | yes |  |  | yes |  | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |
| `elevenlabs` | [docs](https://docs.litellm.ai/docs/providers/elevenlabs) | yes | yes | yes | yes | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `empiriolabs` | [docs](https://docs.litellm.ai/docs/providers/empiriolabs) | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `empower` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `exa_ai` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `fal_ai` | [docs](https://docs.litellm.ai/docs/providers/fal_ai) | yes | yes | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `fastcrw` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `featherless_ai` | [docs](https://docs.litellm.ai/docs/providers/featherless_ai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `firecrawl` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `fireworks_ai` | [docs](https://docs.litellm.ai/docs/providers/fireworks_ai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `friendliai` | [docs](https://docs.litellm.ai/docs/providers/friendliai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `galadriel` | [docs](https://docs.litellm.ai/docs/providers/galadriel) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `gdc` |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `gemini` | [docs](https://docs.litellm.ai/docs/providers/gemini) | yes | yes | yes | yes | yes |  |  |  |  |  |  | yes |  |  |  | yes | yes |  | yes |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |
| `gigachat` |  | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `github` | [docs](https://docs.litellm.ai/docs/providers/github) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `github_copilot` | [docs](https://docs.litellm.ai/docs/providers/github_copilot) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `gmi` | [docs](https://docs.litellm.ai/docs/providers/gmi) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `google_pse` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `gradient_ai` | [docs](https://docs.litellm.ai/docs/providers/gradient_ai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `groq` | [docs](https://docs.litellm.ai/docs/providers/groq) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `helicone` | [docs](https://docs.litellm.ai/docs/providers/helicone) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `heroku` | [docs](https://docs.litellm.ai/docs/providers/heroku) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `hosted_vllm` |  | yes | yes | yes | yes | yes | yes |  |  |  |  | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  | yes | yes |  |  |  |  |  |  |  |  |  |  |
| `huggingface` | [docs](https://docs.litellm.ai/docs/providers/huggingface) | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `hyperbolic` | [docs](https://docs.litellm.ai/docs/providers/hyperbolic) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `inception` | [docs](https://docs.litellm.ai/docs/providers/inception) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `infinity` | [docs](https://docs.litellm.ai/docs/providers/infinity) |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `jina_ai` | [docs](https://docs.litellm.ai/docs/providers/jina_ai) |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `lambda_ai` | [docs](https://docs.litellm.ai/docs/providers/lambda_ai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `langflow` |  | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `langgraph` | [docs](https://docs.litellm.ai/docs/providers/langgraph) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `lemonade` | [docs](https://docs.litellm.ai/docs/providers/lemonade) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `libertai` |  | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `linkup` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `litellm_proxy` | [docs](https://docs.litellm.ai/docs/providers/litellm_proxy) | yes | yes | yes | yes | yes | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `llamafile` | [docs](https://docs.litellm.ai/docs/providers/llamafile) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `llamagate` | [docs](https://docs.litellm.ai/docs/providers/llamagate) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `lm_studio` | [docs](https://docs.litellm.ai/docs/providers/lm_studio) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `manus` | [docs](https://docs.litellm.ai/docs/providers/manus) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |
| `maritalk` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `meta` | [docs](https://docs.litellm.ai/docs/providers/meta) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `meta_llama` | [docs](https://docs.litellm.ai/docs/providers/meta_llama) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `milvus` |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `minimax` | [docs](https://docs.litellm.ai/docs/providers/minimax) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `mistral` | [docs](https://docs.litellm.ai/docs/providers/mistral) | yes | yes | yes | yes | yes | yes |  |  |  |  | yes |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `modelscope` |  | yes | yes | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `mongodb` |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `moonshot` | [docs](https://docs.litellm.ai/docs/providers/moonshot) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `morph` | [docs](https://docs.litellm.ai/docs/providers/morph) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nadir` | [docs](https://docs.litellm.ai/docs/providers/nadir) | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nanogpt` |  | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nebius` | [docs](https://docs.litellm.ai/docs/providers/nebius) | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `neosantara` |  | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nimble` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nlp_cloud` | [docs](https://docs.litellm.ai/docs/providers/nlp_cloud) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `novita` | [docs](https://docs.litellm.ai/docs/providers/novita) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nscale` | [docs](https://docs.litellm.ai/docs/providers/nscale) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nvidia_nim` | [docs](https://docs.litellm.ai/docs/providers/nvidia_nim) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `nvidia_riva` | [docs](https://docs.litellm.ai/docs/providers/nvidia_riva) |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oci` | [docs](https://docs.litellm.ai/docs/providers/oci) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ollama` | [docs](https://docs.litellm.ai/docs/providers/ollama) | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ollama_chat` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oobabooga` |  | yes | yes | yes | yes | yes |  |  |  | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `openai` | [docs](https://docs.litellm.ai/docs/providers/openai) | yes | yes | yes | yes | yes | yes |  | yes | yes | yes | yes | yes |  | yes |  |  | yes | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  | yes |  |  | yes | yes | yes | yes |
| `openai_like` |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `openrouter` | [docs](https://docs.litellm.ai/docs/providers/openrouter) | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `opensandbox` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |
| `ovhcloud` | [docs](https://docs.litellm.ai/docs/providers/ovhcloud) | yes | yes | yes | yes | yes |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `parallel_ai` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `parasail` |  | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `perplexity` | [docs](https://docs.litellm.ai/docs/providers/perplexity) | yes | yes | yes | yes | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `petals` | [docs](https://docs.litellm.ai/docs/providers/petals) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pg_vector` |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pinstripes` |  | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `poe` | [docs](https://docs.litellm.ai/docs/providers/poe) | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `predibase` | [docs](https://docs.litellm.ai/docs/providers/predibase) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `publicai` | [docs](https://docs.litellm.ai/docs/providers/publicai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `pydantic_ai_agents` |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `qwen_ai_platform` |  | yes | yes | yes | yes | yes | yes |  | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `qwencloud` | [docs](https://docs.litellm.ai/docs/providers/qwencloud) | yes | yes | yes | yes | yes | yes |  | yes |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `ragflow` | [docs](https://docs.litellm.ai/docs/providers/ragflow) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `recraft` | [docs](https://docs.litellm.ai/docs/providers/recraft) |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `reducto` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `replicate` | [docs](https://docs.litellm.ai/docs/providers/replicate) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `runwayml` |  |  |  |  |  |  |  |  | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |
| `s3_vectors` | [docs](https://docs.litellm.ai/docs/providers/s3_vectors) |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `sagemaker` |  | yes | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `sagemaker_chat` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `sambanova` | [docs](https://docs.litellm.ai/docs/providers/sambanova) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `sap` | [docs](https://docs.litellm.ai/docs/providers/sap) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `sarvam` | [docs](https://docs.litellm.ai/docs/providers/sarvam) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `scaleway` | [docs](https://docs.litellm.ai/docs/providers/scaleway) | yes | yes | yes | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `scx-ai` |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `searchapi` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `searxng` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `serper` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `snowflake` | [docs](https://docs.litellm.ai/docs/providers/snowflake) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `soniox` |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `stability` | [docs](https://docs.litellm.ai/docs/providers/stability) |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |
| `synthetic` | [docs](https://docs.litellm.ai/docs/providers/synthetic) | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `tavily` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `tencent` | [docs](https://docs.litellm.ai/docs/providers/tencent) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `tensormesh` | [docs](https://docs.litellm.ai/docs/providers/tensormesh) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `text-completion-codestral` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `text-completion-openai` |  | yes | yes | yes | yes | yes |  |  |  | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `tinyfish` |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `together_ai` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `topaz` | [docs](https://docs.litellm.ai/docs/providers/topaz) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |
| `triton` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `v0` | [docs](https://docs.litellm.ai/docs/providers/v0) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `valkey` |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `venice` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `vercel_ai_gateway` | [docs](https://docs.litellm.ai/docs/providers/vercel_ai_gateway) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `vertex_ai` |  | yes | yes | yes | yes | yes | yes |  | yes |  | yes |  | yes |  |  | yes | yes | yes |  | yes | yes |  | yes |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |
| `vertex_ai/agent_engine` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `vllm` | [docs](https://docs.litellm.ai/docs/providers/vllm) | yes | yes | yes | yes | yes | yes |  |  |  |  | yes |  | yes |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |
| `volcengine` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `voyage` | [docs](https://docs.litellm.ai/docs/providers/voyage) |  |  |  |  |  | yes |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `wandb` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `watsonx` | [docs](https://docs.litellm.ai/docs/providers/watsonx/index) | yes | yes | yes | yes | yes | yes |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `watsonx_text` |  | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `xai` | [docs](https://docs.litellm.ai/docs/providers/xai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `xiaomi_mimo` | [docs](https://docs.litellm.ai/docs/providers/xiaomi_mimo) | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `xinference` | [docs](https://docs.litellm.ai/docs/providers/xinference) |  |  |  |  |  | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `zai` | [docs](https://docs.litellm.ai/docs/providers/zai) | yes | yes | yes | yes | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
