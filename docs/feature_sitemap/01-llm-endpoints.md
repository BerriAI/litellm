# 01 LLM endpoints

Everything a caller can use to run inference through the gateway or the SDK: OpenAI-shaped endpoints, Anthropic Messages, Google generateContent, Bedrock native shapes, the non-chat endpoints (embeddings, images, audio, batches, files, vector stores, and friends), pass-through routes, and cross-cutting request and response behavior. Model registration and provider configuration live in [02-models-providers.md](02-models-providers.md), routing and reliability in [03-routing-reliability.md](03-routing-reliability.md), and the guardrails that sit inside these calls in [06-guardrails-policies.md](06-guardrails-policies.md)

## Chat completions and messages

### llm.chat_completions: /chat/completions
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/input, https://docs.litellm.ai/docs/completion/stream, https://docs.litellm.ai/docs/completion/token_usage
code: `litellm/main.py` (`completion`, `acompletion`), `litellm/proxy/proxy_server.py` (`chat_completion`, POST `/chat/completions`), `litellm/proxy/common_request_processing.py`
tests: `tests/test_litellm/test_main.py`, `tests/test_litellm/chat_completions/`, `tests/e2e/llm_translation/test_chat_completions_contract_e2e.py`
registry: llm_conversational.yaml llm.chat_completions.*
verify: curl -X POST http://localhost:4000/chat/completions -H "Authorization: Bearer sk-1234" -H "Content-Type: application/json" -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"hi"}]}'

### llm.chat_completions.tool_calling: Function/tool calling
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/function_call
code: `litellm/main.py` (`completion`), `litellm/litellm_core_utils/streaming_handler.py` (tool call deltas)
tests: `tests/test_litellm/litellm_core_utils/test_token_counter_tool.py`, `tests/e2e/llm_translation/`
registry: llm_conversational.yaml llm.chat_completions.*.tool_use
verify: curl -X POST http://localhost:4000/chat/completions -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"weather in SF"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}' and check `tool_calls` in the response

### llm.chat_completions.structured_output: JSON mode and structured outputs
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/json_mode, https://docs.litellm.ai/docs/anthropic_unified/structured_output
code: `litellm/main.py` (`completion`), `litellm/proxy/proxy_server.py` (`chat_completion`)
tests: `tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_anthropic_messages_structured_outputs.py`, `tests/e2e/llm_translation/`
registry: llm_conversational.yaml llm.chat_completions.*.structured_output
verify: curl -X POST http://localhost:4000/chat/completions -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"give me JSON with a name field"}],"response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object","properties":{"name":{"type":"string"}}}}}}'

### llm.chat_completions.vision_and_files: Vision, audio-in, document (PDF) inputs
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/vision, https://docs.litellm.ai/docs/completion/document_understanding, https://docs.litellm.ai/docs/completion/audio
code: `litellm/main.py` (`completion`), `litellm/litellm_core_utils/prompt_templates/image_handling.py`
tests: `tests/test_litellm/litellm_core_utils/test_image_handling.py`, `tests/test_litellm/litellm_core_utils/test_extract_base64_image.py`
registry: llm_conversational.yaml llm.chat_completions.*.vision, llm.messages.*.pdf_input
verify: curl -X POST http://localhost:4000/chat/completions -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"}}]}]}'

### llm.chat_completions.reasoning: Reasoning content and effort
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/reasoning_content, https://docs.litellm.ai/docs/providers/anthropic_preserved_thinking
code: `litellm/main.py` (`completion`), `litellm/llms/anthropic/experimental_pass_through/messages/` (thinking blocks)
tests: `tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_anthropic_messages_effort.py`, `tests/e2e/llm_translation/test_deepseek_reasoning_e2e.py`
registry: llm_conversational.yaml llm.chat_completions.*.thinking
verify: curl -X POST http://localhost:4000/chat/completions -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"prove sqrt(2) is irrational"}],"reasoning_effort":"low"}' and check `reasoning_content` in the message

### llm.chat_completions.prompt_caching: Provider prompt caching
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/prompt_caching, https://docs.litellm.ai/docs/tutorials/prompt_caching
code: `litellm/main.py` (`completion`), `litellm/integrations/anthropic_cache_control_hook.py`
tests: `tests/test_litellm/integrations/langfuse/test_gemini_cached_tokens.py`, `tests/e2e/llm_translation/test_cache_control.py`
registry: llm_conversational.yaml llm.chat_completions.*.prompt_cache_5m
verify: curl -X POST http://localhost:4000/chat/completions with a long cached system block carrying `"cache_control":{"type":"ephemeral"}` and check `usage.prompt_tokens_details.cached_tokens` on the second identical call

### llm.chat_completions.web_search: Web search tool
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/web_search, https://docs.litellm.ai/docs/completion/web_fetch
code: `litellm/main.py` (`completion`), `litellm/proxy/proxy_server.py` (`chat_completion`)
tests: `tests/e2e/llm_translation/test_bedrock_web_search_server_tool_e2e.py`
registry: llm_conversational.yaml llm.messages.*.web_search
verify: curl -X POST http://localhost:4000/v1/messages -H "Authorization: Bearer sk-1234" -d '{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"latest news"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}'

### llm.chat_completions.web_search_interception: Web search interception (gateway-run search)
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/integrations/websearch_interception
code: `litellm/integrations/websearch_interception/`, `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` (`/get/websearch_interception_settings`)
tests: `tests/test_litellm/integrations/websearch_interception/`
registry: none
verify: set `general_settings.websearch_interception` in config.yaml pointing at a `/search` tool, send a chat completion with a web_search tool to a model without native search, and watch the tool execute through the gateway

### llm.chat_completions.code_interpreter_interception: Code interpreter sandbox interception
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/guides/code_interpreter, https://docs.litellm.ai/docs/sandbox
code: `litellm/integrations/code_interpreter_interception/`
tests: `tests/test_litellm/integrations/` (code interpreter modules)
registry: none
verify: enable the code interpreter interception callback in config.yaml, send `/v1/responses` with `tools:[{"type":"code_interpreter"}]`, and confirm the code runs in the configured sandbox

### llm.chat_completions.computer_use: Computer use tool
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/computer_use
code: `litellm/main.py` (`completion`)
tests: `tests/test_litellm/` (computer use related files)
registry: none
verify: send a chat completion with a computer_use tool to an Anthropic deployment and check the response includes tool_use blocks with the computer tool input

### llm.chat_completions.predicted_outputs: Predicted outputs and assistant prefix
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/predict_outputs, https://docs.litellm.ai/docs/completion/prefix
code: `litellm/main.py` (`completion`)
tests: `tests/test_litellm/` (predicted outputs related files)
registry: none
verify: curl -X POST http://localhost:4000/chat/completions -d '{"model":"gpt-5.5","messages":[...],"prediction":{"type":"content","content":"..."}}' and compare latency against the same call without prediction

### llm.chat_completions.provider_specific_params: Provider-specific params and drop_params
surfaces: sdk, api, config | flags: none
docs: https://docs.litellm.ai/docs/completion/provider_specific_params, https://docs.litellm.ai/docs/completion/drop_params
code: `litellm/main.py` (`completion`), `litellm/proxy/litellm_pre_call_utils.py`, `litellm/proxy/proxy_server.py` (`/utils/supported_openai_params`)
tests: `tests/test_litellm/test_main.py`
registry: none
verify: GET http://localhost:4000/utils/supported_openai_params?model=gpt-5.5 -H "Authorization: Bearer sk-1234", then send a param it does not list with `drop_params=true` in litellm_settings and confirm the call still succeeds

### llm.chat_completions.message_sanitization: Message trimming and sanitization
surfaces: sdk, api, config | flags: none
docs: https://docs.litellm.ai/docs/completion/message_sanitization, https://docs.litellm.ai/docs/completion/message_trimming
code: `litellm/main.py` (`completion`, `trim_messages`)
tests: `tests/test_litellm/` (message trimming related files)
registry: none
verify: send a chat completion with an invalid empty content message plus `litellm_settings: {message_sanitization: true}` and confirm the request is cleaned instead of 400ing

### llm.chat_completions.mock: Mock responses and testing
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/mock_requests, https://docs.litellm.ai/docs/tutorials/mock_completion
code: `litellm/litellm_core_utils/mock_functions.py`, `litellm/main.py` (`completion` mock_response path)
tests: `tests/test_litellm/test_main.py`
registry: none
verify: python -c 'import litellm; print(litellm.completion(model="gpt-5.5", messages=[{"role":"user","content":"hi"}], mock_response="ok"))'

### llm.text_completions: /completions (legacy text)
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/text_completion, https://docs.litellm.ai/docs/tutorials/text_completion
code: `litellm/main.py` (`text_completion`), `litellm/proxy/proxy_server.py` (POST `/completions`)
tests: `tests/e2e/llm_translation/test_completions_endpoint_e2e.py`
registry: llm_nonconversational.yaml llm.completions.*
verify: curl -X POST http://localhost:4000/v1/completions -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","prompt":"hello"}'

## Responses, Messages, and native provider shapes

### llm.responses: /v1/responses
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/response_api, https://docs.litellm.ai/docs/providers/openai/responses_api
code: `litellm/responses/main.py`, `litellm/proxy/response_api_endpoints/endpoints.py` (POST `/v1/responses`)
tests: `tests/test_litellm/responses/`, `tests/e2e/llm_translation/test_responses_e2e.py`
registry: llm_conversational.yaml llm.responses.*
verify: curl -X POST http://localhost:4000/v1/responses -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","input":"hi"}'

### llm.responses.session_state: Response id security and session continuity
surfaces: api, config | flags: db
docs: https://docs.litellm.ai/docs/response_api
code: `litellm/proxy/response_api_endpoints/endpoints.py` (GET `/v1/responses/{response_id}`), `litellm/responses/main.py`
tests: `tests/test_litellm/responses/`
registry: none
verify: create a response, then GET /v1/responses/{id} with a different virtual key and confirm it is denied unless `disable_responses_id_security` is set

### llm.responses.compaction: Native context compaction
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/response_api_compact
code: `litellm/proxy/response_api_endpoints/endpoints.py` (POST `/v1/responses/compact`)
tests: `tests/test_litellm/responses/`
registry: none
verify: curl -X POST http://localhost:4000/v1/responses/compact -H "Authorization: Bearer sk-1234" with a long conversation body and check the compacted output

### llm.messages: /v1/messages (Anthropic Messages API)
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/anthropic_unified/index, https://docs.litellm.ai/docs/anthropic_count_tokens
code: `litellm/proxy/anthropic_endpoints/endpoints.py` (POST `/v1/messages`, `/v1/messages/count_tokens`), `litellm/llms/anthropic/experimental_pass_through/messages/`
tests: `tests/test_litellm/llms/anthropic/experimental_pass_through/messages/`, `tests/e2e/llm_translation/test_messages_e2e.py`
registry: llm_conversational.yaml llm.messages.*
verify: curl -X POST http://localhost:4000/v1/messages -H "Authorization: Bearer sk-1234" -H "x-api-key: dummy" -d '{"model":"claude-sonnet-4-6","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'

### llm.messages.native_passthrough: /v1/messages native passthrough to Anthropic
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/anthropic_unified/native_passthrough
code: `litellm/proxy/anthropic_endpoints/endpoints.py` (POST `/v1/messages`)
tests: `tests/e2e/llm_translation/test_messages_e2e.py`
registry: llm_conversational.yaml llm.messages.anthropic.*
verify: point an anthropic model deployment at the real API and send /v1/messages; response keeps the native Anthropic fields (stop_reason, usage.cache_*)

### llm.generate_content: Google generateContent shape
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/generateContent, https://docs.litellm.ai/docs/providers/gemini
code: `litellm/proxy/google_endpoints/endpoints.py` (POST `/v1beta/models/{model}:generateContent`, `:streamGenerateContent`, `:countTokens`)
tests: `tests/test_litellm/google_genai/`, `tests/e2e/llm_translation/test_google_native_e2e.py`
registry: llm_nonconversational.yaml llm.google_native.*
verify: curl -X POST "http://localhost:4000/v1beta/models/gemini-2.5-flash:generateContent" -H "Authorization: Bearer sk-1234" -d '{"contents":[{"parts":[{"text":"hi"}]}]}'

### llm.bedrock_native: Bedrock converse and invoke shapes
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/bedrock_converse, https://docs.litellm.ai/docs/bedrock_invoke
code: `litellm/llms/bedrock/chat/converse_handler.py`, `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` (`/bedrock/{endpoint:path}`)
tests: `tests/e2e/llm_translation/test_bedrock_native_e2e.py`
registry: llm_nonconversational.yaml llm.bedrock_native.*
verify: curl -X POST http://localhost:4000/bedrock/model/{model}/converse -H "Authorization: Bearer sk-1234" with a converse body and check the native response

## Non-chat inference endpoints

### llm.embeddings: /embeddings
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/embedding/supported_embedding
code: `litellm/main.py` (`embedding`, `aembedding`), `litellm/proxy/proxy_server.py` (POST `/v1/embeddings`)
tests: `tests/test_litellm/embeddings/`, `tests/e2e/llm_translation/test_embeddings_endpoint_e2e.py`
registry: llm_nonconversational.yaml llm.embeddings.*
verify: curl -X POST http://localhost:4000/v1/embeddings -H "Authorization: Bearer sk-1234" -d '{"model":"text-embedding-3-small","input":"hi"}'

### llm.rerank: /rerank
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/rerank
code: `litellm/rerank_api/main.py`, `litellm/proxy/rerank_endpoints/endpoints.py` (POST `/v1/rerank`)
tests: `tests/test_litellm/rerank_api/`, `tests/e2e/llm_translation/test_rerank_e2e.py`
registry: llm_nonconversational.yaml llm.rerank.*
verify: curl -X POST http://localhost:4000/v1/rerank -H "Authorization: Bearer sk-1234" -d '{"model":"cohere/rerank-english-v3.0","query":"q","documents":["a","b"]}'

### llm.images.generations: /images/generations
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/image_generation
code: `litellm/images/main.py`, `litellm/proxy/image_endpoints/endpoints.py` (POST `/v1/images/generations`)
tests: `tests/test_litellm/images/`, `tests/e2e/llm_translation/test_image_generation_e2e.py`
registry: llm_nonconversational.yaml llm.images_generations.*
verify: curl -X POST http://localhost:4000/v1/images/generations -H "Authorization: Bearer sk-1234" -d '{"model":"dall-e-3","prompt":"a cat"}'

### llm.images.edits: /images/edits
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/image_edits
code: `litellm/images/main.py`, `litellm/proxy/image_endpoints/endpoints.py` (POST `/v1/images/edits`)
tests: `tests/test_litellm/images/test_image_edit_utils.py`, `tests/e2e/llm_translation/test_image_edits_e2e.py`
registry: llm_nonconversational.yaml llm.images_edits.*
verify: POST /v1/images/edits multipart with an input image and prompt; check a `b64_json` or url comes back

### llm.images.variations: /images/variations
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/image_variations
code: `litellm/images/main.py`
tests: `tests/test_litellm/images/`
registry: none
verify: python -c 'import litellm; litellm.image_variation(model="dall-e-2", image=open("in.png","rb"))' with a provider key set

### llm.videos: /videos (Veo, Runway, Sora)
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/videos, https://docs.litellm.ai/docs/proxy/veo_video_generation
code: `litellm/videos/main.py`, `litellm/proxy/video_endpoints/endpoints.py` (POST `/v1/videos`, GET `/v1/videos/{video_id}/content`)
tests: `tests/test_litellm/` (video related files)
registry: none
verify: curl -X POST http://localhost:4000/v1/videos -H "Authorization: Bearer sk-1234" -d '{"model":"gemini/veo-3.0-generate-001","prompt":"a wave"}' then poll GET /v1/videos/{id}

### llm.audio.transcriptions: /audio/transcriptions
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/audio_transcription
code: `litellm/main.py` (`transcription`, `atranscription`), `litellm/proxy/proxy_server.py` (POST `/v1/audio/transcriptions`)
tests: `tests/test_litellm/litellm_core_utils/test_audio_utils.py`, `tests/e2e/llm_translation/test_audio_transcriptions_e2e.py`
registry: llm_nonconversational.yaml llm.audio_transcriptions.*
verify: curl -X POST http://localhost:4000/v1/audio/transcriptions -H "Authorization: Bearer sk-1234" -F model=whisper-1 -F file=@speech.mp3

### llm.audio.speech: /audio/speech
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/text_to_speech
code: `litellm/main.py` (`speech`, `aspeech`), `litellm/proxy/proxy_server.py` (POST `/v1/audio/speech`)
tests: `tests/e2e/llm_translation/test_audio_speech_e2e.py`
registry: llm_nonconversational.yaml llm.audio_speech.*
verify: curl -X POST http://localhost:4000/v1/audio/speech -H "Authorization: Bearer sk-1234" -d '{"model":"tts-1","voice":"alloy","input":"hello"}' --output out.mp3

### llm.realtime: /realtime websocket
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/realtime, https://docs.litellm.ai/docs/providers/openai (realtime), https://docs.litellm.ai/docs/proxy/realtime_webrtc
code: `litellm/proxy/proxy_server.py` (websocket `/v1/realtime`), `litellm/proxy/realtime_endpoints/endpoints.py` (`/v1/realtime/client_secrets`)
tests: `tests/test_litellm/litellm_core_utils/test_realtime_streaming.py`, `tests/e2e/llm_translation/test_realtime_http_e2e.py`
registry: llm_nonconversational.yaml llm.realtime.*
verify: websocat ws://localhost:4000/v1/realtime?model=gpt-4o-realtime-preview with an Authorization header and confirm the session.created event arrives

### llm.moderations: /moderations
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/moderation
code: `litellm/main.py` (`moderation`, `amoderation`), `litellm/proxy/proxy_server.py` (POST `/v1/moderations`)
tests: `tests/e2e/llm_translation/test_moderations_e2e.py`
registry: llm_nonconversational.yaml llm.moderations.*
verify: curl -X POST http://localhost:4000/v1/moderations -H "Authorization: Bearer sk-1234" -d '{"model":"omni-moderation-latest","input":"test"}'

### llm.ocr: /ocr
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/ocr
code: `litellm/proxy/ocr_endpoints/endpoints.py` (POST `/v1/ocr`)
tests: `tests/e2e/llm_translation/test_ocr_rust_e2e.py`
registry: llm_nonconversational.yaml llm.ocr.*
verify: curl -X POST http://localhost:4000/v1/ocr -H "Authorization: Bearer sk-1234" -d '{"model":"mistral/mistral-ocr-latest","document":{"type":"document_url","document_url":"https://arxiv.org/pdf/2201.04234"}}'

### llm.count_tokens: Token counting (/utils/token_counter, anthropic count_tokens)
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/count_tokens, https://docs.litellm.ai/docs/anthropic_count_tokens
code: `litellm/litellm_core_utils/token_counter.py` (`token_counter`), `litellm/proxy/proxy_server.py` (POST `/utils/token_counter`), `litellm/proxy/anthropic_endpoints/endpoints.py` (POST `/v1/messages/count_tokens`)
tests: `tests/test_litellm/litellm_core_utils/test_token_counter.py`, `tests/e2e/llm_translation/test_token_counter_gemini_contents_e2e.py`
registry: llm_conversational.yaml llm.messages.*.count_tokens
verify: curl -X POST http://localhost:4000/utils/token_counter -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"hi"}]}'

## Files, batches, fine-tuning, vector stores

### llm.files: /files
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/files_endpoints
code: `litellm/files/main.py`, `litellm/proxy/openai_files_endpoints/files_endpoints.py` (POST `/v1/files`, GET `/v1/files/{file_id}/content`)
tests: `tests/test_litellm/files/`, `tests/e2e/llm_translation/test_files_batches_contract_e2e.py`
registry: llm_nonconversational.yaml llm.files.*
verify: curl -X POST http://localhost:4000/v1/files -H "Authorization: Bearer sk-1234" -F purpose=batch -F file=@in.jsonl

### llm.files.managed: LiteLLM managed files (unified file ids across providers)
surfaces: api | flags: db
docs: https://docs.litellm.ai/docs/proxy/litellm_managed_files, https://docs.litellm.ai/docs/proxy/passthrough_managed_ids
code: `litellm/proxy/openai_files_endpoints/files_endpoints.py`
tests: `tests/test_litellm/files/test_main.py`
registry: llm_nonconversational.yaml llm.files.openai.require_managed_files_*
verify: upload through /v1/files with managed files enabled, then GET /v1/files/{unified_id}/content against a different provider deployment and confirm it resolves

### llm.batches: /batches
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/batches
code: `litellm/batches/main.py`, `litellm/proxy/batches_endpoints/endpoints.py` (POST `/v1/batches`, GET `/v1/batches/{batch_id}`)
tests: `tests/test_litellm/batches/`, `tests/e2e/batches/`, `tests/e2e/llm_translation/test_files_batches_contract_e2e.py`
registry: llm_nonconversational.yaml llm.batches.*
verify: upload a .jsonl via /v1/files, then POST /v1/batches {"model":"gpt-5.5","input_file_id":"...","endpoint":"/v1/chat/completions","completion_window":"24h"} and poll GET /v1/batches/{id}

### llm.batches.managed: LiteLLM managed batches (model-group batches, cost tracking)
surfaces: api | flags: db
docs: https://docs.litellm.ai/docs/proxy/managed_batches, https://docs.litellm.ai/docs/proxy/unmanaged_vertex_batches
code: `litellm/proxy/batches_endpoints/endpoints.py`
tests: `tests/e2e/batches/`
registry: llm_nonconversational.yaml llm.batches.openai_unified.*
verify: POST /v1/batches against a model group name (not a deployment) and confirm spend shows up under the submitting key in /spend/logs

### llm.fine_tuning: /fine_tuning/jobs
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/fine_tuning, https://docs.litellm.ai/docs/guides/finetuned_models
code: `litellm/fine_tuning/main.py`, `litellm/proxy/fine_tuning_endpoints/endpoints.py` (POST `/v1/fine_tuning/jobs`)
tests: `tests/test_litellm/` (fine tuning related files)
registry: none
verify: POST /v1/fine_tuning/jobs with a training file id, then GET /v1/fine_tuning/jobs to see it listed

### llm.fine_tuning.managed: Managed fine-tuning
surfaces: api | flags: db
docs: https://docs.litellm.ai/docs/proxy/managed_finetuning
code: `litellm/proxy/fine_tuning_endpoints/endpoints.py`
tests: `tests/test_litellm/` (fine tuning related files)
registry: none
verify: create a fine-tuning job through the proxy with managed fine-tuning enabled and check the job row appears via GET /v1/fine_tuning/jobs/{id} with proxy-side status

### llm.vector_stores: /vector_stores (create, search, files)
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/vector_stores/index, https://docs.litellm.ai/docs/vector_stores/create, https://docs.litellm.ai/docs/vector_stores/search, https://docs.litellm.ai/docs/vector_store_files
code: `litellm/vector_stores/main.py`, `litellm/proxy/vector_store_endpoints/endpoints.py` (POST `/v1/vector_stores`, POST `/v1/vector_stores/{id}/search`), `litellm/proxy/vector_store_files_endpoints/endpoints.py`
tests: `tests/test_litellm/vector_stores/`, `tests/e2e/llm_translation/test_vector_stores_e2e.py`
registry: llm_nonconversational.yaml llm.vector_stores.*
verify: curl -X POST http://localhost:4000/v1/vector_stores -H "Authorization: Bearer sk-1234" -d '{"name":"t","custom_llm_provider":"openai"}' then POST /v1/vector_stores/{id}/search {"query":"x"}

### llm.vector_stores.managed: Managed vector stores and knowledge base tool
surfaces: api, ui, config | flags: db
docs: https://docs.litellm.ai/docs/vector_stores/managed_vector_stores, https://docs.litellm.ai/docs/completion/knowledgebase
code: `litellm/proxy/vector_store_endpoints/management_endpoints.py` (`/vector_store/new`, `/vector_store/list`), `litellm/proxy/vector_store_endpoints/endpoints.py`
tests: `tests/test_litellm/vector_stores/`
registry: llm_nonconversational.yaml llm.vector_stores.*
verify: open http://localhost:4000/ui/?page=vector-stores, create a store, add `vector_store_registry` to a model, and run a chat completion that returns retrieved chunks

### llm.rag: /v1/rag ingest and query
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/rag_ingest, https://docs.litellm.ai/docs/rag_query
code: `litellm/proxy/rag_endpoints/endpoints.py` (POST `/v1/rag/ingest`, POST `/v1/rag/query`)
tests: `tests/test_litellm/` (rag related files)
registry: none
verify: POST /v1/rag/ingest with a document, then POST /v1/rag/query {"query":"..."} and check the answer cites the ingested content

### llm.search: /search unified web search API
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/search/index, https://docs.litellm.ai/docs/search/tavily, https://docs.litellm.ai/docs/search/perplexity
code: `litellm/proxy/search_endpoints/endpoints.py` (POST `/v1/search`, GET `/v1/search/tools`)
tests: `tests/test_litellm/` (search related files)
registry: none
verify: curl -X POST http://localhost:4000/v1/search -H "Authorization: Bearer sk-1234" -d '{"query":"litellm","search_provider":"tavily"}' with TAVILY_API_KEY set

### llm.assistants: /assistants and /threads
surfaces: sdk, api | flags: dep
docs: https://docs.litellm.ai/docs/assistants
code: `litellm/assistants/`, `litellm/proxy/proxy_server.py` (GET/POST `/v1/assistants`, `/v1/threads`)
tests: `tests/test_litellm/` (assistants related files)
registry: none
verify: curl -X POST http://localhost:4000/v1/assistants -H "Authorization: Bearer sk-1234" -H "OpenAI-Beta: assistants=v2" -d '{"model":"gpt-5.5","name":"t"}'

### llm.containers: /containers and container files
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/containers, https://docs.litellm.ai/docs/container_files
code: `litellm/containers/`, `litellm/proxy/container_endpoints/endpoints.py` (POST `/v1/containers`)
tests: `tests/e2e/llm_translation/test_containers_e2e.py`
registry: none
verify: curl -X POST http://localhost:4000/v1/containers -H "Authorization: Bearer sk-1234" -d '{"name":"t"}' against an OpenAI deployment

### llm.evals: /v1/evals
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/evals_api, https://docs.litellm.ai/docs/tutorials/eval_suites
code: `litellm/proxy/openai_evals_endpoints/endpoints.py` (POST `/v1/evals`, POST `/v1/evals/{eval_id}/runs`)
tests: `tests/test_litellm/` (evals related files)
registry: none
verify: POST /v1/evals with a data source config, then POST /v1/evals/{id}/runs and poll GET /v1/evals/{id}/runs/{run_id}

### llm.interactions: /v1/interactions
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/interactions
code: `litellm/proxy/google_endpoints/endpoints.py` (POST `/v1beta/interactions`, GET/DELETE `/interactions/{id}`)
tests: `tests/test_litellm/interactions/`
registry: none
verify: curl -X POST http://localhost:4000/v1beta/interactions -H "Authorization: Bearer sk-1234" -d '{"model":"gemini-2.5-flash","input":"hi"}'

### llm.apply_guardrail: /guardrails/apply_guardrail (standalone guardrail call)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/apply_guardrail
code: `litellm/proxy/guardrails/guardrail_endpoints.py` (POST `/guardrails/apply_guardrail`)
tests: `tests/e2e/guardrails/`
registry: guardrail.yaml guardrail.litellm_content_filter.apply_endpoint.*
verify: register a litellm_content_filter guardrail, then curl -X POST http://localhost:4000/guardrails/apply_guardrail -H "Authorization: Bearer sk-1234" -d '{"guardrail_name":"...","text":"my ssn is 123"}'

## Pass-through and cross-cutting request behavior

### llm.passthrough.provider_routes: Provider pass-through routes (/anthropic, /gemini, /vertex_ai, /bedrock, /cohere, /openai, /azure, /assemblyai, /mistral, /cursor, /deepgram, ...)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/pass_through/intro, https://docs.litellm.ai/docs/proxy/pass_through, https://docs.litellm.ai/docs/proxy/pass_through_cost_tracking
code: `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` (`/anthropic/{endpoint:path}`, `/bedrock/{endpoint:path}`, `/vertex_ai/{endpoint:path}`, ...), `litellm/proxy/pass_through_endpoints/openai_passthrough_endpoints.py`
tests: `tests/e2e/llm_translation/test_passthrough_e2e.py`, `tests/e2e/llm_translation/test_passthrough_headers_e2e.py`
registry: llm_conversational.yaml llm.chat_completions.*.passthrough, llm.messages.*.passthrough
verify: curl -X POST http://localhost:4000/anthropic/v1/messages -H "Authorization: Bearer sk-1234" -d '{"model":"claude-sonnet-4-6","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' and check spend logged in /spend/logs

### llm.passthrough.custom: Custom pass-through endpoints from config or UI
surfaces: api, config, ui | flags: none
docs: https://docs.litellm.ai/docs/pass_through/intro, https://docs.litellm.ai/docs/proxy/pass_through
code: `litellm/proxy/pass_through_endpoints/pass_through_endpoints.py` (`/config/pass_through_endpoint` CRUD), `litellm/proxy/config_management_endpoints/pass_through_endpoints.py` (`/config/pass_through_endpoints/settings`)
tests: `tests/e2e/llm_translation/test_passthrough_e2e.py`
registry: none
verify: POST /config/pass_through_endpoint with a path and target, then call http://localhost:4000/{path} and confirm the upstream response is proxied

### llm.passthrough.websocket: OpenAI websocket passthrough
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/pass_through/openai_passthrough
code: `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` (websocket `/openai/{endpoint:path}`), `litellm/proxy/pass_through_endpoints/openai_passthrough_endpoints.py`
tests: `tests/e2e/llm_translation/test_passthrough_e2e.py`
registry: llm_conversational.yaml llm.responses.openai.passthrough_websocket
verify: set `general_settings.enable_openai_websocket_passthrough: true`, then websocat ws://localhost:4000/openai/v1/realtime?model=... and confirm the socket upgrades

### llm.streaming: Streaming (SSE) across endpoints
surfaces: sdk, api | flags: none
docs: https://docs.litellm.ai/docs/completion/stream
code: `litellm/litellm_core_utils/streaming_handler.py` (`CustomStreamWrapper`), `litellm/main.py` (`completion` stream path)
tests: `tests/test_litellm/litellm_core_utils/test_streaming_handler.py`, `tests/test_litellm/litellm_core_utils/test_streaming_chunk_builder_utils.py`, `tests/e2e/llm_translation/test_chat_stream_contract_e2e.py`
registry: llm_conversational.yaml llm.chat_completions.*.basic.stream, logging.otel.stream.*
verify: curl -N -X POST http://localhost:4000/chat/completions -H "Authorization: Bearer sk-1234" -d '{"model":"gpt-5.5","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}' and confirm SSE chunks plus a final usage chunk

### llm.request_headers: Request headers (x-litellm-*, timeout, tags, num_retries, extra headers to provider)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/request_headers, https://docs.litellm.ai/docs/proxy/request_tags
code: `litellm/proxy/litellm_pre_call_utils.py`, `litellm/proxy/common_request_processing.py`
tests: `tests/e2e/llm_translation/test_passthrough_headers_e2e.py`
registry: none
verify: send a chat completion with header `x-litellm-tags: ["sitemap-check"]` then GET /spend/logs and confirm the tag is on the spend row

### llm.response_headers: Response headers (x-litellm-response-cost, model id, rate-limit remaining, call id)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/response_headers
code: `litellm/proxy/common_request_processing.py`, `litellm/proxy/proxy_server.py` (`chat_completion`)
tests: `tests/e2e/llm_translation/test_chat_completions_contract_e2e.py`
registry: llm_conversational.yaml llm.chat_completions.bedrock_converse.response_headers.*
verify: curl -i -X POST http://localhost:4000/chat/completions ... and inspect `x-litellm-response-cost`, `x-litellm-model-id`, `x-litellm-call-id` headers

### llm.forward_client_headers: Forward client headers to LLM API
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/forward_client_headers
code: `litellm/proxy/litellm_pre_call_utils.py`, `litellm/proxy/common_request_processing.py`
tests: `tests/e2e/llm_translation/test_passthrough_headers_e2e.py`
registry: none
verify: set `general_settings.forward_client_headers_to_llm_api: true`, send a custom header on /chat/completions, and confirm it reaches the provider (check provider-side logs or a request-echo deployment)

### llm.clientside_auth: Client-side provider credentials (api_key/api_base in request)
surfaces: api, config | flags: none
docs: https://docs.litellm.ai/docs/proxy/clientside_auth
code: `litellm/proxy/litellm_pre_call_utils.py`, `litellm/proxy/proxy_server.py` (`chat_completion`)
tests: `tests/test_litellm/proxy/` (clientside auth related files)
registry: none
verify: set `general_settings.allow_client_side_credentials: true`, then POST /chat/completions with `"api_key":"<provider key>"` in the body against a model with no server-side credential

### llm.model_discovery: /models, /model/info, /model_group/info, model discovery for clients
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/model_discovery
code: `litellm/proxy/proxy_server.py` (GET `/v1/models`, `/model/info`, `/model_group/info`, `/v1/models/{model_id}`)
tests: `tests/test_litellm/proxy/` (model info related files)
registry: none
verify: curl http://localhost:4000/v1/models -H "Authorization: Bearer sk-1234" and GET /model_group/info to compare key-scoped vs admin view

### llm.route_all_chat_to_responses: Route chat completions to Responses API upstream
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/anthropic_unified/messages_to_responses_mapping
code: `litellm/responses/litellm_completion_transformation/`, `litellm/proxy/proxy_server.py` (`chat_completion`)
tests: `tests/test_litellm/completion_extras/litellm_responses_transformation/`
registry: none
verify: set `litellm_settings.route_all_chat_openai_to_responses: true`, send /chat/completions to an OpenAI model, and confirm upstream traffic is a /responses call
