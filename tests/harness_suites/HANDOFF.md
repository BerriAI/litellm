# Non-chat live tests: handoff to Sameer

The chat-scope keep/drop audit (reports/ci-auditor.md, 8c) identified these
live tests as harness material, but they are non-chat surfaces (embeddings,
rerank, transcription, TTS, agents, skills, evals) and therefore Sameer's
scope per 02-mateo-scope.md "Out of scope". They were NOT moved into
tests/harness_suites; they stay where they are until Sameer builds his
suites. Nothing here blocks the chat suites.

## Whole files (still in tests/llm_translation/)

| file | what it is |
|---|---|
| test_containers_api.py | live OpenAI Containers API CRUD |
| test_evals_api.py | live OpenAI Evals CRUD via VCR cassettes |
| test_deepgram.py | live audio-transcription base subclass |
| test_hosted_vllm_embedding_e2e.py | live vLLM embeddings, env-gated |
| test_jina_ai.py | live rerank base subclass + live embedding |
| test_skills_api.py | live Anthropic Skills CRUD base class. NOTE: no concrete subclass exists, so it executes nothing in CI today (audit 9); decide whether to subclass or delete |
| test_skills_e2e.py | skipped local-only e2e: real DB + live model through skills hook |

## Live portions of files that otherwise stayed as unit tests

| file | live tests for Sameer |
|---|---|
| test_azure_agents.py | test_azure_ai_agents_acompletion_non_streaming / _streaming / _conversation_continuity |
| test_bedrock_agentcore.py | test_bedrock_agentcore_basic, test_bedrock_agentcore_with_streaming |
| test_bedrock_agents.py | test_bedrock_agents, test_bedrock_agents_with_streaming (both @skip live) |
| test_bedrock_completion.py | TestBedrockRerank, TestBedrockCohereRerank, TestBedrockEmbedding |
| test_bedrock_embedding.py | test_e2e_bedrock_embedding, test_e2e_bedrock_embedding_image_twelvelabs_marengo |
| test_bedrock_nova_embedding.py | TestNovaEmbeddingIntegration (all @skip) |
| test_cohere.py | test_cohere_embedding_outout_dimensions + the 8 test_cohere_embed_v4_* live tests |
| test_azure_openai.py | TestAzureEmbedding |
| test_openai.py | TestOpenAIGPT4OAudioTranscription |
| test_elevenlabs.py | TestElevenLabsAudioTranscription inherited live transcription tests |
| test_fireworks_ai_translation.py | TestFireworksAIAudioTranscription |
| test_minimax_tts.py | test_speech_basic, test_speech_with_custom_params (placeholder-key live) |
| test_rerank.py | test_basic_rerank, test_rerank_custom_callbacks, test_basic_rerank_caching, test_rerank_cohere_api, test_basic_rerank_together_ai |
| test_voyage_ai.py | TestVoyageAI base subclass |
| test_gemini.py | test_gemini_embedding |
| test_text_completion.py | test_text_completion_include_usage |

These still run in `llm_translation_testing` per-commit until they migrate;
they are the residual live (network) load in that job. The remaining
@pytest.mark.flaky markers in the tree sit on these tests only.
