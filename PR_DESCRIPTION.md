# chat-scope CircleCI keep/drop: delete mock theater, migrate live tests to the 4h harness

Implements the chat-scope CircleCI keep/drop audit (sprint report `ci-auditor.md`) over `llm_translation_testing` and `llm_responses_api_testing`. The rule, from the scope doc: a test migrates if it exercises real transformation logic or a live/recorded provider, and is deleted if it mocks the layer it asserts on. Golden transform tests stay as unit tests beside translation v2; live provider tests move to `tests/harness_suites/` as 4h-harness suites; non-chat live tests are listed in `tests/harness_suites/HANDOFF.md` for Sameer rather than moved.

## Summary

| action | files | lines |
|---|---|---|
| DROP, whole file | 3 deleted | 1,038 |
| DROP, function-level (32 split files + base_llm_unit_tests) | 50 files touched | 6,155 |
| MIGRATE to tests/harness_suites/ | 13 files/1 dir moved whole, 20 files partially extracted | ~7,700 moved (7,950 lines now in harness_suites) |
| KEEP-UNIT | everything else stays in place | ~17.5k in llm_translation, ~3.9k in llm_responses_api_testing |

Total deleted: 7,193 lines (the audit estimated 5.5 to 6.5k, approaching 7k with the opportunistic dead-test and BaseOSeriesModelsTest pruning included; we land at the top of that range because that pruning is included here).

Net diff: 85 files changed, +6,826 / -13,603.

CI: `llm_translation_testing` drops from xlarge to large and loses `--retries 2 --retry-delay 5` (survivors are deterministic; a pass-on-retry is a bug). `llm_responses_api_testing` is deleted as a job; its unit survivors fold into the `llm_translation_testing` glob. The per-commit recorder jobs (`e2e_openai_endpoints`, `proxy_e2e_anthropic_messages_tests`, `proxy_spend_accuracy_tests`) and the realtime/agent/guardrails/google_generate_content/ocr jobs are untouched. Provider keys stay in the job context until Sameer's suites take the non-chat live tests listed in HANDOFF.md.

## Commit map

1. whole-file deletions
2. bedrock/aws drops, 3. azure drops, 4. openai/anthropic drops, 5. databricks drops, 6. long-tail drops, 7. responses drops, 8. base_llm_unit_tests surgery
9. harness scaffold (manifest.yaml stub, conftest with compat_result contract, HANDOFF.md), 10-16. per-suite moves (anthropic, bedrock, azure, openai, gemini, longtail, responses), 17. databricks live subclass
18. CI config

## Whole-file deletions (commit 1)

| file | lines | justification |
|---|---|---|
| test_bedrock_dynamic_auth_params_unit_tests.py | 297 | every test patches HTTPHandler.post/SigV4Auth and asserts region/credential in URL or Authorization header, or that an aws_* kwarg reached the mock |
| test_bedrock_mantle.py | 149 | all 3 tests patch HTTPHandler.post with a fake Anthropic response and assert endpoint URL, SigV4 header prefix, or a trivial prefix strip |
| test_litellm_proxy_provider.py | 592 | every test patches the OpenAI SDK or HTTPHandler then asserts was-called/kwargs/URL/headers or mock-stuffed values; no transformation asserted |

## Per-file dispositions

Drop patterns: (a) patch-then-assert-called, (b) URL/header asserted on mocked HTTP, (c) kwarg-reached-a-mock, (d) asserting values the test itself injected.

### Split files: drops removed, goldens kept, live set migrated

| file | dropped | kept | migrated |
|---|---|---|---|
| test_anthropic_completion.py | custom-headers test (b) | ~25 transform/parse goldens (headers, tool helpers, json mode, metadata filters, body captures) | TestAnthropicCompletion + 20 live tests -> chat_live_anthropic |
| test_aws_base_llm.py | 5 mock-STS/boto3 auth passthroughs (a/d) + dangling mock_credentials fixture | cache key, session token, runtime endpoint | none |
| test_azure_ai.py | 2 azure-ai-services URL/header tests (b) | model-group mapping, image-url body, deepseek reasoning parse | 7 live tests + router callback helper -> chat_live_azure |
| test_azure_o_series.py | 4 kwarg-on-patched-SDK tests (c) | test_override_fake_stream (now module-level) | TestAzureOpenAIO3Mini -> chat_live_azure; TestAzureOpenAIO3 deleted (empty after BaseOSeriesModelsTest removal) |
| test_azure_openai.py | 9 tests, patterns a/b/c | header/url-builder/param-mapping goldens | 3 live tests -> chat_live_azure; TestAzureEmbedding stays (Sameer) |
| test_bedrock_agentcore.py | 6 URL/header-on-patched-post tests (b) | JSON/SSE parse and transform goldens | 2 live tests stay (non-chat, Sameer) |
| test_bedrock_agents.py | mock-called-once test (a) | none | 2 @skip live stay (non-chat, Sameer) |
| test_bedrock_completion.py | 9 tests incl. calibrated test_bedrock_ptu; the monkeypatch cross-region duplicate that shadowed the live keeper; a zero-assertion test | ~45 transform goldens and stubbed deep-body captures | ~30 live tests + converse suites + helpers -> chat_live_bedrock; rerank/embedding classes stay (Sameer) |
| test_bedrock_embedding.py | 2 stuffed-invocationArn (d), 2 region-in-URL (b) | embedding-models parse golden | 2 live e2e stay (non-chat, Sameer) |
| test_bedrock_govcloud.py | patched-init-was-called (a) | 12 config/route/cost goldens | none (audit lists no live functions) |
| test_bedrock_gpt_oss.py | none | 2 goldens, restructured into TestBedrockGPTOSSGoldens | live subclass -> chat_live_bedrock |
| test_bedrock_invoke_tests.py | none | 3 Nova config/stream-decoder goldens | 2 invoke subclasses -> chat_live_bedrock |
| test_bedrock_moonshot.py | 6 mock overrides of live base tests + 1 asserting a dict it constructed (d) | 12 config goldens + developer-role body golden (restructured) | live subclass -> chat_live_bedrock |
| test_cohere.py | 2 request-body-into-mocked-post (c) | none local | 12 live chat tests -> chat_live_longtail; 9 embed v4 live stay (Sameer) |
| test_databricks.py | ~1,150 lines of URL/header/body passthrough and mock-stuffed caching tests (b/c/d) + 11 dangling mock builders | sdk-missing error-path test | TestDatabricksCompletion -> chat_live_longtail |
| test_deepseek_completion.py | 2 mock completions (a/c) | fill_reasoning_content goldens | live class + cost test -> chat_live_longtail |
| test_elevenlabs.py | diarize form-data passthrough (c) | TTS param/transform goldens | live transcription stays (Sameer) |
| test_fireworks_ai_translation.py | none | all transform goldens | live transcription stays (Sameer) |
| test_gemini.py | dead test literally named `l` (never collected) | context caching, image-config mapping, thinking params, exception mapping, unicode, tool-call invoke golden (de-classed, given a real assertion) | TestGoogleAIStudioGemini + 10 live tests + image-gen trio -> chat_live_gemini; embedding stays (Sameer) |
| test_groq.py | none | structured-output + chunk_parser goldens | TestGroq -> chat_live_longtail |
| test_huggingface_chat_completion.py | 3 URL-on-mock tests (b) | stubbed parse/transform suite (no live signal by design) | none |
| test_hyperbolic.py | mock_response smoke test | 6 config/registry goldens | none |
| test_lambda_ai.py / test_v0.py | none | config/registry goldens | 1 env-gated live call each -> chat_live_longtail |
| test_langgraph.py | none | 3 config transforms | 2 live local-server tests -> chat_live_longtail |
| test_minimax_tts.py | mock-handler-was-called (a) | 15 config/registration goldens | 2 live speech tests stay (Sameer) |
| test_nvidia_nim.py | 3 SDK-patch passthroughs (c) | rerank transform goldens | none |
| test_openai.py | 8 SDK-mock passthroughs incl. calibrated prediction/safety_identifier (a/c) | streaming-handler reasoning, pdf-url, xhigh-reasoning goldens | 17 live tests + class -> chat_live_openai; gpt-4o transcription stays (Sameer) |
| test_openai_o1.py | max_completion_tokens passthrough (c) | 4 o1 transform goldens | TestOpenAIO1/O3 + 2 live tests -> chat_live_openai |
| test_optional_params.py | patches get_optional_params itself; 2 forwarding tests (c) | ~75 direct mapping goldens | none |
| test_perplexity_reasoning.py | mock completion (c) | 5 mapping/capability goldens | none |
| test_rerank.py | 4 URL/mock-payload tests (b/d) | response-shape golden | 5 live rerank stay (Sameer) |
| test_text_completion.py | none | 3 conversion goldens | live usage test stays (Sameer) |
| test_text_completion_unit_tests.py | permanent-@skip logprobs test, optimized-client test (a) | parse golden | none |
| test_together_ai.py | dead pass override | supported-params golden (restructured) | TestTogetherAI -> chat_live_longtail |
| test_triton.py | external-stub-endpoint embeddings test | transform goldens | none |
| test_unit_test_bedrock_invoke.py | test_sign_request_basic (mocks botocore, asserts mocks called) | url/transform/header goldens | none |
| test_voyage_ai.py | 5 patch-litellm.embedding tests (a/d) | 8 contextual-embedding goldens | TestVoyageAI stays (Sameer) |
| test_watsonx.py | 3 auth-header/URL-on-stub tests (b) | deployment-routing body goldens | none |
| test_xai.py | dead pass override | 5 param-mapping goldens | 4 live tests/classes -> chat_live_longtail |
| responses/test_anthropic_responses_api.py | 3 patch-acompletion kwarg tests (c) | none; file deleted after move | class + multiturn tool flow -> responses_api_live |
| responses/test_azure_responses_api.py | none | status-stripping + header-prefix goldens | class + preview api-version -> responses_api_live |
| responses/test_google_ai_studio_responses_api.py | none | bridge-transform mock golden | live tools/thought-signature set -> responses_api_live |
| responses/test_openai_responses_api.py | 8 stubbed-HTTP kwarg passthroughs (c) + dead module MockResponse pair | o1-pro mock goldens, store-field transformation, router-no-metadata | class + 15 live tests + logging helpers -> responses_api_live |
| base_llm_unit_tests.py | BaseOSeriesModelsTest (3 methods, textbook a/c), abstract test_tool_call_no_arguments + 18 trivial subclass overrides, all 14 flaky markers | raw-request goldens and fixtures | live methods reach the harness through the moved subclasses |

### Whole files kept as unit tests (no changes)

test_bedrock_anthropic_regression, test_bedrock_common_utils, test_cloudflare, test_convert_dict_to_image, test_crusoe, test_gemini_image_usage, test_gigachat, test_infinity, test_llm_response_utils/ (2), test_model_cost_map_resilience, test_morph, test_openai_record_replay_proxy, test_prompt_factory, test_replicate, test_sambanova_chat_transformation, test_vcr_classification, test_vcr_conftest_common_banner, test_vcr_filters, test_vcr_redis_persister, test_bedrock_nova_embedding, test_azure_agents (the vcr/recorder files guard the kept per-commit replay jobs).

### Whole files moved to harness suites

reasoning_effort_grid/ (dir), test_bedrock_llama, test_bedrock_nova_json -> chat_live_bedrock; test_gpt4o_audio, test_prompt_caching, test_router_llm_translation_tests -> chat_live_openai; test_a2a, test_mistral_api, test_openrouter, test_snowflake -> chat_live_longtail.

### Non-chat live, handed to Sameer (not moved)

See `tests/harness_suites/HANDOFF.md`: containers, evals, deepgram, hosted-vllm embedding, jina, skills (x2), plus the live portions of azure_agents, bedrock agentcore/agents/embedding/nova-embedding, the bedrock rerank/embedding classes, cohere embed v4, azure/openai audio classes, elevenlabs/fireworks transcription, minimax TTS, rerank, voyage, gemini embedding, text-completion usage.

## Audit surprises (verbatim from the audit, all confirmed during implementation)

- test_gemini.py's parametrized status-code exception test was literally named `l`; pytest never collected it, so the coverage it claimed did not exist. Deleted.
- test_bedrock_completion.py had two defs of test_bedrock_cross_region_inference; the mock variant shadowed the live one at module level, so the "keeper" was dead until the mock variant was deleted here. The live variant is collected again as of this PR.
- test_huggingface_chat_completion.py subclasses BaseLLMChatTest but autouse fixtures stub all HTTP; it looks like a live grid cell and is not. Kept as a stubbed parse suite.
- TestBedrockEmbedding.test_bedrock_image_embedding_transformation has a bare `==` expression instead of an assert (no-op test). Left as-is per the audit (fix when porting to v2).
- test_databricks.py's cache-injection tests never asserted the injected body. Deleted.
- test_skills_api.py defines only an abstract base with no concrete subclass; it executes nothing in CI today. Left in place for Sameer's call (HANDOFF.md).
- The flaky-marker count on this branch is 14, not the audit's 15 (the double-stacked pair does not exist here). All sat on live-call tests; all stripped.

## Implementation divergences from the audit (documented, none change a verdict)

- test_optional_params.py::test_supports_system_message is audited KEEP-UNIT but actually makes a live OpenAI call; kept per the audit, flagged for the v2 porting pass.
- TestAzureOpenAIO3 deleted outright: BaseOSeriesModelsTest was its only base, so the audit's "inherited live tests migrate" set is empty for it.
- Audit 5 marks TestDatabricksCompletion MIGRATE but 8c's suite mapping omits databricks; it landed in chat_live_longtail.
- test_bedrock_govcloud.py appears in 8c's chat_live_bedrock line but 5 lists no live functions for it; it stayed put as unit goldens.
- Keep-unit methods that lived inside migrating classes (moonshot developer-role golden, gpt-oss body goldens, azure o-series fake-stream golden, gemini tool-call invoke golden) were restructured into standalone unit classes/functions so the live class could move without dragging goldens to the harness.
- The two anthropic replay tests (basic/streaming completion replay) moved with the live set per 8c; they depend on the vcr fixtures that live in tests/llm_translation/conftest.py, so wiring them up in the harness is the harness-engineer's call (they collect cleanly today).

## Verification

- `pytest --collect-only` on tests/llm_translation (excl. realtime), tests/llm_responses_api_testing, tests/harness_suites: 2,641 tests, zero errors.
- All keep-unit survivors that need no real network were run locally and pass (754 tests across the touched files); the only failures observed required CI-provided env vars (AWS/Anthropic/Gemini/Databricks keys for SigV4 signing or client construction) and pass with dummy values, identical to base since litellm/ itself is untouched by this PR.
- No references to deleted or moved files remain in .circleci/, scripts, or conftest files.
- .circleci/config.yml parses as valid YAML; the deleted job is unreferenced.
