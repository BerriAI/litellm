# LIT-7659 audit: PR #41218 on tip 3c58786534

Behavior under audit: a tag that a pre_call custom guardrail adds to the request is budget checked and reserved like a tag sent in the request body

After (head) proxy: worktree /Users/yucheng/litellm_lit7659 at 3c58786534, port 20660, `--num_workers 2`. Before (base) proxy: merge base 3ed6c19b8d in scratchpad/base-wt, port 20659, one worker. Both: real Postgres 16 (docker `lit7659-pg`), real OpenAI (`live-mini` = gpt-5-mini family, `live-embed`), zero-cost `free-mini`/`free-embed`, `broken-mini` with an invalid provider key, `LITELLM_DISABLE_NO_REDIS_WARNING=true`, no `--detailed_debug`, no Redis. The rig guardrail (`rig/rig_tag_guardrail.py`, a `CustomGuardrail` in pre_call mode) reads `tag=<name>` or `tagraw=<json>` from the prompt and writes `metadata.tags`; with no marker it sets `guardrail-budgeted`. Tags: guardrail-budgeted max_budget $0.000001 (spend above it), guardrail-rich $10, body-budgeted $0.000001 (spend above it), untracked-audit (no row), null-budget (row, no budget), team-budgeted $0.000001 on a team

Every request in this report cost real money against OpenAI unless the model was free-mini or the proxy rejected the request before the provider. Nothing in the product path is mocked

## Surface inventory

Endpoints: /v1/chat/completions, /v1/responses, /v1/messages, /v1/completions, /v1/embeddings (all go through `common_processing_pre_call_logic`, which is where the fix lives). Streaming and non streaming for the four text endpoints. Clients: raw curl for every endpoint, OpenAI SDK sync and async (chat, responses, completions, embeddings), Anthropic SDK sync and async (messages). Modes the fix branches on: tag present in body versus added by the guardrail, tag with a budget versus without a row versus a row without a budget, zero-cost model (reservation skipped), malformed guardrail output (null, bare string, mixed list), provider error before and after the tag check, body metadata absent/null/empty. Management endpoints: /tag/new, /tag/update, /team/new, /key/generate, /health/readiness. Not affected and not driven: passthrough routes, batch/files endpoints, the direct `pre_call_hook` callers outside `common_processing_pre_call_logic` (documented gap in the PR)

## Happy, mode and sad matrix

One real request per cell, verified in `LiteLLM_SpendLogs` by response id (or by `x-litellm-call-id` for errors; streaming /v1/responses ids are opaque in the stream and were matched to the single row with the tag at the same `endTime`). PASS means the head response matched the expectation and exactly one row exists with the expected status and error code. Base columns are observed values on the merge base; the base shows the bug wherever it serves a request the head rejects with the tag 429

| set | scenario | endpoint | client | stream | expected on head | head http | head row | base http | base row | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| core | over | chat | curl | False | 429 | 429 | 0fdf1e49-629c-4bc4-ab07-cb61 (id) | 200 | chatcmpl-EORnasPeQygUqLplVTW (id) | PASS |
| core | over | chat | curl | True | 429 | 429 | 89dcf0a9-2240-4a41-8d0d-8759 (id) | 200 | chatcmpl-EORndWsWwHR0ky9wIVU (id) | PASS |
| core | over | responses | curl | False | 429 | 429 | e330940f-6a9a-4650-bc68-8b55 (id) | 200 | resp_deujfJD58P-IBVNXktgxrLR (id) | PASS |
| core | over | responses | curl | True | 429 | 429 | ea5d6781-88f2-40dd-9399-8d18 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| core | over | messages | curl | False | 429 | 429 | be9c02a2-51ad-405f-ac95-f48b (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| core | over | messages | curl | True | 429 | 429 | d004f4f5-60bc-4b63-82a2-0b0a (id) | 200 | msg_c68be2b7-953f-41d0-95ed- (id) | PASS |
| core | over | completions | curl | False | 429 | 429 | 06f31c16-0a6f-4e82-9ca5-3322 (time) | 200 | chatcmpl-EORnvKF6A4QuSWZFgOo (id) | PASS |
| core | over | completions | curl | True | 429 | 429 | 06f31c16-0a6f-4e82-9ca5-3322 (time) | 200 | chatcmpl-EORnyiop7DiM1tr37DQ (id) | PASS |
| core | over | embeddings | curl | False | 429 | 429 | cc8801aa-8141-4395-8cd5-d5c7 (id) | 200 | e30d16ab-4892-4481-861c-ea96 (id) | PASS |
| core | over | chat | openai-sync | False | 429 | 429 | 3660333c-3a09-4823-99e1-b1a2 (id) | 200 | chatcmpl-EORo3uiWcduedgd2NtE (id) | PASS |
| core | over | chat | openai-sync | True | 429 | 429 | df29b499-2748-44e6-8504-24ab (id) | 200 | chatcmpl-EORo6zzgG2tZUDKwbAM (id) | PASS |
| core | over | responses | openai-sync | False | 429 | 429 | 57e1e04d-3909-45f5-b712-da1d (id) | 200 | resp_boOM1vThMG5kdo4LtFBh3hg (id) | PASS |
| core | over | responses | openai-sync | True | 429 | 429 | 49bb0d7a-3ccc-4b94-b203-4c0a (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| core | over | completions | openai-sync | False | 429 | 429 | 06f31c16-0a6f-4e82-9ca5-3322 (time) | 200 | chatcmpl-EORoGvd27WH9x1Oxs8R (id) | PASS |
| core | over | embeddings | openai-sync | False | 429 | 429 | 85af18d1-f2f0-485c-bd14-c021 (id) | 200 | 1d28b9a6-8f87-45fd-8e55-978b (id) | PASS |
| core | over | chat | openai-async | False | 429 | 429 | e815134a-a5ca-42fc-a170-3dc5 (id) | 200 | chatcmpl-EORoHNVqzCLvbdH2Br6 (id) | PASS |
| core | over | chat | openai-async | True | 429 | 429 | 06f31c16-0a6f-4e82-9ca5-3322 (id) | 200 | chatcmpl-EORoLWlljG34hCAlm48 (id) | PASS |
| core | over | responses | openai-async | False | 429 | 429 | 5cf6c378-8985-4aee-a0cc-ba01 (id) | 200 | resp_eqgOexlD-PlLdavwQN6l1An (id) | PASS |
| core | over | responses | openai-async | True | 429 | 429 | 2d5f0360-8806-414f-82ff-37b9 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| core | over | messages | anthropic-sync | False | 429 | 429 | 3d9426a9-23e3-4291-98a9-e8d3 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| core | over | messages | anthropic-sync | True | 429 | 429 | d2a7cc4d-a343-4816-b905-a1c2 (id) | 200 | msg_8352a1c9-42db-466c-8376- (id) | PASS |
| core | over | messages | anthropic-async | False | 429 | 429 | 0012469c-b9ee-4220-bc51-4ff0 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| core | over | messages | anthropic-async | True | 429 | 429 | b67cab8b-6c5a-4aa9-9cb6-fbca (id) | 200 | msg_ed4ff0be-52c9-4c9e-b594- (id) | PASS |
| core | rich | chat | curl | False | 200 | 200 | chatcmpl-EORmGPl9zfqdxBDGNyD (id) | 200 | chatcmpl-EORomK3nZjY222QwjWH (id) | PASS |
| core | rich | chat | curl | True | 200 | 200 | chatcmpl-EORmIgnxB2EetZ7EP5N (id) | 200 | chatcmpl-EORoq8MOuczfZZgg1zv (id) | PASS |
| core | rich | responses | curl | False | 200 | 200 | resp_6PR8IaHyxxcAyCBM0-sE42w (id) | 200 | resp_Yu1R4v_0TzVslcoWCjxpaSn (id) | PASS |
| core | rich | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| core | rich | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| core | rich | messages | curl | True | 200 | 200 | msg_5641bd6f-ebb2-4ca8-b8d4- (id) | 200 | msg_4b4c5ce1-3d6d-4569-82aa- (id) | PASS |
| core | rich | completions | curl | False | 200 | 200 | chatcmpl-EORmfDij9LAgf2XwWRF (id) | 200 | chatcmpl-EORp37akvKzqnIFmTQC (id) | PASS |
| core | rich | completions | curl | True | 200 | 200 | chatcmpl-EORmjCTFSbi8lu14Y0d (id) | 200 | chatcmpl-EORp4BDoLJ8bwvV3iZ0 (id) | PASS |
| core | rich | embeddings | curl | False | 200 | 200 | 9e14aef5-59b3-474e-9d79-1094 (id) | 200 | 0a0ffa0c-b527-45a7-8577-6712 (id) | PASS |
| core | rich | chat | openai-sync | False | 200 | 200 | chatcmpl-EORmlpHzb7xmydgdgpR (id) | 200 | chatcmpl-EORp7VCqX8uXkYGUEom (id) | PASS |
| core | rich | chat | openai-sync | True | 200 | 200 | chatcmpl-EORmpnYYMgPMxb4mW0m (id) | 200 | chatcmpl-EORpBr6sTHcATFNPmud (id) | PASS |
| core | rich | responses | openai-sync | False | 200 | 200 | resp_ydMbB3zoX64Mq1ORn4v329s (id) | 200 | resp_OrdTyxF94-vzijwvAjP2Nlc (id) | PASS |
| core | rich | responses | openai-sync | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| core | rich | completions | openai-sync | False | 200 | 200 | chatcmpl-EORn3iBONQ0zuXuqFYz (id) | 200 | chatcmpl-EORpPSjrVlfcLUBHzHB (id) | PASS |
| core | rich | embeddings | openai-sync | False | 200 | 200 | 39ae8f88-d07c-4663-96c1-9b86 (id) | 200 | 9378702b-0f70-40a1-9d5d-900e (id) | PASS |
| core | rich | chat | openai-async | False | 200 | 200 | chatcmpl-EORn7mBfDV8eRQBfZb8 (id) | 200 | chatcmpl-EORpR2mJXPdYvNtWWN6 (id) | PASS |
| core | rich | chat | openai-async | True | 200 | 200 | chatcmpl-EORn7TpoAuelmgxr6wC (id) | 200 | chatcmpl-EORpU9nyLA3Y9wwsEIB (id) | PASS |
| core | rich | responses | openai-async | False | 200 | 200 | resp_gU5Cc0UyKkN3s78oe26FJHV (id) | 200 | resp_hLWyx58zraFWyyV_C67JJev (id) | PASS |
| core | rich | responses | openai-async | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| core | rich | messages | anthropic-sync | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| core | rich | messages | anthropic-sync | True | 200 | 200 | msg_b078f2cc-a098-40d1-8fe9- (id) | 200 | msg_eec19228-0a46-4620-bd71- (id) | PASS |
| core | rich | messages | anthropic-async | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| core | rich | messages | anthropic-async | True | 200 | 200 | msg_d7152c0e-f3a8-4b8e-b217- (id) | 200 | msg_1f8d12cb-4c8d-4077-b3a2- (id) | PASS |
| modes | untracked | chat | curl | False | 200 | 200 | chatcmpl-EORpw3ndd1kmyEMlBDh (id) | 200 | chatcmpl-EORrH8gnQ01vZNiVBKK (id) | PASS |
| modes | untracked | chat | curl | True | 200 | 200 | chatcmpl-EORq0dmeC5JTp0Pf9bP (id) | 200 | chatcmpl-EORrL28fflSdmh3RAiP (id) | PASS |
| modes | untracked | responses | curl | False | 200 | 200 | resp_hDvc6eQS7wp71aVaq0n7uRg (id) | 200 | resp_eiSAhEmDmYGhD-iRAChKWAW (id) | PASS |
| modes | untracked | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| modes | untracked | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| modes | untracked | messages | curl | True | 200 | 200 | msg_ea3153e7-0182-42db-9d5a- (id) | 200 | msg_233171b7-a388-4bbc-8eda- (id) | PASS |
| modes | untracked | completions | curl | False | 200 | 200 | chatcmpl-EORqKBjZSavWl3Xqxar (id) | 200 | chatcmpl-EORrbDrsUy0WTnVTBIY (id) | PASS |
| modes | untracked | completions | curl | True | 200 | 200 | chatcmpl-EORqOc2LQ1UaBPGiTzH (id) | 200 | chatcmpl-EORre0puWMOckohuIIP (id) | PASS |
| modes | untracked | embeddings | curl | False | 200 | 200 | 5b38b7f8-c5d9-4822-9b3a-b87b (id) | 200 | 4f49ee27-4b99-4ff8-bf38-cd97 (id) | PASS |
| modes | free | chat | curl | False | 200 | 200 | chatcmpl-EORqRf2Ed7RwQdDmRzQ (id) | 200 | chatcmpl-EORrjxFAxg0zoYADf58 (id) | SUPERSEDED |
| modes | free | chat | curl | True | 200 | 200 | chatcmpl-EORqVneojG5EhRvWR4A (id) | 200 | chatcmpl-EORrjN3tWYn7PZZmlY3 (id) | SUPERSEDED |
| modes | free | responses | curl | False | 200 | 200 | resp_wFgB0a7U0xzqU39xZ8lsKYH (id) | 200 | resp_eOhnPIIWPn8hd1jq2pFKosA (id) | SUPERSEDED |
| modes | free | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | SUPERSEDED |
| modes | free | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | SUPERSEDED |
| modes | free | messages | curl | True | 200 | 200 | msg_3b6bd6ff-d586-4925-a4f1- (id) | 200 | msg_bd79a0d4-fde4-4ae4-9fb5- (id) | SUPERSEDED |
| modes | free | completions | curl | False | 200 | 200 | chatcmpl-EORqlAh252QYPrtdr8M (id) | 200 | chatcmpl-EORs1l1SNy4rTt5rpOy (id) | SUPERSEDED |
| modes | free | completions | curl | True | 200 | 200 | chatcmpl-EORqp0zARAscjGbIY6T (id) | 200 | chatcmpl-EORs6TtYTvY4Tzg2JAn (id) | SUPERSEDED |
| modes | free | embeddings | curl | False | 200 | 403 | ba11ad32-6fbf-46f9-931f-52a3 (id) | 403 | 34e165ee-4d8f-4cc9-9da0-1cf6 (id) | SUPERSEDED |
| modes | bodytag | chat | curl | False | 429 | 200 | chatcmpl-EORqxyFyxcTxh7L3qmK (id) | 429 | cde7d66a-1663-48b1-947c-3abd (time) | SUPERSEDED |
| modes | bodytag | chat | curl | True | 429 | 429 | 6fe342f7-cc54-405d-825d-9eea (time) | 429 | cde7d66a-1663-48b1-947c-3abd (time) | SUPERSEDED |
| modes | bodytag | responses | curl | False | 200 | 200 | resp_qu6RLeb0B5DnjI8ugE9g9pe (id) | 200 | resp_m4ibnuU9Pg-pPsunAM1znsy (id) | SUPERSEDED |
| modes | bodytag | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | SUPERSEDED |
| modes | bodytag | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | SUPERSEDED |
| modes | bodytag | messages | curl | True | 200 | 200 | msg_d8518e3b-10a6-416b-b5b3- (id) | 200 | msg_ed765358-71b0-47c2-a6ff- (id) | SUPERSEDED |
| modes | bodytag | completions | curl | False | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | 2c48546e-fb83-424e-b8ef-13a5 (time) | SUPERSEDED |
| modes | bodytag | completions | curl | True | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | 2c48546e-fb83-424e-b8ef-13a5 (time) | SUPERSEDED |
| modes | bodytag | embeddings | curl | False | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | 2c48546e-fb83-424e-b8ef-13a5 (time) | SUPERSEDED |
| modes | both | chat | curl | False | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | 2c48546e-fb83-424e-b8ef-13a5 (time) | PASS |
| modes | both | chat | curl | True | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | 2c48546e-fb83-424e-b8ef-13a5 (time) | PASS |
| modes | both | responses | curl | False | 429 | 429 | c7eed6e0-e02a-4959-a498-15c7 (id) | 200 | resp_-hhmSurMFMZN_cru6oMqtiI (id) | PASS |
| modes | both | responses | curl | True | 429 | 429 | 0756e1fa-c813-4131-874d-8364 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| modes | both | messages | curl | False | 429 | 429 | 6f8b0d30-2f95-4090-acaf-ec79 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| modes | both | messages | curl | True | 429 | 429 | 5210d695-b9ee-49eb-9390-738c (id) | 200 | msg_9995f2bb-bcbc-4788-9ed2- (id) | PASS |
| modes | both | completions | curl | False | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | e0cbc4ca-de1d-45c1-96cc-244b (time) | PASS |
| modes | both | completions | curl | True | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | e0cbc4ca-de1d-45c1-96cc-244b (time) | PASS |
| modes | both | embeddings | curl | False | 429 | 429 | e5a0506a-78c4-4e9c-95f5-5606 (time) | 429 | e0cbc4ca-de1d-45c1-96cc-244b (time) | PASS |
| modes | emptytags | chat | curl | False | 429 | 429 | a3c7e55f-9fdf-428e-8488-bb66 (id) | 200 | chatcmpl-EORsY8GAbpM3WXf34Jx (id) | PASS |
| modes | emptytags | chat | curl | True | 429 | 429 | 32bda827-9105-41f3-b3f6-f4bd (id) | 200 | chatcmpl-EORsbBtcp0YeWXFpxLd (id) | PASS |
| modes | emptytags | responses | curl | False | 429 | 429 | 25621c74-5406-4796-9f50-9a36 (id) | 200 | resp_tQSr8qa27OYbVu2U39Qsy31 (id) | PASS |
| modes | emptytags | responses | curl | True | 429 | 429 | 45cf24b5-d8d2-4712-81b7-2fc4 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| modes | emptytags | messages | curl | False | 429 | 429 | e62a0765-40ea-482d-b842-1f35 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| modes | emptytags | messages | curl | True | 429 | 429 | ea064d52-ec24-4783-a81f-d443 (id) | 200 | msg_76962ae1-f631-4e28-933e- (id) | PASS |
| modes | emptytags | completions | curl | False | 429 | 429 | d6703a97-89cf-4a7c-84b9-ab0c (time) | 200 | chatcmpl-EORssN9Fr6B8m2BE2xs (id) | PASS |
| modes | emptytags | completions | curl | True | 429 | 429 | d6703a97-89cf-4a7c-84b9-ab0c (time) | 200 | chatcmpl-EORss5HjZ7GRBUg3Xz9 (id) | PASS |
| modes | emptytags | embeddings | curl | False | 429 | 429 | 24d59989-560e-468a-b1f9-fe4e (id) | 200 | 674a364d-10c8-496d-8b5b-d462 (id) | PASS |
| modes | nullmeta | chat | curl | False | 429 | 429 | c67252c2-eeb4-4a94-8899-75b5 (id) | 200 | chatcmpl-EORsyYOJbOUM9NPqiWi (id) | PASS |
| modes | nullmeta | chat | curl | True | 429 | 429 | 33923eee-ead7-4841-b74e-7c1a (id) | 200 | chatcmpl-EORt2lM1qE6H1I10mQh (id) | PASS |
| modes | nullmeta | responses | curl | False | 429 | 429 | 322596f2-4570-44c9-9019-514f (id) | 200 | resp_vKayKU7fdG8jFsqYaCxsAHM (id) | PASS |
| modes | nullmeta | responses | curl | True | 429 | 429 | c19c580c-c271-4f37-adb7-8cb0 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| modes | nullmeta | messages | curl | False | 429 | 429 | 8bbaade2-f487-41fb-804e-4548 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| modes | nullmeta | messages | curl | True | 429 | 429 | 8cab4a10-7e87-48f6-a566-b231 (id) | 200 | msg_7bf21e83-cf09-47e7-aae8- (id) | PASS |
| modes | nullmeta | completions | curl | False | 429 | 429 | d6703a97-89cf-4a7c-84b9-ab0c (time) | 200 | chatcmpl-EORtLiIiRThFI8ebnpZ (id) | PASS |
| modes | nullmeta | completions | curl | True | 429 | 429 | d6703a97-89cf-4a7c-84b9-ab0c (time) | 200 | chatcmpl-EORtMGUG8DZ89AMBpwf (id) | PASS |
| modes | nullmeta | embeddings | curl | False | 429 | 429 | d6703a97-89cf-4a7c-84b9-ab0c (id) | 200 | 891fa78d-d0a3-4b11-9df3-ce35 (id) | PASS |
| sad | hostile-null | chat | curl | False | 200 | 200 | chatcmpl-EORtPPBmDoFT0oL0R2e (id) | 200 | chatcmpl-EORu2GrwBNB07zTXMeg (id) | PASS |
| sad | hostile-null | chat | curl | True | 200 | 200 | chatcmpl-EORtRFsSfU1suO9cylr (id) | 200 | chatcmpl-EORu3YnXifdNvGSTn1e (id) | PASS |
| sad | hostile-null | responses | curl | False | 200 | 200 | resp_WP5PtL5ek5ggouQ7XxrRNwF (id) | 200 | resp_pV4R1yutp7DmvxbbAek9wv1 (id) | PASS |
| sad | hostile-null | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| sad | hostile-null | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| sad | hostile-null | messages | curl | True | 200 | 200 | msg_134b0bac-cc27-44f1-8407- (id) | 200 | msg_06c998d2-374e-4359-a9bf- (id) | PASS |
| sad | hostile-null | completions | curl | False | 200 | 200 | chatcmpl-EORtiO2qLxXCnGhlcWk (id) | 200 | chatcmpl-EORuFUfdHyKmDUrOy4C (id) | PASS |
| sad | hostile-null | completions | curl | True | 200 | 200 | chatcmpl-EORtlvn9eWLQBG8Tm2q (id) | 200 | chatcmpl-EORuGoBog1BNkHan8pO (id) | PASS |
| sad | hostile-null | embeddings | curl | False | 200 | 200 | 83354dbc-7fe2-4ffe-ad50-f8c9 (id) | 200 | 192159b0-237a-4adf-a9a6-e3b0 (id) | PASS |
| sad | hostile-str | chat | curl | False | 200 | 200 | chatcmpl-EORtohejoAKuFRTXgyO (id) | 200 | chatcmpl-EORuHbHj0048rnRI430 (id) | PASS |
| sad | hostile-str | chat | curl | True | 200 | 200 | chatcmpl-EORtqMDwTwkE6gWszgU (id) | 200 | chatcmpl-EORuJJ96MJZj3H5ZMdH (id) | PASS |
| sad | hostile-str | responses | curl | False | 200 | 200 | resp_kPNO0j_QMV19CP6RyMXZdbr (id) | 200 | resp_b-Z-Yb5NZAELidOMuwqO8VD (id) | PASS |
| sad | hostile-str | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| sad | hostile-str | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| sad | hostile-str | messages | curl | True | 200 | 200 | msg_74ced3b9-1aa6-42c7-b7f2- (id) | 200 | msg_3a8fa85e-6702-48e6-80ae- (id) | PASS |
| sad | hostile-str | completions | curl | False | 200 | 200 | chatcmpl-EORtyUAuAE2SywRg1QQ (id) | 200 | chatcmpl-EORuS0ICE9oxXeQrMsn (id) | PASS |
| sad | hostile-str | completions | curl | True | 200 | 200 | chatcmpl-EORtzYPVGoRbKAHeAvf (id) | 200 | chatcmpl-EORuTPSdZKSa4SUmF2J (id) | PASS |
| sad | hostile-str | embeddings | curl | False | 200 | 200 | fd0fe651-ab45-4be8-a341-ab4f (id) | 200 | dee7bae9-8496-4e3a-a640-a2d9 (id) | PASS |
| sad | hostile-mixed | chat | curl | False | 429 | 429 | 1b58054b-893b-480a-9c77-20c1 (id) | 200 | chatcmpl-EORuVEzmUFj2QxpLsHd (id) | PASS |
| sad | hostile-mixed | chat | curl | True | 429 | 429 | 46828d75-99b2-420b-ba40-3564 (id) | 200 | chatcmpl-EORuV6lAPGiQXdKH3PE (id) | PASS |
| sad | hostile-mixed | responses | curl | False | 429 | 429 | 7f77c329-13da-48d6-991b-515d (id) | 200 | resp_5y-35yjWy7-uQwJQKPNzGak (id) | PASS |
| sad | hostile-mixed | responses | curl | True | 429 | 429 | 5d089de2-21bb-40db-a139-bf22 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| sad | hostile-mixed | messages | curl | False | 429 | 429 | 2950cc0e-f505-41fd-a0ef-1475 (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| sad | hostile-mixed | messages | curl | True | 429 | 429 | 8d35ad59-a1bf-4cf9-9f3e-77c8 (id) | 200 | msg_2b8d296e-3255-458a-a36b- (id) | PASS |
| sad | hostile-mixed | completions | curl | False | 429 | 429 | 770bdf3b-bf0c-4596-8f65-07e8 (time) | 200 | chatcmpl-EORucf1PX0tbW9HQvPI (id) | PASS |
| sad | hostile-mixed | completions | curl | True | 429 | 429 | 770bdf3b-bf0c-4596-8f65-07e8 (time) | 200 | chatcmpl-EORucmwQzGov4rZDmji (id) | PASS |
| sad | hostile-mixed | embeddings | curl | False | 429 | 429 | 770bdf3b-bf0c-4596-8f65-07e8 (id) | 200 | 47c16699-0b48-49e2-9345-5d0f (id) | PASS |
| sad | badmodel | chat | curl | False | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | chat | curl | True | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | responses | curl | False | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | responses | curl | True | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | messages | curl | False | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | messages | curl | True | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | completions | curl | False | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | completions | curl | True | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| sad | badmodel | embeddings | curl | False | 403 | 403 | df6e2fb7-8f64-4f53-b7e1-cb21 (time) | 403 | 29d9521b-d1d9-4f36-b026-9463 (time) | PASS |
| redo | bodytag | chat | curl | False | 429 | 429 | 4f69cb59-336b-4fa4-8f93-618c (time) | 429 | 8d3dcadd-9213-4735-8d8f-da7c (time) | PASS |
| redo | bodytag | chat | curl | True | 429 | 429 | 4f69cb59-336b-4fa4-8f93-618c (time) | 429 | 8d3dcadd-9213-4735-8d8f-da7c (time) | PASS |
| redo | bodytag | responses | curl | False | 200 | 200 | resp_5sWt8ug7kYRTCv-2FkxR4_r (id) | 200 | resp_bJlty7NtOswwPAPgB9wRmV5 (id) | PASS |
| redo | bodytag | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| redo | bodytag | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| redo | bodytag | messages | curl | True | 200 | 200 | msg_eb043c6b-4ebe-4122-ad62- (id) | 200 | msg_9687894b-c8fe-40ca-85ee- (id) | PASS |
| redo | bodytag | completions | curl | False | 429 | 429 | f0b29e7b-6f97-43d5-99fa-59eb (time) | 429 | d61ee414-f193-4deb-af28-e69f (time) | PASS |
| redo | bodytag | completions | curl | True | 429 | 429 | f0b29e7b-6f97-43d5-99fa-59eb (time) | 429 | d61ee414-f193-4deb-af28-e69f (time) | PASS |
| redo | bodytag | embeddings | curl | False | 429 | 429 | f0b29e7b-6f97-43d5-99fa-59eb (time) | 429 | d61ee414-f193-4deb-af28-e69f (time) | PASS |
| redo | free | chat | curl | False | 200 | 200 | chatcmpl-EORvxRQX3burce3VTnz (id) | 200 | chatcmpl-EORwDWCto8GqlYnBhxJ (id) | PASS |
| redo | free | chat | curl | True | 200 | 200 | chatcmpl-EORvyAILjJjzlUMxzJQ (id) | 200 | chatcmpl-EORwE89I67u4cnfBiY9 (id) | PASS |
| redo | free | responses | curl | False | 200 | 200 | resp_IqoUuBhqxXkIMyG_VR1Wkqq (id) | 200 | resp_UIfrvRRQMkHAVX093vurytt (id) | PASS |
| redo | free | responses | curl | True | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (time) | PASS |
| redo | free | messages | curl | False | 200 | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | 200 | resp_bGl0ZWxsbTpjdXN0b21fbGx (id) | PASS |
| redo | free | messages | curl | True | 200 | 200 | msg_9f9475ef-cc01-4494-bec9- (id) | 200 | msg_b5facfe8-ac04-4d6f-879d- (id) | PASS |
| redo | free | completions | curl | False | 200 | 200 | chatcmpl-EORw4M6YSSvxHuBHMMe (id) | 200 | chatcmpl-EORwK8nXZcgimmwK8nN (id) | PASS |
| redo | free | completions | curl | True | 200 | 200 | chatcmpl-EORw5dfpk1I7IHytelZ (id) | 200 | chatcmpl-EORwL8UGnxagYL2HJi6 (id) | PASS |
| redo | free | embeddings | curl | False | 200 | 200 | 15badfa6-8ac4-4eeb-937d-744a (id) | 200 | d6366a1d-3d13-483c-87c0-c9b2 (id) | PASS |
| sad2 | providererror-over | chat | curl | False | 429 | 429 | 7dde5b65-38cb-4d90-99a9-fc00 (id) | 401 | 53b26d20-4c99-4642-a158-6dc7 (id) | PASS |
| sad2 | providererror-over | chat | curl | True | 429 | 429 | 077c29dd-7d74-4bc2-9423-9e61 (id) | 429 | 80c3e5cd-9bc9-4aa3-8489-2b24 (id) | PASS |
| sad2 | providererror-over | responses | curl | False | 429 | 429 | 24a8a58e-59e3-432d-bb0f-fbcd (id) | 429 | 00b7ed77-f98b-41be-9668-780a (id) | PASS |
| sad2 | providererror-over | responses | curl | True | 429 | 429 | e4438061-54cc-4e09-942d-ae1f (id) | 429 | f7ae0896-20ee-411f-895e-5e6e (id) | PASS |
| sad2 | providererror-over | messages | curl | False | 429 | 429 | bae5b65c-ff00-408a-a12b-5166 (id) | 429 | 17cfc3cc-28e1-471a-8dc9-01fa (id) | PASS |
| sad2 | providererror-over | messages | curl | True | 429 | 429 | 7e469786-cf83-4fcd-9650-c6eb (id) | 429 | 34431862-55e2-4834-ac09-e8ec (id) | PASS |
| sad2 | providererror-over | completions | curl | False | 429 | 429 | c4216688-cee1-4aec-af2e-b919 (time) | 429 | 34e0b4d6-7003-44de-ba04-bdae (time) | PASS |
| sad2 | providererror-over | completions | curl | True | 429 | 429 | c4216688-cee1-4aec-af2e-b919 (time) | 429 | 34e0b4d6-7003-44de-ba04-bdae (time) | PASS |
| sad2 | providererror-over | embeddings | curl | False | 429 | 429 | c4216688-cee1-4aec-af2e-b919 (id) | 429 | 34e0b4d6-7003-44de-ba04-bdae (id) | PASS |
| sad2 | providererror-rich | chat | curl | False | 401|429 | 401 | cafa371d-6858-4f45-a73b-36e7 (id) | 429 | b6651d4c-4583-4f5d-98f1-b7f7 (id) | PASS |
| sad2 | providererror-rich | chat | curl | True | 401|429 | 429 | 0efcc7b0-8212-4807-8290-a50f (id) | 429 | 1c1244ca-c7d0-474e-b60f-730a (id) | PASS |
| sad2 | providererror-rich | responses | curl | False | 401|429 | 429 | 061a2de0-281f-48f0-8aa8-69a0 (id) | 429 | 4346aa22-8617-4dcb-a0e2-00e5 (id) | PASS |
| sad2 | providererror-rich | responses | curl | True | 401|429 | 401 | bc6501f7-7e6d-456a-b565-f494 (id) | 429 | 24d4e917-e5e3-49a0-bfe7-cc6f (id) | PASS |
| sad2 | providererror-rich | messages | curl | False | 401|429 | 429 | 5c34a637-4515-4b88-a31b-6a2d (id) | 429 | 0cebf6f8-4ac5-4356-9b28-6bde (id) | PASS |
| sad2 | providererror-rich | messages | curl | True | 401|429 | 429 | 3883c443-8ed0-478f-8d37-6722 (id) | 429 | 240cced1-0a38-45ee-b925-9bb5 (id) | PASS |
| sad2 | providererror-rich | completions | curl | False | 401|429 | 429 | f398d8b1-b5d5-4f02-af0d-9c03 (time) | 429 | 52f22b44-ac85-47ea-b7b8-d71f (time) | PASS |
| sad2 | providererror-rich | completions | curl | True | 401|429 | 429 | f398d8b1-b5d5-4f02-af0d-9c03 (time) | 429 | 52f22b44-ac85-47ea-b7b8-d71f (time) | PASS |
| sad2 | providererror-rich | embeddings | curl | False | 401|429 | 429 | f398d8b1-b5d5-4f02-af0d-9c03 (id) | 429 | 52f22b44-ac85-47ea-b7b8-d71f (id) | PASS |

Totals on head: 154 PASS, 0 FAIL, 18 superseded (see below). Base: 172 cells recorded, every over-budget guardrail tag scenario served (over, emptytags, nullmeta, hostile-mixed, providererror-over) instead of returning the tag 429

bodytag on /v1/responses and /v1/messages expects 200 on both sides: those routes read request tags from `litellm_metadata`, not `metadata`, so the body tag is not seen by auth there (pre-existing, same on base). Sent as `litellm_metadata.tags` the same body gets the auth 429 on both sides: head responses 429, head messages 429, base responses 429, base messages 429 (curl, non streaming, run once each on 3c58786534 and 3ed6c19b8d)

Superseded cells: set modes scenario bodytag: superseded by set redo: body-budgeted had no spend yet when modes ran, so the first cells were 200 by design; set modes scenario free: superseded by set redo: the embeddings cell ran before free-embed existed in the rig config

Scenario expectations: over: guardrail adds guardrail-budgeted (spend above its $0.000001 budget): tag budget 429 before the provider; rich: guardrail adds guardrail-rich ($10 budget): served, one spend row with the tag; untracked: guardrail adds a tag with no LiteLLM_TagTable row: served; free: free-mini costs $0, so the tag reservation is skipped even though the tag is over budget; bodytag: body metadata.tags carries body-budgeted (over budget): auth path 429, unchanged; both: body tag over budget plus guardrail tag: auth path 429 first; emptytags: body metadata.tags is []: the guardrail tag is still budget checked; nullmeta: body metadata is null: the guardrail tag is still budget checked; hostile-null: guardrail sets tags to null: ignored, served; hostile-str: guardrail sets tags to a bare string: ignored, served; hostile-mixed: guardrail sets [1, null, {}, "guardrail-budgeted"]: junk entries ignored, the string entry is budget checked; badmodel: unknown model: the key model check rejects it, no crash; providererror-over: broken-mini has a bad provider key, tag over budget: the tag 429 comes first, provider never called; providererror-rich: broken-mini with guardrail-rich: the provider 401 (then router cooldown 429) reaches the caller

Request shapes per cell (curl form; the SDK cells send the same body through the OpenAI or Anthropic client with max_retries=0):

- over /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- over /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- over /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16}'`
- over /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16,"stream":true}'`
- over /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- over /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- over /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5}'`
- over /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5,"stream":true}'`
- over /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi"}'`
- rich /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5}'`
- rich /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"stream":true}'`
- rich /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tag=guardrail-rich","max_output_tokens":16}'`
- rich /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tag=guardrail-rich","max_output_tokens":16,"stream":true}'`
- rich /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5}'`
- rich /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"stream":true}'`
- rich /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tag=guardrail-rich","max_tokens":5}'`
- rich /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tag=guardrail-rich","max_tokens":5,"stream":true}'`
- rich /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi tag=guardrail-rich"}'`
- untracked /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=untracked-audit"}],"max_tokens":5}'`
- untracked /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=untracked-audit"}],"max_tokens":5,"stream":true}'`
- untracked /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tag=untracked-audit","max_output_tokens":16}'`
- untracked /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tag=untracked-audit","max_output_tokens":16,"stream":true}'`
- untracked /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=untracked-audit"}],"max_tokens":5}'`
- untracked /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=untracked-audit"}],"max_tokens":5,"stream":true}'`
- untracked /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tag=untracked-audit","max_tokens":5}'`
- untracked /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tag=untracked-audit","max_tokens":5,"stream":true}'`
- untracked /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi tag=untracked-audit"}'`
- free /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- free /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- free /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","input":"say hi","max_output_tokens":16}'`
- free /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","input":"say hi","max_output_tokens":16,"stream":true}'`
- free /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- free /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- free /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","prompt":"say hi","max_tokens":5}'`
- free /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-mini","prompt":"say hi","max_tokens":5,"stream":true}'`
- free /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"free-embed","input":"say hi"}'`
- bodytag /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]}}'`
- bodytag /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- bodytag /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tag=guardrail-rich","max_output_tokens":16,"metadata":{"tags":["body-budgeted"]}}'`
- bodytag /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tag=guardrail-rich","max_output_tokens":16,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- bodytag /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]}}'`
- bodytag /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- bodytag /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tag=guardrail-rich","max_tokens":5,"metadata":{"tags":["body-budgeted"]}}'`
- bodytag /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tag=guardrail-rich","max_tokens":5,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- bodytag /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi tag=guardrail-rich","metadata":{"tags":["body-budgeted"]}}'`
- both /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]}}'`
- both /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- both /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16,"metadata":{"tags":["body-budgeted"]}}'`
- both /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- both /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]}}'`
- both /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- both /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5,"metadata":{"tags":["body-budgeted"]}}'`
- both /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5,"metadata":{"tags":["body-budgeted"]},"stream":true}'`
- both /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi","metadata":{"tags":["body-budgeted"]}}'`
- emptytags /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":[]}}'`
- emptytags /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":[]},"stream":true}'`
- emptytags /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16,"metadata":{"tags":[]}}'`
- emptytags /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16,"metadata":{"tags":[]},"stream":true}'`
- emptytags /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":[]}}'`
- emptytags /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":{"tags":[]},"stream":true}'`
- emptytags /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5,"metadata":{"tags":[]}}'`
- emptytags /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5,"metadata":{"tags":[]},"stream":true}'`
- emptytags /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi","metadata":{"tags":[]}}'`
- nullmeta /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":null}'`
- nullmeta /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":null,"stream":true}'`
- nullmeta /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16,"metadata":null}'`
- nullmeta /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi","max_output_tokens":16,"metadata":null,"stream":true}'`
- nullmeta /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":null}'`
- nullmeta /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"metadata":null,"stream":true}'`
- nullmeta /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5,"metadata":null}'`
- nullmeta /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi","max_tokens":5,"metadata":null,"stream":true}'`
- nullmeta /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi","metadata":null}'`
- hostile-null /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=null"}],"max_tokens":5}'`
- hostile-null /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=null"}],"max_tokens":5,"stream":true}'`
- hostile-null /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tagraw=null","max_output_tokens":16}'`
- hostile-null /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tagraw=null","max_output_tokens":16,"stream":true}'`
- hostile-null /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=null"}],"max_tokens":5}'`
- hostile-null /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=null"}],"max_tokens":5,"stream":true}'`
- hostile-null /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tagraw=null","max_tokens":5}'`
- hostile-null /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tagraw=null","max_tokens":5,"stream":true}'`
- hostile-null /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi tagraw=null"}'`
- hostile-str /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=\"guardrail-budgeted\""}],"max_tokens":5}'`
- hostile-str /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=\"guardrail-budgeted\""}],"max_tokens":5,"stream":true}'`
- hostile-str /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tagraw=\"guardrail-budgeted\"","max_output_tokens":16}'`
- hostile-str /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tagraw=\"guardrail-budgeted\"","max_output_tokens":16,"stream":true}'`
- hostile-str /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=\"guardrail-budgeted\""}],"max_tokens":5}'`
- hostile-str /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=\"guardrail-budgeted\""}],"max_tokens":5,"stream":true}'`
- hostile-str /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tagraw=\"guardrail-budgeted\"","max_tokens":5}'`
- hostile-str /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tagraw=\"guardrail-budgeted\"","max_tokens":5,"stream":true}'`
- hostile-str /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi tagraw=\"guardrail-budgeted\""}'`
- hostile-mixed /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]"}],"max_tokens":5}'`
- hostile-mixed /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]"}],"max_tokens":5,"stream":true}'`
- hostile-mixed /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]","max_output_tokens":16}'`
- hostile-mixed /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","input":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]","max_output_tokens":16,"stream":true}'`
- hostile-mixed /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]"}],"max_tokens":5}'`
- hostile-mixed /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","messages":[{"role":"user","content":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]"}],"max_tokens":5,"stream":true}'`
- hostile-mixed /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]","max_tokens":5}'`
- hostile-mixed /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-mini","prompt":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]","max_tokens":5,"stream":true}'`
- hostile-mixed /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"live-embed","input":"say hi tagraw=[1,null,{\"a\":1},\"guardrail-budgeted\"]"}'`
- badmodel /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- badmodel /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- badmodel /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","input":"say hi","max_output_tokens":16}'`
- badmodel /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","input":"say hi","max_output_tokens":16,"stream":true}'`
- badmodel /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- badmodel /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- badmodel /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","prompt":"say hi","max_tokens":5}'`
- badmodel /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","prompt":"say hi","max_tokens":5,"stream":true}'`
- badmodel /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"no-such-model","input":"say hi"}'`
- providererror-over /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- providererror-over /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- providererror-over /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","input":"say hi","max_output_tokens":16}'`
- providererror-over /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","input":"say hi","max_output_tokens":16,"stream":true}'`
- providererror-over /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5}'`
- providererror-over /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi"}],"max_tokens":5,"stream":true}'`
- providererror-over /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","prompt":"say hi","max_tokens":5}'`
- providererror-over /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","prompt":"say hi","max_tokens":5,"stream":true}'`
- providererror-over /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","input":"say hi"}'`
- providererror-rich /v1/chat/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5}'`
- providererror-rich /v1/chat/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/chat/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"stream":true}'`
- providererror-rich /v1/responses stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","input":"say hi tag=guardrail-rich","max_output_tokens":16}'`
- providererror-rich /v1/responses stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/responses -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","input":"say hi tag=guardrail-rich","max_output_tokens":16,"stream":true}'`
- providererror-rich /v1/messages stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5}'`
- providererror-rich /v1/messages stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/messages -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","messages":[{"role":"user","content":"say hi tag=guardrail-rich"}],"max_tokens":5,"stream":true}'`
- providererror-rich /v1/completions stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","prompt":"say hi tag=guardrail-rich","max_tokens":5}'`
- providererror-rich /v1/completions stream=True: `curl -sS -X POST http://127.0.0.1:<port>/v1/completions -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","prompt":"say hi tag=guardrail-rich","max_tokens":5,"stream":true}'`
- providererror-rich /v1/embeddings stream=False: `curl -sS -X POST http://127.0.0.1:<port>/v1/embeddings -H 'Authorization: Bearer <key>' -H 'Content-Type: application/json' -d '{"model":"broken-mini","input":"say hi tag=guardrail-rich"}'`

## Edge cases

E1 budget raised then lowered live (head): with guardrail-budgeted over budget, head 429 and base 200; `POST /tag/update max_budget=10` and one second later head 200; `max_budget=0.000001` again and head 429 immediately. Recorded wait: 1s. PASS

E2 budget lowered during a concurrent burst (head, 24 mixed requests across chat/responses/messages, half streaming, `POST /tag/update guardrail-rich max_budget=0.000001` fired 0.4s in): 17x200 and 7x429, no transport errors, every stream terminated, 24 rows in the table (17 success, 7 failure), 22 matched by id and the two streaming /v1/responses cells by endTime. Raising the budget back took up to 9s to reach the second worker because each uvicorn worker holds its own copy of the tag object for `DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL` (60s) and only the worker that served the update invalidates; the same per-process cache serves body tags on base and there is no Redis in this rig. PASS, with the propagation window recorded

E3 five identical requests (head, guardrail-rich): five distinct response ids, one row each, no duplicates. PASS

E4 tag row with no max_budget (null-budget): head 200 and base 200, spend accrues on the tag. PASS

E5 tag merged from a team (`team-budgeted`, $0.000001, set in the team's metadata.tags, prompt selecting guardrail-rich): 200 on both sides in two rounds twenty seconds apart even after the team tag's spend passed its budget, because auth reads team tags from `litellm_metadata` only. Pre-existing, identical on base, out of scope for this PR and documented in the PR body. PASS as parity, gap noted

## Chaos

C1 Postgres paused mid burst (head, 36 requests, 6 shapes x 6, `docker pause lit7659-pg` at 0.8s, burst finished at 3.0s, unpaused at 10s): all 36 answered 200, no unterminated stream, `/health/readiness` returned 503 `db: disconnected` while paused and 200 `connected` 2s after unpause, all 36 rows landed exactly once after the unpause (30 by id, 6 streaming /v1/responses by endTime, sum of row spend $0.0004912). The `LiteLLM_TagTable.spend` delta for guardrail-rich stayed $0 for over 60s: both workers' `update_spend` jobs fired during the pause and failed with prisma P2028 `Unable to start a transaction in the given time`, which is not in `DB_RETRY_SAFE_ERROR_TYPES`, so the batch of tag/key/team spend increments was dropped. The same burst on base (36x200, 36 rows, tag delta $0, same P2028 trace in `proxy-audit-base.log`) shows this is pre-existing spend writer behavior unrelated to the PR. Rows and readiness: PASS. Aggregate spend under a DB outage: pre-existing loss, identical on base, FLAGGED for a follow-up ticket

C2 over-budget guardrail tag with Postgres paused, base vs head: with the tag object cached, head answered the tag 429 in 34ms while paused (fail closed from cache), base served 200 (no check at all). With a key the worker had never seen, both sides hung on the auth DB lookup for the full curl timeout (head 45s, base 30s) and one head probe returned 503 `no_db_connection` after 13s; after unpause both recovered on the next request (head 429, base 200) and every request, including the 503s, has a failure row. Auth's DB outage behavior is pre-existing and identical on both sides. PASS

C3 SIGTERM to the head master mid burst (36 requests, signal at 1.2s): uvicorn drained, all 36 answered 200 with terminated streams, all 36 rows landed on the shutdown flush and the tag spend moved from $0.0018400 to $0.0023424. Relaunched at the same tip with two workers. PASS, no loss

C4 SIGKILL of one of the two head workers mid burst (36 requests, kill at 0.8s): 28x200 on the surviving worker, 8 in-flight requests on the killed worker failed at the transport (`Server disconnected without sending a response`), all 28 completed requests have exactly one row and the six follow-up probes were served by the survivor while uvicorn respawned the worker (`Child process [24397] died`, `Started server process [24854]`). The eight killed requests have no row and no reservation survives, since both lived in the dead process. PASS, expected loss documented

## Config and route legs (base vs head, one request per cell)

S1 /guardrails/apply_guardrail with the master key: 200 `{"response_text":"say ok"}` on both, the route is not budget-checked at auth so the post-hook check skips it. PASS

S2 burst of 12 against a fresh tag with room for one request ($0.00001), body tag vs guardrail tag: base body 1x200/11x429, base guardrail 12x200 with tag spend $0.0001632 and the next request 200 (the bug); head body 2x200/10x429 and head guardrail 2x200/10x429 with `Current cost: 1.44e-05, Max budget: 1e-05`, tag spend $0.0000288 and the next request 429. The split is 2 on head because each of the two workers keeps its own reservation counter without Redis and admits one; body and guardrail tags split identically, which is the contract. PASS

S3 four workers (`--num_workers 4`), 12 concurrent chat requests: base 12x200 for the default tag and 12x200 for guardrail-rich (pids 28703 28705 28707 28709); head 12x429 for the default tag and 12x200 for guardrail-rich (pids 28706 28708 28710 28711), 24 requests spread over all four workers. PASS

S4 `general_settings.custom_auth` without `custom_auth_run_common_checks`: guardrail tag 200 and body tag 200 on both (no common checks run on such a deployment); head with `custom_auth_run_common_checks: true`: 429 and 429. PASS

S5 `general_settings.public_routes: [/v1/chat/completions]`, no Authorization header: 200/200 on both for the guardrail tag and the body tag, auth runs no checks on a public route so the post-hook check does not either. PASS

S6 `/v1/batches` create through a `files_settings` proxy (input file uploaded through the proxy, one live-mini line): base 200 `batch_6aa992608cc48190ba0271785c3e602d` (cancelled), head 429 tag budget error with type `internal_server_error` (the label caveat in the PR). PASS

S7 client disconnect under `sse_keepalive_ping_interval_seconds: 1` with a real 11s upstream stream (1500 word story, `max_tokens: 1200`) cut by `curl --max-time 2.5`, fresh $0.00001 tags: body tag follow-up 429 on both with the reservation on the counter and DB spend 0 at that moment; guardrail tag follow-up base 200 (DB spend $0.000496 thirty seconds later, still 200) and head 429 `Current cost: 0.00045159999999999997`. The proxy kept consuming the upstream stream after the disconnect on both sides and billed the whole story. PASS, the reservation-after-disconnect behavior is pre-existing for body tags and now the same for guardrail tags

S8 guardrail writes `metadata.tags` on /v1/messages and /v1/responses (whose own key is `litellm_metadata`): base 200/200 and the rows carry only the User-Agent tags, so the tag is neither charged nor checked; head 429/429 with failure rows tagged guardrail-budgeted. A rich tag written the same way is served on both and not charged on either (`resp_bGl0...` rows with User-Agent tags only): checked on head, charged on neither. PASS, charging on these routes is pre-existing and unchanged

## UI evidence (head, 3c58786534)

![Tag Management list](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-tags.png)

![guardrail-budgeted detail with the $0.000001 budget](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-tag-detail.png)

![Playground /v1/chat/completions: the 429 tag budget error](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-playground-chat.png)

![Playground /v1/responses: same 429](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-playground-responses.png)

![Playground /v1/messages: same 429](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-playground-messages.png)

![Logs page with the audited requests](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-logs.png)

![Log detail of a 429 row: failure, $0, BudgetExceededError](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-logs-detail.png)

![Same row, metadata with the tag budget error](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-logs-detail-tags.png)

Recording of the whole flow with step and assertion banners: [audit-flow.webm](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-flow.webm) (Playwright capture, no ffmpeg on this machine so no mp4), frame strip [audit-flow.webp](https://raw.githubusercontent.com/BerriAI/litellm/assets-pr41218/evidence/lit7659/audit/audit-flow.webp)

## Cells not run

- OpenAI SDK sync/async for /v1/messages and Anthropic SDK for the OpenAI-shaped endpoints: the SDKs do not expose those routes
- Streaming /v1/embeddings: the endpoint has no streaming mode
- The SDK clients were driven only for the core set (over, rich); the mode and sad sets ran through curl, which reaches the same `common_processing_pre_call_logic` path
- Base with two workers: the base proxy ran one worker, so the E2 propagation window was measured on head only (the cache is per process on both sides)
- Redis-backed rig: no Redis in this rig, so the tag cache and reservation counters were in-memory per worker
- Passthrough, batch and files routes: not in the changed path

## Verdict

PASS on 3c58786534: 154 of 154 live cells match the outcome contract, all edge and chaos legs hold, the only defects found (spend batch dropped on prisma P2028 during a DB outage, team tag budgets not enforced from body metadata) reproduce identically on the merge base and are outside this PR
