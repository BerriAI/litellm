# Cross-model response contracts

Every row sends `CODING_AGENT_REQUEST` to `/v1/chat/completions`, changing only `model`. Initial and follow-up responses must satisfy the independently defined public contract and match the standard response structure of the OpenAI reference in their tier. OpenAI rows compare with Anthropic instead, so no row compares with itself

| Tier | Model versions |
| --- | --- |
| Small | Haiku 4.5 and 5.5, GPT 5 Mini and 5.4 Mini, GPT 5.6 Luna and 6 Luna |
| Mid | Opus 5 and 5.5, GPT 5.6 Sol, 6 Sol and 6.1 Sol, GPT 5.6 Terra |
| High | Fable 5 and 5.1, Gemini 4 Argon, GPT 6 Astra |

The 45 rows cover Anthropic direct, OpenAI direct, Vertex, Azure OpenAI, Azure AI and Bedrock Converse where accessible. Vertex Haiku 5.5 was unavailable during capture and is not represented

The request contains system instructions, conversation history, previous assistant tool calls and results, four function definitions, automatic parallel tools, low reasoning effort and an output limit. Each row exercises a tool-call response, raw streaming, SDK stream assembly and a follow-up that replays the returned assistant message and tool-call IDs

## Assertions

Start with `test_cross_model_contract.py` for the four scenarios and `contracts.py` for their requirements

| Check | Compared across models | Compared with each row's independent expectation |
| --- | --- | --- |
| Initial and final responses | Standard field presence, types and nesting, including one choice at index 0 | Complete response, with only explicitly marked transport-generated values flexible |
| Initial tools | Two distinct IDs, function type, JSON-string arguments, `search_code.query` and `run_tests.path` as nonempty strings | Exact IDs, names and argument values |
| Final answer | Assistant role, nonempty text, stop finish reason, no tool calls | Exact final text and other response fields |
| Raw stream | Same envelope requirements, valid tool indexes, assistant role, one tool-calls finish, final usage and one terminal `[DONE]` | Assembled text, tool IDs, names, parsed argument values, reasoning and reported usage counters |
| SDK stream | Same completed tool-call contract through `OpenAI.chat.completions.stream()` | The same independently recorded message and usage values as the raw stream |
| Provider request | Shared client request apart from model | Exact method, target, relevant headers and complete body, including replayed assistant message and tool results |

`response_structure()` compares recursive shapes, not generated values. IDs, timestamps, text, argument values and token counts may differ between models. The initial tool shape is compared separately, independent of tool order. Content may be text or null. A final message's absent, null or empty `tool_calls` are equivalent

`OPTIONAL_FIELDS` lists standard fields whose presence can legitimately vary, including `system_fingerprint`, `service_tier`, `logprobs`, `refusal`, `annotations`, `audio`, legacy `function_call` and usage detail objects. Their types are validated when present, but their presence is not compared across models. `PROVIDER_EXTENSIONS` explicitly lists provider-specific fields excluded from this comparison, including reasoning formats and Vertex metadata. Unlisted fields are not silently discarded

Streaming cases store `expected_litellm_message` and `expected_litellm_usage`. Message values are compared after joining deltas and parsing tool argument JSON, so whitespace and chunk boundaries are irrelevant. Reasoning text, thinking signatures, redacted data and encrypted reasoning items must survive. An empty reasoning-item `content` list may be omitted. Thinking indexes are transport metadata, not compared as content. Raw assembly accepts the known unindexed Haiku thinking fragments, but does not claim SDK compatibility for them

Usage requires nonnegative integer counts, positive prompt and completion counts, and `total_tokens == prompt_tokens + completion_tokens`. The complete normalized usage must equal the independently expected counters, including cached and reasoning tokens and derived text/cache details. Missing, changed or unexpected counters fail. Only null optional fields, the explicitly listed provider metadata/duplicate counters and computed `cost` are excluded. Thus a wrong prompt count with an adjusted total still fails

The raw stream check assembles tool names and argument fragments by index. The SDK check separately uses `OpenAI.chat.completions.stream()` without removing reasoning fields. Seven Haiku SDK checks are explicitly skipped for [LIT-9653](https://linear.app/litellm-ai/issue/LIT-9653), because unindexed `thinking_blocks` crash the SDK assembler. Their non-streaming, tool-loop and raw-stream checks still run

## Recorded upstreams

The cases use `TranslationTestCase` and `StreamingTranslationTestCase`, shared client and provider templates, and the translation runner. Existing model bases stay unchanged and agent scenarios use `dataclasses.replace`. Models without a base have one in `chat_completions/bases/`. The three tier tuples in `case.py` group those cases without a second capture format or JSON loader

Native provider requests and responses were recorded using the shared synthetic conversation. Normal CI replays them through `SharedProvider` and the real proxy. Streaming cases use independent expected message values and reported usage rather than pinning generated chunk IDs or transport-dependent chunk boundaries. Chunk IDs must remain consistent within a stream. Bedrock frames use the shared runner's AWS event-stream encoding. CI needs no real provider credentials

To refresh a row, follow the capture procedure in the parent translation README with `CODING_AGENT_REQUEST`. Deep-copy each request before every provider call so a provider transformation cannot change later captures. Capture the initial call, the same request with streaming and usage enabled, and the follow-up built from the returned assistant message and `TOOL_RESULTS`

Replace project paths, signatures, encrypted reasoning and redacted blocks with synthetic values, preserving relationships between a response and its follow-up. Keep the documented Gemini skip-validation signature unchanged. Targets use the synthetic deployment's API base, so OpenAI targets omit `/v1` and Vertex Anthropic stream targets include `?alt=sse`

Do not generate expectations with LiteLLM translators. Add the model's synthetic deployment to the shared proxy config. Reuse the common request in `chat_completions/templates.py` and the matching wire request in `translation/templates/<provider>.py`. Store only model-specific differences in the base or agent variant. Populate the stream message and usage expectations from the provider's own frames, including any reported optional counters. Add its initial, follow-up and streaming cases to the appropriate tier tuple

## Run

The assertion verification rejected 2,179 malformed payloads with 137 valid controls passing. It covers missing or extra fields, wrong types, split choices, changed tool arguments, lost reasoning and missing, changed or unexpected usage counters. A synthetic positive control with `cached_tokens=600` also rejects dropping or changing that value. These are assertion-level mutations, not a historical production-version replay, and the recorded matrix itself does not exercise nonzero cache hits

```bash
INTEGRATION_WORKERS=2 .venv/bin/python tests/integration/run.py providers \
  tests/integration/translation/chat_completions/cross_model/test_cross_model_contract.py
```

This first request does not cover image input, strict structured output, nonzero cache hits, search citations, refusals or truncation. It exercises only the client-facing Chat Completions endpoint, even when the provider uses Responses or Messages. Missing trigger scenarios such as multiple Responses message items before tools require separate variants. Timing, timeouts, semantic-cache state, minimal SDK installs and callbacks need other test groups. These assertions alone do not prove the stable release gate blocks regressions
