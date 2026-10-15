# Coding-agent response contracts

Every row sends `CODING_AGENT_REQUEST` to `/v1/chat/completions`, changing only `model`. Each response must satisfy the independently defined public contract before its tool-call shape is compared with the OpenAI reference in its tier. OpenAI rows compare with Anthropic instead, so no row compares with itself

| Tier | Model versions |
| --- | --- |
| Small | Haiku 4.5 and 5.5, GPT 5 Mini and 5.4 Mini, GPT 5.6 Luna and 6 Luna |
| Mid | Opus 5 and 5.5, GPT 5.6 Sol, 6 Sol and 6.1 Sol, GPT 5.6 Terra |
| High | Fable 5 and 5.1, Gemini 4 Argon, GPT 6 Astra |

The 45 rows cover Anthropic direct, OpenAI direct, Vertex, Azure OpenAI, Azure AI and Bedrock Converse where accessible. Vertex Haiku 5.5 was unavailable during capture and is not represented

The request contains system instructions, conversation history, previous assistant tool calls and results, four function definitions, automatic parallel tools, low reasoning effort and an output limit. Each row exercises a tool-call response, raw streaming, SDK stream assembly and a follow-up that replays the returned assistant message and tool-call IDs

`contracts.py` requires the response envelope, assistant role, content key, finish reason, two unique tool-call IDs, JSON-string arguments and nonnegative token counts whose total equals prompt plus completion tokens. Generated text, IDs, timestamps, token values and tool order may differ between models. Each non-streaming case also checks its complete recorded response, including reasoning fields and usage. Optional reasoning and cache token details are type-checked when present

The raw stream check assembles tool names and argument fragments by index. The SDK check separately uses `OpenAI.chat.completions.stream()` without removing reasoning fields. Seven Haiku SDK checks are explicitly skipped for [LIT-9653](https://linear.app/litellm-ai/issue/LIT-9653), because unindexed `thinking_blocks` crash the SDK assembler. Their non-streaming, tool-loop and raw-stream checks still run

## Recorded upstreams

The cases use `TranslationTestCase` and `StreamingTranslationTestCase`, shared client and provider templates, and the translation runner. Existing model bases stay unchanged and agent scenarios use `dataclasses.replace`. Models without a base have one in `chat_completions/bases/`. The three tier tuples in `case.py` group those cases without a second capture format or JSON loader

Native provider requests and responses were recorded using the shared synthetic conversation. Normal CI replays them through `SharedProvider` and the real proxy, asserting the upstream method, target, headers and complete body. Initial and follow-up cases also assert the complete public response. Streaming cases use the independent raw-stream and SDK contracts instead of pinning generated chunk IDs and transport-dependent chunk boundaries. Bedrock frames use the shared runner's AWS event-stream encoding. CI needs no real provider credentials

To refresh a row, follow the capture procedure in the parent translation README with `CODING_AGENT_REQUEST`. Deep-copy each request before every provider call so a provider transformation cannot change later captures. Capture the initial call, the same request with streaming and usage enabled, and the follow-up built from the returned assistant message and `TOOL_RESULTS`

Replace project paths, signatures, encrypted reasoning and redacted blocks with synthetic values, preserving relationships between a response and its follow-up. Keep the documented Gemini skip-validation signature unchanged. Targets use the synthetic deployment's API base, so OpenAI targets omit `/v1` and Vertex Anthropic stream targets include `?alt=sse`

Do not generate expectations with LiteLLM translators. Add the model's synthetic deployment to the shared proxy config. Reuse the common request in `chat_completions/templates.py` and the matching wire request in `translation/templates/<provider>.py`. Store only model-specific differences in the base or agent variant. Add its initial, follow-up and streaming cases to the appropriate tier tuple

## Run

```bash
INTEGRATION_WORKERS=2 .venv/bin/python tests/integration/run.py providers \
  tests/integration/translation/chat_completions/agent/test_agent_contract.py
```

This first request does not cover image input, strict structured output, cache hits, search citations, refusals or truncation. Those require separate recorded scenarios
