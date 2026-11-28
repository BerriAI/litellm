# Streaming contract matrix candidate

This suite evaluates streaming behavior through `completion`, `acompletion`, `text_completion`, and `atext_completion`. It uses actual provider parsing, response conversion, aggregation, and final SDK logging. Only provider HTTP traffic is simulated, using injected HTTP clients. `mock_response` cases exercise the SDK's own mock path

Run from the repository root with existing test dependencies:

```sh
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest tests/test_litellm/test_streaming_contract_matrix.py -q -n 2 -o addopts=
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest tests/test_litellm/test_streaming_contract_matrix.py --collect-only -q -o addopts=
```

The second command prints every exact parameterized case ID. A single case can be selected by appending `::test_name[parameters]` to the filename. Add `--runxfail` to reproduce a known failure without its exclusion

No provider credentials, proxy, database, or Docker stack are needed. Placeholder credentials and injected HTTP transports handle requests; unexpected HTTP traffic is rejected. The local cost-map setting also avoids fetching registry data during import

## Cases and observations

| Cases | Coverage |
| --- | --- |
| `test_sync_without_running_loop` | Ordinary synchronous calls without an active event loop; all five provider paths and both SDK surfaces |
| `test_success` | OpenAI chat, Anthropic, Gemini, native OpenAI text; both SDK surfaces and modes; five usage-option states; prompt/output counts 0, 1, and larger |
| `test_mock_reservations` | Both surfaces and modes; all usage-option states; both metadata fields; absent, null, empty, null count, zero, one, and larger admission counts |
| `test_fragmented_text`, `test_mock_content` | Empty, ordinary, Unicode, and combining characters; provider content fragments; HTTP byte boundaries including inside UTF-8 and SSE frames |
| `test_sse_without_optional_space` | Standard SSE data fields without the optional space after the colon, across all four HTTP provider paths and both modes |
| `test_fragmented_tool`, `test_mock_tool`, `test_tool_with_text` | Tool ID/name/arguments, optional accompanying text, caller terminal reason, assembled callback content and usage |
| `test_midstream_failure` | OpenAI, Anthropic, Gemini; chat/text and sync/async; content observed before a transport failure; caller error and exactly one final failure event |
| `test_early_close`, `test_async_cancel` | Close after receiving content; async cancellation while awaiting more provider data followed by explicit close; no fabricated final response; cleanup before fixture teardown |
| `test_close_inside_cancelled_scope` | OpenAI/Gemini explicit close while AnyIO cancellation is active, with a transport-close await checkpoint |
| `test_exhaustion_cleanup` | Simulated transport closure after draining success through trailing usage |
| `test_native_text_bridge`, `test_stream_nonstream_parity` | Supported chat/text bridges and equivalent deterministic streaming/nonstreaming results |
| `test_tool_usage_options`, `test_mock_tool_zero_reservation` | Tool usage options with/without text, model-sensitive fallback, and admitted zero on structured mock tools |
| `test_model_specific_usage_fallback`, `test_unicode_prompt_usage`, `test_unicode_partial_usage` | Model-sensitive Unicode input/output fallback and exact partial usage after Unicode content is interrupted |
| `test_provider_usage_outranks_admission`, `test_absent_provider_usage` | Real provider counts outrank admission metadata; deterministic fallback when usage is absent |

The callback observer is registered as a real `CustomLogger`. Tests never invoke its final callbacks directly. It records the emitted response and failure usage; tests wait up to five seconds for an event and observe a further 50 milliseconds for duplicates. Streaming assertions cover the observed final event count, outcome, content, and deterministic token totals. Parity cases compare nonstream caller content/usage and final event count/outcome; nonstream callback payload content is not asserted

The observer uses matching SDK/consumer modes and one registered logger. Legacy string callback integrations, internal re-dispatch, and async consumption of a synchronous SDK response are outside this matrix

OpenAI chat defaults to requesting usage for omitted/None options in this revision. Empty/false options suppress upstream usage, so those cases assert fallback counts instead of inventing provider counts that would not have arrived. Native OpenAI text requests usage only when explicitly enabled. Anthropic and Gemini usage remains available internally even when caller-visible usage is disabled. Legacy text responses can carry zero-valued usage placeholders while actual usage is hidden

## Applicability limits

The fixtures use one response choice and one modern function tool. The tool takes explicit city/count arguments; zero-argument functions are not modeled. Each tool stream supplies its function type; omission on every fragment is not modeled. Multiple choices, parallel tool calls, legacy `function_call`, reasoning/thinking, audio/images, annotations/citations, provider-specific metadata, and dollar-cost accounting are outside this evaluation. Token usage means prompt, completion, and total counts; token subcategories are not covered. Tool-name sanitization, exact placement/repetition of optional role fields, and the number of otherwise empty nonterminal chunks are not evaluated

Completed-response cases verify `stop` and `tool_calls`. Token-limit truncation, safety/refusal terminal reasons, and provider-request rejection are not modeled; the failure lifecycle uses a transport interruption after content begins

Content order is checked against the provider's emitted sequence. The suite does not alter private chunk timestamps, simulate clock jumps, or pass shuffled chunks directly to aggregation. Exact timestamp copying and model-router changes between chunks are outside this matrix

Legacy text completion has no chat tool-call representation; tool cases use the chat surface. Gemini GenerateContent function arguments arrive as JSON objects, so its tool cases fragment HTTP bytes rather than inventing OpenAI-style argument-delta fields. Anthropic message usage is required; absence/null usage cases are limited to paths whose response contract permits them

HTTPX is the selected transport; the alternate aiohttp backend is not evaluated. The reservation Cartesian matrix uses string mock responses. Structured mock responses have focused empty/Unicode and tool cases, including an admitted-zero tool case across all usage options; other structured reservation values are not fully crossed

HTTP fixtures use SSE, including both permitted colon-space forms. Alternate accumulated-JSON transport modes are outside this matrix. The SSE whitespace rule follows the [event-stream specification](https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation)

Anthropic text fixtures begin with an empty `content_block_start` and deliver text through deltas, matching the [documented streaming examples](https://platform.claude.com/docs/en/build-with-claude/streaming). Nonempty text in the start event is not covered. Model-sensitive fallback counts reflect the current registry and tokenizer dependencies; an intentional tokenizer change may require updating those expectations

Synchronous calls use explicit close; cancellation requires an async wait. Cancellation cases measure cancellation followed by the SDK's explicit `aclose`, not automatic cleanup of an abandoned iterator. The text bridge does not expose the chat wrapper's close API, so early-close cases use chat. `mock_response` has no provider transport to fail or close and its deterministic string generator does not await provider data. Provider interruption cases therefore use actual HTTP provider paths

Two focused cases also close within an already cancelled AnyIO scope. Their simulated transport yields during asynchronous close, so the assertions can detect cleanup interrupted by cancellation. This follows the [AnyIO finalization contract](https://anyio.readthedocs.io/en/stable/cancellation.html#finalization)

Partial usage is asserted for async failure paths where the SDK supplies it. Sync failure callbacks currently supply no partial usage object. The suite does not invent a contract that requires one. Transport-close observations establish cleanup at the injected HTTP boundary, not live provider connection-pool behavior

## Existing findings

| Reference | Narrow excluded check |
| --- | --- |
| STREAM-001 | Anthropic transport closure on exhaustion/early close/cancel plus close; Gemini sync early-close transport closure |
| STREAM-002 | Exact usage when a provider explicitly reports zero and aggregation substitutes estimated usage |
| STREAM-003 | Final tool callback reason `stop` instead of `tool_calls` for Gemini and structured mock responses |
| STREAM-004 | Empty string `mock_response` sends a provider request |
| STREAM-005 | Exact cold native sync/async text OpenAI/Pydantic `MockValSer` serialization error |

Each exclusion is reached only at the affected assertion or exact exception. STREAM-002 accepts only the observed recount for this fixture: a zero prompt count becomes 8, a zero output count becomes 2, and total usage remains their sum. Nonzero counts must remain unchanged. Both caller-visible and callback usage must match either the correct tuple or that exact known tuple before the expected failure is recorded

Other assertions execute first where the stream can complete. STREAM-005 can interrupt the stream, so later assertions in that occurrence remain unexecuted. Its attribution to a particular package is unresolved. No production workaround or schema warming is performed

These cases must not be reported as fully protected. Use the verbose test report or JUnit output to see exact affected combinations. The findings and local mutation evidence are provided separately from the implementation diff

## Evaluation and integration

The existing root unit-test target discovers `tests/test_litellm/test_*.py`, including this file. No CI schedules or new required gates are added. This is a candidate for evaluation, with local mutation validation and explicit known-behavior exclusions; measured local runtime is not a measured CircleCI runtime

The local validation report records exact source revisions, environment, case IDs, resource samples, handpicked fault results, automated mutation identifiers/diffs, reviewed survivors, and the applicable denominator. It also records functions omitted by mutmut's decorated-function limitation. The test branch contains only the suite, fixtures, and this usage documentation
