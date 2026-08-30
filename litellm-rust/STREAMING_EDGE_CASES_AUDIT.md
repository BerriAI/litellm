# Streaming Edge Cases Audit

## Current Rust Implementation
The Rust gateway has basic streaming support:
- SSE parsing with `data:` prefix handling
- `[DONE]` signal detection
- JSON chunk parsing
- Error handling for network/HTTP errors
- SpendTrackingStream wrapper for cost tracking
- Usage accumulation from chunks (OpenAI's stream_options.include_usage format)

## Python Implementation Gaps

### 1. Provider-Specific Chunk Handlers
Python has dedicated handlers for 20+ providers:
- **Predibase**: Special JSON format with `token.text` and `details.finish_reason`
- **AI21**: Fake streaming (full response in each chunk)
- **Maritalk**: Fake streaming
- **NLP Cloud**: Full response with `[DONE]` marker, dolphin model special handling
- **Aleph Alpha**: Fake streaming
- **Vertex AI**: Proto-based chunks with function call handling, SAFETY blocked responses
- **Petals/PaLM**: Fake streaming with chunk splitting
- **Triton**: Custom format
- **OpenAI text completion**: Different from chat completion
- **Codestral text completion**: Custom format
- **Azure text**: Custom format
- **Cached response**: Special handling for cached streams
- **OpenAI/Azure chat**: Standard handler with function/tool call parsing

**Rust Gap**: Only has generic OpenAI-compatible SSE parsing. No provider-specific handlers.

### 2. Special Token Handling
Python handles special tokens for SageMaker/HF:
- `<|assistant|>`, `<|system|>`, `<|user|>`, `<s>`, `</s>`, `<|im_end|>`, `<|im_start|>`
- Holds chunks until special tokens are complete
- Strips special tokens from output

**Rust Gap**: No special token handling.

### 3. Model Repetition Detection
Python detects infinite loops:
- Tracks last N chunks
- Raises error if same chunk repeated >= REPEATED_STREAMING_CHUNK_LIMIT times
- Allows retries on repetition detection

**Rust Gap**: No repetition detection.

### 4. Stream Usage Tracking
Python has comprehensive usage tracking:
- `stream_options.include_usage` support
- Calculates total usage from all chunks
- Handles provider-reported cost (Perplexity breakdown format)
- Propagates cost to hidden params
- Usage-only chunks (OpenRouter post-finish)

**Rust Gap**: Basic usage accumulation from OpenAI format only. No provider-reported cost handling.

### 5. Function/Tool Call Parsing
Python has extensive function/tool call handling:
- Parses `function_call` and `tool_calls` from delta
- Handles None arguments (Azure, Mistral)
- Converts tool calls to JSON mode when needed
- Handles invalid parallel tool calls
- Tracks tool call state

**Rust Gap**: No function/tool call parsing in streaming.

### 6. Thinking Block Handling
Python handles thinking/reasoning blocks:
- `thinking_blocks` delta attribute
- `reasoning_content` delta attribute
- `signature_delta` for Anthropic
- Merges reasoning content in choices (configurable)
- Tracks thinking block state (first/last)

**Rust Gap**: No thinking block handling.

### 7. Audio/Image Delta Handling
Python handles special delta content:
- `audio` attribute for audio models
- `images` attribute for image models
- Special delta content detection and handling

**Rust Gap**: No audio/image delta handling.

### 8. Role Stripping
Python strips role from delta after first chunk:
- First chunk includes `role: "assistant"`
- Subsequent chunks have role stripped
- Handles Mistral's None role

**Rust Gap**: No role stripping.

### 9. Provider-Specific Fields
Python preserves provider-specific fields:
- `provider_specific_fields` in chunks
- Copies to model response
- Handles hidden params

**Rust Gap**: No provider-specific field handling.

### 10. Max Streaming Duration
Python enforces max streaming duration:
- `LITELLM_MAX_STREAMING_DURATION_SECONDS` config
- Raises Timeout if exceeded
- Allows cleanup

**Rust Gap**: No max streaming duration enforcement.

### 11. Correlation Context Restoration
Python restores trace/session context:
- Restores context in consumer thread/task
- Handles async/sync streams
- Best-effort cleanup on abandonment
- Prevents context leakage

**Rust Gap**: No correlation context handling (tracing middleware handles this differently).

### 12. Finish Reason Handling
Python has complex finish reason logic:
- Tracks `received_finish_reason` and `intermittent_finish_reason`
- Strips finish_reason from content chunks
- Emits finish_reason on trailing empty-delta chunk
- Handles tool_calls finish reason
- Handles error finish reason

**Rust Gap**: Basic finish reason handling only.

### 13. Combined Chunk Splitting
Python splits combined chunks:
- Vertex AI Gemma sends content + finish_reason in same chunk
- Splits into content-only chunk followed by finish-only chunk
- Prevents content dropping

**Rust Gap**: No combined chunk splitting.

### 14. MCP Metadata Handling
Python adds MCP metadata:
- Adds `mcp_list_tools` to first chunk
- Adds MCP metadata to final chunk

**Rust Gap**: No MCP metadata handling.

### 15. Cache Storage
Python handles cached responses:
- Detects cached_response provider
- Sets cache_hit flag
- Handles cached stream format

**Rust Gap**: No cache storage in streaming (caching module exists but not integrated).

### 16. Post-Call Rules
Python runs post-call rules:
- `rules.post_call_rules(input, model)`
- Allows validation/transformation after streaming

**Rust Gap**: No post-call rules.

### 17. Logging Integration
Python has comprehensive logging:
- Logs each chunk asynchronously
- Handles sync/async logging
- Disable streaming logging option
- Completion start time tracking

**Rust Gap**: Basic logging only (no per-chunk logging).

### 18. Stream Chunk Builder
Python builds complete response from chunks:
- `stream_chunk_builder()` assembles full response
- Handles errors during building
- Falls back to best-effort usage on error

**Rust Gap**: No stream chunk builder.

### 19. Response Headers
Python processes response headers:
- Extracts model ID from headers
- Extracts system fingerprint
- Preserves in hidden params

**Rust Gap**: No response header processing.

### 20. Empty Stream Handling
Python handles empty streams:
- Detects empty model response stream
- Skips empty usage chunks
- Handles StopIteration gracefully

**Rust Gap**: Basic empty handling only.

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### High Priority
1. **Function/Tool Call Parsing** - Critical for tool use
2. **Thinking Block Handling** - Required for Anthropic extended thinking
3. **Finish Reason Handling** - Needed for proper stream termination
4. **Stream Usage Tracking** - Required for accurate cost calculation
5. **Provider-Specific Fields** - Needed for provider-specific features

### Medium Priority
6. **Role Stripping** - Needed for proper OpenAI compatibility
7. **Max Streaming Duration** - Safety feature
8. **Model Repetition Detection** - Prevents infinite loops
9. **Combined Chunk Splitting** - Needed for some providers
10. **Response Headers** - Needed for model ID tracking

### Low Priority
11. **Special Token Handling** - Only for specific providers
12. **Audio/Image Delta** - Only for specific models
13. **MCP Metadata** - Only for MCP integration
14. **Cache Storage** - Can be added later
15. **Post-Call Rules** - Can be added later
16. **Correlation Context** - Already handled by tracing middleware
17. **Logging Integration** - Basic logging exists
18. **Stream Chunk Builder** - Can be added later
19. **Empty Stream Handling** - Basic handling exists
20. **Provider-Specific Handlers** - Only needed for non-standard providers

## Implementation Plan

### Phase 1: Core Streaming Enhancements (High Priority)
1. Add function/tool call parsing to streaming chunks
2. Add thinking block handling for Anthropic
3. Improve finish reason handling
4. Enhance stream usage tracking with provider-reported cost
5. Add provider-specific field preservation

### Phase 2: Robustness Features (Medium Priority)
6. Add role stripping from delta
7. Add max streaming duration enforcement
8. Add model repetition detection
9. Add combined chunk splitting
10. Add response header processing

### Phase 3: Advanced Features (Low Priority)
11. Add special token handling (if needed for major providers)
12. Add audio/image delta handling (if needed)
13. Add MCP metadata handling (if needed)
14. Integrate cache storage with streaming
15. Add post-call rules
16. Add stream chunk builder
17. Improve empty stream handling

## Testing Plan
- Unit tests for each streaming enhancement
- Integration tests with real provider streams
- Edge case tests (repetition, timeout, empty streams)
- Provider-specific tests (OpenAI, Anthropic, Bedrock)
