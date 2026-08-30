# Function/Tool Calling Variations Audit

## Current Rust Implementation
The Rust gateway has **minimal to no function/tool calling support**:
- `ChatCompletionsChoiceMessage` only has `role` and `content` fields
- No `tool_calls` or `function_call` fields in response types
- No `ChatCompletionMessageToolCall` or `FunctionCall` types defined
- Tool calling only appears in token counter tests and Anthropic tests (not in actual types)

## Python Implementation

### Core Types
Python has comprehensive tool calling support with multiple types:

#### Message Class (Non-Streaming)
```python
class Message:
    content: str | None
    role: Literal["assistant", "user", "system", "tool", "function"]
    tool_calls: list[ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall] | None
    function_call: FunctionCall | None
    audio: ChatCompletionAudioResponse | None
    images: list[ImageURLListItem] | None
    reasoning_content: str | None
    thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None
    reasoning_items: list[ChatCompletionReasoningItem] | None
    provider_specific_fields: dict[str, Any] | None
    annotations: list[ChatCompletionAnnotation] | None
```

#### Delta Class (Streaming)
```python
class Delta:
    content: str | None
    role: str | None
    function_call: FunctionCall | None
    tool_calls: list[ChatCompletionDeltaToolCall | ChatCompletionDeltaCustomToolCall] | None
    audio: ChatCompletionAudioResponse | None
    images: list[ImageURLListItem] | None
    annotations: list[ChatCompletionAnnotation] | None
    reasoning_content: str | None
    thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None
    reasoning_items: list[ChatCompletionReasoningItem] | None
    provider_specific_fields: dict[str, Any] | None
```

### Tool Call Types
1. **ChatCompletionMessageToolCall** - Standard OpenAI tool call
   - `id`: str
   - `type`: "function"
   - `function`: FunctionCall

2. **ChatCompletionMessageCustomToolCall** - Custom tool call format
   - Provider-specific tool call formats

3. **ChatCompletionDeltaToolCall** - Streaming tool call delta
   - `index`: int
   - `id`: str | None
   - `type`: "function" | None
   - `function`: FunctionCallDelta

4. **ChatCompletionDeltaCustomToolCall** - Custom streaming tool call

5. **FunctionCall** - Function call details
   - `name`: str
   - `arguments`: str (JSON string)

6. **FunctionCallDelta** - Streaming function call delta
   - `name`: str | None
   - `arguments`: str | None

## Provider-Specific Variations

### OpenAI
- Standard tool calling with `tool_calls` array
- Parallel tool calls supported
- `tool_choice` parameter: "auto", "none", "required", or specific tool
- Streaming tool calls with incremental argument building
- Function calling (deprecated but still supported)

### Anthropic
- Tool use with `tool_use` content blocks
- Different format than OpenAI
- `tool_choice` parameter with different semantics
- Streaming tool use with partial JSON arguments
- Extended thinking with tool use

### Bedrock (Converse API)
- Tool use with `toolUse` content blocks
- Similar to Anthropic format
- `toolChoice` parameter
- Streaming tool use

### Azure OpenAI
- Same as OpenAI but with additional validation
- Special handling for None arguments in tool calls
- Content type handling for tool calls

### Other Providers
- **Mistral**: Returns `type: None` for tool calls (needs normalization)
- **Vertex AI**: Proto-based function calls with different format
- **Cohere**: Different tool calling format
- **Databricks**: Custom tool call format

## Major Gaps in Rust Implementation

### 1. Missing Core Types
**Critical Gap**: No tool calling types defined in Rust
- Need `ToolCall` struct with id, type, function fields
- Need `FunctionCall` struct with name, arguments fields
- Need `ToolCallDelta` for streaming
- Need `FunctionCallDelta` for streaming

### 2. Missing Response Fields
**Critical Gap**: Response types don't include tool calling fields
- `ChatCompletionsChoiceMessage` needs `tool_calls` field
- `ChatCompletionsChoiceMessage` needs `function_call` field
- Need to handle both standard and custom tool calls

### 3. Missing Request Support
**High Priority**: No tool calling in request types
- Need `tools` parameter in request
- Need `tool_choice` parameter
- Need tool definition types
- Need to pass tools through to provider

### 4. Missing Streaming Support
**High Priority**: No tool calling in streaming
- Need to parse tool call deltas
- Need to accumulate partial JSON arguments
- Need to handle parallel tool calls in stream
- Need to emit complete tool calls when finished

### 5. Missing Provider Transforms
**High Priority**: No provider-specific tool call transforms
- OpenAI → Anthropic tool call conversion
- OpenAI → Bedrock tool call conversion
- Anthropic → OpenAI tool call conversion
- Bedrock → OpenAI tool call conversion

### 6. Missing Tool Choice Handling
**Medium Priority**: No tool_choice parameter support
- Need to parse tool_choice from request
- Need to transform tool_choice per provider
- Need to handle "auto", "none", "required", specific tool

### 7. Missing Parallel Tool Calls
**Medium Priority**: No parallel tool call support
- Need to handle multiple tool_calls in response
- Need to handle multiple tool calls in streaming
- Need to accumulate multiple parallel tool calls

### 8. Missing Argument Accumulation
**Medium Priority**: No streaming argument accumulation
- Need to accumulate partial JSON arguments across chunks
- Need to handle malformed JSON during streaming
- Need to validate complete JSON when stream ends

### 9. Missing Tool Call ID Generation
**Low Priority**: No tool call ID handling
- Need to preserve provider-generated IDs
- Need to generate IDs when provider doesn't provide them
- Need to handle ID format variations

### 10. Missing Custom Tool Call Support
**Low Priority**: No custom tool call format support
- Need to support provider-specific tool call formats
- Need normalization layer to convert to standard format
- Need to preserve provider-specific fields

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### Critical (Must Have)
1. **Core Tool Call Types** - Define ToolCall, FunctionCall structs
2. **Response Fields** - Add tool_calls/function_call to response types
3. **Request Support** - Add tools/tool_choice to request types
4. **Provider Transforms** - Implement tool call transforms for OpenAI/Anthropic/Bedrock

### High Priority
5. **Streaming Support** - Parse and accumulate tool call deltas
6. **Parallel Tool Calls** - Handle multiple tool calls in response/stream
7. **Argument Accumulation** - Build complete JSON from partial chunks
8. **Tool Choice Handling** - Support tool_choice parameter

### Medium Priority
9. **Error Handling** - Handle malformed tool calls, invalid JSON
10. **Validation** - Validate tool call structure per provider
11. **Custom Tool Calls** - Support provider-specific formats

### Low Priority
12. **ID Generation** - Generate/preserve tool call IDs
13. **Annotations** - Support tool call annotations
14. **Advanced Features** - Tool result handling, multi-turn tool use

## Implementation Plan

### Phase 1: Core Types (Critical)
1. Define `ToolCall` struct
2. Define `FunctionCall` struct
3. Define `ToolCallDelta` struct
4. Define `FunctionCallDelta` struct
5. Add `tool_calls` and `function_call` to `ChatCompletionsChoiceMessage`
6. Add tests for type serialization/deserialization

### Phase 2: Request Support (Critical)
7. Add `tools` parameter to request types
8. Add `tool_choice` parameter to request types
9. Define tool definition types
10. Pass tools through to provider calls

### Phase 3: Provider Transforms (Critical)
11. Implement OpenAI tool call transform (passthrough)
12. Implement Anthropic tool call transform (OpenAI ↔ Anthropic format)
13. Implement Bedrock tool call transform (OpenAI ↔ Bedrock format)
14. Add tests for each provider transform

### Phase 4: Streaming Support (High Priority)
15. Parse tool call deltas in streaming
16. Accumulate partial JSON arguments
17. Handle parallel tool calls in stream
18. Emit complete tool calls when finished
19. Add streaming tests

### Phase 5: Tool Choice (High Priority)
20. Parse tool_choice from request
21. Transform tool_choice per provider
22. Handle "auto", "none", "required", specific tool
23. Add tests for tool_choice handling

### Phase 6: Advanced Features (Medium Priority)
24. Add error handling for malformed tool calls
25. Add validation for tool call structure
26. Support custom tool call formats
27. Add tool call ID handling

## Testing Plan
- Unit tests for each tool call type
- Integration tests with real provider responses
- Streaming tests with partial tool calls
- Parallel tool call tests
- Provider transform tests (OpenAI ↔ Anthropic ↔ Bedrock)
- Tool choice handling tests
- Error case tests (malformed JSON, invalid structure)

## Comparison with Streaming Audit
This gap is **more critical** than streaming edge cases because:
1. Tool calling is a core feature for modern LLM applications
2. Without tool calling, the gateway can't support agentic workflows
3. Major providers (OpenAI, Anthropic, Bedrock) all support tool calling
4. Python has comprehensive tool calling support that we need to match

## Estimated Effort
- Phase 1 (Core Types): 2-3 days
- Phase 2 (Request Support): 1-2 days
- Phase 3 (Provider Transforms): 3-4 days
- Phase 4 (Streaming Support): 3-4 days
- Phase 5 (Tool Choice): 1-2 days
- Phase 6 (Advanced Features): 2-3 days

**Total**: 12-18 days for complete tool calling parity
