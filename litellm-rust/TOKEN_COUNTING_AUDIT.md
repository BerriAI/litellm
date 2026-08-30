# Token Counting Audit

## Current Rust Implementation
The Rust gateway has **basic token counting** in `token_counter` module:
- Uses tiktoken for OpenAI models (o200k_base for gpt-4o, cl100k_base as fallback)
- Uses HuggingFace tokenizer for some Anthropic models (Claude, but not Claude-3)
- Supports text and message counting
- Supports tools and tool_choice counting
- No image token counting support
- No custom tokenizer support
- No provider-specific token counting

### Current Token Counting Support
```rust
// In token_counter/mod.rs
pub fn token_counter(request: &TokenCounterRequest<'_>) -> CoreResult<usize> {
    // Supports text or messages
    // Supports tools and tool_choice
    // Uses tiktoken or HuggingFace tokenizer
}
```

### Current Tokenizer Resolution
```rust
// In token_counter/hf_tokenizer.rs
pub(crate) fn resolve(model: &str) -> ResolvedTokenizer {
    // Try HuggingFace tokenizer for Claude (non-Claude-3)
    // Fall back to tiktoken for OpenAI models
    // Use o200k_base for gpt-4o
    // Use cl100k_base as default fallback
}
```

## Python Implementation

### Comprehensive Token Counting
Python has extensive token counting support:

#### 1. Text and Message Counting
- Count tokens in raw text
- Count tokens in message lists
- Handle different message formats (OpenAI, Anthropic, etc.)
- Support system, user, assistant, tool messages

#### 2. Tool and Function Call Counting
- Count tokens in tool definitions
- Count tokens in tool_choice
- Count tokens in tool_calls (arguments)
- Count tokens in function_call (arguments)
- Handle parallel tool calls

#### 3. Image Token Counting
- Count tokens for image URLs (download and analyze)
- Count tokens for base64 images
- Support different image resolutions (low, high, auto)
- Calculate tokens based on image dimensions
- Support default image token count (no download)
- Handle image tiles for high-resolution images

#### 4. Custom Tokenizer Support
- Support custom HuggingFace tokenizers
- Support custom tiktoken encodings
- Allow users to provide their own tokenizer
- Cache custom tokenizers

#### 5. Provider-Specific Token Counting
- OpenAI: tiktoken-based counting
- Anthropic: HuggingFace tokenizer
- Bedrock: Provider-specific tokenizers
- Vertex AI: Provider-specific tokenizers
- Cohere: Provider-specific tokenizers
- Many more providers

#### 6. Model-Specific Token Counting
- Different tokens_per_message for different models
- Different tokens_per_name for different models
- Model-specific token counting rules
- Handle model name variations (gpt-3.5-turbo-0301, etc.)

#### 7. Response Token Counting
- Count tokens in streaming responses
- Count tokens in complete responses
- Handle response format variations

#### 8. Default Token Count
- Provide default token count for error cases
- Configurable default token count
- Fallback when token counting fails

#### 9. Rust Bridge Integration
- Python calls Rust token counter for performance
- Falls back to Python if Rust fails
- Seamless integration

#### 10. Advanced Features
- Disable token counter flag (for performance)
- Chunked token counting for large texts
- Verbose logging for debugging
- Error handling and fallbacks

## Major Gaps in Rust Implementation

### 1. Missing Image Token Counting
**Critical Gap**: No image token counting support
- Need to count tokens for image URLs
- Need to count tokens for base64 images
- Need to support different image resolutions
- Need to calculate tokens based on image dimensions
- Need to handle image tiles for high-resolution images
- Need to support default image token count

### 2. Missing Custom Tokenizer Support
**High Priority**: No custom tokenizer support
- Need to support custom HuggingFace tokenizers
- Need to support custom tiktoken encodings
- Need to allow users to provide their own tokenizer
- Need to cache custom tokenizers

### 3. Missing Provider-Specific Token Counting
**High Priority**: Limited provider-specific token counting
- Need Bedrock-specific tokenizers
- Need Vertex AI-specific tokenizers
- Need Cohere-specific tokenizers
- Need more provider-specific tokenizers
- Need to handle provider-specific token counting rules

### 4. Missing Claude-3 Tokenizer
**High Priority**: No Claude-3 tokenizer support
- Current implementation only supports Claude (non-Claude-3)
- Need to add Claude-3 tokenizer
- Need to handle Claude-3-specific token counting

### 5. Missing Model-Specific Token Counting
**Medium Priority**: Limited model-specific token counting
- Need different tokens_per_message for different models
- Need different tokens_per_name for different models
- Need model-specific token counting rules
- Need to handle model name variations

### 6. Missing Response Token Counting
**Medium Priority**: No response token counting
- Need to count tokens in streaming responses
- Need to count tokens in complete responses
- Need to handle response format variations

### 7. Missing Default Token Count
**Medium Priority**: No default token count support
- Need to provide default token count for error cases
- Need configurable default token count
- Need fallback when token counting fails

### 8. Missing Advanced Features
**Low Priority**: No advanced token counting features
- Need disable token counter flag
- Need chunked token counting for large texts
- Need verbose logging for debugging
- Need better error handling and fallbacks

### 9. Missing Token Counting for All Message Types
**Low Priority**: Limited message type support
- Need to handle all message types (system, user, assistant, tool, function)
- Need to handle message name tokens
- Need to handle message role tokens

### 10. Missing Token Counting Optimization
**Low Priority**: No token counting optimization
- Need to cache tokenizers
- Need to optimize token counting for performance
- Need to reduce memory usage

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### Critical (Must Have)
1. **Image Token Counting** - Count tokens for images (URLs, base64, resolutions)
2. **Claude-3 Tokenizer** - Add Claude-3 tokenizer support

### High Priority
3. **Custom Tokenizer Support** - Allow users to provide custom tokenizers
4. **Bedrock Tokenizers** - Add Bedrock-specific tokenizers
5. **Provider-Specific Counting** - Handle provider-specific token counting rules

### Medium Priority
6. **Model-Specific Counting** - Different rules for different models
7. **Response Token Counting** - Count tokens in responses
8. **Default Token Count** - Provide defaults for error cases

### Low Priority
9. **Advanced Features** - Disable flag, chunking, logging
10. **Optimization** - Caching, performance, memory usage

## Implementation Plan

### Phase 1: Image Token Counting (Critical)
1. Add image URL download and analysis
2. Add base64 image token counting
3. Support different image resolutions (low, high, auto)
4. Calculate tokens based on image dimensions
5. Handle image tiles for high-resolution images
6. Support default image token count (no download)
7. Add tests for image token counting

### Phase 2: Claude-3 Tokenizer (Critical)
8. Add Claude-3 tokenizer support
9. Handle Claude-3-specific token counting
10. Update tokenizer resolution logic
11. Add tests for Claude-3 token counting

### Phase 3: Custom Tokenizer Support (High Priority)
12. Add custom HuggingFace tokenizer support
13. Add custom tiktoken encoding support
14. Allow users to provide their own tokenizer
15. Cache custom tokenizers
16. Add tests for custom tokenizer support

### Phase 4: Provider-Specific Tokenizers (High Priority)
17. Add Bedrock-specific tokenizers
18. Add Vertex AI-specific tokenizers
19. Add Cohere-specific tokenizers
20. Handle provider-specific token counting rules
21. Add tests for provider-specific tokenizers

### Phase 5: Model-Specific Token Counting (Medium Priority)
22. Add different tokens_per_message for different models
23. Add different tokens_per_name for different models
24. Add model-specific token counting rules
25. Handle model name variations
26. Add tests for model-specific token counting

### Phase 6: Response Token Counting (Medium Priority)
27. Count tokens in streaming responses
28. Count tokens in complete responses
29. Handle response format variations
30. Add tests for response token counting

### Phase 7: Default Token Count (Medium Priority)
31. Provide default token count for error cases
32. Add configurable default token count
33. Add fallback when token counting fails
34. Add tests for default token count

### Phase 8: Advanced Features (Low Priority)
35. Add disable token counter flag
36. Add chunked token counting for large texts
37. Add verbose logging for debugging
38. Improve error handling and fallbacks
39. Add tests for advanced features

### Phase 9: Message Type Support (Low Priority)
40. Handle all message types (system, user, assistant, tool, function)
41. Handle message name tokens
42. Handle message role tokens
43. Add tests for message type support

### Phase 10: Optimization (Low Priority)
44. Cache tokenizers
45. Optimize token counting for performance
46. Reduce memory usage
47. Add benchmarks for token counting performance

## Testing Plan
- Unit tests for each token counting feature
- Integration tests with real provider responses
- Image token counting tests (URLs, base64, resolutions)
- Custom tokenizer tests
- Provider-specific tokenizer tests
- Model-specific token counting tests
- Response token counting tests
- Default token count tests
- Advanced feature tests
- Performance benchmarks

## Comparison with Previous Audits
This gap is **less critical** than function/tool calling, vision/multimodal, and provider-specific features because:
1. Basic token counting already exists in Rust
2. Token counting is used for cost calculation and rate limiting
3. The current token counting works for basic cases
4. Image token counting is the most critical missing feature

However, it's still important because:
1. Image token counting is essential for vision models
2. Custom tokenizers allow flexibility
3. Provider-specific tokenizers improve accuracy
4. Model-specific token counting improves accuracy

## Estimated Effort
- Phase 1 (Image Token Counting): 3-4 days
- Phase 2 (Claude-3 Tokenizer): 1-2 days
- Phase 3 (Custom Tokenizer Support): 2-3 days
- Phase 4 (Provider-Specific Tokenizers): 3-4 days
- Phase 5 (Model-Specific Counting): 2-3 days
- Phase 6 (Response Token Counting): 1-2 days
- Phase 7 (Default Token Count): 1 day
- Phase 8 (Advanced Features): 2-3 days
- Phase 9 (Message Type Support): 1-2 days
- Phase 10 (Optimization): 2-3 days

**Total**: 18-27 days for complete token counting parity
