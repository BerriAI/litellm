# Provider-Specific Features Audit

## Current Rust Implementation
The Rust gateway has **minimal provider-specific feature support**:
- Basic parameter passthrough via `optional_params` field
- No provider-specific validation or transformation
- No provider-specific configuration blocks
- No provider-specific header handling
- No provider-specific feature detection

### Current Provider Support
```rust
// In chat_completions/types.rs
pub struct ChatCompletionsRequest<'a> {
    pub model: &'a str,
    pub messages: Value,
    pub optional_params: Map<String, Value>,  // Generic passthrough
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}
```

## Python Implementation

### OpenAI Provider-Specific Features

#### 1. Response Format
- `response_format`: JSON mode, JSON schema, text
- `json_schema`: Structured output with schema validation
- `strict`: Strict schema validation mode

#### 2. Predicted Outputs
- `prediction`: Predicted output for content reuse
- `content`: Predicted content string

#### 3. Audio Support
- `modalities`: ["text", "audio"]
- `audio`: Audio output configuration (voice, format)

#### 4. Web Search
- `web_search_options`: Web search configuration
- `search_context_size`: low, medium, high
- `user_location`: Approximate location for search

#### 5. Service Tier
- `service_tier`: default, auto, scale_tier

#### 6. Store and Background
- `store`: Enable/disable response storage
- `background`: Enable background processing

#### 7. Logprobs
- `logprobs`: Return log probabilities
- `top_logprobs`: Number of top logprobs to return

#### 8. Seed
- `seed`: Reproducible sampling seed

#### 9. Stop Sequences
- `stop`: Up to 4 stop sequences

#### 10. Tools and Tool Choice
- `tools`: Tool definitions
- `tool_choice`: auto, none, required, specific tool
- `parallel_tool_calls`: Enable/disable parallel tool calls

#### 11. Function Calling (Deprecated)
- `functions`: Function definitions
- `function_call`: auto, none, specific function

### Anthropic Provider-Specific Features

#### 1. Extended Thinking
- `thinking`: Enable extended thinking
- `budget_tokens`: Token budget for thinking
- `thinking_config`: Advanced thinking configuration
- `output_config`: Output configuration for thinking

#### 2. Prompt Caching
- `cache_control`: Cache control markers
- `ephemeral`: Ephemeral cache type
- Cache breakpoints for long conversations

#### 3. Computer Use
- `computer_use_preview`: Computer use tool
- Tool prefixes: computer_use_preview, computer_, bash_, text_editor_
- Screen interaction capabilities

#### 4. Beta Headers
- `anthropic-beta`: Beta feature headers
- Advanced tool use, prompt caching, etc.
- Beta header filtering for unsupported features

#### 5. Top K Sampling
- `top_k`: Top K sampling parameter
- Not supported by all models

#### 6. Metadata
- `metadata`: Request metadata
- User ID tracking

#### 7. System Messages
- `system`: System message handling
- System message caching

#### 8. Tool Use
- `tools`: Tool definitions
- `tool_choice`: auto, any, tool, none
- Tool result handling

### Bedrock Provider-Specific Features

#### 1. Guardrail Configuration
- `guardrailConfig`: Guardrail configuration block
- `guardrailIdentifier`: Guardrail ID
- `guardrailVersion`: Guardrail version
- `trace`: Enable guardrail tracing

#### 2. Performance Configuration
- `performanceConfig`: Performance configuration block
- `latency`: standard, optimized

#### 3. Service Tier
- `serviceTier`: Service tier configuration
- `tier`: Service tier level

#### 4. Request Metadata
- `requestMetadata`: Request metadata dictionary
- Maximum 16 items
- Key/value pattern validation
- Character limits (1-256 for keys, 0-256 for values)

#### 5. Nova 2 Reasoning
- `reasoningConfig`: Nova 2 reasoning configuration
- Different from Anthropic thinking
- Model-specific reasoning support

#### 6. Computer Use Tools
- Tool prefixes: computer_use_preview, computer_, bash_, text_editor_
- Screen interaction capabilities
- Tool name validation

#### 7. Beta Header Filtering
- Filter unsupported beta headers
- advanced-tool-use, prompt-caching, compact-2026-01-12
- Prevent errors from unsupported features

#### 8. Consecutive User Messages
- Convert consecutive user messages to guarded_text
- Required for guardrailConfig
- Automatic message transformation

#### 9. Application Inference Profiles
- Support for application inference profile ARNs
- Profile-based routing

#### 10. Claude 4.5 on Bedrock
- Special handling for Claude 4.5 models
- Output config effort normalization
- Adaptive thinking support

#### 11. Tool Name Transformation
- Transform tool names for Bedrock compatibility
- Make valid Bedrock tool names
- Tool name mapping

#### 12. Parallel Tool Use
- `parallel_tool_use_config`: Parallel tool use configuration
- Model-specific support detection

#### 13. Adaptive Thinking
- `adaptive_thinking`: Adaptive thinking configuration
- Drop unsupported warnings
- Output config effort mapping

## Major Gaps in Rust Implementation

### 1. Missing OpenAI-Specific Features
**Critical Gap**: No OpenAI-specific feature support
- Need response_format (JSON mode, JSON schema)
- Need prediction (predicted outputs)
- Need modalities and audio support
- Need web_search_options
- Need service_tier
- Need store and background
- Need logprobs and top_logprobs
- Need seed
- Need parallel_tool_calls

### 2. Missing Anthropic-Specific Features
**Critical Gap**: No Anthropic-specific feature support
- Need thinking and extended_thinking
- Need budget_tokens and thinking_config
- Need cache_control and prompt caching
- Need computer_use tools
- Need anthropic-beta headers
- Need top_k
- Need metadata
- Need system message handling

### 3. Missing Bedrock-Specific Features
**Critical Gap**: No Bedrock-specific feature support
- Need guardrailConfig
- Need performanceConfig
- Need serviceTier
- Need requestMetadata with validation
- Need reasoningConfig for Nova 2
- Need computer use tool handling
- Need beta header filtering
- Need consecutive user message conversion
- Need application inference profile support
- Need Claude 4.5 special handling
- Need tool name transformation
- Need parallel_tool_use_config
- Need adaptive_thinking

### 4. Missing Provider-Specific Validation
**High Priority**: No provider-specific parameter validation
- Need to validate provider-specific parameters
- Need to check model-specific feature support
- Need to validate parameter combinations
- Need to provide helpful error messages

### 5. Missing Provider-Specific Transformation
**High Priority**: No provider-specific parameter transformation
- Need to transform OpenAI params to Anthropic format
- Need to transform OpenAI params to Bedrock format
- Need to transform Anthropic params to OpenAI format
- Need to transform Bedrock params to OpenAI format

### 6. Missing Provider-Specific Header Handling
**High Priority**: No provider-specific header handling
- Need to handle anthropic-beta headers
- Need to filter unsupported beta headers
- Need to add provider-specific headers
- Need to merge extra_headers correctly

### 7. Missing Feature Detection
**Medium Priority**: No provider feature detection
- Need to detect which features a model supports
- Need to check model capabilities
- Need to provide feature availability information
- Need to warn about unsupported features

### 8. Missing Configuration Blocks
**Medium Priority**: No provider-specific configuration blocks
- Need guardrailConfig block for Bedrock
- Need performanceConfig block for Bedrock
- Need thinking_config block for Anthropic
- Need response_format block for OpenAI

### 9. Missing Parameter Mapping
**Medium Priority**: No parameter mapping between providers
- Need to map thinking to reasoningConfig
- Need to map response_format to provider format
- Need to map tool_choice to provider format
- Need to map cache_control to provider format

### 10. Missing Error Handling
**Low Priority**: No provider-specific error handling
- Need to handle provider-specific errors
- Need to provide helpful error messages
- Need to suggest alternatives for unsupported features
- Need to log provider-specific issues

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### Critical (Must Have)
1. **OpenAI Features** - response_format, prediction, modalities, web_search, service_tier
2. **Anthropic Features** - thinking, cache_control, computer_use, beta headers
3. **Bedrock Features** - guardrailConfig, performanceConfig, requestMetadata, reasoningConfig

### High Priority
4. **Provider Validation** - Validate provider-specific parameters
5. **Provider Transformation** - Transform parameters between providers
6. **Provider Headers** - Handle provider-specific headers

### Medium Priority
7. **Feature Detection** - Detect model capabilities
8. **Configuration Blocks** - Support provider-specific config blocks
9. **Parameter Mapping** - Map parameters between providers

### Low Priority
10. **Error Handling** - Provider-specific error messages

## Implementation Plan

### Phase 1: OpenAI-Specific Features (Critical)
1. Add response_format support (JSON mode, JSON schema)
2. Add prediction support (predicted outputs)
3. Add modalities and audio support
4. Add web_search_options support
5. Add service_tier support
6. Add store and background support
7. Add logprobs and top_logprobs support
8. Add seed support
9. Add parallel_tool_calls support
10. Add tests for OpenAI-specific features

### Phase 2: Anthropic-Specific Features (Critical)
11. Add thinking and extended_thinking support
12. Add budget_tokens and thinking_config support
13. Add cache_control and prompt caching support
14. Add computer_use tool support
15. Add anthropic-beta header handling
16. Add top_k support
17. Add metadata support
18. Add system message handling
19. Add tests for Anthropic-specific features

### Phase 3: Bedrock-Specific Features (Critical)
20. Add guardrailConfig support
21. Add performanceConfig support
22. Add serviceTier support
23. Add requestMetadata with validation
24. Add reasoningConfig for Nova 2
25. Add computer use tool handling
26. Add beta header filtering
27. Add consecutive user message conversion
28. Add application inference profile support
29. Add Claude 4.5 special handling
30. Add tool name transformation
31. Add parallel_tool_use_config support
32. Add adaptive_thinking support
33. Add tests for Bedrock-specific features

### Phase 4: Provider Validation (High Priority)
34. Implement provider-specific parameter validation
35. Check model-specific feature support
36. Validate parameter combinations
37. Provide helpful error messages
38. Add tests for provider validation

### Phase 5: Provider Transformation (High Priority)
39. Implement OpenAI to Anthropic transformation
40. Implement OpenAI to Bedrock transformation
41. Implement Anthropic to OpenAI transformation
42. Implement Bedrock to OpenAI transformation
43. Add tests for provider transformation

### Phase 6: Provider Headers (High Priority)
44. Handle anthropic-beta headers
45. Filter unsupported beta headers
46. Add provider-specific headers
47. Merge extra_headers correctly
48. Add tests for provider headers

### Phase 7: Feature Detection (Medium Priority)
49. Detect which features a model supports
50. Check model capabilities
51. Provide feature availability information
52. Warn about unsupported features
53. Add tests for feature detection

### Phase 8: Configuration Blocks (Medium Priority)
54. Add guardrailConfig block for Bedrock
55. Add performanceConfig block for Bedrock
56. Add thinking_config block for Anthropic
57. Add response_format block for OpenAI
58. Add tests for configuration blocks

### Phase 9: Parameter Mapping (Medium Priority)
59. Map thinking to reasoningConfig
60. Map response_format to provider format
61. Map tool_choice to provider format
62. Map cache_control to provider format
63. Add tests for parameter mapping

### Phase 10: Error Handling (Low Priority)
64. Handle provider-specific errors
65. Provide helpful error messages
66. Suggest alternatives for unsupported features
67. Log provider-specific issues
68. Add tests for error handling

## Testing Plan
- Unit tests for each provider-specific feature
- Integration tests with real provider responses
- Provider transformation tests
- Provider validation tests
- Feature detection tests
- Configuration block tests
- Parameter mapping tests
- Error handling tests

## Comparison with Previous Audits
This gap is **less critical** than function/tool calling and vision/multimodal but **more critical** than streaming edge cases because:
1. Provider-specific features are essential for advanced use cases
2. Without provider-specific features, users can't access provider capabilities
3. Major providers (OpenAI, Anthropic, Bedrock) have unique features
4. Python has comprehensive provider-specific feature support
5. Provider-specific features differentiate the gateway from basic proxies

However, it's less critical than tool calling and vision because:
1. Basic functionality works without provider-specific features
2. Provider-specific features are advanced capabilities
3. Users can work around missing features in some cases
4. Provider-specific features are provider-dependent

## Estimated Effort
- Phase 1 (OpenAI Features): 3-4 days
- Phase 2 (Anthropic Features): 3-4 days
- Phase 3 (Bedrock Features): 4-5 days
- Phase 4 (Provider Validation): 2-3 days
- Phase 5 (Provider Transformation): 3-4 days
- Phase 6 (Provider Headers): 1-2 days
- Phase 7 (Feature Detection): 2-3 days
- Phase 8 (Configuration Blocks): 2-3 days
- Phase 9 (Parameter Mapping): 2-3 days
- Phase 10 (Error Handling): 1-2 days

**Total**: 23-33 days for complete provider-specific feature parity
