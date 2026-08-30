# Cost Calculation Audit

## Current Rust Implementation
The Rust gateway has **basic cost calculation** in `cost_calculator` module:
- Supports input/output token costs
- Supports cache read/creation tokens
- Supports audio tokens (input/output)
- Supports image tokens (input)
- Supports reasoning tokens (output)
- Supports service tier pricing (priority/flex)
- Supports token threshold pricing (128k, 200k)
- Uses pricing database lookup

### Current Cost Calculation Support
```rust
// In cost_calculator/calc.rs
pub fn calculate_cost(request: &CostRequest<'_>) -> CoreResult<CostResponse> {
    // Lookup model pricing
    // Calculate input cost (text, cache, audio, image tokens)
    // Calculate output cost (text, reasoning, audio tokens)
    // Handle service tier and token thresholds
}
```

### Current Pricing Database
```rust
// In cost_calculator/pricing.rs
// Loads pricing from JSON/YAML
// Lookup model pricing by name
// Supports custom pricing databases
```

## Python Implementation

### Comprehensive Cost Calculation
Python has extensive cost calculation support:

#### 1. Provider-Specific Cost Calculators
- **OpenAI**: cost_per_token, cost_per_second (for audio/video)
- **Anthropic**: cost_per_token with cache handling
- **Bedrock**: cost_per_token with regional pricing
- **Azure**: cost_per_token with Azure-specific pricing
- **Azure AI**: cost_per_token with Azure Foundry pricing
- **Vertex AI**: cost_per_token, cost_per_character
- **Gemini**: cost_per_token
- **Databricks**: cost_per_token
- **DeepSeek**: cost_per_token
- **Fireworks AI**: cost_per_token
- **Perplexity**: cost_per_token
- **Tencent**: cost_per_token
- **Together AI**: cost_per_token with registry pricing
- **xAI**: cost_per_token
- **Lemonade**: cost_per_token

#### 2. Call Type-Specific Cost Calculation
- **Chat Completions**: Standard token-based pricing
- **Text Completions**: Standard token-based pricing
- **Embeddings**: Token-based pricing
- **Audio Transcription**: Duration-based or token-based pricing
- **Audio Speech (TTS)**: Character-based or duration-based pricing
- **Image Generation**: Per-image pricing with quality tiers
- **Video Generation**: Duration-based pricing
- **Rerank**: Billed units pricing
- **Search**: Query-based pricing
- **Realtime**: Session-based pricing
- **MCP Tools**: Tool call pricing

#### 3. Advanced Pricing Features
- **Custom Pricing**: User-defined cost per token/second
- **Character-Based Pricing**: For Vertex AI and other character-based models
- **Duration-Based Pricing**: For audio/video/speech
- **Quality Tier Pricing**: For image generation (standard, HD, etc.)
- **Regional Pricing**: Data residency and regional processing uplifts
- **Service Tier Pricing**: Priority, flex, standard tiers
- **Cache Pricing**: Cache read and creation token costs
- **Additional Costs**: Routing fees, infrastructure costs

#### 4. Usage Object Handling
- **OpenAI-Style Usage**: prompt_tokens includes cached_tokens
- **Anthropic-Style Usage**: prompt_tokens excludes cache tokens
- **Transcription Usage**: Duration and token-based usage
- **Rerank Usage**: Billed units
- **Search Usage**: Query counts
- **Realtime Usage**: Session duration

#### 5. Model Variant Support
- **Base Models**: gpt-4, gpt-3.5-turbo, claude-3, etc.
- **Model Variants**: gpt-4-turbo, gpt-4o, claude-3-opus, etc.
- **Date-Stamped Models**: gpt-4-0613, claude-3-opus-20240229, etc.
- **Fine-Tuned Models**: Custom model pricing
- **Regional Models**: Azure regional models, Vertex regional models

#### 6. Pricing Database
- **Comprehensive Pricing**: Thousands of models with pricing
- **Dynamic Updates**: Pricing database can be updated
- **Custom Pricing**: Users can add custom pricing
- **Fallback Pricing**: Default pricing for unknown models

#### 7. Cost Calculation Helpers
- **Token Type Breakdown**: Separate costs for different token types
- **Billable Input Tokens**: Calculate billable tokens excluding cache
- **Cost Component Calculation**: Calculate individual cost components
- **Service Tier Cost Key**: Resolve service tier to cost key
- **Regional Uplift Multiplier**: Calculate regional pricing uplift

## Major Gaps in Rust Implementation

### 1. Missing Provider-Specific Cost Calculators
**Critical Gap**: No provider-specific cost calculation logic
- Need OpenAI-specific cost calculator (audio/video pricing)
- Need Anthropic-specific cost calculator (cache handling)
- Need Bedrock-specific cost calculator (regional pricing)
- Need Azure-specific cost calculator (Azure pricing)
- Need Vertex AI-specific cost calculator (character pricing)
- Need many more provider-specific calculators

### 2. Missing Call Type-Specific Cost Calculation
**Critical Gap**: No call type-specific cost calculation
- Need audio transcription cost calculation (duration-based)
- Need audio speech cost calculation (character/duration-based)
- Need image generation cost calculation (per-image with quality)
- Need video generation cost calculation (duration-based)
- Need rerank cost calculation (billed units)
- Need search cost calculation (query-based)
- Need realtime cost calculation (session-based)

### 3. Missing Character-Based Pricing
**High Priority**: No character-based pricing support
- Need to count characters in prompt/completion
- Need to calculate cost per character
- Need to support Vertex AI character pricing
- Need to support other character-based models

### 4. Missing Duration-Based Pricing
**High Priority**: No duration-based pricing support
- Need to track request duration
- Need to calculate cost per second
- Need to support audio transcription duration pricing
- Need to support audio speech duration pricing
- Need to support video generation duration pricing

### 5. Missing Image Generation Pricing
**High Priority**: No image generation pricing support
- Need per-image pricing
- Need quality tier pricing (standard, HD, etc.)
- Need size-based pricing
- Need to count number of images

### 6. Missing Regional Pricing
**Medium Priority**: No regional pricing support
- Need data residency pricing (EU, US, etc.)
- Need regional processing uplifts
- Need Vertex AI location-based pricing
- Need Azure regional pricing

### 7. Missing Additional Costs
**Medium Priority**: No additional cost support
- Need routing fee calculation
- Need infrastructure cost calculation
- Need provider-specific additional costs
- Need custom additional costs

### 8. Missing Usage Object Handling
**Medium Priority**: Limited usage object handling
- Need to handle Anthropic-style usage (cache tokens separate)
- Need to handle transcription usage (duration + tokens)
- Need to handle rerank usage (billed units)
- Need to handle search usage (query counts)

### 9. Missing Model Variant Support
**Medium Priority**: Limited model variant support
- Need to handle date-stamped models
- Need to handle fine-tuned models
- Need to handle regional models
- Need to handle model aliases

### 10. Missing Cost Calculation Helpers
**Low Priority**: No cost calculation helpers
- Need token type breakdown
- Need billable input token calculation
- Need cost component calculation
- Need service tier cost key resolution
- Need regional uplift multiplier calculation

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### Critical (Must Have)
1. **Provider-Specific Calculators** - OpenAI, Anthropic, Bedrock cost calculators
2. **Call Type Support** - Audio, image, video cost calculation

### High Priority
3. **Character-Based Pricing** - Vertex AI character pricing
4. **Duration-Based Pricing** - Audio/video duration pricing
5. **Image Generation Pricing** - Per-image with quality tiers

### Medium Priority
6. **Regional Pricing** - Data residency and regional uplifts
7. **Additional Costs** - Routing fees, infrastructure costs
8. **Usage Object Handling** - Anthropic-style, transcription, rerank

### Low Priority
9. **Model Variant Support** - Date-stamped, fine-tuned, regional models
10. **Cost Calculation Helpers** - Token breakdown, billable tokens, etc.

## Implementation Plan

### Phase 1: Provider-Specific Cost Calculators (Critical)
1. Add OpenAI-specific cost calculator (audio/video pricing)
2. Add Anthropic-specific cost calculator (cache handling)
3. Add Bedrock-specific cost calculator (regional pricing)
4. Add Azure-specific cost calculator (Azure pricing)
5. Add Vertex AI-specific cost calculator (character pricing)
6. Add tests for provider-specific cost calculators

### Phase 2: Call Type-Specific Cost Calculation (Critical)
7. Add audio transcription cost calculation (duration-based)
8. Add audio speech cost calculation (character/duration-based)
9. Add image generation cost calculation (per-image with quality)
10. Add video generation cost calculation (duration-based)
11. Add rerank cost calculation (billed units)
12. Add search cost calculation (query-based)
13. Add realtime cost calculation (session-based)
14. Add tests for call type-specific cost calculation

### Phase 3: Character-Based Pricing (High Priority)
15. Add character counting for prompt/completion
16. Add cost per character calculation
17. Support Vertex AI character pricing
18. Support other character-based models
19. Add tests for character-based pricing

### Phase 4: Duration-Based Pricing (High Priority)
20. Add request duration tracking
21. Add cost per second calculation
22. Support audio transcription duration pricing
23. Support audio speech duration pricing
24. Support video generation duration pricing
25. Add tests for duration-based pricing

### Phase 5: Image Generation Pricing (High Priority)
26. Add per-image pricing
27. Add quality tier pricing (standard, HD, etc.)
28. Add size-based pricing
29. Add image count tracking
30. Add tests for image generation pricing

### Phase 6: Regional Pricing (Medium Priority)
31. Add data residency pricing (EU, US, etc.)
32. Add regional processing uplifts
33. Add Vertex AI location-based pricing
34. Add Azure regional pricing
35. Add tests for regional pricing

### Phase 7: Additional Costs (Medium Priority)
36. Add routing fee calculation
37. Add infrastructure cost calculation
38. Add provider-specific additional costs
39. Add custom additional costs
40. Add tests for additional costs

### Phase 8: Usage Object Handling (Medium Priority)
41. Handle Anthropic-style usage (cache tokens separate)
42. Handle transcription usage (duration + tokens)
43. Handle rerank usage (billed units)
44. Handle search usage (query counts)
45. Add tests for usage object handling

### Phase 9: Model Variant Support (Medium Priority)
46. Handle date-stamped models
47. Handle fine-tuned models
48. Handle regional models
49. Handle model aliases
50. Add tests for model variant support

### Phase 10: Cost Calculation Helpers (Low Priority)
51. Add token type breakdown
52. Add billable input token calculation
53. Add cost component calculation
54. Add service tier cost key resolution
55. Add regional uplift multiplier calculation
56. Add tests for cost calculation helpers

## Testing Plan
- Unit tests for each cost calculation feature
- Integration tests with real provider responses
- Provider-specific cost calculator tests
- Call type-specific cost calculation tests
- Character-based pricing tests
- Duration-based pricing tests
- Image generation pricing tests
- Regional pricing tests
- Additional costs tests
- Usage object handling tests
- Model variant support tests

## Comparison with Previous Audits
This gap is **less critical** than function/tool calling, vision/multimodal, and provider-specific features because:
1. Basic cost calculation already exists in Rust
2. Cost calculation is used for spend tracking and billing
3. The current cost calculation works for basic cases
4. Provider-specific cost calculators are the most critical missing feature

However, it's still important because:
1. Provider-specific cost calculators improve accuracy
2. Call type-specific cost calculation is essential for non-chat endpoints
3. Character/duration-based pricing is needed for audio/video
4. Regional pricing is needed for compliance

## Estimated Effort
- Phase 1 (Provider-Specific Calculators): 4-5 days
- Phase 2 (Call Type-Specific): 3-4 days
- Phase 3 (Character-Based): 2-3 days
- Phase 4 (Duration-Based): 2-3 days
- Phase 5 (Image Generation): 2-3 days
- Phase 6 (Regional Pricing): 2-3 days
- Phase 7 (Additional Costs): 2-3 days
- Phase 8 (Usage Object): 2-3 days
- Phase 9 (Model Variants): 2-3 days
- Phase 10 (Helpers): 2-3 days

**Total**: 23-34 days for complete cost calculation parity
