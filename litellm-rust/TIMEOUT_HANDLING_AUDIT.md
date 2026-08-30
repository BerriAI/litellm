# Timeout Handling Audit

## Current Rust Implementation
The Rust gateway has **basic timeout handling**:
- Global request timeout via `TimeoutLayer` (default 300s, configurable via `REQUEST_TIMEOUT_SECS` env var)
- Per-request timeout support via `timeout` field in request types
- Timeout is applied to the entire request lifecycle (from receiving request to sending response)

### Current Timeout Configuration
```rust
// In routes/mod.rs
let request_timeout = std::env::var("REQUEST_TIMEOUT_SECS")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(300u64);

Router::new()
    // ... routes ...
    .layer(TimeoutLayer::new(std::time::Duration::from_secs(request_timeout)))
```

### Current Timeout in Requests
```rust
// In chat_completions/types.rs
pub struct ChatCompletionsRequest<'a> {
    // ... other fields ...
    pub timeout: Option<Duration>,
}
```

## Python Implementation

### Timeout Configuration Levels
Python has comprehensive timeout handling at multiple levels:

#### 1. Router-Level Timeout
```python
class RouterConfig(BaseModel):
    timeout: float | None = None  # Global timeout for all requests
```

#### 2. Per-Request Timeout
```python
# In completion request
timeout: float | None = None  # Per-request timeout override
```

#### 3. Per-Deployment Timeout
```python
class LiteLLMParams(BaseModel):
    timeout: float | None = None  # Per-deployment timeout
```

#### 4. HTTP Client Timeout Components
Python's HTTP client supports separate timeout components:
- **connect_timeout**: Time to establish connection
- **read_timeout**: Time to read response (idle timeout between chunks)
- **pool_timeout**: Time to wait for connection from pool
- **total_timeout**: Total time for entire request

### Streaming-Specific Timeout Handling
Python has special handling for streaming timeouts:
- **Idle timeout**: Maximum time between chunks (default 60s)
- **Total timeout**: Maximum time for entire stream
- **Chunk timeout**: Maximum time to receive next chunk
- Configurable via `timeout` parameter or environment variables

### Provider-Specific Timeout Handling
Python has provider-specific timeout logic:
- **OpenAI**: Respects rate limit headers, adjusts timeout dynamically
- **Anthropic**: Handles overloaded errors with exponential backoff
- **Bedrock**: Handles throttling with provider-specific timeouts
- **Azure**: Handles Azure-specific timeout patterns
- **Vertex AI**: Handles quota-based timeouts

### Timeout Error Handling
Python maps timeout errors to specific exception types:
- **Timeout**: Request timeout (408/504)
- **ConnectTimeout**: Connection timeout
- **ReadTimeout**: Read timeout (idle timeout between chunks)
- **PoolTimeout**: Connection pool timeout

### Advanced Timeout Features
- **Dynamic timeout adjustment**: Adjusts timeout based on provider response times
- **Timeout retry logic**: Retries on timeout with exponential backoff
- **Timeout metrics**: Tracks timeout rates per deployment
- **Timeout-based routing**: Routes away from deployments with high timeout rates
- **Streaming keepalive**: Sends keepalive messages to prevent idle timeouts
- **Timeout inheritance**: Child requests inherit parent timeout minus elapsed time

## Major Gaps in Rust Implementation

### 1. Missing Timeout Components
**Critical Gap**: No separate timeout components
- Need connect_timeout (connection establishment)
- Need read_timeout (idle timeout between chunks)
- Need pool_timeout (connection pool wait time)
- Need total_timeout (entire request lifecycle)

### 2. Missing Streaming-Specific Timeouts
**High Priority**: No streaming timeout handling
- Need idle timeout (max time between chunks)
- Need total stream timeout (max time for entire stream)
- Need chunk timeout (max time to receive next chunk)
- Need streaming keepalive support

### 3. Missing Per-Deployment Timeout Configuration
**High Priority**: No per-deployment timeout configuration
- Need timeout field in deployment configuration
- Need to apply deployment-specific timeouts
- Need to override global timeout per deployment
- Need to support timeout in model_list config

### 4. Missing Provider-Specific Timeout Logic
**High Priority**: No provider-specific timeout handling
- Need OpenAI-specific timeout logic (rate limit adjustments)
- Need Anthropic-specific timeout logic (overloaded handling)
- Need Bedrock-specific timeout logic (throttling handling)
- Need Azure-specific timeout logic
- Need Vertex AI-specific timeout logic

### 5. Missing Dynamic Timeout Adjustment
**Medium Priority**: No dynamic timeout adjustment
- Need to track provider response times
- Need to adjust timeout based on historical data
- Need to implement adaptive timeout algorithms
- Need to support manual timeout overrides

### 6. Missing Timeout Metrics
**Medium Priority**: No timeout-specific metrics
- Need to track timeout rates per deployment
- Need to track timeout types (connect, read, total)
- Need to expose timeout metrics via Prometheus
- Need to alert on high timeout rates

### 7. Missing Timeout-Based Routing
**Medium Priority**: No timeout-based routing decisions
- Need to route away from deployments with high timeout rates
- Need to factor timeout rates into routing decisions
- Need to implement timeout-based health checks
- Need to cooldown deployments with excessive timeouts

### 8. Missing Timeout Error Categorization
**Low Priority**: No timeout error categorization
- Need to categorize timeout errors (connect, read, total)
- Need to map timeout errors to specific exception types
- Need to handle timeout errors differently based on type
- Need to log timeout error details

### 9. Missing Timeout Configuration in Config File
**Low Priority**: No timeout configuration in YAML
- Need to support timeout in general_settings
- Need to support timeout in model_list entries
- Need to support timeout in router_settings
- Need to parse timeout from config file

### 10. Missing Timeout Inheritance
**Low Priority**: No timeout inheritance for nested requests
- Need to pass timeout to child requests
- Need to calculate remaining timeout for child requests
- Need to handle timeout propagation through middleware
- Need to support timeout cancellation

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### Critical (Must Have)
1. **Timeout Components** - Separate connect, read, pool, total timeouts
2. **Streaming Timeouts** - Idle timeout, total stream timeout, chunk timeout
3. **Per-Deployment Timeout** - Configure timeout per deployment

### High Priority
4. **Provider-Specific Timeouts** - Handle provider-specific timeout patterns
5. **Timeout Metrics** - Track and expose timeout metrics
6. **Timeout Error Handling** - Categorize and handle timeout errors

### Medium Priority
7. **Dynamic Timeout Adjustment** - Adjust timeout based on provider performance
8. **Timeout-Based Routing** - Route away from high-timeout deployments
9. **Timeout Configuration** - Support timeout in config files

### Low Priority
10. **Timeout Inheritance** - Pass timeout to nested requests
11. **Advanced Features** - Keepalive, adaptive timeouts, etc.

## Implementation Plan

### Phase 1: Timeout Components (Critical)
1. Add connect_timeout field to HTTP client configuration
2. Add read_timeout field to HTTP client configuration
3. Add pool_timeout field to HTTP client configuration
4. Add total_timeout field to HTTP client configuration
5. Update HTTP client to use separate timeout components
6. Add tests for timeout components

### Phase 2: Streaming Timeouts (Critical)
7. Add idle_timeout configuration for streaming
8. Add total_stream_timeout configuration
9. Add chunk_timeout configuration
10. Implement idle timeout detection in streaming
11. Implement streaming keepalive (if needed)
12. Add tests for streaming timeouts

### Phase 3: Per-Deployment Timeout (Critical)
13. Add timeout field to deployment configuration
14. Parse timeout from model_list config
15. Apply deployment-specific timeouts to requests
16. Override global timeout with deployment timeout
17. Add tests for per-deployment timeout

### Phase 4: Provider-Specific Timeouts (High Priority)
18. Implement OpenAI-specific timeout logic
19. Implement Anthropic-specific timeout logic
20. Implement Bedrock-specific timeout logic
21. Add provider-specific timeout detection
22. Add tests for provider-specific timeouts

### Phase 5: Timeout Metrics (High Priority)
23. Track timeout rates per deployment
24. Track timeout types (connect, read, total)
25. Expose timeout metrics via Prometheus
26. Add timeout rate alerts
27. Add tests for timeout metrics

### Phase 6: Timeout Error Handling (High Priority)
28. Categorize timeout errors (connect, read, total)
29. Map timeout errors to specific exception types
30. Handle timeout errors differently based on type
31. Log timeout error details
32. Add tests for timeout error handling

### Phase 7: Dynamic Timeout Adjustment (Medium Priority)
33. Track provider response times
34. Implement adaptive timeout algorithm
35. Adjust timeout based on historical data
36. Support manual timeout overrides
37. Add tests for dynamic timeout adjustment

### Phase 8: Timeout-Based Routing (Medium Priority)
38. Track timeout rates per deployment
39. Factor timeout rates into routing decisions
40. Implement timeout-based health checks
41. Cooldown deployments with excessive timeouts
42. Add tests for timeout-based routing

### Phase 9: Timeout Configuration (Medium Priority)
43. Add timeout to general_settings in config
44. Add timeout to model_list entries in config
45. Add timeout to router_settings in config
46. Parse timeout from YAML config
47. Add tests for timeout configuration

### Phase 10: Timeout Inheritance (Low Priority)
48. Pass timeout to child requests
49. Calculate remaining timeout for child requests
50. Handle timeout propagation through middleware
51. Support timeout cancellation
52. Add tests for timeout inheritance

## Testing Plan
- Unit tests for each timeout component
- Integration tests with simulated timeouts
- Provider-specific timeout tests (OpenAI, Anthropic, Bedrock)
- Streaming timeout tests
- Per-deployment timeout tests
- Timeout metrics tests
- Timeout error handling tests
- Dynamic timeout adjustment tests
- Timeout-based routing tests
- Configuration parsing tests

## Comparison with Previous Audits
This gap is **less critical** than function/tool calling, vision/multimodal, and retry logic because:
1. Basic timeout handling already exists in Rust
2. Timeout is more about reliability than functionality
3. The current timeout works for basic cases
4. Advanced timeout features are optimizations, not requirements

However, it's still important because:
1. Streaming timeouts are essential for long-running requests
2. Per-deployment timeouts allow fine-grained control
3. Provider-specific timeouts improve reliability
4. Timeout metrics help identify performance issues

## Estimated Effort
- Phase 1 (Timeout Components): 1-2 days
- Phase 2 (Streaming Timeouts): 2-3 days
- Phase 3 (Per-Deployment Timeout): 1 day
- Phase 4 (Provider-Specific Timeouts): 2-3 days
- Phase 5 (Timeout Metrics): 1-2 days
- Phase 6 (Timeout Error Handling): 1-2 days
- Phase 7 (Dynamic Adjustment): 2-3 days
- Phase 8 (Timeout-Based Routing): 2-3 days
- Phase 9 (Timeout Configuration): 1 day
- Phase 10 (Timeout Inheritance): 1-2 days

**Total**: 14-22 days for complete timeout handling parity
