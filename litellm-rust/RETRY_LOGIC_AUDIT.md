# Retry Logic Audit

## Current Rust Implementation
The Rust gateway has **basic retry logic** in `auth/retry.rs`:
- Error categories: Network, Http, Application, Provider, Unknown
- Per-category retry strategies with configurable max_retries, base_delay, max_delay, jitter, backoff_multiplier
- Exponential backoff with jitter
- Error categorization from CoreError
- `retry_with_backoff` function for retrying operations

### Current Retry Strategies (Default)
- **Network errors**: 5 retries, 100ms base delay, 30s max delay, jitter enabled
- **HTTP errors (5xx, 429)**: 3 retries, 200ms base delay, 20s max delay, jitter enabled
- **Application errors (4xx)**: 1 retry, 500ms base delay, 5s max delay, no jitter
- **Provider errors**: 2 retries, 500ms base delay, 10s max delay, jitter enabled
- **Unknown errors**: 1 retry, 1000ms base delay, 5s max delay, no jitter

## Python Implementation

### RetryPolicy Type
Python has a comprehensive retry policy system with specific exception types:

```python
class RetryPolicy(BaseModel):
    BadRequestErrorRetries: int | None = None
    AuthenticationErrorRetries: int | None = None
    TimeoutErrorRetries: int | None = None
    RateLimitErrorRetries: int | None = None
    ContentPolicyViolationErrorRetries: int | None = None
    InternalServerErrorRetries: int | None = None
```

### Router-Level Retry Configuration
Python router has multiple retry-related configurations:
- `num_retries`: Global retry count for all errors
- `retry_after`: Delay between retries
- `allowed_fails`: Number of failures before marking deployment as unhealthy
- `cooldown_time`: Time to wait before retrying failed deployment
- `retry_policy`: Per-exception-type retry counts
- `model_group_retry_policy`: Per-model-group retry policies

### Exception Mapping
Python maps provider errors to specific exception types:
- **BadRequestError** (400): Invalid request, malformed JSON, missing parameters
- **AuthenticationError** (401): Invalid API key, expired credentials
- **RateLimitError** (429): Rate limit exceeded, quota exceeded
- **Timeout** (408/504): Request timeout, connection timeout
- **ContentPolicyViolationError** (400 with specific codes): Content filtered by provider
- **InternalServerError** (500/502/503): Provider server errors

### Retry Logic Flow
1. Check if error matches retry policy exception types
2. Get retry count from retry policy (or use default num_retries)
3. Calculate delay using retry_after or exponential backoff
4. Retry the request up to the configured number of times
5. If all retries fail, trigger fallback logic
6. Update allowed_fails counter for the deployment
7. If allowed_fails exceeded, mark deployment as unhealthy and cooldown

### Provider-Specific Retry Handling
Python has provider-specific retry logic:
- **OpenAI**: Respects Retry-After header, handles rate limits
- **Anthropic**: Handles overloaded errors, respects rate limits
- **Bedrock**: Handles throttling errors, respects retry-after
- **Azure**: Handles specific Azure error codes
- **Vertex AI**: Handles quota errors, respects retry-after

### Advanced Retry Features
- **Retry-After header parsing**: Extracts delay from provider responses
- **Exponential backoff with jitter**: Prevents thundering herd
- **Circuit breaker integration**: Stops retrying when circuit is open
- **Fallback routing**: Routes to healthy deployments after retries exhausted
- **Cooldown mechanism**: Temporarily removes unhealthy deployments from rotation
- **Allowed fails tracking**: Tracks failures per deployment
- **Model group retry policy**: Different retry policies for different model groups

## Major Gaps in Rust Implementation

### 1. Missing Exception-Specific Retry Policies
**Critical Gap**: No per-exception-type retry configuration
- Need BadRequestErrorRetries
- Need AuthenticationErrorRetries
- Need TimeoutErrorRetries
- Need RateLimitErrorRetries
- Need ContentPolicyViolationErrorRetries
- Need InternalServerErrorRetries

### 2. Missing Retry-After Header Parsing
**High Priority**: No Retry-After header support
- Need to parse Retry-After header from provider responses
- Need to use Retry-After value as delay between retries
- Need to handle both delta-seconds and HTTP-date formats
- Need to respect Retry-After for rate limit errors

### 3. Missing Provider-Specific Retry Logic
**High Priority**: No provider-specific retry handling
- Need OpenAI-specific retry logic (rate limits, overloaded errors)
- Need Anthropic-specific retry logic (overloaded errors)
- Need Bedrock-specific retry logic (throttling errors)
- Need Azure-specific retry logic (Azure error codes)
- Need Vertex AI-specific retry logic (quota errors)

### 4. Missing Model Group Retry Policies
**Medium Priority**: No per-model-group retry configuration
- Need to support different retry policies for different model groups
- Need to allow configuration via config file
- Need to allow runtime updates to retry policies

### 5. Missing Allowed Fails Tracking
**Medium Priority**: No deployment health tracking
- Need to track failures per deployment
- Need to mark deployment as unhealthy after allowed_fails exceeded
- Need cooldown mechanism for unhealthy deployments
- Need to reintroduce deployments after cooldown period

### 6. Missing Fallback Integration
**Medium Priority**: Retry logic not integrated with fallback routing
- Need to trigger fallback after retries exhausted
- Need to update fallback routing based on retry failures
- Need to coordinate retry logic with circuit breaker

### 7. Missing Content Policy Violation Handling
**Low Priority**: No special handling for content policy violations
- Need to detect content policy violation errors
- Need to configure retry behavior for content policy violations
- Need to log content policy violations separately

### 8. Missing Configuration Support
**Low Priority**: No retry configuration in config file
- Need to support retry_policy in YAML config
- Need to support model_group_retry_policy in YAML config
- Need to support num_retries, retry_after, allowed_fails in config

### 9. Missing Runtime Configuration Updates
**Low Priority**: No runtime retry policy updates
- Need to support updating retry policies at runtime
- Need to support updating retry policies via API
- Need to persist retry policy changes

### 10. Missing Retry Metrics
**Low Priority**: No retry-specific metrics
- Need to track retry attempts per deployment
- Need to track retry success/failure rates
- Need to expose retry metrics via Prometheus

## Priority Gaps for Major Providers (OpenAI, Anthropic, Bedrock)

### Critical (Must Have)
1. **Exception-Specific Retry Policies** - Configure retries per exception type
2. **Retry-After Header Parsing** - Respect provider-specified delays
3. **Provider-Specific Retry Logic** - Handle provider-specific error patterns

### High Priority
4. **Model Group Retry Policies** - Different retry policies for different models
5. **Allowed Fails Tracking** - Track deployment health
6. **Fallback Integration** - Coordinate retries with fallback routing

### Medium Priority
7. **Content Policy Violation Handling** - Special handling for content filters
8. **Configuration Support** - Retry config in YAML files

### Low Priority
9. **Runtime Configuration Updates** - Update retry policies at runtime
10. **Retry Metrics** - Track and expose retry metrics

## Implementation Plan

### Phase 1: Exception-Specific Retry Policies (Critical)
1. Define RetryPolicy struct with per-exception-type retry counts
2. Add retry_policy field to RetryConfig
3. Implement exception type detection (map CoreError to exception types)
4. Update retry_with_backoff to use exception-specific retry counts
5. Add tests for exception-specific retry policies

### Phase 2: Retry-After Header Parsing (Critical)
6. Parse Retry-After header from HTTP responses
7. Handle both delta-seconds and HTTP-date formats
8. Use Retry-After value as delay between retries
9. Add tests for Retry-After parsing

### Phase 3: Provider-Specific Retry Logic (Critical)
10. Implement OpenAI-specific retry logic (rate limits, overloaded errors)
11. Implement Anthropic-specific retry logic (overloaded errors)
12. Implement Bedrock-specific retry logic (throttling errors)
13. Add provider-specific error detection
14. Add tests for provider-specific retry logic

### Phase 4: Model Group Retry Policies (High Priority)
15. Add model_group_retry_policy field to config
16. Implement model group lookup for retry policy
17. Support runtime updates to model group retry policies
18. Add tests for model group retry policies

### Phase 5: Allowed Fails Tracking (High Priority)
19. Add allowed_fails counter per deployment
20. Implement deployment health tracking
21. Add cooldown mechanism for unhealthy deployments
22. Implement deployment reintroduction after cooldown
23. Add tests for allowed fails tracking

### Phase 6: Fallback Integration (High Priority)
24. Integrate retry logic with fallback routing
25. Trigger fallback after retries exhausted
26. Update fallback routing based on retry failures
27. Coordinate retry logic with circuit breaker
28. Add tests for fallback integration

### Phase 7: Content Policy Violation Handling (Medium Priority)
29. Detect content policy violation errors
30. Configure retry behavior for content policy violations
31. Log content policy violations separately
32. Add tests for content policy violation handling

### Phase 8: Configuration Support (Medium Priority)
33. Add retry_policy to YAML config schema
34. Add model_group_retry_policy to YAML config schema
35. Add num_retries, retry_after, allowed_fails to config
36. Parse retry config from YAML
37. Add tests for retry configuration

### Phase 9: Runtime Configuration Updates (Low Priority)
38. Support updating retry policies at runtime
39. Add API endpoints for retry policy updates
40. Persist retry policy changes
41. Add tests for runtime configuration updates

### Phase 10: Retry Metrics (Low Priority)
42. Track retry attempts per deployment
43. Track retry success/failure rates
44. Expose retry metrics via Prometheus
45. Add tests for retry metrics

## Testing Plan
- Unit tests for each retry policy type
- Integration tests with simulated provider errors
- Provider-specific retry tests (OpenAI, Anthropic, Bedrock)
- Retry-After header parsing tests
- Allowed fails tracking tests
- Fallback integration tests
- Configuration parsing tests
- Runtime configuration update tests
- Retry metrics tests

## Comparison with Previous Audits
This gap is **less critical** than function/tool calling and vision/multimodal but **more critical** than streaming edge cases because:
1. Retry logic is essential for production reliability
2. Without proper retry logic, transient errors cause request failures
3. Provider-specific retry logic is needed for optimal error handling
4. Python has comprehensive retry logic that we need to match
5. Retry logic integrates with fallback routing and circuit breakers

However, it's less critical than tool calling and vision because:
1. Basic retry logic already exists in Rust
2. Retry logic is more about reliability than functionality
3. The current retry logic works for basic cases
4. Provider-specific retry logic is an optimization, not a requirement

## Estimated Effort
- Phase 1 (Exception-Specific Policies): 1-2 days
- Phase 2 (Retry-After Parsing): 1 day
- Phase 3 (Provider-Specific Logic): 2-3 days
- Phase 4 (Model Group Policies): 1-2 days
- Phase 5 (Allowed Fails Tracking): 2-3 days
- Phase 6 (Fallback Integration): 2-3 days
- Phase 7 (Content Policy Handling): 1 day
- Phase 8 (Configuration Support): 1-2 days
- Phase 9 (Runtime Updates): 1-2 days
- Phase 10 (Retry Metrics): 1 day

**Total**: 13-20 days for complete retry logic parity
