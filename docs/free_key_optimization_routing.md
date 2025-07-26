# Free Key Optimization Routing Strategy

The `free-key-optimization` routing strategy is designed for optimal usage of API keys with multi-window rate limiting. It provides intelligent deployment selection based on current usage while respecting all configured rate limits across minute, hour, and day time windows.

## Features

### Multi-Window Rate Limiting

- **Per-minute limits**: Traditional RPM/TPM limits for immediate responsiveness
- **Per-hour limits**: RPH/TPH limits for sustained usage patterns
- **Per-day limits**: RPD/TPD limits for quota management and cost control

### AND Logic Enforcement

All configured rate limits must be respected simultaneously. A deployment is only eligible if it's within ALL of its configured limits.

### Intelligent Selection

From eligible deployments, the strategy selects the one with the lowest estimated cost for the current request. It calculates:

- **Input cost**: `input_cost_per_token × actual_input_tokens`
- **Output cost**: `output_cost_per_token × estimated_output_tokens` (assumes 1:1 input:output ratio)
- **Total estimated cost**: `input_cost + output_cost`

**Cost Information Lookup Priority:**

1. **litellm_params Override** (Highest Priority)

   - `input_cost_per_token` and `output_cost_per_token` in `litellm_params`
   - Allows per-deployment cost customization

2. **model_info Configuration** (Second Priority)

   - `input_cost_per_token` and `output_cost_per_token` in `model_info`
   - Supports explicit `0.0` costs for free models

3. **Global Model Cost Map** (Third Priority)

   - Automatic lookup from LiteLLM's built-in model pricing database
   - Covers hundreds of models with up-to-date pricing

4. **Token Usage Fallback** (Lowest Priority)
   - Uses current token usage when no cost data is available

## Configuration

### Cost Configuration Examples

#### 1. Automatic Cost Lookup (Recommended)

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: sk-your-key
      # Cost automatically looked up from global model cost map
    rpm: 60
    tpm: 60000
```

#### 2. User Override in litellm_params (Highest Priority)

```yaml
model_list:
  - model_name: custom-model
    litellm_params:
      model: gpt-3.5-turbo
      api_key: sk-your-key
      # Override cost information (takes priority over everything else)
      input_cost_per_token: 0.0000015
      output_cost_per_token: 0.000002
    rpm: 60
    tpm: 60000
```

#### 3. Free Model in model_info (Second Priority)

```yaml
model_list:
  - model_name: free-gemini
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: your-key
    rpm: 10
    tpm: 250000
    # Cost information in model_info (explicit zero for free models)
    model_info:
      input_cost_per_token: 0.0 # Free model
      output_cost_per_token: 0.0 # Free model
```

### Basic Setup

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: sk-your-key
    # Traditional minute limits
    rpm: 60 # Requests per minute
    tpm: 60000 # Tokens per minute

    # New hour limits
    rph: 3000 # Requests per hour
    tph: 3000000 # Tokens per hour

    # New day limits
    rpd: 50000 # Requests per day
    tpd: 50000000 # Tokens per day

router_settings:
  routing_strategy: "free-key-optimization"
  routing_strategy_args:
    ttl: 60 # Minute window TTL (seconds)
    hour_ttl: 3600 # Hour window TTL (seconds)
    day_ttl: 86400 # Day window TTL (seconds)
```

### Rate Limit Fields

| Field | Description         | Time Window | Default       |
| ----- | ------------------- | ----------- | ------------- |
| `rpm` | Requests per minute | 1 minute    | ∞ (unlimited) |
| `tpm` | Tokens per minute   | 1 minute    | ∞ (unlimited) |
| `rph` | Requests per hour   | 1 hour      | ∞ (unlimited) |
| `tph` | Tokens per hour     | 1 hour      | ∞ (unlimited) |
| `rpd` | Requests per day    | 1 day       | ∞ (unlimited) |
| `tpd` | Tokens per day      | 1 day       | ∞ (unlimited) |

## Use Cases

### 1. Free API Key Management

Optimize usage of free-tier API keys with conservative limits:

```yaml
- model_name: gpt-3.5-turbo
  litellm_params:
    model: gpt-3.5-turbo
    api_key: sk-free-tier-key
  rpm: 3 # Conservative per-minute
  rph: 200 # Reasonable hourly limit
  rpd: 1000 # Daily quota protection
  tpm: 4000
  tph: 200000
  tpd: 1000000
```

### 2. Paid Tier Optimization

Higher limits for paid accounts with burst capability:

```yaml
- model_name: gpt-4
  litellm_params:
    model: gpt-4
    api_key: sk-paid-tier-key
  rpm: 500 # High responsiveness
  rph: 10000 # Sustained usage
  rpd: 100000 # Daily safety net
  tpm: 150000
  tph: 5000000
  tpd: 100000000
```

### 3. Multi-Provider Load Balancing

Different providers with varying rate limits:

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: sk-openai-key
    rpm: 60
    rph: 3000
    rpd: 50000

  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-35-turbo
      api_key: azure-key
    rpm: 240 # Azure has higher limits
    rph: 10000
    rpd: 100000
```

## How It Works

### 1. Deployment Selection Process

1. **Filter by Rate Limits**: Check all deployments against their configured limits
2. **Apply AND Logic**: Exclude deployments that exceed ANY limit
3. **Select Lowest Cost**: From eligible deployments, calculate estimated cost for the request and choose the lowest
4. **Cost Calculation**: `(input_tokens × input_cost_per_token) + (estimated_output_tokens × output_cost_per_token)`
5. **Fallback to Usage**: If no cost information is available, select based on lowest current token usage
6. **Handle No Availability**: Raise `RateLimitError` if no deployments are available

### 2. Cache Key Structure

The strategy uses Redis/cache keys with the following format:

```
{deployment_id}:{model_name}:{metric}:{window}:{timestamp}
```

Examples:

- `deployment-1:gpt-3.5-turbo:rpm:minute:14-30`
- `deployment-1:gpt-3.5-turbo:tpm:hour:2024-01-15-14`
- `deployment-1:gpt-3.5-turbo:rpm:day:2024-01-15`

### 3. TTL Management

Different time windows use appropriate TTL values:

- **Minute keys**: 60 seconds
- **Hour keys**: 3600 seconds (1 hour)
- **Day keys**: 86400 seconds (24 hours)

### 4. Request Flow

1. **Pre-call Check**: Increment request counters for selected deployment
2. **API Call**: Make the actual LLM API call
3. **Success Logging**: Increment token counters across all time windows
4. **Failure Handling**: Graceful degradation on cache failures

## Error Handling

### Rate Limit Exceeded

When all deployments exceed their limits:

```python
litellm.RateLimitError: No deployments available for selected model.
All deployments exceed rate limits.
```

### Cache Failures

The strategy gracefully handles Redis/cache failures:

- Continues operation without rate limiting
- Logs warnings for debugging
- Maintains service availability

## Troubleshooting

### Enable Detailed Logging

To debug cost selection and rate limiting issues, enable verbose logging:

```python
import litellm
litellm.set_verbose = True
```

### Understanding Log Output

The strategy provides comprehensive logging to help debug selection decisions:

#### Selection Summary

```
free_key_optimization: Selected deployment deployment-2 (model: gemini/gemini-2.5-flash) with cost_metric: 0.00000000 (source: cost_calculation)
```

#### Cost Calculation Details

```
free_key_optimization: Cost calculation details - input_cost_per_token: 0.00000000, output_cost_per_token: 0.00000000, input_tokens: 12, estimated_output_tokens: 12, estimated_total_cost: 0.00000000, cost_data_source: model_info
```

#### Current Usage Statistics

```
free_key_optimization: Current usage - TPM: 79, RPM: 1, TPH: 79, RPH: 1, TPD: 79, RPD: 1
```

#### All Deployment Comparison

```
free_key_optimization: All deployment costs:
  1. deployment-1 (openrouter/google/gemini-2.5-flash): cost_metric=0.00003360 (cost_calculation, cost_data_from=global_model_cost_map[openrouter/google/gemini-2.5-flash]), TPM=0, RPM=1
  2. deployment-2 (gemini/gemini-2.5-flash): cost_metric=0.00000000 (cost_calculation, cost_data_from=model_info), TPM=79, RPM=1
```

### Common Issues

#### Cost Data Not Found

If you see `cost_data_source: no_cost_data_found`, the strategy couldn't find cost information:

- Check if the model name exists in LiteLLM's global cost map
- Add explicit cost data in `litellm_params` or `model_info`
- Verify model name spelling

#### Unexpected Deployment Selection

If a more expensive deployment is selected:

- Check the logs to see cost calculations for all deployments
- Verify cost data sources (litellm_params > model_info > global_cost_map)
- Ensure free models have explicit `0.0` costs in `model_info`

#### Rate Limits Not Working

If rate limits seem ineffective:

- Verify cache/Redis connectivity
- Check TTL settings for cache keys
- Enable verbose logging to see rate limit checks

## Best Practices

1. **Use Automatic Cost Lookup**: Let LiteLLM handle cost data for supported models
2. **Explicit Zero Costs**: Use `model_info` with `0.0` costs for free models
3. **Start Conservative**: Begin with lower limits and increase based on monitoring
4. **Monitor Usage**: Use the detailed logging to understand usage patterns
5. **Balance Windows**: Set appropriate limits for each time window based on your use case
6. **Test Thoroughly**: Validate rate limiting behavior in staging environments
7. **Plan for Growth**: Design limits that accommodate expected traffic increases

## Backward Compatibility

The strategy is fully backward compatible:

- Existing `rpm`/`tpm` configurations work unchanged
- New fields (`rph`, `rpd`, `tph`, `tpd`) are optional
- Defaults to unlimited for unspecified time windows

## Performance Considerations

### Batch Operations

- Uses batch cache operations to minimize Redis calls
- Efficient key generation and lookup

### Memory Usage

- Minimal memory overhead
- Efficient cache key management
- Automatic cleanup via TTL
