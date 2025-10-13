# Dynamic Rate Limiter v3 - Adaptive Saturation-Aware Priority-Based Rate Limiting

## Overview

The v3 dynamic rate limiter implements adaptive saturation-aware rate limiting with priority-based allocation. It **automatically detects** which mode to use based on your model configuration:

### Two Modes (Auto-Detected):

**MODE 1: ABSOLUTE LIMITS** (when model has `rpm`/`tpm` configured)
- Enforces actual TPM/RPM capacity limits
- Saturation calculated from usage (e.g., 800/1000 RPM = 80%)
- Use case: Public APIs with known limits (OpenAI, Anthropic, Azure)
- Behavior:
  - Under 80% capacity: Generous mode - allows priority borrowing
  - At/above 80% capacity: Strict mode - enforces normalized priority limits

**MODE 2: PERCENTAGE SPLITTING** (when model has NO `rpm`/`tpm` configured)
- Enforces traffic percentage splits based on priorities
- Saturation calculated from error counts (e.g., 5 x 429 errors = 100%)
- Use case: Self-hosted models, Vertex AI dynamic quotas, unknown limits
- Behavior:
  - Before error threshold: All traffic allowed
  - After error threshold: Enforces percentage allocation (e.g., prod=90%, dev=10%)

**No user configuration needed** - the limiter automatically chooses the right mode!

## How It Works

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Request                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  1. AUTOMATIC MODE DETECTION                                 │
│     - Check if model has rpm/tpm configured                  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
   HAS rpm/tpm                    NO rpm/tpm
         │                               │
         ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  MODE: ABSOLUTE     │         │  MODE: PERCENTAGE   │
│                     │         │                     │
│  Check saturation   │         │  Check error        │
│  from usage:        │         │  saturation:        │
│  current/max        │         │  error_count/       │
│                     │         │  threshold          │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           ▼                               ▼
    ┌────────────┐                  ┌────────────┐
    │Saturation? │                  │Saturated?  │
    └──────┬─────┘                  └──────┬─────┘
           │                               │
  ┌────────┴────────┐              ┌───────┴────────┐
  │                 │              │                │
  ▼                 ▼              ▼                ▼
< 80%          >= 80%          < 100%          >= 100%
  │                 │              │                │
  ▼                 ▼              ▼                ▼
Generous         Strict        Allow All      Percentage
Mode             Mode          Traffic        Enforcement
  │                 │              │                │
  │                 │              │                │
  └────────┬────────┘              └────────┬───────┘
           │                                │
           ▼                                ▼
    ┌──────────────┐              ┌──────────────┐
    │  Enforce     │              │  Track & split│
    │  rpm/tpm     │              │  by % share   │
    │  limits      │              │               │
    └──────┬───────┘              └──────┬────────┘
           │                             │
           └──────────────┬──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  v3 Limiter  │
                   │  Check       │
                   └──────┬───────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    OVER_LIMIT                         OK
          │                               │
          ▼                               ▼
  Return 429 Error               Allow Request
```

## Configuration

### Basic Setup

```yaml
litellm_settings:
  callbacks: ["dynamic_rate_limiter_v3"]
  
  # Priority weights (same for both modes)
  priority_reservation:
    "realtime": 0.9    # Realtime workloads get 90%
    "batch": 0.1       # Batch workloads get 10%
  
  priority_reservation_settings:
    default_priority: 0.5           # Default weight for keys without explicit priority
    saturation_threshold: 0.80      # Threshold for absolute mode (80%)
    
    # Error-based saturation for percentage mode
    saturation_policy:
      RateLimitErrorSaturationThreshold: 3  # Trigger after 3 x 429 errors
```

### Example 1: Absolute Mode (Model with rpm/tpm)

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY
      rpm: 100    # ← Triggers ABSOLUTE mode
      tpm: 100000
```

**Behavior:**
- Saturation calculated from usage: `current_requests / 100`
- Under 80%: All priorities can use up to 100 RPM
- At/above 80%: Realtime gets 90 RPM, Batch gets 10 RPM

### Example 2: Percentage Mode (Model without rpm/tpm)

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: openai/my-model
      api_base: http://localhost:8000/v1
      # No rpm/tpm ← Triggers PERCENTAGE mode
```

**Behavior:**
- Saturation calculated from errors: `error_count / 3`
- Before 3 errors: All traffic allowed (no limiting)
- After 3 errors: Realtime gets 90% of traffic, Batch gets 10%

### Priority Reservation Settings (Detailed)

```python
litellm.priority_reservation_settings = PriorityReservationSettings(
    default_priority=0.5,           # Default weight for users without explicit priority
    saturation_threshold=0.80,      # ABSOLUTE mode: 80% threshold for strict enforcement
    saturation_policy=SaturationPolicy(
        RateLimitErrorSaturationThreshold=3,      # PERCENTAGE mode: trigger after N 429s
        TimeoutErrorSaturationThreshold=10,       # Optional: trigger after N timeouts
        InternalServerErrorSaturationThreshold=8, # Optional: trigger after N 500s
    )
)
```

**Settings:**
- `default_priority` (default: 0.5) - Priority weight for users without explicit priority metadata
- `saturation_threshold` (default: 0.80) - [Absolute mode] Saturation level (0.0-1.0) for strict enforcement
- `saturation_policy` - [Percentage mode] Error thresholds that trigger saturation

### User Priority Assignment

Set priority in user metadata (same for both modes):

```python
user_api_key_dict.metadata = {"priority": "realtime"}
```

## Priority Weight Normalization

If priorities sum to > 1.0, they are automatically normalized:

```
Input:  {key_a: 0.60, key_b: 0.80} = 1.40 total
Output: {key_a: 0.43, key_b: 0.57} = 1.00 total
```

This ensures total allocation never exceeds model capacity.

## Implementation Details

### Automatic Mode Detection

```python
def _has_explicit_limits(self, model_group_info: ModelGroupInfo) -> bool:
    return (
        (model_group_info.rpm is not None and model_group_info.rpm > 0)
        or (model_group_info.tpm is not None and model_group_info.tpm > 0)
    )
```

If model has `rpm` or `tpm`: Use **ABSOLUTE** mode  
If model has neither: Use **PERCENTAGE** mode

### Saturation Detection

**Absolute Mode:**
- Queries v3 limiter's Redis counters for model-wide usage
- Checks both RPM and TPM, returns higher saturation value
- Calculation: `max(current_rpm/max_rpm, current_tpm/max_tpm)`
- Non-blocking reads (doesn't increment counters)

**Percentage Mode:**
- Queries router's failure tracking (60s rolling window)
- Sums failures across all deployments in model group
- Calculation: `min(1.0, error_count / threshold)`
- Returns 1.0 (100%) when threshold reached

### Enforcement Logic

**ABSOLUTE MODE:**

*Generous Mode (< 80% saturation):*
- Creates single model-wide descriptor
- Enforces total capacity only
- Allows any priority to use available capacity
- Prevents over-subscription via model-wide limit

*Strict Mode (>= 80% saturation):*
- Creates priority-specific descriptors with normalized weights
- Each priority gets its reserved allocation
- Tracks model-wide usage separately (non-blocking, 10x multiplier)
- Ensures fairness under load

**PERCENTAGE MODE:**

*Before Saturation (< 100%):*
- Tracks aggregate + per-priority traffic
- No enforcement - all traffic allowed
- Counters updated for future percentage calculations

*After Saturation (>= 100%):*
- Calculates: `priority_share = priority_count / aggregate_count`
- Enforces: `priority_share <= priority_weight`
- Allows small buffer (1%) for edge cases
- Self-adjusting based on actual traffic volume

### Test Scenarios Covered

**Absolute Mode:**
1. No rate limiting when under capacity
2. Priority queue behavior during saturation
3. Spillover capacity for default keys
4. Over-allocated priorities with normalization
5. Default priority value handling

**Percentage Mode:**
1. No rate limiting before error threshold
2. Percentage enforcement after saturation
3. Traffic split accuracy (90/10 split validated)
4. Concurrent request handling

### `_PROXY_DynamicRateLimitHandlerV3`

Main handler class inheriting from `CustomLogger`.

**Key Methods:**
- `async_pre_call_hook()` - Main entry point, detects mode and routes
- `_has_explicit_limits()` - Automatic mode detection
- `_check_model_saturation()` - [Absolute] Queries Redis for current usage
- `_check_error_saturation()` - [Percentage] Queries router failure tracking
- `_check_rate_limits()` - Mode-aware rate limit checking
- `_check_percentage_rate_limits()` - [Percentage] Traffic split enforcement
- `_normalize_priority_weights()` - Handles over-allocation
- `_create_priority_based_descriptors()` - [Absolute] Creates absolute limit descriptors
- `_create_priority_traffic_descriptors()` - [Percentage] Creates tracking descriptors
- `_create_aggregate_traffic_descriptor()` - [Percentage] Creates aggregate counter


