# Dynamic Rate Limiter v3 - Saturation-Aware Priority-Based Rate Limiting

## Overview

The v3 dynamic rate limiter implements saturation-aware rate limiting with priority-based allocation. It balances resource efficiency (allowing unused capacity to be borrowed) with fairness guarantees (enforcing priorities during high load).

**Key Behavior:**
- When system is under 80% capacity: Generous mode - allows priority borrowing
- When system is at/above 80% capacity: Strict mode - enforces normalized priority limits

## How It Works

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Request                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Check Model Saturation                                   │
│     - Query v3 limiter's Redis counters                      │
│     - Calculate: current_usage / capacity                    │
│     - Returns: 0.0 (empty) to 1.0+ (saturated)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                ┌────────┴────────┐
                │  Saturation?    │
                └────────┬────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
   < 80% (Generous)                >= 80% (Strict)
         │                               │
         ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  Generous Mode      │         │  Strict Mode        │
│                     │         │                     │
│  - Enforce model-   │         │  - Normalize        │
│    wide capacity    │         │    priority weights │
│  - No priority      │         │    (if over 1.0)    │
│    restrictions     │         │                     │
│  - Allows borrowing │         │  - Create priority- │
│                     │         │    specific         │
│  - First-come-      │         │    descriptors      │
│    first-served     │         │                     │
│    until capacity   │         │  - Enforce strict   │
│                     │         │    limits per       │
│                     │         │    priority         │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │                               ▼
           │                    ┌──────────────────────┐
           │                    │  Track model usage   │
           │                    │  for future          │
           │                    │  saturation checks   │
           │                    └──────────┬───────────┘
           │                               │
           └───────────────┬───────────────┘
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
     OVER_LIMIT                        OK
           │                               │
           ▼                               ▼
   Return 429 Error              Allow Request
```

## Configuration

### Priority Reservation

Set priority weights in your proxy configuration:

```python
litellm.priority_reservation = {
    "premium": 0.75,    # 75% of capacity
    "standard": 0.25    # 25% of capacity
}
```

### Priority Reservation Settings

Configure saturation-aware behavior:

```python
litellm.priority_reservation_settings = PriorityReservationSettings(
    default_priority=0.5,           # Default weight for users without explicit priority
    saturation_threshold=0.80,      # 80% - threshold for strict mode enforcement
    tracking_multiplier=10          # 10x - multiplier for non-blocking tracking in strict mode
)
```

**Settings:**
- `default_priority` (default: 0.5) - Priority weight for users without explicit priority metadata
- `saturation_threshold` (default: 0.80) - Saturation level (0.0-1.0) at which strict priority enforcement begins
- `tracking_multiplier` (default: 10) - Multiplier for model-wide tracking limits in strict mode

### User Priority Assignment

Set priority in user metadata:

```python
user_api_key_dict.metadata = {"priority": "premium"}
```

## Priority Weight Normalization

If priorities sum to > 1.0, they are automatically normalized:

```
Input:  {key_a: 0.60, key_b: 0.80} = 1.40 total
Output: {key_a: 0.43, key_b: 0.57} = 1.00 total
```

This ensures total allocation never exceeds model capacity.

## Implementation Details

### Saturation Detection

- Queries v3 limiter's Redis counters for model-wide usage
- Checks both RPM and TPM, returns higher saturation value
- Non-blocking reads (doesn't increment counters)

### Mode Selection

**Generous Mode (< 80% saturation):**
- Creates single model-wide descriptor
- Enforces total capacity only
- Allows any priority to use available capacity
- Prevents over-subscription via model-wide limit

**Strict Mode (>= 80% saturation):**
- Creates priority-specific descriptors with normalized weights
- Each priority gets its reserved allocation
- Tracks model-wide usage separately (non-blocking, 10x multiplier)
- Ensures fairness under load

Test scenarios covered:
1. No rate limiting when under capacity
2. Priority queue behavior during saturation
3. Spillover capacity for default keys
4. Over-allocated priorities with normalization
5. Default priority value handling


### Implementation

The limiter is a Rust CustomLoggerV2 callback in `litellm/rust/crates/callbacks-model-capacity`. `litellm/proxy/hooks/model_capacity_native.py` builds it and registers it under the callback name `dynamic_rate_limiter_v3`, in the same place in `litellm.callbacks` and with the same settings as before, so proxy configs do not change.

The Rust side owns the saturation check, priority normalization, class and team shares, admission and token settlement. It runs on the pre-call, post-call success, post-call failure, response headers and attempt ended hooks. The Python side (`CapacityDelegate`) only serves the host data the Rust side asks for: the priority settings, model group limits, the v3 limiter's counters, token estimates, the active team registry and Prometheus metrics.

Admission on one worker runs one request at a time while the counters live in memory, so concurrent requests on that worker see each other's holds. With Redis, the v3 limiter's atomic check and increment keeps admission consistent across workers.

`PROXY_DynamicRateLimitHandlerV3` in `litellm/proxy/hooks/dynamic_rate_limiter_v3.py` names the Rust-backed class (`NativeDynamicRateLimiter`), so `isinstance` checks against it keep working. Build an instance with `native_dynamic_rate_limiter(internal_usage_cache, llm_router)`, which returns the one registered instance.
