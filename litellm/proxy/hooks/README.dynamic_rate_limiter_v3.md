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


### `_PROXY_DynamicRateLimitHandlerV3`

Main handler class inheriting from `CustomLogger`.

**Key Methods:**
- `async_pre_call_hook()` - Main entry point, routes to generous/strict mode
- `_check_model_saturation()` - Queries Redis for current usage
- `_handle_generous_mode()` - Enforces model-wide capacity only
- `_handle_strict_mode()` - Enforces normalized priority limits
- `_normalize_priority_weights()` - Handles over-allocation
- `_create_priority_based_descriptors()` - Creates rate limit descriptors


