# Askii Coin API Reference

## Overview

This document provides a comprehensive reference for all Askii Coin-related functions, classes, and APIs. The Askii Coin system provides currency conversion, budget enforcement, and billing integration capabilities.

## Core Functions

### Currency Conversion Functions

#### `usd_to_askii_coins(usd_amount: float) -> float`

Converts USD amounts to Askii Coins using the configured exchange rate.

**Parameters:**
- `usd_amount` (float): Amount in USD to convert

**Returns:**
- `float`: Equivalent amount in Askii Coins

**Example:**
```python
from litellm.cost_calculator import usd_to_askii_coins

# Convert $1.50 to Askii Coins
askii_coins = usd_to_askii_coins(1.50)
print(f"$1.50 = {askii_coins:,.0f} Askii Coins")
# Output: $1.50 = 1,500,000 Askii Coins (with default rate)
```

**Error Handling:**
- Returns `0.0` for negative amounts
- Handles very large and very small amounts gracefully
- Uses configured exchange rate or falls back to default

---

#### `get_askii_coin_exchange_rate() -> float`

Retrieves the current USD to Askii Coin exchange rate from environment configuration.

**Parameters:** None

**Returns:**
- `float`: Current exchange rate (Askii Coins per USD)

**Example:**
```python
from litellm.cost_calculator import get_askii_coin_exchange_rate

rate = get_askii_coin_exchange_rate()
print(f"Current rate: 1 USD = {rate:,.0f} Askii Coins")
```

**Configuration:**
- Reads from `ASKII_COIN_EXCHANGE_RATE` environment variable
- Default: `1,000,000` (1 USD = 1M Askii Coins)
- Supports scientific notation (e.g., `1e6`)

---

#### `convert_budget_to_askii_coins(budget_usd: Optional[float]) -> Optional[float]`

Converts USD budget limits to Askii Coins for budget enforcement comparisons.

**Parameters:**
- `budget_usd` (Optional[float]): Budget limit in USD, or None if no budget

**Returns:**
- `Optional[float]`: Budget limit in Askii Coins, or None if no budget

**Example:**
```python
from litellm.cost_calculator import convert_budget_to_askii_coins

# Convert budget limits
budget_askii_coins = convert_budget_to_askii_coins(100.0)
print(f"$100 budget = {budget_askii_coins:,.0f} Askii Coins")

# Handle None budget (no limit)
no_budget = convert_budget_to_askii_coins(None)
print(f"No budget limit: {no_budget}")  # Output: None
```

**Special Cases:**
- `None` input returns `None` (no budget limit)
- `0.0` input returns `0.0` (zero budget)
- Invalid exchange rates fall back to default with error logging

---

### Cost Calculation Functions

#### `cost_per_token(model: str, prompt_tokens: int, completion_tokens: int, **kwargs) -> float`

Calculates the cost for LLM API calls in Askii Coins.

**Parameters:**
- `model` (str): Model name (e.g., "gpt-3.5-turbo")
- `prompt_tokens` (int): Number of prompt tokens
- `completion_tokens` (int): Number of completion tokens
- `**kwargs`: Additional parameters (custom pricing, etc.)

**Returns:**
- `float`: Cost in Askii Coins

**Example:**
```python
from litellm.cost_calculator import cost_per_token

# Calculate cost for a completion
cost = cost_per_token(
    model="gpt-3.5-turbo",
    prompt_tokens=100,
    completion_tokens=50
)

print(f"Cost: {cost:,.0f} Askii Coins")
```

**Note:** This function has been modified to return Askii Coin amounts instead of USD.

---

## Budget Enforcement Functions

### Authentication and Budget Checks

#### `_virtual_key_max_budget_check(valid_token: UserAPIKeyAuth, proxy_logging_obj: ProxyLogging)`

Checks if a virtual key has exceeded its maximum budget limit.

**Parameters:**
- `valid_token` (UserAPIKeyAuth): User API key authentication object
- `proxy_logging_obj` (ProxyLogging): Proxy logging instance

**Raises:**
- `litellm.BudgetExceededError`: If budget is exceeded

**Example:**
```python
from litellm.proxy.auth.auth_checks import _virtual_key_max_budget_check
from litellm.proxy._types import UserAPIKeyAuth

# This function is called internally during request processing
# Budget limits in USD are automatically converted to Askii Coins
```

---

#### `_team_max_budget_check(team_object: LiteLLM_TeamTable, valid_token: UserAPIKeyAuth, proxy_logging_obj: ProxyLogging)`

Checks if a team has exceeded its maximum budget limit.

**Parameters:**
- `team_object` (LiteLLM_TeamTable): Team object with budget information
- `valid_token` (UserAPIKeyAuth): User API key authentication object  
- `proxy_logging_obj` (ProxyLogging): Proxy logging instance

**Raises:**
- `litellm.BudgetExceededError`: If team budget is exceeded

---

### Router Budget Limiting

#### `RouterBudgetLimiting._filter_out_deployments_above_budget(...)`

Filters out deployments that have exceeded their budget limits.

**Parameters:**
- `healthy_deployments` (List[Dict]): List of available deployments
- `provider_configs` (Dict): Provider budget configurations
- `deployment_configs` (Dict): Deployment budget configurations
- `spend_map` (Dict): Current spend amounts in Askii Coins
- `request_tags` (List[str]): Request tags for tag-based budgets

**Returns:**
- `Tuple[List[Dict], str]`: Filtered deployments and debug information

**Example:**
```python
# This function is called internally by the router
# Budget limits in USD are automatically converted to Askii Coins for comparison
```

---

## Lago Integration Functions

### LagoLogger Class

#### `LagoLogger._send_usage_event(external_subscription_id: str, cost: float, call_id: str)`

Sends usage events to Lago billing system with Askii Coin amounts.

**Parameters:**
- `external_subscription_id` (str): Customer subscription ID
- `cost` (float): Cost in Askii Coins
- `call_id` (str): Unique call identifier

**Returns:**
- None (async function)

**Example:**
```python
from moneta.lago_logger import LagoLogger

lago_logger = LagoLogger()

await lago_logger._send_usage_event(
    external_subscription_id="customer-123",
    cost=1_500_000,  # 1.5M Askii Coins
    call_id="unique-call-id"
)
```

**Payload Format:**
```json
{
  "event": {
    "external_subscription_id": "customer-123",
    "code": "askii_coins",
    "timestamp": "2024-01-01T00:00:00Z",
    "transaction_id": "unique-transaction-id",
    "properties": {
      "askii_coins": 1500000,
      "call_id": "unique-call-id"
    }
  }
}
```

---

## Error Classes

### BudgetExceededError

Extended to include Askii Coin amounts in error messages.

**Attributes:**
- `current_cost` (float): Current spend in Askii Coins
- `max_budget` (float): Budget limit in Askii Coins
- `message` (str): Enhanced error message with both USD and Askii Coin amounts

**Example:**
```python
try:
    # Budget check that might fail
    pass
except litellm.BudgetExceededError as e:
    print(f"Budget exceeded!")
    print(f"Current cost: {e.current_cost:,.0f} Askii Coins")
    print(f"Budget limit: {e.max_budget:,.0f} Askii Coins")
    print(f"Message: {e.message}")
```

---

## Configuration Functions

### Environment Configuration

#### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ASKII_COIN_EXCHANGE_RATE` | string | "1000000" | USD to Askii Coin exchange rate |

**Example:**
```bash
# Set exchange rate
export ASKII_COIN_EXCHANGE_RATE=500000

# Use scientific notation
export ASKII_COIN_EXCHANGE_RATE=5e5
```

---

## Usage Patterns

### Basic Currency Conversion

```python
from litellm.cost_calculator import usd_to_askii_coins, get_askii_coin_exchange_rate

# Check current configuration
rate = get_askii_coin_exchange_rate()
print(f"Exchange rate: 1 USD = {rate:,.0f} Askii Coins")

# Convert amounts
amounts_usd = [0.001, 0.01, 0.1, 1.0, 10.0]
for usd in amounts_usd:
    askii_coins = usd_to_askii_coins(usd)
    print(f"${usd} = {askii_coins:,.0f} Askii Coins")
```

### Budget Enforcement Integration

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

def check_user_budget(user_spend_askii_coins: float, user_budget_usd: float) -> bool:
    """Check if user is within budget."""
    
    # Convert USD budget to Askii Coins
    budget_askii_coins = convert_budget_to_askii_coins(user_budget_usd)
    
    if budget_askii_coins is None:
        return True  # No budget limit
    
    return user_spend_askii_coins <= budget_askii_coins

# Example usage
user_spent = 750_000  # 750K Askii Coins
user_budget = 1.0     # $1 USD budget

within_budget = check_user_budget(user_spent, user_budget)
print(f"User within budget: {within_budget}")
```

### Cost Calculation Integration

```python
from litellm.cost_calculator import cost_per_token

def calculate_request_cost(model: str, prompt: str, completion: str) -> float:
    """Calculate cost for a request in Askii Coins."""
    
    # Estimate token counts (simplified)
    prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
    completion_tokens = len(completion.split()) * 1.3
    
    # Calculate cost in Askii Coins
    cost = cost_per_token(
        model=model,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens)
    )
    
    return cost

# Example usage
cost = calculate_request_cost(
    model="gpt-3.5-turbo",
    prompt="What is the capital of France?",
    completion="The capital of France is Paris."
)

print(f"Request cost: {cost:,.0f} Askii Coins")
```

---

## Error Handling Patterns

### Graceful Degradation

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate, usd_to_askii_coins

def safe_currency_conversion(usd_amount: float) -> float:
    """Safely convert USD to Askii Coins with error handling."""
    
    try:
        # Check if exchange rate is valid
        rate = get_askii_coin_exchange_rate()
        if rate <= 0:
            raise ValueError("Invalid exchange rate")
        
        # Perform conversion
        askii_coins = usd_to_askii_coins(usd_amount)
        return askii_coins
        
    except Exception as e:
        print(f"Currency conversion error: {e}")
        # Fall back to default rate
        return usd_amount * 1_000_000  # Default rate

# Example usage
askii_coins = safe_currency_conversion(1.50)
print(f"Converted amount: {askii_coins:,.0f} Askii Coins")
```

### Budget Validation

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

def validate_budget_configuration(budget_usd: float) -> bool:
    """Validate budget configuration."""
    
    try:
        # Test budget conversion
        budget_askii_coins = convert_budget_to_askii_coins(budget_usd)
        
        if budget_askii_coins is None:
            return False  # Conversion failed
        
        if budget_askii_coins <= 0:
            return False  # Invalid budget amount
        
        return True
        
    except Exception as e:
        print(f"Budget validation error: {e}")
        return False

# Example usage
is_valid = validate_budget_configuration(100.0)
print(f"Budget configuration valid: {is_valid}")
```
