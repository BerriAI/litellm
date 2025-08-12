# Askii Coin Virtual Currency System

## Overview

The Askii Coin virtual currency system is a comprehensive solution that transforms LiteLLM's cost calculation and billing from USD-based to a configurable virtual currency system. This system provides:

- **Configurable Exchange Rates**: Set custom USD to Askii Coin conversion rates via environment variables
- **Transparent Cost Calculation**: All costs are calculated and tracked in Askii Coins
- **Budget Enforcement**: Automatic conversion of USD budget limits to Askii Coins for comparison
- **Lago Integration**: Direct billing integration sending Askii Coin amounts instead of USD cents
- **Backward Compatibility**: No database schema changes required

## Key Features

### 🪙 Virtual Currency System
- Convert USD costs to Askii Coins using configurable exchange rates
- Default rate: 1 USD = 1,000,000 Askii Coins
- Support for scientific notation (e.g., `1e6`)
- Robust error handling with fallback to default rates

### 💰 Budget Enforcement
- Automatic conversion of USD budget limits to Askii Coins during comparison
- Support for all budget types:
  - User budgets (personal keys)
  - Team budgets
  - End user budgets
  - Global proxy budgets
  - Model-specific budgets
  - Provider budgets
  - Deployment budgets
  - Tag-based budgets

### 📊 Lago Billing Integration
- Direct integration with Lago billing system
- Sends Askii Coin amounts instead of USD cents
- Event code changed from `credits_in_cent` to `askii_coins`
- Maintains all existing error handling and retry logic

### 🔧 Configuration Management
- Environment variable-based configuration
- Runtime budget conversion (no database changes)
- Consistent exchange rate usage across all components

## Quick Start

### 1. Installation

The Askii Coin system is built into LiteLLM. No additional installation is required.

### 2. Basic Configuration

Set the exchange rate using an environment variable:

```bash
# Set custom exchange rate (1 USD = 500,000 Askii Coins)
export ASKII_COIN_EXCHANGE_RATE=500000

# Or use scientific notation
export ASKII_COIN_EXCHANGE_RATE=5e5
```

### 3. Verify Configuration

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate, usd_to_askii_coins

# Check current exchange rate
rate = get_askii_coin_exchange_rate()
print(f"Current exchange rate: 1 USD = {rate:,.0f} Askii Coins")

# Test conversion
askii_coins = usd_to_askii_coins(1.0)
print(f"1 USD = {askii_coins:,.0f} Askii Coins")
```

### 4. Budget Configuration

Budget limits remain in USD in your database/configuration, but are automatically converted to Askii Coins during enforcement:

```python
# Example: User with $10 USD budget limit
user_budget_usd = 10.0  # Stored in database as USD

# System automatically converts during budget checks
from litellm.cost_calculator import convert_budget_to_askii_coins
budget_askii_coins = convert_budget_to_askii_coins(user_budget_usd)
print(f"Budget: ${user_budget_usd} USD = {budget_askii_coins:,.0f} Askii Coins")
```

## System Architecture

### Phase 1: Currency Conversion Foundation
- **Core Functions**: `usd_to_askii_coins()`, `get_askii_coin_exchange_rate()`
- **Configuration**: Environment variable `ASKII_COIN_EXCHANGE_RATE`
- **Integration**: Modified `cost_per_token()` to return Askii Coin amounts

### Phase 2: Lago Integration
- **Event Format**: Changed from `credits_in_cent` to `askii_coins`
- **Payload**: Sends integer Askii Coin amounts directly
- **Logging**: Enhanced messages showing Askii Coin amounts

### Phase 3: Budget System Integration
- **Budget Conversion**: `convert_budget_to_askii_coins()` utility function
- **Enforcement Points**: All budget checks convert USD limits to Askii Coins
- **Error Messages**: Enhanced to show both USD and Askii Coin amounts

## Configuration Options

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ASKII_COIN_EXCHANGE_RATE` | USD to Askii Coin exchange rate | `1000000` | `500000` |

### Exchange Rate Examples

```bash
# Default rate (1 USD = 1M Askii Coins)
# No environment variable needed

# Custom rate (1 USD = 500K Askii Coins)
export ASKII_COIN_EXCHANGE_RATE=500000

# Scientific notation (1 USD = 2M Askii Coins)
export ASKII_COIN_EXCHANGE_RATE=2e6

# High precision rate
export ASKII_COIN_EXCHANGE_RATE=1234567
```

## Usage Examples

### Cost Calculation

```python
from litellm.cost_calculator import cost_per_token

# Calculate cost for a completion
cost_askii_coins = cost_per_token(
    model="gpt-3.5-turbo",
    prompt_tokens=100,
    completion_tokens=50
)

print(f"Cost: {cost_askii_coins:,.0f} Askii Coins")
```

### Budget Enforcement

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

# Convert budget for comparison
budget_usd = 50.0  # $50 USD budget limit
budget_askii_coins = convert_budget_to_askii_coins(budget_usd)

# Check if spend exceeds budget
current_spend_askii_coins = 45_000_000  # Current spend in Askii Coins

if current_spend_askii_coins > budget_askii_coins:
    print(f"Budget exceeded! Spend: {current_spend_askii_coins:,.0f}, Budget: {budget_askii_coins:,.0f}")
```

### Lago Integration

```python
from moneta.lago_logger import LagoLogger

# Lago logger automatically sends Askii Coin amounts
lago_logger = LagoLogger()

await lago_logger._send_usage_event(
    external_subscription_id="customer-123",
    cost=1_500_000,  # 1.5M Askii Coins
    call_id="unique-call-id"
)
```

## Migration Guide

### From USD-Only System

1. **No Database Changes Required**: Budget limits remain stored in USD
2. **Set Exchange Rate**: Configure `ASKII_COIN_EXCHANGE_RATE` environment variable
3. **Verify Conversion**: Test cost calculations return expected Askii Coin amounts
4. **Update Monitoring**: Adjust dashboards to display Askii Coin amounts
5. **Test Budget Enforcement**: Verify budget checks work with converted amounts

### Rollback Procedure

To rollback to USD-only system:

1. Remove `ASKII_COIN_EXCHANGE_RATE` environment variable
2. System will use default rate (1 USD = 1M Askii Coins)
3. Or set rate to `1` for direct USD amounts (not recommended)

## Error Handling

### Invalid Exchange Rates

The system gracefully handles invalid exchange rates:

```python
# These invalid rates fall back to default (1,000,000)
invalid_rates = ["invalid", "-1000000", "0", "", "inf", "nan"]
```

### Budget Conversion Errors

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

# Handles None budgets
result = convert_budget_to_askii_coins(None)  # Returns None

# Handles zero budgets
result = convert_budget_to_askii_coins(0.0)   # Returns 0.0

# Handles invalid exchange rates
# Falls back to default rate with error logging
```

## Testing

### Run Core Tests

```bash
cd litellm
python -m pytest tests/local_testing/test_askii_coins_core.py -v
```

### Run Lago Integration Tests

```bash
python -m pytest tests/local_testing/test_lago_askii_coins_integration.py -v --asyncio-mode=auto
```

### Run Comprehensive Tests

```bash
python -m pytest tests/local_testing/test_askii_coins_comprehensive.py -v --asyncio-mode=auto
```

## Monitoring and Debugging

### Enable Debug Logging

```python
import litellm
litellm.set_verbose = True
```

### Check Exchange Rate

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate
print(f"Current rate: {get_askii_coin_exchange_rate():,.0f}")
```

### Verify Cost Calculations

```python
from litellm.cost_calculator import usd_to_askii_coins

# Test conversion
usd_amount = 0.001
askii_coins = usd_to_askii_coins(usd_amount)
print(f"${usd_amount} = {askii_coins:,.0f} Askii Coins")
```

## Support and Documentation

- **Configuration Guide**: [CONFIGURATION.md](./CONFIGURATION.md)
- **API Reference**: [API_REFERENCE.md](./API_REFERENCE.md)
- **Integration Guide**: [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

## License

This system is part of LiteLLM and follows the same licensing terms.
