# Askii Coin Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues with the Askii Coin virtual currency system. It includes troubleshooting steps, debugging techniques, frequently asked questions, and migration guidance.

## Common Issues

### 1. Exchange Rate Not Taking Effect

**Symptoms:**
- Currency conversion returns unexpected values
- System appears to use default rate despite setting environment variable

**Diagnosis:**
```python
import os
from litellm.cost_calculator import get_askii_coin_exchange_rate

# Check environment variable
env_rate = os.getenv('ASKII_COIN_EXCHANGE_RATE')
print(f"Environment variable: {env_rate}")

# Check parsed rate
parsed_rate = get_askii_coin_exchange_rate()
print(f"Parsed rate: {parsed_rate:,.0f}")
```

**Solutions:**
1. **Check variable name spelling**: Ensure `ASKII_COIN_EXCHANGE_RATE` is spelled correctly
2. **Restart application**: Environment variables may require application restart
3. **Check variable scope**: Ensure variable is set in the correct environment
4. **Verify permissions**: Check if application has permission to read environment variables

**Example Fix:**
```bash
# Correct spelling and restart
export ASKII_COIN_EXCHANGE_RATE=1000000
# Restart your LiteLLM application
```

### 2. Budget Enforcement Not Working

**Symptoms:**
- Users can exceed budget limits
- Budget checks always pass or always fail
- Incorrect budget calculations

**Diagnosis:**
```python
from litellm.cost_calculator import convert_budget_to_askii_coins

# Test budget conversion
budget_usd = 10.0
budget_askii_coins = convert_budget_to_askii_coins(budget_usd)

print(f"Budget USD: ${budget_usd}")
print(f"Budget Askii Coins: {budget_askii_coins:,.0f}")

# Check if conversion is working
if budget_askii_coins is None:
    print("ERROR: Budget conversion returned None")
elif budget_askii_coins <= 0:
    print("ERROR: Budget conversion returned invalid amount")
else:
    print("Budget conversion working correctly")
```

**Solutions:**
1. **Verify exchange rate**: Ensure exchange rate is set correctly
2. **Check budget values**: Ensure budget limits are stored in USD
3. **Validate spend amounts**: Ensure spend amounts are in Askii Coins
4. **Review error logs**: Check for conversion errors in logs

### 3. Lago Integration Issues

**Symptoms:**
- Lago events not being sent
- Incorrect amounts in Lago
- HTTP errors from Lago API

**Diagnosis:**
```python
import asyncio
from moneta.lago_logger import LagoLogger

async def test_lago_integration():
    """Test Lago integration."""
    
    lago_logger = LagoLogger()
    
    try:
        await lago_logger._send_usage_event(
            external_subscription_id="test-customer",
            cost=1_000_000,  # 1M Askii Coins
            call_id="test-call-id"
        )
        print("Lago integration working")
    except Exception as e:
        print(f"Lago integration error: {e}")

# Run test
asyncio.run(test_lago_integration())
```

**Solutions:**
1. **Check Lago configuration**: Verify API keys and endpoints
2. **Verify network connectivity**: Ensure application can reach Lago API
3. **Check payload format**: Ensure events use `askii_coins` format
4. **Review Lago logs**: Check Lago dashboard for error details

### 4. Incorrect Cost Calculations

**Symptoms:**
- Costs are too high or too low
- Inconsistent cost calculations
- Costs don't match expected values

**Diagnosis:**
```python
from litellm.cost_calculator import cost_per_token, get_askii_coin_exchange_rate

# Test cost calculation
rate = get_askii_coin_exchange_rate()
print(f"Exchange rate: {rate:,.0f}")

cost = cost_per_token(
    model="gpt-3.5-turbo",
    prompt_tokens=100,
    completion_tokens=50
)

print(f"Cost: {cost:,.0f} Askii Coins")

# Convert back to USD for verification
cost_usd = cost / rate
print(f"Cost in USD: ${cost_usd:.6f}")
```

**Solutions:**
1. **Verify model pricing**: Check if model pricing is up to date
2. **Validate token counts**: Ensure token counts are accurate
3. **Check exchange rate**: Verify exchange rate is reasonable
4. **Compare with USD calculations**: Cross-check with original USD calculations

## Debugging Techniques

### 1. Enable Debug Logging

```python
import litellm
import logging

# Enable LiteLLM verbose logging
litellm.set_verbose = True

# Enable Python logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('askii_coin')

# Add debug logging to your functions
def debug_currency_conversion(usd_amount: float):
    from litellm.cost_calculator import usd_to_askii_coins, get_askii_coin_exchange_rate
    
    rate = get_askii_coin_exchange_rate()
    askii_coins = usd_to_askii_coins(usd_amount)
    
    logger.debug(f"Currency conversion: ${usd_amount} * {rate} = {askii_coins}")
    
    return askii_coins
```

### 2. Configuration Validation Script

```python
import os
from litellm.cost_calculator import get_askii_coin_exchange_rate, usd_to_askii_coins, convert_budget_to_askii_coins

def validate_askii_coin_configuration():
    """Comprehensive configuration validation."""
    
    print("=== Askii Coin Configuration Validation ===")
    
    # 1. Check environment variable
    env_rate = os.getenv('ASKII_COIN_EXCHANGE_RATE')
    print(f"1. Environment variable: {env_rate}")
    
    # 2. Check parsed rate
    try:
        parsed_rate = get_askii_coin_exchange_rate()
        print(f"2. Parsed rate: {parsed_rate:,.0f}")
    except Exception as e:
        print(f"2. ERROR parsing rate: {e}")
        return False
    
    # 3. Test currency conversion
    try:
        test_usd = 1.0
        test_askii = usd_to_askii_coins(test_usd)
        print(f"3. Test conversion: ${test_usd} = {test_askii:,.0f} Askii Coins")
    except Exception as e:
        print(f"3. ERROR in conversion: {e}")
        return False
    
    # 4. Test budget conversion
    try:
        test_budget = 10.0
        budget_askii = convert_budget_to_askii_coins(test_budget)
        print(f"4. Budget conversion: ${test_budget} = {budget_askii:,.0f} Askii Coins")
    except Exception as e:
        print(f"4. ERROR in budget conversion: {e}")
        return False
    
    # 5. Test edge cases
    try:
        none_budget = convert_budget_to_askii_coins(None)
        zero_budget = convert_budget_to_askii_coins(0.0)
        print(f"5. Edge cases: None={none_budget}, Zero={zero_budget}")
    except Exception as e:
        print(f"5. ERROR in edge cases: {e}")
        return False
    
    print("✅ All validation checks passed!")
    return True

# Run validation
validate_askii_coin_configuration()
```

### 3. Budget Enforcement Testing

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

def test_budget_enforcement():
    """Test budget enforcement scenarios."""
    
    scenarios = [
        {"name": "Within budget", "budget_usd": 10.0, "spend_askii": 5_000_000, "should_pass": True},
        {"name": "At budget limit", "budget_usd": 10.0, "spend_askii": 10_000_000, "should_pass": True},
        {"name": "Over budget", "budget_usd": 10.0, "spend_askii": 15_000_000, "should_pass": False},
        {"name": "No budget limit", "budget_usd": None, "spend_askii": 100_000_000, "should_pass": True},
        {"name": "Zero budget", "budget_usd": 0.0, "spend_askii": 1, "should_pass": False},
    ]
    
    for scenario in scenarios:
        budget_askii = convert_budget_to_askii_coins(scenario["budget_usd"])
        
        if budget_askii is None:
            within_budget = True  # No limit
        else:
            within_budget = scenario["spend_askii"] <= budget_askii
        
        status = "✅ PASS" if within_budget == scenario["should_pass"] else "❌ FAIL"
        
        print(f"{status} {scenario['name']}: Budget=${scenario['budget_usd']}, "
              f"Spend={scenario['spend_askii']:,.0f}, Result={within_budget}")

test_budget_enforcement()
```

## Frequently Asked Questions

### Q1: Why are my costs much higher than before?

**A:** Costs are now displayed in Askii Coins instead of USD. With the default exchange rate (1 USD = 1,000,000 Askii Coins), a $0.001 USD cost becomes 1,000 Askii Coins. This is normal and expected.

**Solution:** To convert back to USD for comparison:
```python
from litellm.cost_calculator import get_askii_coin_exchange_rate

askii_coins = 1_500_000  # Your cost in Askii Coins
rate = get_askii_coin_exchange_rate()
usd_equivalent = askii_coins / rate
print(f"{askii_coins:,.0f} Askii Coins = ${usd_equivalent:.6f} USD")
```

### Q2: Can I change the exchange rate after deployment?

**A:** Yes, you can change the exchange rate by updating the `ASKII_COIN_EXCHANGE_RATE` environment variable. The new rate takes effect immediately for new calculations.

**Important:** Changing the exchange rate affects:
- New cost calculations
- Budget enforcement comparisons
- Lago billing amounts

Existing stored spend amounts remain in their original currency.

### Q3: Do I need to update my database schema?

**A:** No database schema changes are required. Budget limits remain stored in USD and are converted to Askii Coins during runtime comparisons.

### Q4: What happens if I set an invalid exchange rate?

**A:** The system gracefully falls back to the default rate (1,000,000) and logs an error. Invalid rates include:
- Non-numeric values
- Negative numbers
- Zero
- Infinity or NaN

### Q5: How do I migrate from USD-only to Askii Coins?

**A:** The migration is automatic:
1. Set the `ASKII_COIN_EXCHANGE_RATE` environment variable
2. Restart your application
3. New calculations will use Askii Coins
4. Existing budget limits continue to work (converted at runtime)

### Q6: Can I use different exchange rates for different customers?

**A:** Currently, the system uses a single global exchange rate. For customer-specific rates, you would need to implement custom logic in your application layer.

### Q7: How do I verify Lago is receiving correct amounts?

**A:** Check your Lago dashboard for events with:
- Event code: `askii_coins` (not `credits_in_cent`)
- Properties containing `askii_coins` field with integer values
- Amounts that match your expected Askii Coin calculations

## Migration Guide

### From USD-Only System

#### Pre-Migration Checklist

- [ ] Backup current configuration and data
- [ ] Test Askii Coin system in staging environment
- [ ] Calculate appropriate exchange rate for your use case
- [ ] Update monitoring and alerting systems
- [ ] Train team on new currency system

#### Migration Steps

1. **Set Exchange Rate**
   ```bash
   export ASKII_COIN_EXCHANGE_RATE=1000000
   ```

2. **Test Configuration**
   ```python
   # Run validation script
   validate_askii_coin_configuration()
   ```

3. **Update Monitoring**
   - Update dashboards to show Askii Coin amounts
   - Adjust alert thresholds for new currency scale
   - Add USD conversion for reporting

4. **Deploy and Monitor**
   - Deploy to production
   - Monitor cost calculations
   - Verify budget enforcement
   - Check Lago integration

#### Post-Migration Verification

```python
def verify_migration():
    """Verify migration was successful."""
    
    from litellm.cost_calculator import cost_per_token, get_askii_coin_exchange_rate
    
    # 1. Verify exchange rate
    rate = get_askii_coin_exchange_rate()
    print(f"Exchange rate: {rate:,.0f}")
    
    # 2. Test cost calculation
    cost = cost_per_token(model="gpt-3.5-turbo", prompt_tokens=100, completion_tokens=50)
    print(f"Sample cost: {cost:,.0f} Askii Coins")
    
    # 3. Verify cost is reasonable
    cost_usd = cost / rate
    if 0.0001 <= cost_usd <= 1.0:  # Reasonable range for this request
        print("✅ Cost calculation appears correct")
    else:
        print(f"⚠️  Cost may be incorrect: ${cost_usd:.6f} USD")

verify_migration()
```

### Rollback Procedure

If you need to rollback:

1. **Remove environment variable**
   ```bash
   unset ASKII_COIN_EXCHANGE_RATE
   ```

2. **Or set to minimal rate**
   ```bash
   export ASKII_COIN_EXCHANGE_RATE=1  # 1 USD = 1 Askii Coin
   ```

3. **Restart application**

4. **Verify rollback**
   ```python
   from litellm.cost_calculator import usd_to_askii_coins
   
   # Should return 1.0 with rate=1
   result = usd_to_askii_coins(1.0)
   print(f"Rollback test: {result}")
   ```

## Getting Help

### Log Collection

When reporting issues, include:

1. **Configuration details**
   ```python
   import os
   from litellm.cost_calculator import get_askii_coin_exchange_rate
   
   print(f"Environment: {os.getenv('ASKII_COIN_EXCHANGE_RATE')}")
   print(f"Parsed rate: {get_askii_coin_exchange_rate()}")
   ```

2. **Error messages and stack traces**

3. **Sample inputs and expected outputs**

4. **LiteLLM version and environment details**

### Debug Information Script

```python
def collect_debug_info():
    """Collect debug information for support."""
    
    import os
    import sys
    from litellm.cost_calculator import get_askii_coin_exchange_rate, usd_to_askii_coins
    
    info = {
        "python_version": sys.version,
        "environment_variable": os.getenv('ASKII_COIN_EXCHANGE_RATE'),
        "parsed_exchange_rate": get_askii_coin_exchange_rate(),
        "test_conversion": usd_to_askii_coins(1.0),
        "platform": sys.platform,
    }
    
    print("=== Debug Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    return info

collect_debug_info()
```

### Support Channels

- **Documentation**: Check [README.md](./README.md) and [API_REFERENCE.md](./API_REFERENCE.md)
- **Configuration**: Review [CONFIGURATION.md](./CONFIGURATION.md)
- **Integration**: Follow [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md)
- **Issues**: Report bugs with debug information and reproduction steps
