# Askii Coin Configuration Guide

## Overview

This guide provides detailed configuration instructions for the Askii Coin virtual currency system. The system is designed to be highly configurable while maintaining backward compatibility and robust error handling.

## Environment Variables

### ASKII_COIN_EXCHANGE_RATE

The primary configuration for the Askii Coin system is the exchange rate between USD and Askii Coins.

**Variable**: `ASKII_COIN_EXCHANGE_RATE`  
**Type**: String (parsed as float)  
**Default**: `1000000` (1 USD = 1,000,000 Askii Coins)  
**Required**: No  

#### Valid Formats

```bash
# Standard integer notation
export ASKII_COIN_EXCHANGE_RATE=1000000

# Scientific notation
export ASKII_COIN_EXCHANGE_RATE=1e6

# Decimal notation
export ASKII_COIN_EXCHANGE_RATE=1000000.0

# Custom rates
export ASKII_COIN_EXCHANGE_RATE=500000    # 1 USD = 500K Askii Coins
export ASKII_COIN_EXCHANGE_RATE=2000000   # 1 USD = 2M Askii Coins
```

#### Invalid Values and Fallback Behavior

The system gracefully handles invalid exchange rates by falling back to the default rate:

```bash
# These values will fall back to default (1,000,000)
export ASKII_COIN_EXCHANGE_RATE=invalid
export ASKII_COIN_EXCHANGE_RATE=-1000000  # Negative values
export ASKII_COIN_EXCHANGE_RATE=0         # Zero values
export ASKII_COIN_EXCHANGE_RATE=""        # Empty string
export ASKII_COIN_EXCHANGE_RATE=inf       # Infinity
export ASKII_COIN_EXCHANGE_RATE=nan       # Not a number
```

## Configuration Examples

### Development Environment

For development and testing, you might want a simple 1:1 ratio:

```bash
# 1 USD = 1 Askii Coin (for easy mental math)
export ASKII_COIN_EXCHANGE_RATE=1

# Or a simple 1000:1 ratio
export ASKII_COIN_EXCHANGE_RATE=1000
```

### Production Environment

For production, use the default or a custom rate that fits your business model:

```bash
# Default production rate (recommended)
export ASKII_COIN_EXCHANGE_RATE=1000000

# Custom production rate
export ASKII_COIN_EXCHANGE_RATE=500000
```

### High-Volume Environment

For high-volume environments where you need fine-grained control:

```bash
# Higher precision rate
export ASKII_COIN_EXCHANGE_RATE=10000000  # 1 USD = 10M Askii Coins
```

## Configuration Validation

### Checking Current Configuration

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate

# Get current exchange rate
rate = get_askii_coin_exchange_rate()
print(f"Current exchange rate: 1 USD = {rate:,.0f} Askii Coins")

# Check if using default rate
if rate == 1_000_000.0:
    print("Using default exchange rate")
else:
    print("Using custom exchange rate")
```

### Testing Configuration

```python
from litellm.cost_calculator import usd_to_askii_coins, convert_budget_to_askii_coins

# Test currency conversion
test_amounts = [0.001, 0.01, 0.1, 1.0, 10.0]

for usd in test_amounts:
    askii_coins = usd_to_askii_coins(usd)
    print(f"${usd} USD = {askii_coins:,.0f} Askii Coins")

# Test budget conversion
budget_usd = 100.0
budget_askii_coins = convert_budget_to_askii_coins(budget_usd)
print(f"Budget: ${budget_usd} USD = {budget_askii_coins:,.0f} Askii Coins")
```

## Docker Configuration

### Docker Compose

```yaml
version: '3.8'
services:
  litellm:
    image: litellm/litellm:latest
    environment:
      - ASKII_COIN_EXCHANGE_RATE=1000000
    ports:
      - "4000:4000"
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm
spec:
  template:
    spec:
      containers:
      - name: litellm
        image: litellm/litellm:latest
        env:
        - name: ASKII_COIN_EXCHANGE_RATE
          value: "1000000"
```

### Docker Run

```bash
docker run -e ASKII_COIN_EXCHANGE_RATE=1000000 litellm/litellm:latest
```

## Configuration Management

### Environment-Specific Configuration

#### Development (.env.dev)
```bash
ASKII_COIN_EXCHANGE_RATE=1000
```

#### Staging (.env.staging)
```bash
ASKII_COIN_EXCHANGE_RATE=100000
```

#### Production (.env.prod)
```bash
ASKII_COIN_EXCHANGE_RATE=1000000
```

### Configuration Loading

```python
import os
from dotenv import load_dotenv

# Load environment-specific configuration
env = os.getenv('ENVIRONMENT', 'development')
load_dotenv(f'.env.{env}')

# Verify configuration
from litellm.cost_calculator import get_askii_coin_exchange_rate
print(f"Loaded exchange rate: {get_askii_coin_exchange_rate():,.0f}")
```

## Advanced Configuration

### Dynamic Configuration Updates

The exchange rate is read from the environment variable each time it's needed, allowing for dynamic updates:

```python
import os
from litellm.cost_calculator import get_askii_coin_exchange_rate, usd_to_askii_coins

# Initial rate
print(f"Initial rate: {get_askii_coin_exchange_rate():,.0f}")

# Update environment variable
os.environ['ASKII_COIN_EXCHANGE_RATE'] = '2000000'

# New rate takes effect immediately
print(f"Updated rate: {get_askii_coin_exchange_rate():,.0f}")
```

### Configuration Monitoring

```python
import os
import time
from litellm.cost_calculator import get_askii_coin_exchange_rate

def monitor_exchange_rate():
    """Monitor exchange rate changes."""
    last_rate = None
    
    while True:
        current_rate = get_askii_coin_exchange_rate()
        
        if last_rate != current_rate:
            print(f"Exchange rate changed: {current_rate:,.0f}")
            last_rate = current_rate
        
        time.sleep(60)  # Check every minute

# Run monitoring
monitor_exchange_rate()
```

## Configuration Best Practices

### 1. Use Consistent Rates Across Environments

Maintain consistency between development, staging, and production environments to avoid confusion:

```bash
# All environments use the same rate
export ASKII_COIN_EXCHANGE_RATE=1000000
```

### 2. Document Your Exchange Rate Choice

Document why you chose a specific exchange rate:

```bash
# 1 USD = 1M Askii Coins
# Chosen for:
# - Easy mental math (1 cent = 10K Askii Coins)
# - Sufficient precision for micro-transactions
# - Compatibility with existing billing systems
export ASKII_COIN_EXCHANGE_RATE=1000000
```

### 3. Test Configuration Changes

Always test configuration changes in a non-production environment:

```python
# Test script for configuration validation
from litellm.cost_calculator import usd_to_askii_coins, convert_budget_to_askii_coins

def test_configuration():
    """Test current Askii Coin configuration."""
    
    # Test basic conversion
    assert usd_to_askii_coins(1.0) > 0, "Basic conversion failed"
    
    # Test budget conversion
    assert convert_budget_to_askii_coins(10.0) > 0, "Budget conversion failed"
    
    # Test edge cases
    assert convert_budget_to_askii_coins(None) is None, "None budget handling failed"
    assert convert_budget_to_askii_coins(0.0) == 0.0, "Zero budget handling failed"
    
    print("Configuration test passed!")

test_configuration()
```

### 4. Monitor Exchange Rate Usage

Monitor how the exchange rate affects your costs and budgets:

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate, usd_to_askii_coins

def analyze_exchange_rate_impact():
    """Analyze the impact of current exchange rate."""
    
    rate = get_askii_coin_exchange_rate()
    
    # Common cost scenarios
    scenarios = [
        ("Small API call", 0.001),
        ("Medium API call", 0.01),
        ("Large API call", 0.1),
        ("Daily budget", 10.0),
        ("Monthly budget", 100.0)
    ]
    
    print(f"Exchange Rate Analysis (1 USD = {rate:,.0f} Askii Coins)")
    print("-" * 60)
    
    for scenario, usd_amount in scenarios:
        askii_coins = usd_to_askii_coins(usd_amount)
        print(f"{scenario:15}: ${usd_amount:6.3f} = {askii_coins:10,.0f} Askii Coins")

analyze_exchange_rate_impact()
```

## Troubleshooting Configuration

### Common Issues

1. **Exchange rate not taking effect**
   - Verify environment variable is set correctly
   - Check for typos in variable name
   - Restart application if needed

2. **Unexpected conversion results**
   - Verify exchange rate value
   - Check for scientific notation parsing
   - Test with simple values first

3. **Budget enforcement not working**
   - Ensure budget limits are in USD
   - Verify conversion function is being called
   - Check error logs for conversion failures

### Debug Configuration

```python
import os
from litellm.cost_calculator import get_askii_coin_exchange_rate

def debug_configuration():
    """Debug current configuration."""
    
    print("=== Askii Coin Configuration Debug ===")
    
    # Check environment variable
    env_rate = os.getenv('ASKII_COIN_EXCHANGE_RATE')
    print(f"Environment variable: {env_rate}")
    
    # Check parsed rate
    parsed_rate = get_askii_coin_exchange_rate()
    print(f"Parsed rate: {parsed_rate:,.0f}")
    
    # Check if using default
    is_default = parsed_rate == 1_000_000.0
    print(f"Using default rate: {is_default}")
    
    # Test conversion
    test_usd = 1.0
    test_askii = usd_to_askii_coins(test_usd)
    print(f"Test conversion: ${test_usd} = {test_askii:,.0f} Askii Coins")

debug_configuration()
```

## Security Considerations

### Environment Variable Security

- Store sensitive configuration in secure environment variable management systems
- Use secrets management for production deployments
- Avoid hardcoding exchange rates in source code

### Configuration Validation

- Validate exchange rates are within expected ranges
- Log configuration changes for audit trails
- Implement configuration change approval processes for production

## Performance Considerations

### Caching

The exchange rate is read from environment variables on each call. For high-performance scenarios, consider caching:

```python
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cached_exchange_rate():
    """Get cached exchange rate (updates when environment changes)."""
    return float(os.getenv('ASKII_COIN_EXCHANGE_RATE', '1000000'))

# Clear cache when environment changes
def update_exchange_rate(new_rate):
    os.environ['ASKII_COIN_EXCHANGE_RATE'] = str(new_rate)
    get_cached_exchange_rate.cache_clear()
```

### Batch Operations

For batch operations, minimize environment variable reads:

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate

def batch_convert_to_askii_coins(usd_amounts):
    """Convert multiple USD amounts to Askii Coins efficiently."""
    
    rate = get_askii_coin_exchange_rate()  # Read once
    
    return [usd * rate for usd in usd_amounts]
```
