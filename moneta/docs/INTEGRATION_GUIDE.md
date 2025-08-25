# Askii Coin Integration Guide

## Overview

This guide provides step-by-step instructions for integrating with the Askii Coin virtual currency system. Whether you're setting up a new LiteLLM deployment or migrating from a USD-only system, this guide will walk you through the process.

## Prerequisites

- LiteLLM with Askii Coin system installed
- Access to environment variable configuration
- Understanding of your current billing and budget setup

## Integration Steps

### Step 1: Environment Setup

#### 1.1 Configure Exchange Rate

Set the USD to Askii Coin exchange rate:

```bash
# Option 1: Set in environment
export ASKII_COIN_EXCHANGE_RATE=1000000

# Option 2: Add to .env file
echo "ASKII_COIN_EXCHANGE_RATE=1000000" >> .env

# Option 3: Docker environment
docker run -e ASKII_COIN_EXCHANGE_RATE=1000000 litellm/litellm:latest
```

#### 1.2 Verify Configuration

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate, usd_to_askii_coins

# Check configuration
rate = get_askii_coin_exchange_rate()
print(f"Exchange rate: 1 USD = {rate:,.0f} Askii Coins")

# Test conversion
test_amount = 1.0
askii_coins = usd_to_askii_coins(test_amount)
print(f"${test_amount} = {askii_coins:,.0f} Askii Coins")
```

### Step 2: Cost Calculation Integration

#### 2.1 Basic Cost Calculation

```python
from litellm.cost_calculator import cost_per_token

def calculate_completion_cost(model: str, prompt_tokens: int, completion_tokens: int):
    """Calculate cost for a completion in Askii Coins."""
    
    cost_askii_coins = cost_per_token(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
    
    return cost_askii_coins

# Example usage
cost = calculate_completion_cost(
    model="gpt-3.5-turbo",
    prompt_tokens=100,
    completion_tokens=50
)

print(f"Completion cost: {cost:,.0f} Askii Coins")
```

#### 2.2 Batch Cost Calculation

```python
from litellm.cost_calculator import cost_per_token

def calculate_batch_costs(requests):
    """Calculate costs for multiple requests."""
    
    total_cost = 0
    request_costs = []
    
    for request in requests:
        cost = cost_per_token(
            model=request['model'],
            prompt_tokens=request['prompt_tokens'],
            completion_tokens=request['completion_tokens']
        )
        
        request_costs.append({
            'request_id': request['id'],
            'cost_askii_coins': cost,
            'model': request['model']
        })
        
        total_cost += cost
    
    return {
        'total_cost_askii_coins': total_cost,
        'individual_costs': request_costs
    }

# Example usage
requests = [
    {'id': '1', 'model': 'gpt-3.5-turbo', 'prompt_tokens': 100, 'completion_tokens': 50},
    {'id': '2', 'model': 'gpt-4', 'prompt_tokens': 200, 'completion_tokens': 100},
]

results = calculate_batch_costs(requests)
print(f"Total cost: {results['total_cost_askii_coins']:,.0f} Askii Coins")
```

### Step 3: Budget Enforcement Integration

#### 3.1 User Budget Enforcement

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

class UserBudgetManager:
    def __init__(self, user_id: str, budget_usd: float):
        self.user_id = user_id
        self.budget_usd = budget_usd
        self.budget_askii_coins = convert_budget_to_askii_coins(budget_usd)
        self.current_spend_askii_coins = 0
    
    def can_afford(self, cost_askii_coins: float) -> bool:
        """Check if user can afford a request."""
        if self.budget_askii_coins is None:
            return True  # No budget limit
        
        return (self.current_spend_askii_coins + cost_askii_coins) <= self.budget_askii_coins
    
    def add_spend(self, cost_askii_coins: float):
        """Add spend to user's total."""
        self.current_spend_askii_coins += cost_askii_coins
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget in Askii Coins."""
        if self.budget_askii_coins is None:
            return float('inf')
        
        return max(0, self.budget_askii_coins - self.current_spend_askii_coins)

# Example usage
user_budget = UserBudgetManager(user_id="user123", budget_usd=10.0)

# Check if user can afford a request
request_cost = 500_000  # 500K Askii Coins
if user_budget.can_afford(request_cost):
    print("Request approved")
    user_budget.add_spend(request_cost)
else:
    print("Request denied - insufficient budget")

print(f"Remaining budget: {user_budget.get_remaining_budget():,.0f} Askii Coins")
```

#### 3.2 Team Budget Enforcement

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

class TeamBudgetManager:
    def __init__(self, team_id: str, team_budget_usd: float):
        self.team_id = team_id
        self.team_budget_usd = team_budget_usd
        self.team_budget_askii_coins = convert_budget_to_askii_coins(team_budget_usd)
        self.team_spend_askii_coins = 0
        self.user_spends = {}  # Track individual user spends
    
    def add_user_spend(self, user_id: str, cost_askii_coins: float):
        """Add spend for a specific user."""
        if user_id not in self.user_spends:
            self.user_spends[user_id] = 0
        
        self.user_spends[user_id] += cost_askii_coins
        self.team_spend_askii_coins += cost_askii_coins
    
    def is_within_budget(self) -> bool:
        """Check if team is within budget."""
        if self.team_budget_askii_coins is None:
            return True
        
        return self.team_spend_askii_coins <= self.team_budget_askii_coins
    
    def get_budget_status(self) -> dict:
        """Get detailed budget status."""
        return {
            'team_id': self.team_id,
            'budget_usd': self.team_budget_usd,
            'budget_askii_coins': self.team_budget_askii_coins,
            'spend_askii_coins': self.team_spend_askii_coins,
            'remaining_askii_coins': self.team_budget_askii_coins - self.team_spend_askii_coins if self.team_budget_askii_coins else None,
            'within_budget': self.is_within_budget(),
            'user_spends': self.user_spends
        }

# Example usage
team_budget = TeamBudgetManager(team_id="team456", team_budget_usd=100.0)

# Add spends for different users
team_budget.add_user_spend("user1", 1_000_000)  # 1M Askii Coins
team_budget.add_user_spend("user2", 2_000_000)  # 2M Askii Coins

status = team_budget.get_budget_status()
print(f"Team budget status: {status}")
```

### Step 4: Lago Integration

#### 4.1 Basic Lago Integration

```python
from moneta.lago_logger import LagoLogger
import uuid

class AskiiCoinLagoIntegration:
    def __init__(self):
        self.lago_logger = LagoLogger()
    
    async def send_usage_event(self, customer_id: str, cost_askii_coins: float):
        """Send usage event to Lago with Askii Coin amount."""
        
        call_id = str(uuid.uuid4())
        
        await self.lago_logger._send_usage_event(
            external_subscription_id=customer_id,
            cost=cost_askii_coins,  # Already in Askii Coins
            call_id=call_id
        )
        
        return call_id

# Example usage
lago_integration = AskiiCoinLagoIntegration()

# Send usage event
await lago_integration.send_usage_event(
    customer_id="customer-123",
    cost_askii_coins=1_500_000  # 1.5M Askii Coins
)
```

#### 4.2 Batch Lago Events

```python
import asyncio
from moneta.lago_logger import LagoLogger

class BatchLagoIntegration:
    def __init__(self):
        self.lago_logger = LagoLogger()
    
    async def send_batch_usage_events(self, events):
        """Send multiple usage events to Lago."""
        
        tasks = []
        for event in events:
            task = self.lago_logger._send_usage_event(
                external_subscription_id=event['customer_id'],
                cost=event['cost_askii_coins'],
                call_id=event['call_id']
            )
            tasks.append(task)
        
        # Send all events concurrently
        await asyncio.gather(*tasks)

# Example usage
batch_integration = BatchLagoIntegration()

events = [
    {'customer_id': 'customer-1', 'cost_askii_coins': 1_000_000, 'call_id': 'call-1'},
    {'customer_id': 'customer-2', 'cost_askii_coins': 2_000_000, 'call_id': 'call-2'},
    {'customer_id': 'customer-3', 'cost_askii_coins': 500_000, 'call_id': 'call-3'},
]

await batch_integration.send_batch_usage_events(events)
```

### Step 5: Monitoring and Analytics

#### 5.1 Cost Analytics

```python
from litellm.cost_calculator import get_askii_coin_exchange_rate

class AskiiCoinAnalytics:
    def __init__(self):
        self.exchange_rate = get_askii_coin_exchange_rate()
    
    def askii_coins_to_usd(self, askii_coins: float) -> float:
        """Convert Askii Coins back to USD for reporting."""
        return askii_coins / self.exchange_rate
    
    def generate_cost_report(self, usage_data):
        """Generate cost report with both currencies."""
        
        total_askii_coins = sum(item['cost_askii_coins'] for item in usage_data)
        total_usd = self.askii_coins_to_usd(total_askii_coins)
        
        report = {
            'total_cost_askii_coins': total_askii_coins,
            'total_cost_usd': total_usd,
            'exchange_rate': self.exchange_rate,
            'breakdown': []
        }
        
        for item in usage_data:
            report['breakdown'].append({
                'user_id': item['user_id'],
                'cost_askii_coins': item['cost_askii_coins'],
                'cost_usd': self.askii_coins_to_usd(item['cost_askii_coins']),
                'model': item['model'],
                'timestamp': item['timestamp']
            })
        
        return report

# Example usage
analytics = AskiiCoinAnalytics()

usage_data = [
    {'user_id': 'user1', 'cost_askii_coins': 1_000_000, 'model': 'gpt-3.5-turbo', 'timestamp': '2024-01-01'},
    {'user_id': 'user2', 'cost_askii_coins': 2_000_000, 'model': 'gpt-4', 'timestamp': '2024-01-01'},
]

report = analytics.generate_cost_report(usage_data)
print(f"Total cost: {report['total_cost_askii_coins']:,.0f} Askii Coins (${report['total_cost_usd']:.2f} USD)")
```

#### 5.2 Budget Monitoring

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

class BudgetMonitor:
    def __init__(self):
        self.alerts = []
    
    def check_budget_utilization(self, user_id: str, budget_usd: float, spend_askii_coins: float):
        """Check budget utilization and generate alerts."""
        
        budget_askii_coins = convert_budget_to_askii_coins(budget_usd)
        
        if budget_askii_coins is None:
            return None  # No budget limit
        
        utilization = spend_askii_coins / budget_askii_coins
        
        alert = {
            'user_id': user_id,
            'budget_usd': budget_usd,
            'budget_askii_coins': budget_askii_coins,
            'spend_askii_coins': spend_askii_coins,
            'utilization_percent': utilization * 100,
            'status': 'ok'
        }
        
        if utilization >= 1.0:
            alert['status'] = 'exceeded'
            alert['message'] = 'Budget exceeded'
        elif utilization >= 0.9:
            alert['status'] = 'warning'
            alert['message'] = 'Budget 90% utilized'
        elif utilization >= 0.8:
            alert['status'] = 'caution'
            alert['message'] = 'Budget 80% utilized'
        
        return alert

# Example usage
monitor = BudgetMonitor()

alert = monitor.check_budget_utilization(
    user_id="user123",
    budget_usd=10.0,
    spend_askii_coins=9_000_000  # 9M Askii Coins (90% of 10M budget)
)

print(f"Budget alert: {alert}")
```

### Step 6: Testing Integration

#### 6.1 Unit Tests

```python
import pytest
from litellm.cost_calculator import usd_to_askii_coins, convert_budget_to_askii_coins

def test_currency_conversion():
    """Test currency conversion functionality."""
    
    # Test basic conversion
    result = usd_to_askii_coins(1.0)
    assert result == 1_000_000  # Default rate
    
    # Test budget conversion
    budget = convert_budget_to_askii_coins(10.0)
    assert budget == 10_000_000
    
    # Test None budget
    no_budget = convert_budget_to_askii_coins(None)
    assert no_budget is None

def test_budget_enforcement():
    """Test budget enforcement logic."""
    
    budget_usd = 5.0
    budget_askii_coins = convert_budget_to_askii_coins(budget_usd)
    
    # Within budget
    spend = 3_000_000  # 3M Askii Coins
    assert spend < budget_askii_coins
    
    # Over budget
    spend = 7_000_000  # 7M Askii Coins
    assert spend > budget_askii_coins

# Run tests
pytest.main([__file__])
```

#### 6.2 Integration Tests

```python
import asyncio
from unittest.mock import patch, AsyncMock

async def test_end_to_end_integration():
    """Test end-to-end integration."""
    
    # Mock Lago integration
    with patch('moneta.lago_logger.LagoLogger._send_usage_event') as mock_lago:
        mock_lago.return_value = AsyncMock()
        
        # Calculate cost
        from litellm.cost_calculator import cost_per_token
        cost = cost_per_token(
            model="gpt-3.5-turbo",
            prompt_tokens=100,
            completion_tokens=50
        )
        
        # Check budget
        from litellm.cost_calculator import convert_budget_to_askii_coins
        budget = convert_budget_to_askii_coins(1.0)  # $1 budget
        
        if cost <= budget:
            # Send to Lago
            from moneta.lago_logger import LagoLogger
            lago_logger = LagoLogger()
            await lago_logger._send_usage_event(
                external_subscription_id="test-customer",
                cost=cost,
                call_id="test-call"
            )
            
            # Verify Lago was called
            assert mock_lago.called
            
            # Verify correct parameters
            call_args = mock_lago.call_args
            assert call_args.kwargs['cost'] == cost
            assert call_args.kwargs['external_subscription_id'] == "test-customer"

# Run integration test
asyncio.run(test_end_to_end_integration())
```

## Best Practices

### 1. Error Handling

Always implement proper error handling for currency conversion and budget checks:

```python
from litellm.cost_calculator import convert_budget_to_askii_coins

def safe_budget_check(budget_usd: float, spend_askii_coins: float) -> bool:
    """Safely check budget with error handling."""
    
    try:
        budget_askii_coins = convert_budget_to_askii_coins(budget_usd)
        
        if budget_askii_coins is None:
            return True  # No budget limit
        
        return spend_askii_coins <= budget_askii_coins
        
    except Exception as e:
        print(f"Budget check error: {e}")
        return False  # Fail safe - deny request
```

### 2. Logging and Monitoring

Implement comprehensive logging for debugging and monitoring:

```python
import logging
from litellm.cost_calculator import get_askii_coin_exchange_rate

logger = logging.getLogger(__name__)

def log_currency_conversion(usd_amount: float, askii_coins: float):
    """Log currency conversion for monitoring."""
    
    rate = get_askii_coin_exchange_rate()
    
    logger.info(
        f"Currency conversion: ${usd_amount} -> {askii_coins:,.0f} Askii Coins "
        f"(rate: {rate:,.0f})"
    )
```

### 3. Configuration Management

Use environment-specific configuration:

```python
import os

def get_environment_config():
    """Get environment-specific Askii Coin configuration."""
    
    env = os.getenv('ENVIRONMENT', 'development')
    
    config = {
        'development': {'exchange_rate': '1000'},
        'staging': {'exchange_rate': '100000'},
        'production': {'exchange_rate': '1000000'}
    }
    
    return config.get(env, config['production'])
```

## Migration Checklist

- [ ] Set `ASKII_COIN_EXCHANGE_RATE` environment variable
- [ ] Verify currency conversion functions work correctly
- [ ] Test budget enforcement with converted amounts
- [ ] Update monitoring dashboards to show Askii Coin amounts
- [ ] Test Lago integration sends correct Askii Coin amounts
- [ ] Run comprehensive test suite
- [ ] Update documentation and training materials
- [ ] Monitor system behavior after deployment
- [ ] Verify billing accuracy with Lago
- [ ] Set up alerts for budget threshold violations
