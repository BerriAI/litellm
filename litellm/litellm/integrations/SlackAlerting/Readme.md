# Slack Alerting on LiteLLM Gateway 

This folder contains the Slack Alerting integration for LiteLLM Gateway. 

## Folder Structure 

- `slack_alerting.py`: This is the main file that handles sending different types of alerts
- `batching_handler.py`: Handles Batching + sending Httpx Post requests to slack. Slack alerts are sent every 10s or when events are greater than X events. Done to ensure litellm has good performance under high traffic
- `types.py`: This file contains the AlertType enum which is used to define the different types of alerts that can be sent to Slack.
- `utils.py`: This file contains common utils used specifically for slack alerting

## Budget Alert Types

The `budget_alert_types.py` module provides a flexible framework for handling different types of budget alerts:

- `BaseBudgetAlertType`: An abstract base class with abstract methods that all alert types must implement:
  - `get_event_group()`: Returns the Litellm_EntityType for the alert
  - `get_event_message()`: Returns the message prefix for the alert
  - `get_id(user_info)`: Returns the ID to use for caching/tracking the alert

Concrete implementations include:
- `ProxyBudgetAlert`: Alerting for proxy-level budget concerns
- `SoftBudgetAlert`: Alerting when soft budgets are crossed
- `UserBudgetAlert`: Alerting for user-level budget concerns
- `TeamBudgetAlert`: Alerting for team-level budget concerns
- `TokenBudgetAlert`: Alerting for API key budget concerns
- `ProjectedLimitExceededAlert`: Alerting when projected spend will exceed budget

Use the `get_budget_alert_type()` factory function to get the appropriate alert type class for a given alert type string:

```python
from litellm.integrations.SlackAlerting.budget_alert_types import get_budget_alert_type

# Get the appropriate handler
budget_alert_class = get_budget_alert_type("user_budget")

# Use the handler methods
event_group = budget_alert_class.get_event_group()  # Returns Litellm_EntityType.USER
event_message = budget_alert_class.get_event_message()  # Returns "User Budget: "
cache_id = budget_alert_class.get_id(user_info)  # Returns user_id
```

To add a new budget alert type, simply create a new class that extends `BaseBudgetAlertType` and implements all the required methods, then add it to the dictionary in the `get_budget_alert_type()` function.

## Further Reading
- [Doc setting up Alerting on LiteLLM Proxy (Gateway)](https://docs.litellm.ai/docs/proxy/alerting)