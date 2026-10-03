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

## Filter Slack budget alerts by key alias

Set `general_settings.alerting_args.slack_budget_alert_key_aliases` to send Slack budget alerts only for matching virtual key aliases:

```yaml
general_settings:
  alerting: ["slack"]
  alert_types: ["budget_alerts"]
  alerting_args:
    slack_budget_alert_key_aliases:
      - "github-example-*"
```

Patterns are nonempty strings matched against the whole alias using Python's case-sensitive `fnmatchcase` glob rules. Exact aliases, `*`, `?` and character classes such as `[ab]` are supported. Any matching pattern permits the alert

Omitting the setting or setting it to `null` preserves existing behavior. An empty list `[]` disables Slack budget alerts. When a list is configured, only `KEY` budget events with a present, nonempty matching `key_alias` are sent to Slack. User, team, organization, project, proxy and all other non-key budget events are excluded, even if they carry an associated key alias

This only filters Slack delivery for `budget_alerts`, including immediate and digest alerts. Budget enforcement, thresholds, alert cache and deduplication remain unchanged. Webhook, email and Microsoft Teams delivery, and other Slack alert types, are unaffected

The filter applies when an alert enters the Slack queue or digest. Changing it does not retract alerts already queued or accumulated in a digest

The manual Slack service test sends a budget alert without an entity alias. With this filter configured, that budget test is suppressed even if the endpoint reports success. Other enabled alert types tested by that endpoint are unaffected. Omit the setting or use `null` to check Slack connectivity with the manual budget test

## Further Reading
- [Doc setting up Alerting on LiteLLM Proxy (Gateway)](https://docs.litellm.ai/docs/proxy/alerting)