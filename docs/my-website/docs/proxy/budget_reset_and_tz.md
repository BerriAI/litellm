## Budget Reset Times and Timezones

LiteLLM now supports predictable budget reset times that align with natural calendar boundaries:

- All budgets reset at midnight (00:00:00) in the configured timezone
- Special handling for common durations:
  - Daily (24h/1d): Reset at midnight every day
  - Weekly (7d): Reset on Monday at midnight
  - Monthly (30d): Reset on the 1st of each month at midnight

### Configuring the Timezone

You can specify the timezone for all budget resets in your configuration file:

```yaml
litellm_settings:
  max_budget: 100 # (float) sets max budget as $100 USD
  budget_duration: 30d # (number)(s/m/h/d)
  timezone: "US/Eastern" # Any valid timezone string
```

This ensures that all budget resets happen at midnight in your specified timezone rather than in UTC.
If no timezone is specified, UTC will be used by default.

Common timezone values:

- `UTC` - Coordinated Universal Time
- `US/Eastern` - Eastern Time
- `US/Pacific` - Pacific Time
- `Europe/London` - UK Time
- `Asia/Kolkata` - Indian Standard Time (IST)
- `Asia/Bangkok` - Indochina Time (ICT)
- `Asia/Tokyo` - Japan Standard Time
- `Australia/Sydney` - Australian Eastern Time
