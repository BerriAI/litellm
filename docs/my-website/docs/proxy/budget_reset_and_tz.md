# Budget Reset Times and Timezones

LiteLLM supports predictable budget reset times that align with natural calendar boundaries.

## How Budget Resets Work

All budgets reset at midnight (00:00:00) in the configured timezone with special handling for common durations:

| Duration | Reset Behavior |
| --- | --- |
| Daily (24h/1d) | Resets at midnight every day |
| Weekly (7d) | Resets on Monday at midnight |
| Monthly (30d) | Resets on the 1st of each month at midnight |

## Configuring the Timezone

Specify the timezone for all budget resets in your configuration file:

```yaml
litellm_settings:
  max_budget: 100 # (float) sets max budget as $100 USD
  budget_duration: 30d # (number)(s/m/h/d)
  timezone: "US/Eastern" # Any valid timezone string
```

This ensures that all budget resets happen at midnight in your specified timezone rather than in UTC. If no timezone is specified, UTC will be used by default.

## Supported Timezones

Any valid [IANA timezone string](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones) is supported (powered by Python's `zoneinfo` module). DST transitions are handled automatically.

**Common timezone values:**

| Timezone | Description |
| --- | --- |
| `UTC` | Coordinated Universal Time |
| `US/Eastern` | Eastern Time |
| `US/Pacific` | Pacific Time |
| `Europe/London` | UK Time |
| `Asia/Kolkata` | Indian Standard Time (IST) |
| `Asia/Bangkok` | Indochina Time (ICT) |
| `Asia/Tokyo` | Japan Standard Time |
| `Australia/Sydney` | Australian Eastern Time |
