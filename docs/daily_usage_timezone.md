# Daily usage reporting timezone

To group new request usage by a deployment's calendar days, set an IANA timezone:

```yaml
litellm_settings:
  daily_usage_timezone: Asia/Singapore
```

The setting applies to daily request spend, tool spend and gateway request counts. Usage date presets and daily cost charts display this reporting timezone for every viewer, regardless of the browser's timezone. Daily API date ranges refer to the same calendar. Request timestamps remain UTC

For example, a request starting at `2026-09-25T17:00:00Z` belongs to September 26 in Singapore. Selecting September 26 includes its usage; selecting September 25 does not

Leaving the setting unset preserves existing reporting behavior. The existing `litellm_settings.timezone` and `budget_reset_time` settings continue to control budget resets independently

Upgrade all writers and readers before enabling the setting consistently across replicas sharing a database. Existing rows keep their original UTC dates; there is no migration or automatic backfill. Historical local-day totals remain approximate, and dates around activation can contain mixed calendars. Do not change the reporting timezone later while expecting old rows to be converted

This setting does not add a per-viewer timezone selector. Timestamp-based APIs retain their own timestamp contracts. PTU reserved-capacity flat costs retain their existing UTC accounting calendar
