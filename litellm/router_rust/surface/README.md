# surface

The Python `Router` API, kept over router-core's state so Python callers see today's behavior. router-core holds the state and the rules; these modules are views and adapters.

- `registry`, `model_list`, `indexes`, `members`: the deployment list, indexes and lookups; writes become changesets
- `legacy_members`: the old public members outside the entry points
- `model_info`: model and group info, answering the `model_info` step's lookups
- `replay`: re-registers pricing after a price data reload
- `settings`: `update_settings` rules and the settings view
- `health`, `health_routing`, `retries`: outcome callbacks, health-check routing and retry answers
- `fallbacks`: fallback config validation
- `alerting`: Slack alerting

A Rust host needs none of this.
