---
sidebar_position: 6
---

# Handle budget exhausted

When a key / team / app hits its budget cap, the proxy returns **400** with
a structured detail. Handle it gracefully:

```python
from xct_litellm import XctError

try:
    reply = xct.chat.completions.create(...)
except XctError as e:
    if e.status == 400 and isinstance(e.body, dict):
        detail = e.body.get("detail", "")
        if "budget" in str(detail).lower():
            # Show "out of credit" UX, not a generic error
            ...
            raise
    raise
```

Or subscribe to the `budget.exhausted` webhook (proactive — see
[Subscribe to webhooks](./subscribe-webhook.md)).

The cap is enforced on the spend-tracking write path; the failing
request still consumes the API call quota (no retry-friendly distinction
yet). Plan to fallback to a cheaper model when the primary is near limit.

For a per-app budget alarm in your monitoring system, query Prometheus:

```promql
sum by (app_id) (rate(litellm_capability_spend_total[1h])) > 5
```
