# Team Budgets and Search Cost Attribution

When search requests are made through LiteLLM with a team-bound key, spend is attributed to that team.

## Cost attribution for search

Search calls (`search` / `asearch`) are logged with:

- `metadata.user_api_key_team_id`
- spend rows in `LiteLLM_SpendLogs.team_id`

This means each team's search usage can be queried independently even when using the same model/provider family.

## Why per-team search keys matter

Using one shared Tavily key makes upstream provider billing opaque by team.
With team-specific provider keys:

- provider-side billing is isolated per team
- LiteLLM spend logs still aggregate by team id
- finance can reconcile provider invoices + LiteLLM spend logs

## Recommended setup

1. Issue per-team virtual keys in LiteLLM.
2. Configure `metadata.search_provider_config` per team.
3. Keep a fallback tool-level key only for teams without explicit config.

## Example team update

```bash
curl -X POST "http://localhost:4000/team/update" \
  -H "Authorization: Bearer sk-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-research",
    "metadata": {
      "search_provider_config": {
        "tavily": {
          "api_key": "tvly-research-key"
        },
        "perplexity": {
          "api_key": "pplx-research-key"
        }
      }
    }
  }'
```

## Example spend query

```sql
SELECT team_id, call_type, SUM(spend) AS total_spend, COUNT(*) AS requests
FROM "LiteLLM_SpendLogs"
WHERE call_type IN ('search', 'asearch')
GROUP BY team_id, call_type
ORDER BY total_spend DESC;
```
