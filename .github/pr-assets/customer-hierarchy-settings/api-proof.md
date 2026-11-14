## Captured API proof

These are actual curl requests to a local proxy and a controlled HTTP provider, using synthetic records and keys. Addresses, header files and IDs below are placeholders. Response excerpts omit unrelated fields and credentials.

### Permission change: verified before and after

The baseline process ran clean relevant source at `6685525fa2a809e0d4bc2c13cb07aef8eec242c3`. The requests used the same bound key and org policy; the after request was rerun after the schema update: the org allowed only `gpt-4.1-mini`; the key allowed `usage-proof-model` and was bound with `metadata.customer_org_id`. No team/member was needed for this org-level comparison.

```sh
curl --silent --show-error --request POST "$PROXY_BASE/v1/chat/completions" \
  --header "@$SCOPED_KEY_HEADERS" --header 'Content-Type: application/json' \
  --data-binary @request.json
```

`request.json`:
```json
{
  "model": "usage-proof-model",
  "messages": [
    {
      "role": "user",
      "content": "Synthetic before-after permission check"
    }
  ]
}
```

Baseline: **HTTP 200**.
```json
{
  "object": "chat.completion",
  "model": "usage-proof-model",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Controlled usage proof response.",
        "role": "assistant"
      },
      "provider_specific_fields": {}
    }
  ]
}
```

Current implementation: **HTTP 403**, with **zero controlled-provider calls**.
```json
{
  "error": {
    "message": "The requested model 'usage-proof-model' is not in the allowed models for this customer. Check the models this customer can use and try again.",
    "type": "customer_model_access_denied",
    "param": "model",
    "code": "403"
  }
}
```

### Current hierarchy and key intersection

Admin requests created the records below; each returned **201**. Member `budget_id` selects its template and `budget_override_id` identifies its optional override:

`POST /management/v1/customer_organizations`
```json
{
  "customer_org_id": "<org-id>",
  "alias": "Permissions and deletion proof",
  "budget": {
    "max_budget": 500,
    "allowed_models": [
      "gpt-4.1-mini",
      "gpt-5.5"
    ]
  },
  "default_budgets": [
    {
      "name": "Standard",
      "is_default": true,
      "max_budget": 100,
      "allowed_models": [
        "gpt-4.1-mini"
      ]
    }
  ]
}
```

`POST /management/v1/customer_teams`
```json
{
  "customer_team_id": "<team-id>",
  "parent_org_id": "<org-id>",
  "alias": "Proof customer team",
  "budget": {
    "max_budget": 250,
    "allowed_models": [
      "gpt-4.1-mini",
      "gpt-5.5"
    ]
  }
}
```

`POST /management/v1/customer_members`
```json
{
  "customer_member_id": "<member-id>",
  "parent_team_id": "<team-id>",
  "alias": "Proof customer member",
  "budget_override": {
    "max_budget": 100,
    "allowed_models": [
      "gpt-4.1-mini"
    ]
  }
}
```

A synthetic `internal_user` was created with `auto_create_key: false`. The admin then created these keys through `POST /key/generate` (each **200**; returned credentials omitted):

| Key | models | metadata / budget binding |
| --- | --- | --- |
| A | `["gpt-4.1-mini", "gpt-5.5"]` | `{"customer_org_id":"<org-id>","customer_member_id":"<member-id>"}` |
| B | `["gpt-4.1-mini"]` | Same org/member binding |
| C | `["gpt-4.1-mini"]` | `budget_id: "<shared-budget-id>"`; no customer binding |

Inference curl uses the appropriate key header file. Captured request bodies:
```json
{
  "model": "gpt-4.1-mini",
  "messages": [
    {
      "role": "user",
      "content": "Synthetic permission check"
    }
  ]
}
```
```json
{
  "model": "gpt-4.1-mini",
  "input": "Synthetic permission check"
}
```

| State / key | Requested model | Chat HTTP | Responses HTTP | Provider calls per request |
| --- | --- | --- | --- | --- |
| Member restriction / A | gpt-4.1-mini | 200 | 200 | 1 |
| Member restriction / A | gpt-5.5 | 403 | 403 | 0 |
| Member widened; key B remains narrow | gpt-5.5 | 403 | 403 | 0 |
| Member widened; org narrowed / A | gpt-5.5 | 403 | 403 | 0 |

The widening PATCH was `{"budget_override":{"allowed_models":["gpt-4.1-mini","gpt-5.5"]}}` on the member. The org narrowing PATCH was `{"budget":{"allowed_models":["gpt-4.1-mini"]}}`. Key B still denied the model after the member widened.

### Cascade, retained history and shared key budget

**Same-revision comparison:** deleting the populated org without `cascade=true` returned **409**:
```json
{
  "type": "urn:litellm:error:customer-hierarchy-has-children",
  "title": "Has children",
  "status": 409,
  "detail": "Delete the customer's children before deleting the parent."
}
```

The confirmed cascade request returned **204**, with an empty body:

```sh
curl --silent --show-error --request DELETE \
  "$PROXY_BASE/management/v1/customer_organizations/$ORG_ID?cascade=true" \
  --header "@$ADMIN_HEADERS"
```

Subsequent GETs for that org, its team and its member each returned **404**. The existing demo organization still returned **200**.

Key C referenced the org-owned budget. After PATCHing that budget to `max_budget: 0`, its inference returned **422 `budget_exceeded` before and after cascade**, with zero provider calls. Its authenticated `GET /key/info` retained the same relation and limit (selected fields):
```json
{
  "info": {
    "budget_id": "<shared-budget-id>",
    "litellm_budget_table": {
      "max_budget": 0.0
    }
  }
}
```

The same admin daily query returned **200** before and after cascade:

```sh
curl --silent --show-error \
  "$PROXY_BASE/management/v1/customer_organizations/daily/activity?customer_org_ids=$ORG_ID&start_date=$UTC_DATE&end_date=$UTC_DATE" \
  --header "@$ADMIN_HEADERS"
```

Selected captured metadata was identical before and after; API-key/entity breakdowns are omitted:
```json
{
  "metadata": {
    "total_spend": 0.28,
    "total_prompt_tokens": 200,
    "total_completion_tokens": 40,
    "total_tokens": 240,
    "total_api_requests": 4,
    "total_successful_requests": 2,
    "total_failed_requests": 2
  }
}
```
