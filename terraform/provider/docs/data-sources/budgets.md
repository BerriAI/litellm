# litellm_budgets Data Source

Retrieves all budgets configured on the LiteLLM proxy

## Example Usage

```hcl
data "litellm_budgets" "all" {}

output "budget_ids" {
  value = data.litellm_budgets.all.ids
}
```

## Argument Reference

This data source takes no arguments

## Attribute Reference

- `budgets` - All budgets configured on the proxy. Each entry has:
  - `budget_id` - The budget ID
  - `max_budget` - Hard budget limit in USD
  - `soft_budget` - Soft budget limit in USD that triggers alerts
  - `max_parallel_requests` - Maximum concurrent requests allowed for this budget
  - `tpm_limit` - Maximum tokens per minute allowed for this budget
  - `rpm_limit` - Maximum requests per minute allowed for this budget
  - `budget_duration` - Budget reset period
  - `model_max_budget` - JSON string of per-model budget config
  - `budget_reset_at` - Datetime when the budget is reset
- `ids` - IDs of all budgets configured on the proxy
