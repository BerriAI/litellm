# litellm_budget Data Source

Retrieves information about an existing LiteLLM budget by ID

## Example Usage

```hcl
data "litellm_budget" "engineering" {
  budget_id = "engineering-monthly"
}

output "engineering_max_budget" {
  value = data.litellm_budget.engineering.max_budget
}
```

## Argument Reference

- `budget_id` (Required) - ID of the budget to retrieve

## Attribute Reference

- `id` - The budget ID
- `max_budget` - Hard budget limit in USD
- `soft_budget` - Soft budget limit in USD that triggers alerts
- `max_parallel_requests` - Maximum concurrent requests allowed for this budget
- `tpm_limit` - Maximum tokens per minute allowed for this budget
- `rpm_limit` - Maximum requests per minute allowed for this budget
- `budget_duration` - Budget reset period
- `model_max_budget` - JSON string of per-model budget config
- `budget_reset_at` - Datetime when the budget is reset
