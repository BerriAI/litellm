# litellm_user Data Source

Retrieves information about an existing LiteLLM user by ID

## Example Usage

```hcl
data "litellm_user" "alice" {
  user_id = "alice-user-id"
}

output "alice_email" {
  value = data.litellm_user.alice.user_email
}
```

## Argument Reference

- `user_id` (Required) - ID of the user to retrieve

## Attribute Reference

- `id` - The user ID
- `user_email` - Email address of the user
- `user_alias` - Descriptive name for the user
- `user_role` - Role of the user on the proxy
- `teams` - List of team IDs the user belongs to
- `models` - Models the user is allowed to call
- `max_budget` - Maximum budget in USD for the user
- `spend` - Current spend in USD for the user
- `budget_duration` - Budget reset period for the user
- `tpm_limit` - Tokens per minute limit
- `rpm_limit` - Requests per minute limit
- `max_parallel_requests` - Maximum number of parallel requests
- `metadata` - Map of metadata for the user
- `model_max_budget` - JSON string of per-model budget config
