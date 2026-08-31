# litellm_users Data Source

Retrieves a page of LiteLLM users, with optional server-side filters

## Example Usage

```hcl
data "litellm_users" "internal" {
  role      = "internal_user"
  page      = 1
  page_size = 100
}

output "internal_user_ids" {
  value = data.litellm_users.internal.ids
}
```

## Argument Reference

- `role` (Optional) - Filter users by role
- `user_ids` (Optional) - Comma-separated list of user IDs to filter by
- `user_email` (Optional) - Filter users by partial email match
- `team` (Optional) - Filter users by team ID
- `page` (Optional, Default `1`) - Page number to fetch
- `page_size` (Optional, Default `25`) - Number of users per page, max 100
- `sort_by` (Optional) - Column to sort by, e.g. `user_id`, `user_email`, `created_at`
- `sort_order` (Optional) - Sort order, `asc` or `desc`

## Attribute Reference

- `users` - Users returned for the requested page. Each entry has:
  - `user_id` - The user ID
  - `user_email` - Email address of the user
  - `user_alias` - Descriptive name for the user
  - `user_role` - Role of the user on the proxy
  - `teams` - List of team IDs the user belongs to
  - `models` - Models the user is allowed to call
  - `max_budget` - Maximum budget in USD
  - `spend` - Current spend in USD
  - `tpm_limit` - Tokens per minute limit
  - `rpm_limit` - Requests per minute limit
  - `key_count` - Number of API keys owned by the user
  - `created_at` - Timestamp when the user was created
- `ids` - IDs of the users returned for the requested page
- `total` - Total number of users matching the filters
- `total_pages` - Total number of pages available
