# litellm_tags (Data Source)

Retrieves the list of all LiteLLM tags. This includes stored tags created via `litellm_tag` or the API, and dynamic tags that were passed on requests

## Example Usage

```hcl
data "litellm_tags" "all" {}

output "tag_names" {
  value = data.litellm_tags.all.ids
}
```

## Example Usage with Date Filter

```hcl
# Limit dynamic tags to those active in a window; stored tags are always returned
data "litellm_tags" "january" {
  start_date = "2026-01-01"
  end_date   = "2026-01-31"
}
```

## Argument Reference

The following arguments are supported:

* `start_date` - (Optional) Start date (YYYY-MM-DD) limiting dynamic tags to those active in the window. Must be given with `end_date`
* `end_date` - (Optional) End date (YYYY-MM-DD). Must be given with `start_date`

## Attribute Reference

The following attributes are exported:

* `ids` - Names of all tags (tag names are their IDs)
* `tags` - List of tags. Each entry exports:
  * `name` - The tag name
  * `description` - Description of the tag
  * `models` - Model IDs this tag applies to
  * `budget_id` - Budget ID associated with this tag
  * `max_budget` - Max budget in USD
  * `soft_budget` - Soft budget in USD
  * `max_parallel_requests` - Max concurrent requests allowed
  * `tpm_limit` - Max tokens per minute
  * `rpm_limit` - Max requests per minute
  * `budget_duration` - Duration for budget reset
  * `created_at` - Timestamp when the tag was created
  * `updated_at` - Timestamp when the tag was last updated
  * `created_by` - User that created the tag
