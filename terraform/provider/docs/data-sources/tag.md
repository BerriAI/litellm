# litellm_tag (Data Source)

Retrieves information about an existing LiteLLM tag, including its budget settings

## Example Usage

```hcl
data "litellm_tag" "production" {
  name = "production"
}

output "production_tag_budget" {
  value = data.litellm_tag.production.max_budget
}
```

## Argument Reference

The following arguments are supported:

* `name` - (Required) Name of the tag to retrieve

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `description` - Description of the tag
* `models` - Model IDs this tag applies to
* `budget_id` - Budget ID associated with this tag
* `max_budget` - Max budget in USD for this tag
* `soft_budget` - Soft budget in USD for this tag
* `max_parallel_requests` - Max concurrent requests allowed for this tag
* `tpm_limit` - Max tokens per minute for this tag
* `rpm_limit` - Max requests per minute for this tag
* `budget_duration` - Duration for budget reset
* `created_at` - Timestamp when the tag was created
* `updated_at` - Timestamp when the tag was last updated
* `created_by` - User that created the tag
