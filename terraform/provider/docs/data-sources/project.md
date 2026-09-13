# litellm_project (Data Source)

Retrieves information about an existing LiteLLM project, including its budget settings

## Example Usage

```hcl
data "litellm_project" "ml_experiments" {
  project_id = "4a422a4c-e246-4d02-a1eb-13e835cd0725"
}

output "project_spend" {
  value = data.litellm_project.ml_experiments.spend
}
```

## Argument Reference

The following arguments are supported:

* `project_id` - (Required) Unique identifier of the project to retrieve

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `project_alias` - Human-friendly name for the project
* `description` - Description of the project
* `team_id` - The team ID this project belongs to
* `budget_id` - Budget ID associated with this project
* `models` - List of models the project can access
* `max_budget` - Maximum budget for this project
* `soft_budget` - Soft budget limit for warnings
* `budget_duration` - Budget reset duration
* `tpm_limit` - Tokens per minute limit
* `rpm_limit` - Requests per minute limit
* `max_parallel_requests` - Maximum parallel requests allowed
* `blocked` - Whether the project is blocked from making requests
* `spend` - Current spend for the project
* `created_at` - Timestamp when the project was created
* `updated_at` - Timestamp when the project was last updated
* `created_by` - User that created the project
* `updated_by` - User that last updated the project
