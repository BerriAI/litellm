# litellm_project Resource

Manages a project in LiteLLM. Projects sit between teams and keys in the hierarchy, allowing fine-grained budget and model access control within a team

## Example Usage

```hcl
resource "litellm_team" "research" {
  team_alias = "research-team"
}

resource "litellm_project" "ml_experiments" {
  team_id       = litellm_team.research.id
  project_alias = "ml-experiments"
  description   = "ML experimentation project"
  models        = ["gpt-5.6", "claude-opus-5"]

  max_budget      = 1000.0
  soft_budget     = 800.0
  budget_duration = "30d"
  tpm_limit       = 500000
  rpm_limit       = 5000

  tags = ["research", "gpu"]

  metadata = {
    cost_center = "R&D-001"
  }
}
```

## Argument Reference

The following arguments are supported:

* `team_id` - (Required, Forces new resource) The team ID this project belongs to
* `project_alias` - (Optional) Human-friendly name for the project
* `description` - (Optional) Description of the project's purpose and use case
* `models` - (Optional) List of models the project can access
* `metadata` - (Optional) Map of metadata for the project
* `tags` - (Optional) Tags associated with the project
* `max_budget` - (Optional) Maximum budget for this project
* `soft_budget` - (Optional) Soft budget limit for warnings
* `budget_duration` - (Optional) Budget reset duration, for example `1h`, `30d`
* `budget_id` - (Optional) Budget ID to associate with this project
* `tpm_limit` - (Optional) Tokens per minute limit
* `rpm_limit` - (Optional) Requests per minute limit
* `max_parallel_requests` - (Optional) Maximum parallel requests allowed
* `model_max_budget` - (Optional) Map of per-model budget limits
* `model_rpm_limit` - (Optional) Map of per-model RPM limits
* `model_tpm_limit` - (Optional) Map of per-model TPM limits
* `blocked` - (Optional) Whether the project is blocked from making requests

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `id` - The project ID assigned by LiteLLM
* `spend` - Current spend for the project
* `created_at` - Timestamp when the project was created
* `updated_at` - Timestamp when the project was last updated
* `created_by` - User that created the project
* `updated_by` - User that last updated the project

## Import

Projects can be imported using the project ID:

```shell
terraform import litellm_project.example 4a422a4c-e246-4d02-a1eb-13e835cd0725
```
