# litellm_projects (Data Source)

Retrieves the list of all LiteLLM projects visible to the caller

## Example Usage

```hcl
data "litellm_projects" "all" {}

output "project_ids" {
  value = data.litellm_projects.all.ids
}

output "project_aliases" {
  value = [for p in data.litellm_projects.all.projects : p.project_alias]
}
```

## Argument Reference

This data source takes no arguments

## Attribute Reference

The following attributes are exported:

* `ids` - IDs of all projects
* `projects` - List of projects. Each entry exports:
  * `project_id` - The project ID
  * `project_alias` - Human-friendly name for the project
  * `description` - Description of the project
  * `team_id` - The team ID this project belongs to
  * `budget_id` - Budget ID associated with this project
  * `models` - List of models the project can access
  * `blocked` - Whether the project is blocked from making requests
  * `spend` - Current spend for the project
  * `created_at` - Timestamp when the project was created
  * `updated_at` - Timestamp when the project was last updated
  * `created_by` - User that created the project
  * `updated_by` - User that last updated the project
