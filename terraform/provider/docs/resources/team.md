# litellm_team Resource

Manages a team configuration in LiteLLM. Teams allow you to group users and manage their access to models and usage limits.

## Example Usage

### Basic Team Configuration

```hcl
resource "litellm_team" "engineering" {
  team_alias = "engineering-team"
  models     = ["gpt-4-proxy", "claude-2"]
  max_budget = 1000.0
}
```

### Team with Comprehensive Configuration

```hcl
resource "litellm_team" "advanced_team" {
  team_alias      = "ai-research-team"
  organization_id = "org_123456"
  models          = ["gpt-4-proxy", "claude-2", "gpt-3.5-turbo"]

  # Budget and rate limiting
  max_budget      = 1000.0
  budget_duration = "1mo"
  tpm_limit       = 500000
  rpm_limit       = 5000
  blocked         = false

  # Team member permissions
  team_member_permissions = [
    "create_key",
    "delete_key",
    "view_spend",
    "edit_team"
  ]

  # Metadata for organization
  metadata = {
    department = "Engineering"
    project    = "AI Research"
    cost_center = "R&D-001"
  }
}
```

### Team with Model Dependencies

```hcl
# First create models
resource "litellm_model" "gpt4" {
  model_name          = "gpt-4-proxy"
  custom_llm_provider = "openai"
  base_model          = "gpt-4"
  model_api_key       = var.openai_api_key
}

resource "litellm_model" "claude" {
  model_name          = "claude-proxy"
  custom_llm_provider = "anthropic"
  base_model          = "claude-3-sonnet-20240229"
  model_api_key       = var.anthropic_api_key
}

# Then create team with access to these models
resource "litellm_team" "model_dependent_team" {
  team_alias = "model-users"
  models = [
    litellm_model.gpt4.model_name,
    litellm_model.claude.model_name
  ]
  
  max_budget      = 500.0
  budget_duration = "1mo"
  
  team_member_permissions = [
    "view_spend"
  ]
}
```

## Argument Reference

The following arguments are supported:

* `team_alias` - (Required) A human-readable identifier for the team.

* `organization_id` - (Optional) The ID of the organization this team belongs to.

* `models` - (Optional) List of model names that this team can access.

* `metadata` - (Optional) A map of metadata key-value pairs associated with the team.

* `blocked` - (Optional) Whether the team is blocked from making requests. Default is `false`.

* `tpm_limit` - (Optional) Team-wide tokens per minute limit.

* `rpm_limit` - (Optional) Team-wide requests per minute limit.

* `max_budget` - (Optional) Maximum budget allocated to the team.

* `budget_duration` - (Optional) Duration for the budget cycle. Valid values are:
  * `daily`
  * `weekly`
  * `monthly`
  * `yearly`

* `team_member_permissions` - (Optional) List of permissions granted to team members. This controls what actions team members can perform within the team context.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The unique identifier for the team.

## Import

Teams can be imported using the team ID:

```shell
terraform import litellm_team.engineering <team-id>
```

Note: The team ID is generated when the team is created and is different from the `team_alias`.

## Note on Team Members

Team members are managed through the separate `litellm_team_member` resource. This allows for more granular control over team membership and permissions. See the `litellm_team_member` resource documentation for details on managing team members.
