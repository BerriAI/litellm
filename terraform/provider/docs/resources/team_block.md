# litellm_team_block Resource

Manages the blocked state of an existing LiteLLM team. Creating this resource blocks the team (all calls from its keys are rejected); destroying it unblocks the team.

If the team is unblocked outside of Terraform (or deleted), the resource is removed from state and Terraform plans to re-block it on the next apply.

## Example Usage

```hcl
resource "litellm_team" "example" {
  team_alias = "suspended-team"
}

resource "litellm_team_block" "example" {
  team_id = litellm_team.example.id
}
```

## Argument Reference

The following arguments are supported:

* `team_id` - (Required, Forces new resource) The ID of the team to block.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The team ID.
* `blocked` - Whether the team is currently blocked. Always `true` while this resource exists.

## Import

Team blocks can be imported using the team ID:

```shell
terraform import litellm_team_block.example team-1234
```
