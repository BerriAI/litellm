---
page_title: "litellm_unified_access_group Resource - terraform-provider-litellm"
subcategory: ""
description: |-
  Manages a LiteLLM unified access group.
---

# litellm_unified_access_group (Resource)

Manages a LiteLLM unified access group. Unified access groups grant access to models, MCP servers, and agents in one bundle, and can be assigned to teams and keys.

## Example Usage

```terraform
resource "litellm_unified_access_group" "engineering" {
  access_group_name = "engineering-access"
  description       = "Models and tools for the engineering org"

  access_model_names    = ["gpt-4", "claude-3-sonnet"]
  access_mcp_server_ids = [litellm_mcp_server.github.id]

  assigned_team_ids = [litellm_team.engineering.id]
}
```

## Argument Reference

* `access_group_name` - (Required) Display name of the unified access group.

* `description` - (Optional) Description of the unified access group.

* `access_model_names` - (Optional) Model names this access group grants access to.

* `access_mcp_server_ids` - (Optional) MCP server IDs this access group grants access to.

* `access_agent_ids` - (Optional) Agent IDs this access group grants access to.

* `assigned_team_ids` - (Optional) Team IDs the access group is assigned to.

* `assigned_key_ids` - (Optional) Key IDs (token hashes) the access group is assigned to.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The unique identifier of the unified access group.

* `access_group_id` - Same as `id`.

* `created_at` - Timestamp when the access group was created.

* `created_by` - User who created the access group.

* `updated_at` - Timestamp when the access group was last updated.

* `updated_by` - User who last updated the access group.

## Import

Unified access groups can be imported using the access group ID:

```shell
terraform import litellm_unified_access_group.engineering <access-group-id>
```
