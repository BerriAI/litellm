---
page_title: "litellm_unified_access_group Data Source - terraform-provider-litellm"
subcategory: ""
description: |-
  Retrieves information about an existing LiteLLM unified access group.
---

# litellm_unified_access_group (Data Source)

Retrieves information about an existing LiteLLM unified access group by ID.

## Example Usage

```terraform
data "litellm_unified_access_group" "engineering" {
  access_group_id = "b6e5f9d0-..."
}

output "engineering_models" {
  value = data.litellm_unified_access_group.engineering.access_model_names
}
```

## Argument Reference

* `access_group_id` - (Required) ID of the unified access group to look up.

## Attribute Reference

* `id` - The unified access group ID.

* `access_group_name` - Display name of the unified access group.

* `description` - Description of the unified access group.

* `access_model_names` - Model names the access group grants access to.

* `access_mcp_server_ids` - MCP server IDs the access group grants access to.

* `access_agent_ids` - Agent IDs the access group grants access to.

* `assigned_team_ids` - Team IDs the access group is assigned to.

* `assigned_key_ids` - Key IDs the access group is assigned to.

* `created_at` - Timestamp when the access group was created.

* `created_by` - User who created the access group.

* `updated_at` - Timestamp when the access group was last updated.

* `updated_by` - User who last updated the access group.
