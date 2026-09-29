---
page_title: "litellm_unified_access_groups Data Source - terraform-provider-litellm"
subcategory: ""
description: |-
  Retrieves all LiteLLM unified access groups.
---

# litellm_unified_access_groups (Data Source)

Retrieves all LiteLLM unified access groups configured on the proxy.

## Example Usage

```terraform
data "litellm_unified_access_groups" "all" {}

output "unified_access_group_ids" {
  value = data.litellm_unified_access_groups.all.ids
}
```

## Argument Reference

This data source takes no arguments.

## Attribute Reference

* `access_groups` - List of unified access groups. Each entry exports the same attributes as the `litellm_unified_access_group` data source: `access_group_id`, `access_group_name`, `description`, `access_model_names`, `access_mcp_server_ids`, `access_agent_ids`, `assigned_team_ids`, `assigned_key_ids`, `created_at`, `created_by`, `updated_at`, and `updated_by`.

* `ids` - List of all unified access group IDs.
