---
page_title: "litellm_access_groups Data Source - terraform-provider-litellm"
subcategory: ""
description: |-
  Retrieves all LiteLLM model access groups.
---

# litellm_access_groups (Data Source)

Retrieves all LiteLLM model access groups configured on the proxy.

## Example Usage

```terraform
data "litellm_access_groups" "all" {}

output "access_group_names" {
  value = data.litellm_access_groups.all.ids
}
```

## Argument Reference

This data source takes no arguments.

## Attribute Reference

* `access_groups` - List of access groups. Each entry exports:
  * `access_group` - The access group name.
  * `model_names` - List of model names in the access group.
  * `deployment_count` - Number of deployments tagged with this access group.

* `ids` - List of all access group names.
