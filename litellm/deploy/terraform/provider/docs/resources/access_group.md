---
page_title: "litellm_access_group Resource - terraform-provider-litellm"
subcategory: ""
description: |-
  Manages a LiteLLM model access group.
---

# litellm_access_group (Resource)

Manages a LiteLLM model access group. Access groups bundle model deployments under one name so keys and teams can be granted access to the whole group at once.

## Example Usage

```terraform
resource "litellm_access_group" "production" {
  access_group = "production-models"
  model_names  = ["gpt-4", "claude-3-sonnet"]
}

# Target specific deployments by model ID instead of model name
resource "litellm_access_group" "pinned" {
  access_group = "pinned-deployments"
  model_ids    = ["4dbd9f43-...", "9a1e2c77-..."]
}
```

## Argument Reference

* `access_group` - (Required, Forces new resource) Name of the access group.

* `model_names` - (Optional) List of model names (the `model_name` of each deployment) to include in the group. At least one of `model_names` or `model_ids` must be set.

* `model_ids` - (Optional) List of specific deployment model IDs to include in the group. Takes precedence over `model_names` when both are set.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The access group name.

* `deployment_count` - Number of deployments currently tagged with this access group.

## Import

Access groups can be imported using the access group name:

```shell
terraform import litellm_access_group.production production-models
```
