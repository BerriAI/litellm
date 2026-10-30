# litellm_prompts Data Source

Retrieves the list of all prompts configured on the LiteLLM proxy.

## Example Usage

```hcl
data "litellm_prompts" "all" {}

output "prompt_ids" {
  value = data.litellm_prompts.all.ids
}
```

### Filter by Environment

```hcl
data "litellm_prompts" "production" {
  environment = "production"
}
```

## Argument Reference

* `environment` - (Optional) Filter prompts by environment (e.g. `development`, `production`).

## Attribute Reference

* `prompts` - List of prompts. Each entry contains:
  * `prompt_id` - Unique identifier of the prompt.
  * `prompt_integration` - The prompt integration provider.
  * `prompt_type` - Type of prompt: `config` or `db`.
  * `version` - Version number of the prompt.
  * `environment` - Environment the prompt belongs to.
  * `created_at` - Timestamp when the prompt was created.
  * `updated_at` - Timestamp when the prompt was last updated.
* `ids` - List of all prompt IDs.
