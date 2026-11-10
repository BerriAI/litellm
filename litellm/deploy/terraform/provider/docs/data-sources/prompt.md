# litellm_prompt Data Source

Retrieves information about an existing LiteLLM prompt by ID. The provider API key is not exposed.

## Example Usage

```hcl
data "litellm_prompt" "existing" {
  prompt_id = "my-langfuse-prompt"
}

output "prompt_integration" {
  value = data.litellm_prompt.existing.prompt_integration
}
```

### With Environment

```hcl
data "litellm_prompt" "prod" {
  prompt_id   = "my-langfuse-prompt"
  environment = "production"
}
```

## Argument Reference

* `prompt_id` - (Required) Unique identifier of the prompt to retrieve.
* `environment` - (Optional) Environment to fetch the prompt from (e.g. `development`, `production`).

## Attribute Reference

* `prompt_integration` - The prompt integration provider.
* `api_base` - Base URL for the prompt provider API.
* `provider_specific_query_params` - JSON string of provider-specific query parameters.
* `ignore_prompt_manager_model` - Whether the model specified in the prompt manager is ignored.
* `ignore_prompt_manager_optional_params` - Whether optional params from the prompt manager are ignored.
* `dotprompt_content` - Content for the dotprompt integration.
* `prompt_type` - Type of prompt: `config` or `db`.
* `version` - Version number of the prompt.
* `environments` - List of environments this prompt exists in.
* `created_at` - Timestamp when the prompt was created.
* `updated_at` - Timestamp when the prompt was last updated.
