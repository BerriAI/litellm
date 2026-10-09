# litellm_prompt Resource

Manages a prompt in LiteLLM. Prompts let you manage prompt templates from external providers such as Langfuse, or inline dotprompt content.

## Example Usage

```hcl
resource "litellm_prompt" "langfuse_prompt" {
  prompt_id          = "my-langfuse-prompt"
  prompt_integration = "langfuse"
  api_base           = "https://cloud.langfuse.com"
  api_key            = var.langfuse_api_key
  prompt_type        = "db"

  litellm_params = jsonencode({
    prompt_id = "prompt-name-in-langfuse"
  })
}
```

### Dotprompt

```hcl
resource "litellm_prompt" "greeting" {
  prompt_id          = "greeting"
  prompt_integration = "dotprompt"
  prompt_type        = "db"

  dotprompt_content = <<-EOT
    ---
    model: gpt-5.2
    ---
    Say hello to {{name}}.
  EOT
}
```

## Argument Reference

* `prompt_id` - (Required, Forces new resource) Unique identifier for the prompt.
* `prompt_integration` - (Required) The prompt integration provider (e.g. `langfuse`, `dotprompt`).
* `api_base` - (Optional) Base URL for the prompt provider API.
* `api_key` - (Optional, Sensitive) API key for the prompt provider. Never read back into state.
* `provider_specific_query_params` - (Optional) JSON string of provider-specific query parameters.
* `ignore_prompt_manager_model` - (Optional) If true, ignore the model specified in the prompt manager.
* `ignore_prompt_manager_optional_params` - (Optional) If true, ignore optional params from the prompt manager.
* `dotprompt_content` - (Optional) Content for the dotprompt integration.
* `litellm_params` - (Optional, Sensitive) JSON string with additional `litellm_params` merged into the request, e.g. the integration's own `prompt_id`, `prompt_directory` or `prompt_data`. Never read back into state.
* `prompt_type` - (Optional) Type of prompt: `config` or `db`.

## Attribute Reference

* `id` - The prompt ID (same as `prompt_id`).

## Import

Prompts can be imported using the prompt ID:

```shell
terraform import litellm_prompt.example my-langfuse-prompt
```
