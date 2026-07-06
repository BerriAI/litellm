# litellm_model Resource

Manages a LiteLLM model configuration. This resource allows you to create, update, and delete model configurations in your LiteLLM instance.

## Example Usage

### Basic OpenAI Model

```hcl
resource "litellm_model" "gpt4" {
  model_name          = "gpt-4-proxy"
  custom_llm_provider = "openai"
  model_api_key       = var.openai_api_key
  base_model          = "gpt-4"
  tier                = "paid"
  mode                = "chat"
  
  input_cost_per_million_tokens  = 30.0
  output_cost_per_million_tokens = 60.0
}
```

### Advanced Model with All Features

```hcl
resource "litellm_model" "advanced_gpt4" {
  model_name          = "gpt-4-advanced"
  custom_llm_provider = "openai"
  model_api_key       = var.openai_api_key
  model_api_base      = "https://api.openai.com/v1"
  api_version         = "2023-05-15"
  base_model          = "gpt-4"
  tier                = "paid"
  team_id             = "team-123"
  mode                = "chat"
  reasoning_effort    = "medium"
  thinking_enabled    = true
  thinking_budget_tokens = 1024
  merge_reasoning_content_in_choices = true
  tpm                 = 100000
  rpm                 = 1000
  
  # Cost configuration (per million tokens)
  input_cost_per_million_tokens  = 30.0    # $0.03 per 1k tokens = $30 per million
  output_cost_per_million_tokens = 60.0    # $0.06 per 1k tokens = $60 per million
}
```

### AWS Bedrock Model with Cross-Account Access

```hcl
resource "litellm_model" "bedrock_claude" {
  model_name          = "bedrock-claude-proxy"
  custom_llm_provider = "bedrock"
  base_model          = "anthropic.claude-3-sonnet-20240229-v1:0"
  tier                = "paid"
  mode                = "chat"
  
  # AWS configuration with cross-account access
  aws_access_key_id     = var.aws_access_key_id
  aws_secret_access_key = var.aws_secret_access_key
  aws_region_name       = "us-east-1"
  aws_session_name      = "litellm-cross-account-session"
  aws_role_name         = "arn:aws:iam::123456789012:role/LiteLLMCrossAccountRole"
  
  input_cost_per_million_tokens  = 3.0
  output_cost_per_million_tokens = 15.0
}
```

### Anthropic Model

```hcl
resource "litellm_model" "claude" {
  model_name          = "claude-proxy"
  custom_llm_provider = "anthropic"
  model_api_key       = var.anthropic_api_key
  base_model          = "claude-3-sonnet-20240229"
  tier                = "paid"
  mode                = "chat"
  
  input_cost_per_million_tokens  = 3.0
  output_cost_per_million_tokens = 15.0
}
```

### Azure OpenAI Model

```hcl
resource "litellm_model" "azure_gpt4" {
  model_name          = "azure-gpt4-proxy"
  custom_llm_provider = "azure"
  model_api_key       = var.azure_openai_key
  model_api_base      = var.azure_openai_endpoint
  api_version         = "2023-12-01-preview"
  base_model          = "gpt-4"
  tier                = "paid"
  mode                = "chat"
  
  input_cost_per_million_tokens  = 30.0
  output_cost_per_million_tokens = 60.0
}
```

## Argument Reference

The following arguments are supported:

* `model_name` - (Required) string. The name of the model configuration used to identify the model in API calls.

* `custom_llm_provider` - (Required) string. The LLM provider for this model (e.g., "openai", "anthropic", "azure", "bedrock").

* `model_api_key` - (Optional) string (Sensitive). The API key for the underlying model provider.

* `model_api_base` - (Optional) string. The base URL for the model provider's API.

* `api_version` - (Optional) string. The API version to use for the model provider.

* `base_model` - (Required) string. The actual model identifier from the provider (e.g., "gpt-4", "claude-2").

* `litellm_credential_name` - (Optional) string. Name of a LiteLLM credential to use for this model.

* `tier` - (Optional) string. The usage tier for this model. Valid values are `"free"` or `"paid"`. Default: `"free"`.

* `team_id` - (Optional) string. Associate the model with a specific team.

* `mode` - (Optional) string. The intended use of the model. Valid values are:
  * `completion`
  * `embedding`
  * `image_generation`
  * `chat`
  * `moderation`
  * `audio_transcription`
  * `audio_speech`
  * `rerank`

* `tpm` - (Optional) integer. Tokens per minute limit for this model.

* `rpm` - (Optional) integer. Requests per minute limit for this model.

* `reasoning_effort` - (Optional) string. Configures the model's reasoning effort level. Valid values are:
  * `low`
  * `medium`
  * `high`

* `thinking_enabled` - (Optional) boolean. Enables the model's thinking capability. Default: `false`.

* `thinking_budget_tokens` - (Optional) integer. Sets the token budget for the model's thinking capability. Default: `1024`. Note: this field is only relevant when `thinking_enabled = true`.

* `merge_reasoning_content_in_choices` - (Optional) boolean. When set to `true`, merges reasoning content into the model's choices.

* `input_cost_per_million_tokens` - (Optional) float. Cost per million input tokens. The provider converts this to a per-token cost sent to the API.

* `output_cost_per_million_tokens` - (Optional) float. Cost per million output tokens. The provider converts this to a per-token cost sent to the API.

* `input_cost_per_pixel` - (Optional) float. Cost applied per input pixel for models that charge by image size.

* `output_cost_per_pixel` - (Optional) float. Cost applied per output pixel for image-generation models.

* `input_cost_per_second` - (Optional) float. Cost applied per input second for audio/transcription models.

* `output_cost_per_second` - (Optional) float. Cost applied per output second for audio/transcription models.

* `vertex_project` - (Optional) string. Vertex AI project id (for `custom_llm_provider = "vertex"`).

* `vertex_location` - (Optional) string. Vertex AI location (e.g., `us-central1`).

* `vertex_credentials` - (Optional) string. Vertex credentials (JSON string or path depending on your setup).

* `additional_litellm_params` - (Optional) map(string). A map of arbitrary additional parameters that will be merged into the `litellm_params` object sent to the LiteLLM API. This is intended for provider-specific or experimental options not exposed as dedicated arguments.

  Conversion and behavior rules (how the provider handles values):
  * When values in the map are strings the provider will attempt to coerce them:
    * `"true"` / `"false"` (strings) -> boolean true / false
    * Numeric strings are parsed first as integers; if integer parsing fails, parsed as floats (e.g., `"16384"` -> 16384, `"0.75"` -> 0.75)
    * JSON strings (starting with `[` or `{`) are parsed as JSON objects/arrays
    * Non-convertible strings remain strings
  * Non-string map values (if supplied) are passed through unchanged.
  * The provider merges these keys into the `litellm_params` payload sent to the API.
  * Note: the remote API may not echo back all custom parameters; this provider preserves `additional_litellm_params` in state when present in configuration.

  **Special parameter: `additional_drop_params`**
  * When `additional_drop_params` is provided as a JSON array string, it specifies parameters to remove from the final `litellm_params` before sending to the API
  * This allows you to override or remove built-in parameters if needed
  * The `additional_drop_params` key itself is not included in the final parameters

  Example showing booleans, integers, floats, strings, and parameter dropping:

  ```hcl
  resource "litellm_model" "with_additional" {
    model_name          = "custom-model"
    custom_llm_provider = "openai"
    model_api_key       = var.openai_api_key
    base_model          = "gpt-4"
    mode                = "chat"

    additional_litellm_params = {
      "use_fine_tune"          = "true"                    # becomes boolean true
      "max_context"            = "16384"                   # becomes integer 16384
      "scale"                  = "0.75"                    # becomes float 0.75
      "note"                   = "for testing"             # stays string
      "complex_config"         = "{\"nested\": {\"value\": 42}}"  # parsed as JSON object
      "additional_drop_params" = "[\"reasoningEffort\"]"   # removes reasoningEffort parameter
    }
  }
  ```

### AWS-specific Configuration

* `aws_access_key_id` - (Optional) string (Sensitive). AWS access key ID for AWS-based models.

* `aws_secret_access_key` - (Optional) string (Sensitive). AWS secret access key for AWS-based models.

* `aws_region_name` - (Optional) string. AWS region name for AWS-based models.

* `aws_session_name` - (Optional) string (Sensitive). AWS session name for cross-account access scenarios.

* `aws_role_name` - (Optional) string (Sensitive). AWS IAM role name for cross-account access scenarios.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The ID of the model configuration.

## Import

Model configurations can be imported using the model ID:

```shell
terraform import litellm_model.gpt4 <model-id>
```

Note: The model ID is generated when the model is created and is different from the `model_name`.

## Security Note

When using this resource, ensure that sensitive information such as API keys and AWS credentials are stored securely. It's recommended to use environment variables or a secure secret management solution rather than hardcoding these values in your Terraform configuration files.
