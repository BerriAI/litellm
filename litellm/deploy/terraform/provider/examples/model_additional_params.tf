provider "litellm" {
  api_base = "https://your-litellm-proxy.com"
  api_key  = var.litellm_api_key
}

# Example: using additional_litellm_params to pass provider-specific options.
# Notes:
# - String values "true"/"false" will be coerced to booleans.
# - Numeric strings will be parsed to integer (if possible) otherwise float.
# - JSON strings (starting with [ or {) will be parsed as JSON objects/arrays.
# - Non-convertible strings remain strings.
# - "additional_drop_params" takes a JSON array of request parameters the proxy drops before calling the model provider.

resource "litellm_model" "with_additional" {
  model_name          = "custom-model"
  custom_llm_provider = "openai"
  model_api_key       = var.openai_api_key
  base_model          = "gpt-4"
  mode                = "chat"

  # Additional parameters not exposed as first-class arguments
  additional_litellm_params = {
    "use_fine_tune"          = "true"                          # becomes boolean true
    "max_context"            = "16384"                         # becomes integer 16384
    "temperature_scale"      = "0.75"                          # becomes float 0.75
    "experimental_feature"   = "enabled"                       # stays string "enabled"
    "complex_config"         = "{\"nested\": {\"value\": 42}}" # parsed as JSON object
    "additional_drop_params" = "[\"reasoning_effort\"]"        # proxy drops reasoning_effort from requests
  }

  # Cost configuration (optional)
  input_cost_per_million_tokens  = 30.0
  output_cost_per_million_tokens = 60.0
}

# Example: Azure model with parameter dropping
resource "litellm_model" "azure_with_drop_params" {
  model_name          = "gpt-5-mini-coder"
  custom_llm_provider = "azure"
  model_api_key       = "your-azure-api-key"
  model_api_base      = "https://your-azure-endpoint.openai.azure.com/"
  api_version         = "2025-03-01-preview"
  base_model          = "gpt-5-mini"
  tier                = "paid"
  mode                = "completion"

  # Have the proxy drop reasoning_effort from requests before they reach Azure
  additional_litellm_params = {
    "additional_drop_params" = "[\"reasoning_effort\"]"
  }

  input_cost_per_million_tokens  = 0.25
  output_cost_per_million_tokens = 2.00
}
