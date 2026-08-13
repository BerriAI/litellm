# LiteLLM Provider

The LiteLLM provider allows Terraform to manage LiteLLM resources. LiteLLM is a proxy service that standardizes the input/output across different LLM APIs, providing a unified interface for various language model providers.

## Example Usage

```hcl
terraform {
  required_providers {
    litellm = {
      source = "registry.terraform.io/BerriAI/litellm"
    }
  }
}

provider "litellm" {
  api_base = "https://your-litellm-proxy.com"
  api_key  = var.litellm_api_key
}

# Basic model configuration
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

# Team configuration
resource "litellm_team" "dev_team" {
  team_alias = "development-team"
  models     = [litellm_model.gpt4.model_name]
  max_budget = 100.0
}
```

## Available Resources

The LiteLLM provider supports the following resources:

* [`litellm_model`](./resources/model) - Manage LiteLLM model configurations
* [`litellm_team`](./resources/team) - Manage teams and their permissions
* [`litellm_team_member`](./resources/team_member) - Manage team member configurations
* [`litellm_team_member_add`](./resources/team_member_add) - Add members to teams
* [`litellm_key`](./resources/key) - Manage API keys
* [`litellm_mcp_server`](./resources/mcp_server) - Manage MCP (Model Context Protocol) servers
* [`litellm_credential`](./resources/credential) - Manage credentials for various providers
* [`litellm_vector_store`](./resources/vector_store) - Manage vector stores

## Available Data Sources

The LiteLLM provider supports the following data sources:

* [`litellm_credential`](./data-sources/credential) - Retrieve credential information
* [`litellm_vector_store`](./data-sources/vector_store) - Retrieve vector store information

## Authentication

The LiteLLM provider requires an API key and base URL for authentication. These can be provided in the provider configuration block or via environment variables.

### Environment Variables

- `LITELLM_API_BASE` - The base URL of your LiteLLM instance
- `LITELLM_API_KEY` - Your LiteLLM API key

### Example with Environment Variables

```bash
export LITELLM_API_BASE="https://your-litellm-proxy.com"
export LITELLM_API_KEY="your-api-key"
```

```hcl
terraform {
  required_providers {
    litellm = {
      source = "registry.terraform.io/BerriAI/litellm"
    }
  }
}

# Provider will automatically use environment variables
provider "litellm" {}
```

## Provider Arguments

The following arguments are supported in the provider block:

* `api_base` - (Required) The base URL of your LiteLLM instance. This can also be provided via the `LITELLM_API_BASE` environment variable.
* `api_key` - (Required) The API key used to authenticate with LiteLLM. This can also be provided via the `LITELLM_API_KEY` environment variable.

## Getting Started

1. Install the provider by adding it to your Terraform configuration
2. Configure your LiteLLM instance URL and API key
3. Start creating resources like models, teams, and credentials
4. Use data sources to reference existing configurations

For detailed examples and configuration options, see the individual resource and data source documentation pages.

## Examples

This repository includes an `examples/` directory with curated, ready-to-run HCL examples that demonstrate common and advanced usages of the provider. Examples are grouped by resource and illustrate provider-specific configuration, handling of sensitive values, and advanced options such as `additional_litellm_params`.

See:
* `examples/model_additional_params.tf` — demonstrates how to use `additional_litellm_params` (booleans, integers, floats, and strings).
* Other example files will be added to `examples/` for credentials, vector stores, and MCP servers.

You can reference these examples directly or copy snippets into your Terraform configurations for quick starts.

For detailed examples and configuration options, see the individual resource and data source documentation pages.
