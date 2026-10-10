variable "location" {
  description = "Azure region to deploy into."
  type        = string
  default     = "eastus"
}

variable "tenant" {
  description = "Tenant slug used as the prefix for every Azure resource the stack creates."
  type        = string
  default     = "acme"
}

variable "env" {
  description = "Environment suffix (e.g. `stage`, `prod`, `dev`)."
  type        = string
  default     = "stage"
}

variable "azs" {
  description = "Azure availability zone identifiers (e.g. [\"1\", \"2\"])."
  type        = list(string)
  default     = ["1", "2"]
}

variable "litellm_master_key" {
  description = "Pre-existing LiteLLM master key (must begin `sk-`). Leave empty to have the stack auto-generate."
  type        = string
  default     = ""
  sensitive   = true
}

variable "litellm_license" {
  description = "Optional LiteLLM enterprise license."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ui_password" {
  description = "Optional UI admin password."
  type        = string
  default     = ""
  sensitive   = true
}

variable "key_vault_certificate_id" {
  description = "Resource ID of an existing Key Vault certificate for the App Gateway HTTPS listener. Leave empty to deploy HTTP-only (set `allow_plaintext_app_gateway = true` too)."
  type        = string
  default     = ""
}

variable "allow_plaintext_app_gateway" {
  description = "Allow the Application Gateway to serve HTTP without TLS. Dev / trial only."
  type        = bool
  default     = true
}

variable "storage_force_destroy" {
  description = "Allow `terraform destroy` to remove a non-empty storage container."
  type        = bool
  default     = false
}

variable "proxy_config" {
  description = "LiteLLM proxy config map (mirrors helm chart gateway.config.proxy_config)."
  type        = any
  default = {
    model_list = [
      {
        model_name = "gpt-4o"
        litellm_params = {
          model   = "openai/gpt-4o"
          api_key = "os.environ/OPENAI_API_KEY"
        }
      },
    ]
    general_settings = {
      master_key   = "os.environ/LITELLM_MASTER_KEY"
      database_url = "os.environ/DATABASE_URL"
    }
  }
}
