# Curated surface for the one-command deploy path. The module (../../)
# exposes far more knobs (per-component CPU/memory, autoscaling, RDS/Redis
# sizing, …). To tune those, set them directly on the `module "litellm"`
# block in main.tf, or call the module from your own root config. Full
# per-variable docs live in ../../variables.tf — the module is the source
# of truth; descriptions here are intentionally terse.

variable "region" {
  description = "AWS region to deploy into."
  type        = string
}

variable "tenant" {
  description = "Tenant slug — prefix for every resource (<tenant>-litellm-<env>)."
  type        = string
}

variable "env" {
  description = "Environment suffix (stage, prod, dev)."
  type        = string
}

variable "azs" {
  description = "Availability zones for subnets. At least 2 (RDS + ALB)."
  type        = list(string)
}

# Sensitive — prefer TF_VAR_litellm_master_key / TF_VAR_litellm_license /
# TF_VAR_ui_password so values stay out of any committed tfvars file.
variable "litellm_master_key" {
  description = "Pre-existing LITELLM_MASTER_KEY (sk-…). Empty → auto-generated."
  type        = string
  default     = ""
  sensitive   = true
}

variable "litellm_license" {
  description = "LiteLLM enterprise license. Empty → OSS-only."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ui_password" {
  description = "UI admin password. Empty → falls back to LITELLM_MASTER_KEY."
  type        = string
  default     = ""
  sensitive   = true
}

# TLS — provide an ACM cert for production, or opt into HTTP-only for dev.
variable "acm_certificate_arn" {
  description = "ACM cert ARN for the ALB HTTPS listener. Empty → no TLS."
  type        = string
  default     = ""
}

variable "allow_plaintext_alb" {
  description = "Opt into HTTP-only ALB (trial/dev only)."
  type        = bool
  default     = false
}

variable "s3_force_destroy" {
  description = "Allow destroy of a non-empty S3 bucket (ephemeral/CI only)."
  type        = bool
  default     = false
}

variable "skip_final_snapshot" {
  description = "Skip the Aurora final snapshot on destroy (ephemeral/CI only)."
  type        = bool
  default     = false
}

variable "proxy_config" {
  description = "LiteLLM proxy config (contents of config.yaml). Empty → defaults."
  type        = any
  default     = {}
}

variable "gateway_extra_env" {
  description = "Plain-text env vars layered onto the gateway."
  type        = map(string)
  default     = {}
}

variable "backend_extra_env" {
  description = "Plain-text env vars layered onto the backend."
  type        = map(string)
  default     = {}
}

variable "gateway_extra_secrets" {
  description = "Gateway env vars sourced from Secrets Manager (name → ARN)."
  type        = map(string)
  default     = {}
}

variable "backend_extra_secrets" {
  description = "Backend env vars sourced from Secrets Manager (name → ARN)."
  type        = map(string)
  default     = {}
}
