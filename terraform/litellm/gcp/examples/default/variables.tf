# Curated surface for the one-command deploy path. The module (../../)
# exposes far more knobs (per-component CPU/memory/instances, Cloud SQL
# tier/edition, Memorystore tier, per-component image overrides, …). To
# tune those, set them directly on the `module "litellm"` block in
# main.tf, or call the module from your own root config. Full per-variable
# docs live in ../../variables.tf — the module is the source of truth.

variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region for VPC, Cloud SQL, Memorystore, Cloud Run, and the LB IP."
  type        = string
  default     = "us-central1"
}

variable "tenant" {
  description = "Tenant slug — prefix for every resource (<tenant>-litellm-<env>)."
  type        = string
}

variable "env" {
  description = "Environment suffix (stage, prod, dev)."
  type        = string
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

# Image source. Cloud Run rejects ghcr.io, so a real deploy must point
# image_registry at an Artifact Registry remote repo (see README "Image
# pulls"). Per-component overrides live in ../../variables.tf.
variable "image_registry" {
  description = "Registry path prefix; images composed as <image_registry>/litellm-<component>:<image_tag>."
  type        = string
  default     = "ghcr.io/berriai"
}

variable "image_tag" {
  description = "Tag applied to all four litellm-* images. Bump in lockstep."
  type        = string
  default     = "v1.86.0-dev"
}

# TLS — provide DNS names for a managed cert, or opt into HTTP-only for dev.
variable "lb_domains" {
  description = "DNS names (already pointing at lb_ip) for a Google-managed cert. Empty → no TLS."
  type        = list(string)
  default     = []
}

variable "allow_plaintext_lb" {
  description = "Opt into HTTP-only LB (trial/dev only)."
  type        = bool
  default     = false
}

variable "cloudsql_deletion_protection" {
  description = "Cloud SQL deletion protection (writer + reader)."
  type        = bool
  default     = true
}

variable "gcs_force_destroy" {
  description = "Allow destroy of a non-empty GCS bucket (ephemeral/CI only)."
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
  description = "Gateway env vars sourced from Secret Manager (name → secret resource ID)."
  type        = map(string)
  default     = {}
}

variable "backend_extra_secrets" {
  description = "Backend env vars sourced from Secret Manager (name → secret resource ID)."
  type        = map(string)
  default     = {}
}
