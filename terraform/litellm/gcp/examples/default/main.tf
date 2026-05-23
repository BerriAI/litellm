# One-command deploy of the LiteLLM GCP stack.
#
#   cd terraform/litellm/gcp/examples/default
#   cp terraform.tfvars.example terraform.tfvars   # edit it
#   terraform init
#   terraform apply
#
# This root just wires the providers (see providers.tf) to the module. The
# module itself (../../) declares no provider, so it can also be consumed
# from your own config with count/for_each or impersonated-SA providers:
#
#   module "litellm" {
#     source  = "github.com/BerriAI/litellm//terraform/litellm/gcp?ref=<tag>"
#     ...
#   }
#
# Knobs not surfaced as variables here (per-component sizing/instances,
# Cloud SQL tier/edition, Memorystore tier, per-component image overrides)
# can be set directly on this block — see ../../variables.tf.

# Resolve the effective project: explicit var.project wins, otherwise fall
# back to whatever the provider inferred from gcloud / ADC (Cloud Shell sets
# this). The module needs a concrete project ID for project-scoped IAM.
data "google_client_config" "current" {}

locals {
  project = var.project != "" ? var.project : data.google_client_config.current.project
}

module "litellm" {
  source = "../../"

  project = local.project
  region  = var.region
  tenant  = var.tenant
  env     = var.env

  litellm_master_key = var.litellm_master_key
  litellm_license    = var.litellm_license
  ui_password        = var.ui_password

  image_registry = var.image_registry
  image_tag      = var.image_tag

  lb_domains                   = var.lb_domains
  allow_plaintext_lb           = var.allow_plaintext_lb
  cloudsql_deletion_protection = var.cloudsql_deletion_protection
  gcs_force_destroy            = var.gcs_force_destroy

  proxy_config          = var.proxy_config
  gateway_extra_env     = var.gateway_extra_env
  backend_extra_env     = var.backend_extra_env
  gateway_extra_secrets = var.gateway_extra_secrets
  backend_extra_secrets = var.backend_extra_secrets
}
