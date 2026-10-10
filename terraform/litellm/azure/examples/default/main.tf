# One-command deploy of the LiteLLM Azure stack.
#
#   cd terraform/litellm/azure/examples/default
#   cp terraform.tfvars.example terraform.tfvars   # edit it
#   terraform init
#   terraform apply
#
# This root just wires the provider (see providers.tf) to the module. The
# module itself (../../) declares no provider, so it can also be consumed
# from your own config with count/for_each/aliased providers:
#
#   module "litellm" {
#     source  = "github.com/BerriAI/litellm//terraform/litellm/azure?ref=<tag>"
#     ...
#   }
#
# Knobs not surfaced as variables here (per-component sizing, autoscaling,
# Postgres / Redis / Storage tuning) can be set directly on this block:
# see ../../variables.tf.

module "litellm" {
  source = "../../"

  location = var.location
  tenant   = var.tenant
  env      = var.env
  azs      = var.azs

  litellm_master_key = var.litellm_master_key
  litellm_license    = var.litellm_license
  ui_password        = var.ui_password

  key_vault_certificate_id    = var.key_vault_certificate_id
  allow_plaintext_app_gateway = var.allow_plaintext_app_gateway

  storage_force_destroy = var.storage_force_destroy

  proxy_config          = var.proxy_config
  gateway_extra_env     = {}
  backend_extra_env     = {}
  gateway_extra_secrets = {}
  backend_extra_secrets = {}
}
