# One-command deploy of the LiteLLM AWS stack.
#
#   cd terraform/litellm/aws/examples/default
#   cp terraform.tfvars.example terraform.tfvars   # edit it
#   terraform init
#   terraform apply
#
# This root just wires the provider (see providers.tf) to the module. The
# module itself (../../) declares no provider, so it can also be consumed
# from your own config with count/for_each/aliased or assume-role providers:
#
#   module "litellm" {
#     source  = "github.com/BerriAI/litellm//terraform/litellm/aws?ref=<tag>"
#     ...
#   }
#
# Knobs not surfaced as variables here (per-component sizing, autoscaling,
# RDS/Redis tuning) can be set directly on this block — see ../../variables.tf.

# When azs is left empty, pick the first two AZs in the region so a
# zero-config apply works in any region without the caller naming them.
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = length(var.azs) > 0 ? var.azs : slice(data.aws_availability_zones.available.names, 0, 2)
}

module "litellm" {
  source = "../../"

  region = var.region
  tenant = var.tenant
  env    = var.env
  azs    = local.azs

  litellm_master_key = var.litellm_master_key
  litellm_license    = var.litellm_license
  ui_password        = var.ui_password

  acm_certificate_arn = var.acm_certificate_arn
  allow_plaintext_alb = var.allow_plaintext_alb
  s3_force_destroy    = var.s3_force_destroy

  proxy_config          = var.proxy_config
  gateway_extra_env     = var.gateway_extra_env
  backend_extra_env     = var.backend_extra_env
  gateway_extra_secrets = var.gateway_extra_secrets
  backend_extra_secrets = var.backend_extra_secrets
}
