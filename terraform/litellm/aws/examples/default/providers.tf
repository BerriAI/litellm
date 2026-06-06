# The provider is configured HERE, in the root, not in the module. That is
# the whole point of the split: a module that declares its own configured
# `provider` block can't be called with count/for_each/depends_on and gives
# the caller no way to set assume-role, custom endpoints, or aliases.
#
# `default_tags` set here still flow into every resource the module creates
# (provider default_tags propagate through module calls) and merge with the
# module's own `litellm:stack` / `managed-by` / var.tags. Use this block for
# org-wide tags; use the module's `tags` input for per-deployment tags.
provider "aws" {
  region = var.region

  # Reserve `default_tags` for pure org-wide tags the module shouldn't know
  # about (cost center, team, compliance scope, …). They propagate through the
  # module call and merge with the module's own `litellm:stack` / `managed-by`
  # / var.tags. The module already stamps `managed-by = "terraform"`, so don't
  # duplicate it here — set per-deployment tags via the module's `tags` input.
  #
  # default_tags {
  #   tags = {
  #     "cost-center" = "platform"
  #   }
  # }
}
