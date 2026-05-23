# Providers are configured HERE, in the root, not in the module. A module
# that declares its own configured `provider` block can't be called with
# count/for_each/depends_on and gives the caller no way to set an
# impersonated service account, a different project, or aliases.
#
# The module's resources inherit these default (unaliased) `google` /
# `google-beta` configs automatically through the module call, so project
# and region set here flow into every resource that doesn't pass its own.
# project = null when var.project is empty, which lets the provider infer
# the project from the active gcloud config / ADC (set in Cloud Shell). The
# resolved value is read back via data.google_client_config in main.tf and
# passed explicitly to the module.
provider "google" {
  project = var.project != "" ? var.project : null
  region  = var.region
}

provider "google-beta" {
  project = var.project != "" ? var.project : null
  region  = var.region
}
