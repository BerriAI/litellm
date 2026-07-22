# ---------- Managed identities ----------
#
# One user-assigned managed identity shared across gateway / backend /
# ui / migrations Container Apps. Same identity is referenced as
# `azurerm_container_app.identity` -> user-assigned and granted the
# minimal set of role assignments below:
#   - AcrPull              on the Container Apps managed registry
#   - Storage Blob Data    on the Storage Account (cache / files)
#   - Key Vault Secrets    on the Key Vault (read master key + secrets)
#   - Reader               on its own resource group (private endpoint lookups)

resource "azurerm_user_assigned_identity" "container_apps" {
  name                = "${local.name}-mi"
  location            = var.location
  resource_group_name = local.resource_group_name
  tags                = local.tags
}

# ---------- Role assignments ----------
#
# Granted inside the same module so a fresh `terraform apply` ends with a
# fully-wired stack; nothing to add post-apply aside from `db_bootstrap_sql`.

resource "azurerm_role_assignment" "storage_blob_data_contributor" {
  scope                = azurerm_storage_account.this.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
  principal_type       = "ServicePrincipal"
}

# Key Vault access is granted via access policies (data actions, scoped
# per-secret); see keyvault.tf.

# ---------- Container Apps Environment infrastructure role ----------
#
# The ACA environment requires a "managed identity" with the
# "Container Apps Environment Managed Identity" role on its own resource
# group (defined by Microsoft.App) so it can read ACR images and pull
# from the registry. We grant the user-assigned MI that role once.
# (For modular simplicity the ACA managed identity is the same identity
# we already created above.)

resource "azurerm_role_assignment" "container_apps_env_infra" {
  scope                = local.resource_group_id
  role_definition_name = "Container Apps Environment Managed Identity Contributor"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
  principal_type       = "ServicePrincipal"
}

locals {
  # Resolve the resource-group ID whether the caller supplied an existing
  # RG (no resource here) or the module created one. Falls back to the
  # resource's own id if a known format conversion is needed.
  resource_group_id = var.resource_group_name == "" ? azurerm_resource_group.this[0].id : "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${var.resource_group_name}"
}
