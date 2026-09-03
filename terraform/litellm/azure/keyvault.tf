# Key Vault holds the proxy's secret material:
#   - master-key secret (LITELLM_MASTER_KEY)
#   - optional license (LITELLM_LICENSE)
#   - optional UI password (UI_PASSWORD)
#   - Postgres admin password (PG_ADMIN_PASSWORD) for the bootstrap step
#
# The Container Apps managed identity is granted per-secret access via
# access policies. Caller-supplied secrets (`gateway_extra_secrets`,
# `backend_extra_secrets`) are referenced by their secret IDs and not
# stored in this module; callers wire provider keys with their own
# `azurerm_key_vault_secret` resources (or supply them externally).

resource "random_password" "db_admin_password" {
  length           = 32
  special          = true
  override_special = "@%*_+-:?#"
}

resource "random_password" "litellm_master_key" {
  length  = 43
  special = false
}

# Generate the master key with the `sk-` prefix the proxy expects.
locals {
  litellm_master_key_value = var.litellm_master_key != "" ? var.litellm_master_key : "sk-${substr(random_password.litellm_master_key.result, 0, 43)}"
}

resource "azurerm_key_vault" "this" {
  name                = replace("${var.tenant}${var.env}kv", "-", "") # no dashes allowed in KV name; truncated by Azure
  location            = var.location
  resource_group_name = local.resource_group_name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  tags                = local.tags

  # Networking: lock down to the VNet's private endpoint subnet. Caller's
  # terraform principal needs to also be granted access to add secrets.
  public_network_access_enabled   = false
  enable_rbac_authorization       = false
  enabled_for_deployment          = false
  enabled_for_disk_encryption     = false
  enabled_for_template_deployment = false
  purge_protection_enabled        = true
  soft_delete_retention_days      = 7

  network_acls {
    bypass         = "AzureServices"
    default_action = "Deny"
    ip_rules       = []
    virtual_network_subnet_ids = [
      azurerm_subnet.private_endpoints.id,
    ]
  }
}

# ---------- Key Vault access policies ----------
#
# Three identities get secret access:
#   1. The Container Apps managed identity (read-only on all secrets)
#   2. The current terraform principal (so `terraform apply` can add
#      secrets during creation; matches the AWS module's bootstrap pattern)
#   3. Possibly AAD admins for break-glass (not configured by default)

resource "azurerm_key_vault_access_policy" "container_apps" {
  key_vault_id = azurerm_key_vault.this.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.container_apps.principal_id

  secret_permissions = [
    "Get", "List",
  ]
}

resource "azurerm_key_vault_access_policy" "terraform" {
  key_vault_id = azurerm_key_vault.this.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Get", "List", "Set", "Delete", "Purge", "Recover",
  ]
}

# ---------- Secrets ----------
#
# The DB admin password lives in the vault and is consumed by the
# bootstrap job (which runs `psql` once to create the Entra token-auth
# app user; see `bootstrap.tf` / `iam.tf`). The proxy itself never sees
# it; gateway/backend/migrations use Entra tokens via the Container
# Apps managed identity.

resource "azurerm_key_vault_secret" "db_admin_password" {
  name         = "db-admin-password"
  value        = random_password.db_admin_password.result
  key_vault_id = azurerm_key_vault.this.id

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "master_key" {
  name         = "litellm-master-key"
  value        = local.litellm_master_key_value
  key_vault_id = azurerm_key_vault.this.id
  tags         = local.tags

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "license" {
  count        = var.litellm_license != "" ? 1 : 0
  name         = "litellm-license"
  value        = var.litellm_license
  key_vault_id = azurerm_key_vault.this.id
  tags         = local.tags

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

resource "azurerm_key_vault_secret" "ui_password" {
  count        = var.ui_password != "" ? 1 : 0
  name         = "ui-password"
  value        = var.ui_password
  key_vault_id = azurerm_key_vault.this.id
  tags         = local.tags

  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# ---------- Private endpoint ----------
#
# Pulls the vault onto a private IP in `private_endpoints` subnet so the
# Container Apps don't need public access to reach secrets.

resource "azurerm_private_endpoint" "keyvault" {
  name                = "${local.name}-kv-pe"
  location            = var.location
  resource_group_name = local.resource_group_name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "${local.name}-kv"
    private_connection_resource_id = azurerm_key_vault.this.id
    is_manual_connection           = false
    subresource_names              = ["vault"]
  }

  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.keyvault.id]
  }

  depends_on = [azurerm_private_dns_zone_virtual_network_link.keyvault]
}

resource "azurerm_private_dns_zone" "keyvault" {
  name                = "privatelink.vaultcore.azure.net"
  resource_group_name = local.resource_group_name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "keyvault" {
  name                  = "${local.name}-kv-dnslink"
  resource_group_name   = local.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.keyvault.name
  virtual_network_id    = azurerm_virtual_network.this.id
  registration_enabled  = false
}
