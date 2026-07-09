# Azure Database for PostgreSQL Flexible Server, single instance, with
# Entra (Azure AD) authentication enabled for the proxy's managed
# identity. (Single zone by default; set `zone = "1"` plus a high-
# availability SKU for zone redundancy.)
#
# The bootstrap job (run once during initial setup, see bootstrap.tf)
# creates the literal-login user used for password-based app access;
# the gateway / backend / migrations Container Apps use Entra tokens at
# runtime.

resource "azurerm_postgresql_flexible_server" "this" {
  name                = replace("${var.tenant}-${var.env}-pg", "-", "")
  location            = var.location
  resource_group_name = local.resource_group_name
  sku_name            = var.db_sku_name
  storage_mb          = var.db_storage_mb
  version             = var.db_version
  tags                = local.tags

  # The AAD admin: this object's principal can connect to the server and
  # bootstrap the application user. Defaults to the current terraform
  # principal; override var.db_ad_admin_object_id for a dedicated admin.
  administrator_login    = "litellm_admin"
  administrator_password = random_password.db_admin_password.result

  authentication {
    password_auth_enabled         = true
    active_directory_auth_enabled = true
  }

  # Default to private access in the VNet; public access disabled.
  public_network_access_enabled = false
  delegated_subnet_id           = azurerm_subnet.private_endpoints.id
  private_dns_zone_id           = azurerm_private_dns_zone.postgres.id

  # Zone redundancy is off by default to control cost; flip on via
  # `zone = "1"` + matching SKU.
  zone = length(var.azs) > 1 ? var.azs[0] : null

  depends_on = [
    azurerm_role_assignment.container_apps_env_infra,
    azurerm_private_dns_zone_virtual_network_link.postgres,
  ]
}

resource "azurerm_postgresql_flexible_server_database" "this" {
  server_id = azurerm_postgresql_flexible_server.this.id
  name      = var.db_name
  collation = "en_US.utf8"
  charset   = "UTF8"
}

# Set the AAD admin via a separate resource. The first principal in the
# list wins; we use the current terraform principal by default.
resource "azurerm_postgresql_flexible_server_active_directory_administrator" "this" {
  server_name         = azurerm_postgresql_flexible_server.this.name
  resource_group_name = local.resource_group_name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  object_id           = data.azurerm_client_config.current.object_id
  principal_name      = "terraform-admin"
  principal_type      = "User"

  depends_on = [azurerm_postgresql_flexible_server.this]
}

resource "azurerm_private_dns_zone" "postgres" {
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = local.resource_group_name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "${local.name}-pg-dnslink"
  resource_group_name   = local.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.this.id
  registration_enabled  = false
}
