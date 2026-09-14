# Storage Account: holds:
#   - proxy_config.yaml (uploaded from var.proxy_config; refreshed on
#     proxy_config changes to trigger Container App revision swap)
#   - /v1/files passthrough storage (request log archival, file uploads)
#   - log archive bucket (optional, future)
#
# LiteLLM reads `AZURE_STORAGE_ACCOUNT_NAME` and uses the Container Apps
# managed identity's Storage Blob Data Contributor role on this account
# to do CRUD with the SDK. No shared keys are issued.

resource "azurerm_storage_account" "this" {
  name                     = replace("${var.tenant}${var.env}proxy", "-", "") # 24 char cap; lowercase + no dashes
  location                 = var.location
  resource_group_name      = local.resource_group_name
  account_tier             = var.storage_account_tier
  account_replication_type = var.storage_replication_type
  account_kind             = "StorageV2"
  tags                     = local.tags

  # Security defaults: TLS 1.2 minimum, blob public access blocked, shared
  # keys off (managed identity only).
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false
  public_network_access_enabled   = false

  blob_properties {
    versioning_enabled = true

    container_delete_retention_policy {
      days = 7
    }

    delete_retention_policy {
      days = 7
    }
  }

  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
    ip_rules       = []
    virtual_network_subnet_ids = [
      azurerm_subnet.private_endpoints.id,
      azurerm_subnet.containers.id,
    ]
  }
}

# Container that holds the proxy config + /v1/files uploads.
resource "azurerm_storage_container" "proxy" {
  name                  = "proxy"
  storage_account_name  = azurerm_storage_account.this.name
  container_access_type = "private"
}

# Files subcontainer: dedicated for the /v1/files endpoint so callers can
# isolate the data plane from the proxy_config blob.
resource "azurerm_storage_container" "files" {
  name                  = "files"
  storage_account_name  = azurerm_storage_account.this.name
  container_access_type = "private"
}

# Private endpoint for the Storage Account (blob sub-resource).
resource "azurerm_private_endpoint" "storage_blob" {
  name                = "${local.name}-sa-pe"
  location            = var.location
  resource_group_name = local.resource_group_name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = local.tags

  private_service_connection {
    name                           = "${local.name}-sa-blob"
    private_connection_resource_id = azurerm_storage_account.this.id
    is_manual_connection           = false
    subresource_names              = ["blob"]
  }

  private_dns_zone_group {
    name                 = "default"
    private_dns_zone_ids = [azurerm_private_dns_zone.blob.id]
  }

  depends_on = [azurerm_private_dns_zone_virtual_network_link.blob]
}

resource "azurerm_private_dns_zone" "blob" {
  name                = "privatelink.blob.core.windows.net"
  resource_group_name = local.resource_group_name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "blob" {
  name                  = "${local.name}-sa-dnslink"
  resource_group_name   = local.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.blob.name
  virtual_network_id    = azurerm_virtual_network.this.id
  registration_enabled  = false
}

# ---------- proxy_config blob upload ----------
#
# Encoded from `var.proxy_config` (a typed map). Each apply that changes
# proxy_config produces a new blob content, which the gateway and
# backend Container Apps container start uses as a versioning trigger.

resource "azurerm_storage_blob" "proxy_config" {
  count                  = var.proxy_config != {} ? 1 : 0
  name                   = "config/litellm-config.yaml"
  storage_account_name   = azurerm_storage_account.this.name
  storage_container_name = azurerm_storage_container.proxy.name
  type                   = "Block"
  content_type           = "application/yaml"

  # Force a fresh upload on content change. The lifecycle keeps the
  # previous version accessible via the versioned blob endpoint.
  source_content = local.proxy_config_yaml

  depends_on = [
    azurerm_role_assignment.storage_blob_data_contributor,
    azurerm_storage_container.proxy,
  ]
}
