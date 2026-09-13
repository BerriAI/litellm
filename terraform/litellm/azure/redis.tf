# Azure Cache for Redis: single instance (Basic tier) by default.
# Production should move to Standard / Premium with replication.
#
# The proxy connects via `rediss://` (TLS) when `var.redis_enable_ssl`
# is true (the default, matching AWS module's transit_encryption_enabled).

resource "azurerm_redis_cache" "this" {
  name                = replace("${var.tenant}-${var.env}-redis", "-", "")
  location            = var.location
  resource_group_name = local.resource_group_name
  sku_name            = var.redis_sku
  family              = var.redis_family
  capacity            = var.redis_capacity
  enable_non_ssl_port = !var.redis_enable_ssl
  minimum_tls_version = "1.2"
  tags                = local.tags

  # Redis access keys are managed by Azure; the proxy uses the `rediss://`
  # URL with the primary key as auth. Caller can rotate via Key Vault
  # access policy if tighter secret-scoped handling is required.
  redis_configuration {
    # Renamed to authentication_enabled in v4. Until we drop 3.x support,
    # both forms exist; this module targets 3.117+ which still accepts
    # enable_authentication.
    enable_authentication = true
    maxmemory_policy      = "allkeys-lru"
  }

  # Access via private endpoint only.
  public_network_access_enabled = false
  subnet_id                     = azurerm_subnet.private_endpoints.id
}

resource "azurerm_private_dns_zone" "redis" {
  name                = "privatelink.redis.cache.windows.net"
  resource_group_name = local.resource_group_name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "redis" {
  name                  = "${local.name}-redis-dnslink"
  resource_group_name   = local.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.redis.name
  virtual_network_id    = azurerm_virtual_network.this.id
  registration_enabled  = false
}

# Outputs that the gateway/backend pass via `REDIS_HOST` / `REDIS_PORT`.
locals {
  redis_host_for_container_apps = azurerm_redis_cache.this.hostname
  redis_port_for_container_apps = azurerm_redis_cache.this.ssl_port
}
