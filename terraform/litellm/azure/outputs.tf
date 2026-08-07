output "app_gateway_url" {
  description = "Proxy URL. Scheme is https if key_vault_certificate_id is set, http otherwise. The dashboard is served at /, the API at /v1/*."
  value       = "${local.tls_enabled ? "https" : "http"}://${azurerm_public_ip.app_gateway.fqdn}"
}

output "app_gateway_fqdn" {
  description = "Public FQDN of the LiteLLM Application Gateway (derived from the public IP)."
  value       = azurerm_public_ip.app_gateway.fqdn
}

output "container_apps_environment_id" {
  description = "Resource ID of the Container Apps Environment hosting the gateway / backend / ui Container Apps."
  value       = azurerm_container_app_environment.this.id
}

output "postgres_fqdn" {
  description = "FQDN of the Azure Database for PostgreSQL Flexible Server. Used by gateway / backend / migrations as `DATABASE_HOST`."
  value       = azurerm_postgresql_flexible_server.this.fqdn
}

output "postgres_database_name" {
  description = "PostgreSQL database name."
  value       = var.db_name
}

output "redis_hostname" {
  description = "Hostname of the Azure Cache for Redis (use the `rediss://` scheme if var.redis_enable_ssl is true)."
  value       = azurerm_redis_cache.this.hostname
}

output "redis_port" {
  description = "Port of the Azure Cache for Redis."
  value       = azurerm_redis_cache.this.port
}

output "storage_account_name" {
  description = "Name of the Storage Account hosting the proxy blob container for cache backend, request log archival, and /v1/files storage. Exposed to gateway + backend as `AZURE_STORAGE_ACCOUNT_NAME` (LiteLLM reads this env var to assemble credentials)."
  value       = azurerm_storage_account.this.name
}

output "storage_blob_container" {
  description = "Blob container name on the Storage Account for proxy state."
  value       = azurerm_storage_container.proxy.name
}

output "key_vault_uri" {
  description = "URI of the Key Vault holding LITELLM_MASTER_KEY, the Aurora master password bootstrap, optional LITELLM_LICENSE, and optional UI_PASSWORD."
  value       = azurerm_key_vault.this.vault_uri
}

output "master_key_secret_id" {
  description = "Resource ID of the Key Vault secret holding LITELLM_MASTER_KEY."
  value       = azurerm_key_vault_secret.master_key.id
}

output "managed_identity_client_id" {
  description = "Client ID of the user-assigned managed identity assigned to the Container Apps. Use this when granting additional role assignments outside the module."
  value       = azurerm_user_assigned_identity.container_apps.client_id
}

output "managed_identity_principal_id" {
  description = "Object (principal) ID of the user-assigned managed identity. Use this for `az role assignment create` against additional scopes."
  value       = azurerm_user_assigned_identity.container_apps.principal_id
}

# Pre-baked SQL to run once as the Entra admin (after the first apply) to
# create the application user that gateway / backend / migration will
# authenticate as. Azure Database for PostgreSQL Flexible Server uses
# `azure_ad_admin` for the AAD-enabled login.
output "db_bootstrap_sql" {
  description = "Run this once as the AAD admin (after the first apply) to create the application user that will authenticate via Entra-managed-identity tokens at runtime."
  value       = <<-SQL
    CREATE USER "${var.db_username}" WITH LOGIN;
    GRANT ALL PRIVILEGES ON DATABASE ${var.db_name} TO "${var.db_username}";
    GRANT ALL ON SCHEMA public TO "${var.db_username}";
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO "${var.db_username}";
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO "${var.db_username}";
  SQL
}

# Pre-baked az CLI command for the one-off migration Container App Job.
output "migration_run_command" {
  description = "az CLI command that triggers the one-off prisma migration Container App Job. Run after the Entra admin has executed the db_bootstrap_sql above."
  value = format(
    "az containerapp job start --name %s --resource-group %s --subscription <your-sub-id>",
    azurerm_container_app_job.migrations.name,
    local.resource_group_name,
  )
}
