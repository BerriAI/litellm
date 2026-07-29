output "lb_ip" {
  description = "Global anycast IP of the external load balancer."
  value       = module.litellm.lb_ip
}

output "lb_url" {
  description = "Proxy URL. Dashboard at /, API at /v1/*."
  value       = module.litellm.lb_url
}

output "gateway_service_url" {
  description = "Default Cloud Run URL for the gateway (bypasses the LB)."
  value       = module.litellm.gateway_service_url
}

output "backend_service_url" {
  description = "Default Cloud Run URL for the backend (bypasses the LB)."
  value       = module.litellm.backend_service_url
}

output "ui_service_url" {
  description = "Default Cloud Run URL for the UI (bypasses the LB)."
  value       = module.litellm.ui_service_url
}

output "cloudsql_writer_ip" {
  description = "Private IP of the Cloud SQL writer."
  value       = module.litellm.cloudsql_writer_ip
}

output "cloudsql_reader_ip" {
  description = "Private IP of the Cloud SQL read replica."
  value       = module.litellm.cloudsql_reader_ip
}

output "redis_endpoint" {
  description = "Memorystore Redis endpoint."
  value       = module.litellm.redis_endpoint
}

output "gcs_bucket" {
  description = "GCS bucket name."
  value       = module.litellm.gcs_bucket
}

output "master_key_secret_id" {
  description = "Secret Manager resource ID holding LITELLM_MASTER_KEY."
  value       = module.litellm.master_key_secret_id
}

output "db_password_secret_id" {
  description = "Secret Manager resource ID holding the Cloud SQL app-user password."
  value       = module.litellm.db_password_secret_id
}

output "migration_run_command" {
  description = "Break-glass command to re-run the one-off migration job."
  value       = module.litellm.migration_run_command
}
