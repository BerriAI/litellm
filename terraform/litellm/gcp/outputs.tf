output "lb_ip" {
  description = "Global anycast IP of the external HTTPS load balancer. Null when create_runtime is false."
  value       = var.create_runtime ? one(google_compute_global_address.lb[*].address) : null
}

output "lb_url" {
  description = "Proxy URL, or null when create_runtime is false. Switches scheme based on whether lb_domains is set."
  value       = var.create_runtime ? (local.tls_enabled ? "https://${var.lb_domains[0]}" : "http://${one(google_compute_global_address.lb[*].address)}") : null
}

output "gateway_service_url" {
  description = "Default Cloud Run URL for the gateway, or null when create_runtime is false."
  value       = var.create_runtime ? one(google_cloud_run_v2_service.gateway[*].uri) : null
}

output "backend_service_url" {
  description = "Default Cloud Run URL for the backend, or null when create_runtime is false."
  value       = var.create_runtime ? one(google_cloud_run_v2_service.backend[*].uri) : null
}

output "ui_service_url" {
  description = "Default Cloud Run URL for the UI, or null when create_runtime is false."
  value       = var.create_runtime ? one(google_cloud_run_v2_service.ui[*].uri) : null
}

output "cloudsql_writer_ip" {
  description = "Private IP of the Cloud SQL writer."
  value       = google_sql_database_instance.writer.private_ip_address
}

output "cloudsql_reader_ip" {
  description = "Private IP of the Cloud SQL read replica."
  value       = google_sql_database_instance.reader.private_ip_address
}

output "redis_endpoint" {
  description = "Memorystore Redis endpoint."
  value       = "${google_redis_instance.this.host}:${google_redis_instance.this.port}"
}

output "runtime_service_account_email" {
  description = "Runtime service account email for Cloud Run or GKE Workload Identity."
  value       = google_service_account.runtime.email
}

output "redis_host" {
  description = "Memorystore Redis host."
  value       = google_redis_instance.this.host
}

output "redis_port" {
  description = "Memorystore Redis port."
  value       = google_redis_instance.this.port
}

output "redis_server_ca_pem" {
  description = "Memorystore server CA PEM. Mount it in the pod and set REDIS_SSL=true and REDIS_SSL_CA_CERTS=<path> via extraEnv when transit encryption is enabled."
  value       = var.redis_transit_encryption ? google_redis_instance.this.server_ca_certs[0].cert : null
}

output "db_username" {
  description = "Cloud SQL application username."
  value       = var.db_username
}

output "db_name" {
  description = "Cloud SQL database name."
  value       = var.db_name
}

output "gcs_bucket" {
  description = "GCS bucket name. Exposed to gateway + backend as GCS_BUCKET_NAME. Reference from proxy_config via `os.environ/GCS_BUCKET_NAME`."
  value       = google_storage_bucket.this.name
}

output "master_key_secret_id" {
  description = "Secret Manager resource ID holding LITELLM_MASTER_KEY. Fetch with `gcloud secrets versions access latest --secret=<id>`."
  value       = google_secret_manager_secret.master_key.secret_id
}

output "db_password_secret_id" {
  description = "Secret Manager resource ID holding the Cloud SQL app-user password."
  value       = google_secret_manager_secret.db_password.secret_id
}

output "migration_run_command" {
  description = "Shell command that executes the one-off migration job against Cloud SQL, or null when create_runtime is false."
  value = var.create_runtime ? format(
    "gcloud run jobs execute %s --region %s --project %s --wait",
    one(google_cloud_run_v2_job.migrations[*].name),
    var.region,
    var.project_id,
  ) : null
}
