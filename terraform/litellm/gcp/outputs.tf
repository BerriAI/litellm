output "lb_ip" {
  description = "Global anycast IP of the external HTTPS load balancer."
  value       = google_compute_global_address.lb.address
}

output "lb_url" {
  description = "Proxy URL. Switches scheme based on whether lb_domains is set; when TLS is enabled the URL points at the first listed domain (since managed certs are tied to the hostname, not the anycast IP). The dashboard is served at /, the API at /v1/*."
  value       = local.tls_enabled ? "https://${var.lb_domains[0]}" : "http://${google_compute_global_address.lb.address}"
}

output "gateway_service_url" {
  description = "Default Cloud Run URL for the gateway (bypasses the LB)."
  value       = google_cloud_run_v2_service.gateway.uri
}

output "backend_service_url" {
  description = "Default Cloud Run URL for the backend (bypasses the LB)."
  value       = google_cloud_run_v2_service.backend.uri
}

output "ui_service_url" {
  description = "Default Cloud Run URL for the UI (bypasses the LB)."
  value       = google_cloud_run_v2_service.ui.uri
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
  description = "Shell command that executes the one-off migration job against Cloud SQL. Run this once after the first apply."
  value = format(
    "gcloud run jobs execute %s --region %s --project %s --wait",
    google_cloud_run_v2_job.migrations.name,
    var.region,
    var.project_id,
  )
}
