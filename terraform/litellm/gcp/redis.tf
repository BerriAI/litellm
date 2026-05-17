resource "google_redis_instance" "this" {
  name           = local.name
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_size_gb
  region         = var.region

  authorized_network = google_compute_network.this.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version = "REDIS_7_0"

  # In-transit encryption between Cloud Run and Memorystore. The instance
  # exposes its self-signed CA via `server_ca_certs` (read in cloudrun.tf
  # and passed to the proxy as REDIS_CA_PEM_B64); the proxy decodes it to
  # /tmp/redis-ca.pem at startup and uses it to validate the rediss://
  # handshake. Mirrors `transit_encryption_enabled = true` on AWS.
  transit_encryption_mode = "SERVER_AUTHENTICATION"

  depends_on = [google_service_networking_connection.psa]
}
