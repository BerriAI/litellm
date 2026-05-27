# Cloud SQL for PostgreSQL — one primary + one read replica.
#
# Note on auth: LiteLLM's IAM-auth helper (rds_iam_token.py) mints AWS RDS
# tokens via boto3 and doesn't speak GCP IAM. Cloud SQL IAM auth from Cloud
# Run requires the Cloud SQL Auth Proxy as a sidecar, which complicates the
# Cloud Run service spec. We instead use password auth: a random password
# lives in Secret Manager and is injected into the Cloud Run services as
# DATABASE_PASSWORD. The writer's DATABASE_URL is assembled inside the
# container at startup; the reader URL is built from the replica's IP.

resource "google_sql_database_instance" "writer" {
  name             = local.name
  region           = var.region
  database_version = var.db_version

  depends_on = [google_service_networking_connection.psa]

  settings {
    # ENTERPRISE accepts the db-custom-* and db-n1-* tiers we default to.
    # ENTERPRISE_PLUS only accepts db-perf-optimized-* and is ~3x cost — set
    # var.db_edition = "ENTERPRISE_PLUS" + change var.db_tier together if you
    # want it.
    edition           = var.db_edition
    tier              = var.db_tier
    availability_type = "REGIONAL"
    disk_size         = 20
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
      start_time                     = "07:00"
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.this.id
    }

    insights_config {
      query_insights_enabled  = true
      record_application_tags = true
      record_client_address   = true
    }
  }

  deletion_protection = var.cloudsql_deletion_protection
}

resource "google_sql_database_instance" "reader" {
  name                 = "${local.name}-reader"
  region               = var.region
  database_version     = var.db_version
  master_instance_name = google_sql_database_instance.writer.name

  depends_on = [google_service_networking_connection.psa]

  settings {
    edition           = var.db_edition
    tier              = var.db_tier
    availability_type = "ZONAL"
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.this.id
    }
  }

  deletion_protection = var.cloudsql_deletion_protection
}

resource "google_sql_database" "this" {
  name     = var.db_name
  instance = google_sql_database_instance.writer.name
}

resource "random_password" "db_password" {
  length      = 32
  special     = false
  min_lower   = 4
  min_upper   = 4
  min_numeric = 4
}

resource "google_sql_user" "app" {
  name     = var.db_username
  instance = google_sql_database_instance.writer.name
  password = random_password.db_password.result
}

resource "google_secret_manager_secret" "db_password" {
  secret_id = "${local.name}-db-password"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db_password.result
}
