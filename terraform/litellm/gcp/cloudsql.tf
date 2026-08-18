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

    user_labels = local.labels

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

  lifecycle {
    # disk_autoresize grows storage but never shrinks it. Without this,
    # the first plan after any auto-grow reads disk_size as a shrink, which
    # is an immutable change and forces a destroy/recreate of the instance
    # (full data loss). Set the initial size only; let Cloud SQL own it
    # thereafter.
    ignore_changes = [settings[0].disk_size]
  }
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

    user_labels = local.labels

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.this.id
    }
  }

  deletion_protection = var.cloudsql_deletion_protection

  lifecycle {
    # Same autoresize footgun as the writer — the replica grows its disk
    # independently. Never let a perceived shrink replace the instance.
    ignore_changes = [settings[0].disk_size]
  }
}

resource "google_sql_database" "this" {
  name     = var.db_name
  instance = google_sql_database_instance.writer.name

  deletion_policy = "ABANDON"
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

  deletion_policy = "ABANDON"
}

resource "google_secret_manager_secret" "db_password" {
  secret_id = "${local.name}-db-password"
  labels    = local.labels
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db_password.result
}
