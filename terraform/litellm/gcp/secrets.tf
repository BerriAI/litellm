resource "random_password" "master_key" {
  length      = 48
  special     = false
  min_lower   = 4
  min_upper   = 4
  min_numeric = 4
}

# LITELLM_MASTER_KEY (sk-…) lives in Secret Manager. The Cloud Run service
# account gets accessor permission on it (see iam.tf).
resource "google_secret_manager_secret" "master_key" {
  secret_id = "${local.name}-master-key"
  labels    = local.labels
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "master_key" {
  secret = google_secret_manager_secret.master_key.id
  # When the operator passes litellm_master_key, use it verbatim. Otherwise
  # fall back to the auto-generated `sk-…` value (trial / OSS path).
  secret_data = coalesce(var.litellm_master_key, "sk-${random_password.master_key.result}")
}

# LITELLM_LICENSE — only created when the operator supplies one. The runtime
# SA gets accessor permission via iam.tf, and gateway + backend pick it up
# through shared_env_secrets in cloudrun.tf.
resource "google_secret_manager_secret" "license" {
  count = var.litellm_license == "" ? 0 : 1

  secret_id = "${local.name}-license"
  labels    = local.labels
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "license" {
  count = var.litellm_license == "" ? 0 : 1

  secret      = google_secret_manager_secret.license[0].id
  secret_data = var.litellm_license
}

# UI_PASSWORD — backend-only. Same pattern as license: only created when
# the operator supplies one. The runtime SA gets accessor permission via
# iam.tf, and the backend service picks the env var up through
# backend_managed_env_secrets in cloudrun.tf.
resource "google_secret_manager_secret" "ui_password" {
  count = var.ui_password == "" ? 0 : 1

  secret_id = "${local.name}-ui-password"
  labels    = local.labels
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "ui_password" {
  count = var.ui_password == "" ? 0 : 1

  secret      = google_secret_manager_secret.ui_password[0].id
  secret_data = var.ui_password
}

# Billing-metrics mTLS material — only created when metering is enabled
# (billing_metrics_endpoint non-empty) and the operator supplied the PEM.
# The runtime SA gets accessor permission via iam.tf, and gateway + backend
# pick the env vars up through billing_metrics_env_secrets in cloudrun.tf.
resource "google_secret_manager_secret" "billing_metrics_client_cert" {
  count = local.billing_metrics_client_cert_enabled ? 1 : 0

  secret_id = "${local.name}-billing-metrics-client-cert"
  labels    = local.labels
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "billing_metrics_client_cert" {
  count = local.billing_metrics_client_cert_enabled ? 1 : 0

  secret      = google_secret_manager_secret.billing_metrics_client_cert[0].id
  secret_data = var.billing_metrics_client_cert_pem
}

resource "google_secret_manager_secret" "billing_metrics_client_key" {
  count = local.billing_metrics_client_key_enabled ? 1 : 0

  secret_id = "${local.name}-billing-metrics-client-key"
  labels    = local.labels
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "billing_metrics_client_key" {
  count = local.billing_metrics_client_key_enabled ? 1 : 0

  secret      = google_secret_manager_secret.billing_metrics_client_key[0].id
  secret_data = var.billing_metrics_client_key_pem
}

resource "google_secret_manager_secret" "billing_metrics_ca_cert" {
  count = local.billing_metrics_ca_cert_enabled ? 1 : 0

  secret_id = "${local.name}-billing-metrics-ca-cert"
  labels    = local.labels
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "billing_metrics_ca_cert" {
  count = local.billing_metrics_ca_cert_enabled ? 1 : 0

  secret      = google_secret_manager_secret.billing_metrics_ca_cert[0].id
  secret_data = var.billing_metrics_ca_cert_pem
}
