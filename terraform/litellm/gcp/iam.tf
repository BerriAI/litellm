# Runtime SA used by the gateway, backend, and migration job — has Cloud
# SQL client + Secret Manager accessor on every managed/extra secret. The
# UI deliberately uses a *different* SA (below) so a compromised UI
# container can't read master_key / db_password / license / ui_password /
# provider creds via the metadata service.
resource "google_service_account" "runtime" {
  account_id   = "${local.name}-runtime"
  display_name = "LiteLLM Cloud Run runtime"
}

# UI runtime SA — no role bindings. The UI is static nginx with no DB,
# Redis, or Secret Manager dependencies, so its task identity should not
# be able to read any of those. Cloud Run pulls the UI image via the
# project's serverless service agent (not this SA), so it doesn't need
# artifactregistry.reader either.
resource "google_service_account" "ui_runtime" {
  account_id   = "${local.name}-ui-runtime"
  display_name = "LiteLLM Cloud Run UI runtime (no data-plane access)"
}

# Cloud SQL client — lets the Cloud Run services connect to the instance
# over private IP via the VPC connector.
resource "google_project_iam_member" "runtime_cloudsql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# Secret Manager accessor — managed secrets first (split out as separate
# resources because their IDs are computed-at-apply and can't drive a
# for_each).
resource "google_secret_manager_secret_iam_member" "master_key" {
  secret_id = google_secret_manager_secret.master_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret_iam_member" "db_password" {
  secret_id = google_secret_manager_secret.db_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# License secret accessor — only created when var.litellm_license is set.
resource "google_secret_manager_secret_iam_member" "license" {
  count = var.litellm_license == "" ? 0 : 1

  secret_id = google_secret_manager_secret.license[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# UI password secret accessor — only created when var.ui_password is set.
resource "google_secret_manager_secret_iam_member" "ui_password" {
  count = var.ui_password == "" ? 0 : 1

  secret_id = google_secret_manager_secret.ui_password[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# User-supplied extras. Dedupe on the secret resource ID — two different
# env-var names could reference the same secret, and we want exactly one
# IAM binding per (secret, role, member) tuple in state.
resource "google_secret_manager_secret_iam_member" "extras" {
  for_each = toset(values(merge(var.gateway_extra_secrets, var.backend_extra_secrets)))

  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# OTEL_HEADERS secret accessor — only created when var.otel_headers_secret
# is set. Carries the OTLP collector's auth header(s).
resource "google_secret_manager_secret_iam_member" "otel_headers" {
  count = var.otel_headers_secret == "" ? 0 : 1

  secret_id = var.otel_headers_secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}
