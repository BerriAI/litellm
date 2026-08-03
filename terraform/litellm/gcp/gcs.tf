# General-purpose GCS bucket — same role as the AWS S3 bucket. The bucket
# name is exposed to gateway + backend as GCS_BUCKET_NAME; reference it
# from proxy_config via `os.environ/GCS_BUCKET_NAME`.

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "this" {
  name                        = "${var.project_id}-${local.name}-${random_id.bucket_suffix.hex}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.gcs_force_destroy

  versioning {
    enabled = true
  }

  public_access_prevention = "enforced"

  labels = local.labels
}

# Cloud Run runtime SA gains object admin on this bucket only.
resource "google_storage_bucket_iam_member" "runtime" {
  bucket = google_storage_bucket.this.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

# Dedicated bucket holding only config.yaml. Mounted read-only into the
# gateway and backend via Cloud Run v2's gcsfuse volume. Kept separate from
# the data-plane bucket above so the runtime SA can hold a narrower
# objectViewer binding here (config is read-only at runtime) while keeping
# objectAdmin on the data-plane bucket. Only created when proxy_config is
# non-empty.
resource "google_storage_bucket" "proxy_config" {
  count = local.proxy_config_enabled ? 1 : 0

  name                        = "${var.project_id}-${local.name}-config-${random_id.bucket_suffix.hex}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.gcs_force_destroy

  versioning {
    enabled = true
  }

  public_access_prevention = "enforced"

  labels = local.labels
}

resource "google_storage_bucket_object" "proxy_config" {
  count = local.proxy_config_enabled ? 1 : 0

  name         = local.proxy_config_file_name
  bucket       = google_storage_bucket.proxy_config[0].name
  content      = local.proxy_config_yaml
  content_type = "application/yaml"
}

resource "google_storage_bucket_iam_member" "proxy_config_runtime" {
  count = local.proxy_config_enabled ? 1 : 0

  bucket = google_storage_bucket.proxy_config[0].name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime.email}"
}
