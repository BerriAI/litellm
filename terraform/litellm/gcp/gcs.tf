# General-purpose GCS bucket — same role as the AWS S3 bucket. The bucket
# name is exposed to gateway + backend as GCS_BUCKET_NAME; reference it
# from proxy_config via `os.environ/GCS_BUCKET_NAME`.

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "this" {
  name                        = "${var.project}-${local.name}-${random_id.bucket_suffix.hex}"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = var.gcs_force_destroy

  versioning {
    enabled = true
  }

  public_access_prevention = "enforced"

  labels = var.labels
}

# Cloud Run runtime SA gains object admin on this bucket only.
resource "google_storage_bucket_iam_member" "runtime" {
  bucket = google_storage_bucket.this.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
}
