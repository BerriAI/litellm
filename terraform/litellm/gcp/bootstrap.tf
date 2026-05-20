# Auto-runs the prisma schema migration as part of `terraform apply`. Mirrors
# the AWS stack's terraform_data.migration in spirit. Cloud SQL doesn't need a
# separate user-bootstrap step because google_sql_user.app already creates the
# application user — so the only post-cluster work is the migration.
#
# Gateway/backend Cloud Run services depend on this resource (in cloudrun.tf)
# so they don't go live until the schema is in place.
#
# Triggers:
#   - re-runs if the migrations image changes (new release ships new prisma
#     migration files).
#   - re-runs if the migration job is recreated.
#
# Requires `gcloud` on the machine running terraform, with user creds live
# enough to invoke Cloud Run admin APIs (`gcloud auth login`).

resource "terraform_data" "migration" {
  triggers_replace = {
    job_id    = google_cloud_run_v2_job.migrations.id
    job_image = local.migrations_image
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    environment = {
      JOB     = google_cloud_run_v2_job.migrations.name
      REGION  = var.region
      PROJECT = var.project
    }
    command = <<-EOT
      set -euo pipefail
      gcloud run jobs execute "$JOB" \
        --region "$REGION" \
        --project "$PROJECT" \
        --wait
    EOT
  }

  depends_on = [
    google_cloud_run_v2_job.migrations,
    google_sql_user.app,
  ]
}
