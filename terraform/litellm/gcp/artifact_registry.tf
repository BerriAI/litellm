# Cloud Run can only pull from Artifact Registry, [region.]gcr.io, or
# docker.io — it rejects ghcr.io URIs at apply time. The four LiteLLM
# images live on GHCR upstream, so by default this stack provisions an
# Artifact Registry *remote repository* that transparently proxies
# https://ghcr.io. Cloud Run then pulls `…-docker.pkg.dev/<project>/<repo>/
# berriai/litellm-<component>` and AR fetches+caches from GHCR on first pull.
#
# This is what makes a zero-config deploy possible: with the proxy in place
# the default `image_registry` resolves to the proxy path (see locals.tf),
# so no manual `gcloud artifacts repositories create` step is needed.
#
# Set create_image_proxy_repo = false (and supply your own image_registry /
# *_image) to skip it — e.g. when mirroring images into a standard AR repo.

data "google_project" "this" {
  project_id = var.project
}

resource "google_artifact_registry_repository" "ghcr_proxy" {
  count = var.create_image_proxy_repo ? 1 : 0

  location      = var.region
  repository_id = "${local.name}-ghcr"
  description   = "GitHub Container Registry (ghcr.io) passthrough for LiteLLM images"
  format        = "DOCKER"
  mode          = "REMOTE_REPOSITORY"
  labels        = var.labels

  remote_repository_config {
    description = "ghcr.io"
    docker_repository {
      custom_repository {
        uri = "https://ghcr.io"
      }
    }
  }
}

# Cloud Run pulls images with the per-project serverless service agent, not
# the runtime SA. Grant that agent read on the proxy repo so the pull (and
# the upstream fetch) succeeds.
resource "google_artifact_registry_repository_iam_member" "serverless_agent_reader" {
  count = var.create_image_proxy_repo ? 1 : 0

  location   = google_artifact_registry_repository.ghcr_proxy[0].location
  repository = google_artifact_registry_repository.ghcr_proxy[0].name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:service-${data.google_project.this.number}@serverless-robot-prod.iam.gserviceaccount.com"
}
