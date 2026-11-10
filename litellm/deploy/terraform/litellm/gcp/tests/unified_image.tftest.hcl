# Plan-only coverage for the single-image contract: every Cloud Run service
# and the migrations Job pull the same image and name their process through
# the image entrypoint's component argument.

mock_provider "google" {
  mock_resource "google_redis_instance" {
    defaults = {
      host = "10.0.0.4"
      port = 6379
      server_ca_certs = [{
        cert = "-----BEGIN CERTIFICATE-----\nmock\n-----END CERTIFICATE-----"
      }]
    }
  }
}

mock_provider "google-beta" {}
mock_provider "random" {}

variables {
  project_id          = "test-project"
  tenant              = "tenant"
  env                 = "test"
  allow_plaintext_lb  = true
  image_registry      = "us-central1-docker.pkg.dev/test-project/litellm/berriai"
  image_tag           = "v9.9.9-stable"
  gateway_num_workers = 3
}

run "registry_and_tag_compose_one_image_for_every_workload" {
  command = plan

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.gateway[0].template[0].containers[0].image == "us-central1-docker.pkg.dev/test-project/litellm/berriai/litellm:v9.9.9-stable",
      google_cloud_run_v2_service.backend[0].template[0].containers[0].image == "us-central1-docker.pkg.dev/test-project/litellm/berriai/litellm:v9.9.9-stable",
      google_cloud_run_v2_service.ui[0].template[0].containers[0].image == "us-central1-docker.pkg.dev/test-project/litellm/berriai/litellm:v9.9.9-stable",
      google_cloud_run_v2_job.migrations[0].template[0].template[0].containers[0].image == "us-central1-docker.pkg.dev/test-project/litellm/berriai/litellm:v9.9.9-stable",
      terraform_data.migration[0].triggers_replace.job_image == "us-central1-docker.pkg.dev/test-project/litellm/berriai/litellm:v9.9.9-stable",
    ])
    error_message = "All workloads must pull <image_registry>/litellm:<image_tag> and the migration re-run trigger must follow it."
  }

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.ui[0].template[0].containers[0].command == null,
      google_cloud_run_v2_service.ui[0].template[0].containers[0].args == tolist(["ui"]),
      google_cloud_run_v2_job.migrations[0].template[0].template[0].containers[0].command == null,
      google_cloud_run_v2_job.migrations[0].template[0].template[0].containers[0].args == tolist(["migrations"]),
      google_cloud_run_v2_service.gateway[0].template[0].containers[0].command == tolist(["sh", "-c"]),
      endswith(google_cloud_run_v2_service.gateway[0].template[0].containers[0].args[0], " && exec /app/docker-entrypoint.sh gateway --workers 3"),
      google_cloud_run_v2_service.backend[0].template[0].containers[0].command == tolist(["sh", "-c"]),
      endswith(google_cloud_run_v2_service.backend[0].template[0].containers[0].args[0], " && exec /app/docker-entrypoint.sh backend"),
    ])
    error_message = "UI and migrations must keep the image entrypoint with their component as args; gateway and backend must exec it with their component after the DATABASE_URL bootstrap."
  }
}

run "image_overrides_the_composed_uri_everywhere" {
  command = plan

  variables {
    image = "us-central1-docker.pkg.dev/test-project/mirror/litellm:custom"
  }

  assert {
    condition = alltrue([
      google_cloud_run_v2_service.gateway[0].template[0].containers[0].image == var.image,
      google_cloud_run_v2_service.backend[0].template[0].containers[0].image == var.image,
      google_cloud_run_v2_service.ui[0].template[0].containers[0].image == var.image,
      google_cloud_run_v2_job.migrations[0].template[0].template[0].containers[0].image == var.image,
      terraform_data.migration[0].triggers_replace.job_image == var.image,
    ])
    error_message = "A full image URI must win over image_registry + image_tag for every workload."
  }
}
