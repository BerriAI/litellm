mock_provider "aws" {
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }
}
mock_provider "random" {}

variables {
  image_tag           = "v9.9.9"
  region              = "us-east-1"
  tenant              = "acme"
  env                 = "test"
  allow_plaintext_alb = true
  azs                 = ["us-east-1a", "us-east-1b"]
}

run "every_component_follows_image_registry_and_image_tag" {
  command = plan

  variables {
    image_registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com/mirror"
    image_tag      = "v9.9.9"
  }

  assert {
    condition = (
      local.gateway_image == "123456789012.dkr.ecr.us-east-1.amazonaws.com/mirror/litellm-gateway:v9.9.9" &&
      local.backend_image == "123456789012.dkr.ecr.us-east-1.amazonaws.com/mirror/litellm-backend:v9.9.9" &&
      local.ui_image == "123456789012.dkr.ecr.us-east-1.amazonaws.com/mirror/litellm-ui:v9.9.9" &&
      local.migrations_image == "123456789012.dkr.ecr.us-east-1.amazonaws.com/mirror/litellm-migrations:v9.9.9"
    )
    error_message = "All four images must be composed as <image_registry>/litellm-<component>:<image_tag>."
  }
}

run "an_empty_image_tag_fails_closed_unless_every_image_is_overridden" {
  command = plan

  variables {
    image_tag = ""
  }

  expect_failures = [var.image_tag]
}

run "a_null_image_tag_takes_the_module_default" {
  command = plan

  variables {
    image_tag = null
  }

  expect_failures = [var.image_tag]
}

run "four_explicit_uris_need_no_image_tag" {
  command = plan

  variables {
    image_tag        = ""
    gateway_image    = "example.com/custom/gateway:nightly"
    backend_image    = "example.com/custom/backend:nightly"
    ui_image         = "example.com/custom/ui:nightly"
    migrations_image = "example.com/custom/migrations:nightly"
  }

  assert {
    condition     = local.migrations_image == "example.com/custom/migrations:nightly"
    error_message = "Explicit per-component URIs must satisfy the module without an image_tag."
  }
}

run "composed_defaults_pull_from_ghcr_at_the_image_tag" {
  command = plan

  assert {
    condition = alltrue([
      for image in [local.gateway_image, local.backend_image, local.ui_image, local.migrations_image] :
      startswith(image, "ghcr.io/berriai/litellm-") && endswith(image, ":${var.image_tag}")
    ])
    error_message = "With no overrides every component must come from ghcr.io/berriai at image_tag."
  }
}

run "a_per_component_uri_wins_over_the_composed_image" {
  command = plan

  variables {
    image_tag     = "v9.9.9"
    gateway_image = "example.com/custom/gateway:nightly"
  }

  assert {
    condition = (
      local.gateway_image == "example.com/custom/gateway:nightly" &&
      local.backend_image == "ghcr.io/berriai/litellm-backend:v9.9.9" &&
      local.ui_image == "ghcr.io/berriai/litellm-ui:v9.9.9" &&
      local.migrations_image == "ghcr.io/berriai/litellm-migrations:v9.9.9"
    )
    error_message = "A per-component override must replace only that component's image."
  }
}
