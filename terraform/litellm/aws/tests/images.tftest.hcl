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

run "the_image_follows_image_registry_and_image_tag" {
  command = plan

  variables {
    image_registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com/mirror"
    image_tag      = "v9.9.9"
  }

  assert {
    condition     = local.image == "123456789012.dkr.ecr.us-east-1.amazonaws.com/mirror/litellm:v9.9.9"
    error_message = "The image must be composed as <image_registry>/litellm:<image_tag>."
  }
}

run "an_empty_image_tag_fails_closed_unless_image_is_set" {
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

run "an_explicit_uri_needs_no_image_tag" {
  command = plan

  variables {
    image_tag = ""
    image     = "example.com/custom/litellm:nightly"
  }

  assert {
    condition     = local.image == "example.com/custom/litellm:nightly"
    error_message = "An explicit image URI must satisfy the module without an image_tag."
  }
}

run "the_composed_default_pulls_from_ghcr_at_the_image_tag" {
  command = plan

  assert {
    condition     = local.image == "ghcr.io/berriai/litellm:${var.image_tag}"
    error_message = "With no override the image must come from ghcr.io/berriai at image_tag."
  }
}

run "an_explicit_uri_wins_over_the_composed_image" {
  command = plan

  variables {
    image_tag = "v9.9.9"
    image     = "example.com/custom/litellm:nightly"
  }

  assert {
    condition     = local.image == "example.com/custom/litellm:nightly"
    error_message = "A full image URI must replace the composed image."
  }
}
