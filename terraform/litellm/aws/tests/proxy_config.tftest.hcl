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
  proxy_config        = { general_settings = { alerting = ["slack"] } }
}

run "config_hash_in_task_definition_is_known_at_plan" {
  command = plan

  assert {
    condition = contains(
      [for e in local.proxy_config_env : e.value if e.name == "LITELLM_PROXY_CONFIG_S3_ETAG"],
      md5(yamlencode(var.proxy_config)),
    )
    error_message = "LITELLM_PROXY_CONFIG_S3_ETAG must be the md5 of the uploaded proxy_config YAML so it is known at plan time."
  }
}
