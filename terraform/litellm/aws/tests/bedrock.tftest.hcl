# Plan-only coverage for the three Bedrock knobs: the invoke policy, the Mantle
# grant, and the Custom Model Import service role. Run from
# terraform/litellm/aws with `terraform test`.
#
# `mock_provider` replaces aws_iam_policy_document with a fixed empty document,
# so these runs assert which resources exist and how the gating composes, not the
# rendered statements. Actions, resources, and the import role's confused-deputy
# conditions are unreachable offline and are covered by reading the plan output
# against a real provider instead.

mock_provider "aws" {
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }

  # The Mantle policy's precondition compares the provider's region against
  # var.region, and a generated placeholder would fail it every time.
  mock_data "aws_region" {
    defaults = {
      name = "us-east-1"
    }
  }
}
mock_provider "random" {}

variables {
  region              = "us-east-1"
  tenant              = "acme"
  env                 = "test"
  allow_plaintext_alb = true
  azs                 = ["us-east-1a", "us-east-1b"]
  create_database     = false
  create_redis        = false
  # Single process, so the Redis-less rate-limit check stays quiet: none of these
  # runs are about the data stores.
  gateway_autoscaling_enabled = false
  gateway_desired_count       = 1
  gateway_num_workers         = 1
}

run "no_bedrock_variables_creates_nothing" {
  command = plan

  assert {
    condition = alltrue([
      length(aws_iam_policy.bedrock_invoke) == 0,
      length(aws_iam_role_policy_attachment.task_bedrock_invoke) == 0,
      length(aws_iam_role.bedrock_model_import) == 0,
      length(aws_iam_role_policy.bedrock_model_import) == 0,
    ])
    error_message = "Every Bedrock resource is opt-in: the defaults must grant the task role nothing."
  }
}

run "model_arns_attach_an_invoke_policy_to_the_task_role" {
  command = plan

  variables {
    bedrock_model_arns = ["arn:aws:bedrock:*::foundation-model/anthropic.*"]
  }

  assert {
    condition = alltrue([
      length(aws_iam_policy.bedrock_invoke) == 1,
      length(aws_iam_role_policy_attachment.task_bedrock_invoke) == 1,
      aws_iam_role_policy_attachment.task_bedrock_invoke[0].role == aws_iam_role.task.name,
    ])
    error_message = "A non-empty bedrock_model_arns must create the invoke policy and attach it to the task role, not merely define it."
  }

  assert {
    condition     = length(aws_iam_role.bedrock_model_import) == 0
    error_message = "Invoking a model must not drag in the Custom Model Import service role."
  }
}

# The gate is an OR, and Mantle is the arm that is easy to break: it authorizes
# against bedrock-mantle:CreateInference rather than an ARN, so it has to produce
# a policy with bedrock_model_arns left empty.
run "mantle_alone_still_produces_a_policy" {
  command = plan

  variables {
    enable_bedrock_mantle = true
  }

  assert {
    condition = alltrue([
      length(aws_iam_policy.bedrock_invoke) == 1,
      length(aws_iam_role_policy_attachment.task_bedrock_invoke) == 1,
    ])
    error_message = "enable_bedrock_mantle must create and attach the policy on its own, with no bedrock_model_arns set."
  }
}

run "custom_model_import_creates_a_service_role_and_no_task_grant" {
  command = plan

  # A role's id is unknown until apply, so pin it to a literal the attachment
  # assertion below can name.
  override_resource {
    target          = aws_iam_role.bedrock_model_import
    override_during = plan
    values          = { id = "acme-test-bedrock-model-import" }
  }

  variables {
    enable_bedrock_custom_model_import = true
  }

  assert {
    condition = alltrue([
      length(aws_iam_role.bedrock_model_import) == 1,
      length(aws_iam_role_policy.bedrock_model_import) == 1,
    ])
    error_message = "enable_bedrock_custom_model_import must create the service role Bedrock assumes plus its inline read policy."
  }

  # The import role is Bedrock's, not the tasks'. Serving the imported model
  # afterwards goes through bedrock_model_arns like any other model.
  assert {
    condition = alltrue([
      length(aws_iam_policy.bedrock_invoke) == 0,
      length(aws_iam_role_policy_attachment.task_bedrock_invoke) == 0,
    ])
    error_message = "The import role must not imply any invoke grant on the task role."
  }

  assert {
    condition     = aws_iam_role_policy.bedrock_model_import[0].role == aws_iam_role.bedrock_model_import[0].id
    error_message = "The read-model-weights policy must attach to the import role rather than to the task role."
  }
}

# Guards the one input that lands in an IAM policy's resource list verbatim. A
# bare "*" would let the internet-facing proxy invoke every Bedrock resource in
# the account, including other teams' provisioned throughput and imported models.
run "a_bare_wildcard_is_rejected" {
  command = plan

  variables {
    bedrock_model_arns = ["*"]
  }

  expect_failures = [
    var.bedrock_model_arns,
  ]
}

run "a_non_bedrock_arn_is_rejected" {
  command = plan

  variables {
    bedrock_model_arns = ["arn:aws:s3:::some-bucket/*"]
  }

  expect_failures = [
    var.bedrock_model_arns,
  ]
}

run "a_china_partition_bedrock_arn_is_accepted" {
  command = plan

  variables {
    bedrock_model_arns = ["arn:aws-cn:bedrock:cn-north-1::foundation-model/anthropic.claude-v2"]
  }

  assert {
    condition     = length(aws_iam_policy.bedrock_invoke) == 1
    error_message = "The ARN guard must admit non-commercial partitions, not just arn:aws:."
  }
}
