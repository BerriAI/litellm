# Bedrock authenticates with the task role's credentials through the standard
# boto3 chain, so a `bedrock/...` entry in proxy_config needs no api_key and no
# gateway_extra_secrets entry. The region comes from AWS_REGION /
# AWS_REGION_NAME, already in every task's env (see locals.shared_env in ecs.tf).

data "aws_region" "current" {}

data "aws_iam_policy_document" "bedrock_invoke" {
  count = local.bedrock_policy_enabled ? 1 : 0

  dynamic "statement" {
    for_each = length(var.bedrock_model_arns) > 0 ? [1] : []

    content {
      sid = "BedrockInvokeModel"
      actions = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
      ]
      resources = var.bedrock_model_arns
    }
  }

  dynamic "statement" {
    for_each = var.enable_bedrock_mantle ? [1] : []

    content {
      sid       = "BedrockMantleCreateInference"
      actions   = ["bedrock-mantle:CreateInference"]
      resources = ["*"]

      condition {
        test     = "StringEquals"
        variable = "aws:RequestedRegion"
        values   = [data.aws_region.current.name]
      }
    }
  }
}

resource "aws_iam_policy" "bedrock_invoke" {
  count  = local.bedrock_policy_enabled ? 1 : 0
  name   = "${local.name}-bedrock-invoke"
  policy = data.aws_iam_policy_document.bedrock_invoke[0].json

  tags = local.tags

  # The module declares no provider block, so the provider takes its region from
  # the environment while var.region drives naming and subnets. If the two
  # disagree, the Mantle statement's aws:RequestedRegion condition silently
  # pins to the provider's region and every Mantle call is denied with no hint
  # why. Fail at plan time instead.
  lifecycle {
    precondition {
      condition     = data.aws_region.current.name == var.region
      error_message = "Provider region ${data.aws_region.current.name} != var.region ${var.region}. Export AWS_REGION=${var.region} (the Makefile does this from dev.tfvars)."
    }
  }
}

resource "aws_iam_role_policy_attachment" "task_bedrock_invoke" {
  count      = local.bedrock_policy_enabled ? 1 : 0
  role       = aws_iam_role.task.name
  policy_arn = aws_iam_policy.bedrock_invoke[0].arn
}
