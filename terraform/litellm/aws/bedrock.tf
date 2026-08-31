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

# Custom Model Import reads Hugging Face safetensors out of S3 under a role it
# assumes itself, so this is a service role for Bedrock rather than anything the
# tasks use. Serving the imported model afterwards is plain `bedrock:InvokeModel`
# on the imported-model ARN, which belongs in var.bedrock_model_arns.
data "aws_iam_policy_document" "bedrock_model_import_assume" {
  count = var.enable_bedrock_custom_model_import ? 1 : 0

  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }

    # Without these a caller in another account who learns this role's name
    # could have Bedrock assume it on their behalf and read the bucket.
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:model-import-job/*"]
    }
  }
}

data "aws_iam_policy_document" "bedrock_model_import" {
  count = var.enable_bedrock_custom_model_import ? 1 : 0

  # Bedrock assumes this role with its own scoped session policy, so the grant
  # here has to be a superset of that policy or the intersection denies the
  # difference. It asks for GetObjectAttributes alongside GetObject, needed to
  # read a multi-GB safetensors file's part layout before ranged reads. Omitting
  # it fails the import job with "Encountered an internal error".
  statement {
    sid       = "ReadModelWeights"
    actions   = ["s3:GetObject", "s3:GetObjectAttributes"]
    resources = ["${aws_s3_bucket.this.arn}/models/*"]
  }

  # No s3:prefix condition: Bedrock chooses its own list parameters during
  # import, and a mismatch surfaces as an opaque job failure. Reads stay pinned
  # to models/ by the statement above.
  statement {
    sid       = "ListModelWeights"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.this.arn]
  }
}

resource "aws_iam_role" "bedrock_model_import" {
  count              = var.enable_bedrock_custom_model_import ? 1 : 0
  name               = "${local.name}-bedrock-model-import"
  assume_role_policy = data.aws_iam_policy_document.bedrock_model_import_assume[0].json

  tags = local.tags
}

resource "aws_iam_role_policy" "bedrock_model_import" {
  count  = var.enable_bedrock_custom_model_import ? 1 : 0
  name   = "read-model-weights"
  role   = aws_iam_role.bedrock_model_import[0].id
  policy = data.aws_iam_policy_document.bedrock_model_import[0].json
}
