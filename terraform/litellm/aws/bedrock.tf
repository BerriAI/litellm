# Bedrock authenticates with the task role's credentials through the standard
# boto3 chain, so a `bedrock/...` entry in proxy_config needs no api_key and no
# gateway_extra_secrets entry. The region comes from AWS_REGION /
# AWS_REGION_NAME, already in every task's env (see locals.shared_env in ecs.tf).

data "aws_iam_policy_document" "bedrock_invoke" {
  count = length(var.bedrock_model_arns) > 0 ? 1 : 0

  statement {
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = var.bedrock_model_arns
  }
}

resource "aws_iam_policy" "bedrock_invoke" {
  count  = length(var.bedrock_model_arns) > 0 ? 1 : 0
  name   = "${local.name}-bedrock-invoke"
  policy = data.aws_iam_policy_document.bedrock_invoke[0].json

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "task_bedrock_invoke" {
  count      = length(var.bedrock_model_arns) > 0 ? 1 : 0
  role       = aws_iam_role.task.name
  policy_arn = aws_iam_policy.bedrock_invoke[0].arn
}
