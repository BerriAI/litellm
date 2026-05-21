# ECS task execution role — used by the agent to pull images, write logs,
# and resolve secrets at task start.
data "aws_iam_policy_document" "task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "${local.name}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# User-provided extra secrets may be passed as the bare secret ARN
# ("arn:aws:secretsmanager:...:secret:name-AbCdEf") or the JSON-key form
# ECS supports — fully spelled out as
# "arn:...:secret:name-AbCdEf:jsonKey:versionStage:versionId" with any of
# the trailing parts blank ("...:jsonKey::" being the most common). The IAM
# policy resource must always be the bare ARN, so we split on ':' and keep
# the first 7 components — robust to any combination of empty/non-empty
# version-stage/version-id suffixes that a regex would otherwise have to
# enumerate.
locals {
  extra_secret_value_froms = concat(
    values(var.gateway_extra_secrets),
    values(var.backend_extra_secrets),
  )

  extra_secret_arns = distinct([
    for v in local.extra_secret_value_froms :
    join(":", slice(split(":", v), 0, 7))
  ])
}

# Execution role can read the managed secrets + any caller-provided extras
# so ECS can resolve them when launching tasks. Image pulls inherit the
# managed AmazonECSTaskExecutionRolePolicy.
data "aws_iam_policy_document" "secrets_access" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = concat(
      [aws_secretsmanager_secret.master_key.arn],
      aws_secretsmanager_secret.license[*].arn,
      aws_secretsmanager_secret.ui_password[*].arn,
      local.extra_secret_arns,
    )
  }
}

resource "aws_iam_policy" "secrets_access" {
  name   = "${local.name}-secrets-access"
  policy = data.aws_iam_policy_document.secrets_access.json
}

resource "aws_iam_role_policy_attachment" "task_execution_secrets" {
  role       = aws_iam_role.task_execution.name
  policy_arn = aws_iam_policy.secrets_access.arn
}

# ---------- Task role ----------
#
# Assumed by the running container. Gets `rds-db:connect` so the proxy can
# mint IAM-signed Postgres tokens for the app user. Layer additional
# policies here (e.g. Bedrock invoke, S3 read) when the proxy needs them.

resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

data "aws_caller_identity" "current" {}

data "aws_iam_policy_document" "rds_iam_connect" {
  statement {
    actions = ["rds-db:connect"]
    resources = [
      "arn:aws:rds-db:${var.region}:${data.aws_caller_identity.current.account_id}:dbuser:${aws_rds_cluster.this.cluster_resource_id}/${var.db_username}",
    ]
  }
}

resource "aws_iam_policy" "rds_iam_connect" {
  name   = "${local.name}-rds-iam-connect"
  policy = data.aws_iam_policy_document.rds_iam_connect.json
}

resource "aws_iam_role_policy_attachment" "task_rds_iam_connect" {
  role       = aws_iam_role.task.name
  policy_arn = aws_iam_policy.rds_iam_connect.arn
}

# ---------- UI task role ----------
#
# The UI is static nginx with no DB, S3, or Secrets Manager dependencies,
# so it deliberately does NOT inherit the shared `task` role's
# rds-db:connect / S3 / extra-secrets policies. Empty policy set — the
# only thing exposed via the task metadata endpoint is an identity that
# can't reach any LiteLLM data-plane resource. The shared
# `task_execution` role still pulls the image and writes logs (its
# credentials aren't surfaced to the container).

resource "aws_iam_role" "ui_task" {
  name               = "${local.name}-ui-task"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}
