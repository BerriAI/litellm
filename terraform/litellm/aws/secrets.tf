resource "random_password" "master_key" {
  length      = 48
  special     = false
  min_lower   = 4
  min_upper   = 4
  min_numeric = 4
}

# Master DB password — used once to bootstrap the IAM-authed application
# user (see rds.tf header). Runtime services authenticate via IAM tokens
# and never read this secret.
resource "random_password" "db_master_password" {
  length      = 32
  special     = false
  min_lower   = 4
  min_upper   = 4
  min_numeric = 4
}

# LITELLM_MASTER_KEY — must begin with `sk-` per the proxy's validator.
resource "aws_secretsmanager_secret" "master_key" {
  name                    = "${local.name}-master-key"
  description             = "LITELLM_MASTER_KEY for gateway + backend."
  recovery_window_in_days = 0

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "master_key" {
  secret_id = aws_secretsmanager_secret.master_key.id
  # When the operator passes litellm_master_key, use it verbatim. Otherwise
  # fall back to the auto-generated `sk-…` value (trial / OSS path).
  secret_string = coalesce(var.litellm_master_key, "sk-${random_password.master_key.result}")
}

# LITELLM_LICENSE — only created when the operator supplies one. The
# task-execution role gets GetSecretValue via iam.tf, and gateway + backend
# pick the env var up through shared_secrets in ecs.tf.
resource "aws_secretsmanager_secret" "license" {
  count = var.litellm_license == "" ? 0 : 1

  name                    = "${local.name}-license"
  description             = "LITELLM_LICENSE for gateway + backend."
  recovery_window_in_days = 0

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "license" {
  count = var.litellm_license == "" ? 0 : 1

  secret_id     = aws_secretsmanager_secret.license[0].id
  secret_string = var.litellm_license
}

# UI_PASSWORD — backend-only. Same pattern as license: only created when
# the operator supplies one. The execution role gets GetSecretValue via
# iam.tf, and the backend task picks the env var up through
# backend_managed_secrets in ecs.tf.
resource "aws_secretsmanager_secret" "ui_password" {
  count = var.ui_password == "" ? 0 : 1

  name                    = "${local.name}-ui-password"
  description             = "UI_PASSWORD for the backend (UI admin login)."
  recovery_window_in_days = 0

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "ui_password" {
  count = var.ui_password == "" ? 0 : 1

  secret_id     = aws_secretsmanager_secret.ui_password[0].id
  secret_string = var.ui_password
}

resource "aws_secretsmanager_secret" "db_master_password" {
  name                    = "${local.name}-db-master-password"
  description             = "Aurora master-user password - bootstrap only. Runtime auth is IAM-token."
  recovery_window_in_days = 0

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "db_master_password" {
  secret_id = aws_secretsmanager_secret.db_master_password.id
  secret_string = jsonencode({
    username = var.db_master_username
    password = random_password.db_master_password.result
    host     = aws_rds_cluster.this.endpoint
    port     = aws_rds_cluster.this.port
    dbname   = var.db_name
  })
}
