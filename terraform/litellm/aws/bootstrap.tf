# Auto-runs the two manual steps that used to follow `terraform apply`:
#
#   1. Create the IAM-authed Postgres user (litellm_app) — uses the postgres:16
#      image with the master password from Secrets Manager.
#   2. Run prisma migrate deploy — reuses the existing aws_ecs_task_definition
#      .migrations task def from migrations.tf.
#
# Both are invoked via `terraform_data` provisioners. Gateway/backend services
# in ecs.tf depend on `terraform_data.migration`, so on a fresh apply they
# don't start until the schema is in place — no crash-loop window.
#
# Triggers:
#   - bootstrap_db re-runs if the Aurora cluster is recreated, or if the
#     bootstrap task definition (image/SQL) changes.
#   - migration re-runs if the migration task def revision changes (e.g., new
#     backend image with new prisma migration files) or if bootstrap re-ran.
#
# Requires `aws` CLI on the machine running terraform. For laptop usage that's
# fine; for CI/CD the runner image needs `aws`.

# ---------- IAM ----------
# Execution role can already read the runtime secrets (master_key, user-provided
# extras — see iam.tf). The DB master password lives in a separate secret used
# only here, so we grant access in an additive policy.
resource "aws_iam_policy" "bootstrap_secrets" {
  name = "${local.name}-bootstrap-secrets-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.db_master_password.arn]
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "task_execution_bootstrap_secrets" {
  role       = aws_iam_role.task_execution.name
  policy_arn = aws_iam_policy.bootstrap_secrets.arn
}

# ---------- Bootstrap task def ----------
resource "aws_cloudwatch_log_group" "bootstrap_db" {
  name              = "/ecs/${local.name}/bootstrap-db"
  retention_in_days = var.log_retention_days

  tags = local.tags
}

locals {
  # Idempotent: CREATE USER is wrapped in DO/EXCEPTION; GRANTs are
  # idempotent by definition (re-granting is a no-op). Safe to re-run on
  # any subsequent apply.
  bootstrap_sql = <<-SQL
    DO $$
    BEGIN
      CREATE USER ${var.db_username};
    EXCEPTION WHEN duplicate_object THEN NULL;
    END $$;
    GRANT rds_iam TO ${var.db_username};
    GRANT ALL PRIVILEGES ON DATABASE ${var.db_name} TO ${var.db_username};
    GRANT ALL ON SCHEMA public TO ${var.db_username};
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${var.db_username};
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${var.db_username};
  SQL
}

resource "aws_ecs_task_definition" "bootstrap_db" {
  family                   = "${local.name}-bootstrap-db"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 256
  memory                   = 512
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "psql"
    image     = "postgres:16-alpine"
    essential = true

    environment = [
      { name = "PGHOST", value = aws_rds_cluster.this.endpoint },
      { name = "PGPORT", value = tostring(aws_rds_cluster.this.port) },
      { name = "PGUSER", value = var.db_master_username },
      { name = "PGDATABASE", value = var.db_name },
      { name = "BOOTSTRAP_SQL", value = local.bootstrap_sql },
    ]
    secrets = [
      # `:password::` extracts the password field out of the JSON secret.
      { name = "PGPASSWORD", valueFrom = "${aws_secretsmanager_secret.db_master_password.arn}:password::" },
    ]

    entryPoint = ["sh", "-c"]
    command    = ["echo \"$BOOTSTRAP_SQL\" | psql -v ON_ERROR_STOP=1"]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.bootstrap_db.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "bootstrap"
      }
    }
  }])

  tags = local.tags
}

# ---------- Bootstrap trigger ----------
resource "terraform_data" "bootstrap_db" {
  triggers_replace = {
    cluster_resource_id = aws_rds_cluster.this.cluster_resource_id
    task_def_revision   = aws_ecs_task_definition.bootstrap_db.revision
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    environment = {
      CLUSTER  = aws_ecs_cluster.this.name
      TASK_DEF = aws_ecs_task_definition.bootstrap_db.arn
      SUBNETS  = join(",", aws_subnet.private[*].id)
      SG       = aws_security_group.tasks.id
      REGION   = var.region
      LOG_GRP  = aws_cloudwatch_log_group.bootstrap_db.name
    }
    command = <<-EOT
      set -euo pipefail
      task_arn=$(aws ecs run-task --region "$REGION" --cluster "$CLUSTER" \
        --launch-type FARGATE --task-definition "$TASK_DEF" \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=DISABLED}" \
        --query 'tasks[0].taskArn' --output text)
      echo "bootstrap task: $task_arn"
      aws ecs wait tasks-stopped --region "$REGION" --cluster "$CLUSTER" --tasks "$task_arn"
      task_id=$(echo "$task_arn" | awk -F/ '{print $NF}')
      exit_code=$(aws ecs describe-tasks --region "$REGION" --cluster "$CLUSTER" --tasks "$task_id" \
        --query 'tasks[0].containers[0].exitCode' --output text)
      if [ "$exit_code" != "0" ]; then
        echo "Bootstrap failed (exit=$exit_code). Logs: $LOG_GRP" >&2
        exit 1
      fi
    EOT
  }

  depends_on = [
    aws_rds_cluster_instance.writer,
    aws_iam_role_policy_attachment.task_execution_bootstrap_secrets,
  ]
}

# ---------- Migration trigger ----------
# Reuses the task definition from migrations.tf — this resource just invokes
# it and waits.
resource "terraform_data" "migration" {
  triggers_replace = {
    task_def_revision = aws_ecs_task_definition.migrations.revision
    bootstrap_id      = terraform_data.bootstrap_db.id
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    environment = {
      CLUSTER  = aws_ecs_cluster.this.name
      TASK_DEF = aws_ecs_task_definition.migrations.arn
      SUBNETS  = join(",", aws_subnet.private[*].id)
      SG       = aws_security_group.tasks.id
      REGION   = var.region
      LOG_GRP  = aws_cloudwatch_log_group.migrations.name
    }
    command = <<-EOT
      set -euo pipefail
      task_arn=$(aws ecs run-task --region "$REGION" --cluster "$CLUSTER" \
        --launch-type FARGATE --task-definition "$TASK_DEF" \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=DISABLED}" \
        --query 'tasks[0].taskArn' --output text)
      echo "migration task: $task_arn"
      aws ecs wait tasks-stopped --region "$REGION" --cluster "$CLUSTER" --tasks "$task_arn"
      task_id=$(echo "$task_arn" | awk -F/ '{print $NF}')
      exit_code=$(aws ecs describe-tasks --region "$REGION" --cluster "$CLUSTER" --tasks "$task_id" \
        --query 'tasks[0].containers[0].exitCode' --output text)
      if [ "$exit_code" != "0" ]; then
        echo "Migration failed (exit=$exit_code). Logs: $LOG_GRP" >&2
        exit 1
      fi
    EOT
  }

  depends_on = [terraform_data.bootstrap_db]
}
