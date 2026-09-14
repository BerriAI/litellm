# Auto-runs the two manual steps that used to follow `terraform apply`:
#
#   1. Create the IAM-authed Postgres user (litellm_app) — uses the postgres:16
#      image with the master password from Secrets Manager. Only relevant to
#      the Aurora cluster this module creates, so it is skipped when
#      create_database = false.
#   2. Run prisma migrate deploy — reuses the existing aws_ecs_task_definition
#      .migrations task def from migrations.tf. Runs against an existing
#      database too, and only disappears when there is no database at all.
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
  count = var.create_database ? 1 : 0
  name  = "${local.name}-bootstrap-secrets-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [aws_secretsmanager_secret.db_master_password[0].arn]
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "task_execution_bootstrap_secrets" {
  count      = var.create_database ? 1 : 0
  role       = aws_iam_role.task_execution.name
  policy_arn = aws_iam_policy.bootstrap_secrets[0].arn
}

# ---------- Bootstrap task def ----------
resource "aws_cloudwatch_log_group" "bootstrap_db" {
  count             = var.create_database ? 1 : 0
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
  count                    = var.create_database ? 1 : 0
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
      { name = "PGHOST", value = aws_rds_cluster.this[0].endpoint },
      { name = "PGPORT", value = tostring(aws_rds_cluster.this[0].port) },
      { name = "PGUSER", value = var.db_master_username },
      { name = "PGDATABASE", value = var.db_name },
      { name = "BOOTSTRAP_SQL", value = local.bootstrap_sql },
    ]
    secrets = [
      # `:password::` extracts the password field out of the JSON secret.
      { name = "PGPASSWORD", valueFrom = "${aws_secretsmanager_secret.db_master_password[0].arn}:password::" },
    ]

    entryPoint = ["sh", "-c"]
    command    = ["echo \"$BOOTSTRAP_SQL\" | psql -v ON_ERROR_STOP=1"]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.bootstrap_db[0].name
        awslogs-region        = var.region
        awslogs-stream-prefix = "bootstrap"
      }
    }
  }])

  tags = local.tags
}

# ---------- Bootstrap trigger ----------
resource "terraform_data" "bootstrap_db" {
  count = var.create_database ? 1 : 0

  triggers_replace = {
    cluster_resource_id = aws_rds_cluster.this[0].cluster_resource_id
    task_def_revision   = aws_ecs_task_definition.bootstrap_db[0].revision
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    environment = {
      CLUSTER  = aws_ecs_cluster.this.name
      TASK_DEF = aws_ecs_task_definition.bootstrap_db[0].arn
      SUBNETS  = join(",", local.private_subnet_ids)
      SG       = join(",", local.task_security_group_ids)
      REGION   = var.region
      LOG_GRP  = aws_cloudwatch_log_group.bootstrap_db[0].name
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

  # Same secret-by-ARN gap as the migration below. The margin here is wide,
  # since the writer instance takes minutes while the version write does not,
  # but both hang off the cluster in parallel and nothing orders them.
  depends_on = [
    aws_rds_cluster_instance.writer,
    aws_iam_role_policy_attachment.task_execution_bootstrap_secrets,
    aws_secretsmanager_secret_version.db_master_password,
  ]
}

# ---------- Migration trigger ----------
# Reuses the task definition from migrations.tf — this resource just invokes
# it and waits.
resource "terraform_data" "migration" {
  count = local.database_enabled ? 1 : 0

  triggers_replace = {
    task_def_revision = aws_ecs_task_definition.migrations[0].revision
    bootstrap_id      = join(",", terraform_data.bootstrap_db[*].id)
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    environment = {
      CLUSTER  = aws_ecs_cluster.this.name
      TASK_DEF = aws_ecs_task_definition.migrations[0].arn
      SUBNETS  = join(",", local.private_subnet_ids)
      SG       = join(",", local.task_security_group_ids)
      REGION   = var.region
      LOG_GRP  = aws_cloudwatch_log_group.migrations[0].name
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

  # A container reads a secret by ARN, so Terraform sees no edge from the
  # ARN to the _version that gives it a value. The managed-Aurora path hides
  # that: the cluster create takes long enough that the version always lands
  # first. A bring-your-own database has nothing slow in between, so without
  # this the run-task below can fire against a valueless secret and fail the
  # apply with ResourceInitializationError.
  depends_on = [
    terraform_data.bootstrap_db,
    aws_secretsmanager_secret_version.database_url,
  ]
}
