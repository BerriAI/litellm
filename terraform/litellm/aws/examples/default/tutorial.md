# Deploy LiteLLM on AWS

This walkthrough deploys the **LiteLLM AI Gateway** into your own AWS account
using Terraform. You get the full componentized proxy — gateway, backend, and
dashboard on ECS Fargate, fronted by an Application Load Balancer, backed by
Aurora Postgres, ElastiCache (Redis), and an S3 bucket.

The fastest path is **AWS CloudShell**, which already has the AWS CLI and your
credentials wired up. Click the button in the
[module README](https://github.com/BerriAI/litellm/blob/main/terraform/litellm/aws/README.md),
or open <https://console.aws.amazon.com/cloudshell> and follow along.

## 1. Get the code

In CloudShell (or any machine with the AWS CLI configured):

```bash
git clone --depth 1 https://github.com/BerriAI/litellm.git
cd litellm/terraform/litellm/aws/examples/default
```

## 2. Deploy

`deploy.sh` installs a pinned, checksum-verified Terraform (CloudShell doesn't
ship one), then runs `terraform init` + `terraform apply`:

```bash
./deploy.sh
```

Review the plan and type `yes`. The apply provisions the VPC, Aurora cluster,
Redis, S3, and the three ECS services, bootstraps the database, runs the
schema migration, and only then starts the services — so it takes **15-20
minutes** on the first run.

> Prefer to drive Terraform yourself? `terraform init && terraform apply` works
> too, as long as Terraform is already installed.

## 3. You're live

```bash
terraform output alb_url
```

The dashboard is at `/`, the OpenAI-compatible API at `/v1/*`. Log in with
username `admin` and the auto-generated master key:

```bash
aws secretsmanager get-secret-value \
  --secret-id "$(terraform output -raw master_key_secret_arn)" \
  --query SecretString --output text
```

The ALB takes a few minutes to pass health checks after apply returns.

## 4. Customize (optional)

This trial deploy serves plain HTTP and registers no models. For a real
deployment, copy and edit the tfvars file, then re-apply:

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit: region, tenant/env, acm_certificate_arn (TLS), proxy_config (models),
# gateway_extra_secrets (provider API keys).
terraform apply
```

Provider API keys go in AWS Secrets Manager and are referenced by ARN — see
the [module README](https://github.com/BerriAI/litellm/blob/main/terraform/litellm/aws/README.md)
for the full configuration surface (TLS, models, sizing, multi-tenant).

## 5. Clean up

```bash
terraform destroy
```

Aurora takes a final snapshot and the S3 bucket refuses to delete while
non-empty (data-loss guards). Set `skip_final_snapshot = true` /
`s3_force_destroy = true` for an ephemeral trial you don't mind losing.
