# LiteLLM Terraform stacks

Two self-contained, reusable Terraform **modules** that deploy the
**componentized** LiteLLM proxy — the gateway, backend, and UI as three
independent containers (see `helm/litellm/` for the canonical chart with the
same split).

Each module declares **no `provider` block of its own**, so it can be called
with `count` / `for_each` / `depends_on` and the caller controls region,
assume-role / impersonation, aliases, and `default_tags`. A ready-to-run root
that wires the provider lives at `<stack>/examples/default/` — that's the
one-command deploy path. To embed a stack in your own config, call the module
by source:

```hcl
module "litellm" {
  source = "github.com/BerriAI/litellm//terraform/litellm/aws?ref=<tag>"
  # ... inputs ...
}
```

| Stack  | Compute     | Database (writer + reader)         | Cache       | Object store | Public entrypoint  |
| ------ | ----------- | ---------------------------------- | ----------- | ------------ | ------------------ |
| `aws/` | ECS Fargate | Aurora Postgres (IAM auth)         | ElastiCache | S3           | Application LB     |
| `gcp/` | Cloud Run   | Cloud SQL Postgres (password auth) | Memorystore | GCS          | External HTTPS LB  |

Each stack creates its own VPC and managed data stores — from
`<stack>/examples/default/`, drop in a tfvars file and run `terraform apply`.
Both stacks support a typed `proxy_config` input (mirrors `helm/litellm`'s
`gateway.config.proxy_config`) and per-component extra env vars /
secret-manager refs.

## Components

The proxy is split into three deployables:

| Component | Default image                            | Port | Role                                                                 |
| --------- | ---------------------------------------- | ---- | -------------------------------------------------------------------- |
| `gateway` | `ghcr.io/berriai/litellm-gateway:main-stable` | 4000 | LLM data plane (`/v1/chat/completions`, `/v1/embeddings`, …)         |
| `backend` | `ghcr.io/berriai/litellm-backend:main-stable` | 4001 | Management API (`/key/*`, `/user/*`, `/team/*`, `/model/*`, …)       |
| `ui`      | `ghcr.io/berriai/litellm-ui:main-stable`      | 3000 | Static Next.js dashboard served by nginx                             |

The load balancer routes gateway path prefixes (mirrored verbatim from
`gateway/routes/allowlist.py`) to the gateway, UI asset paths (`/`,
`/litellm-asset-prefix/*`, `/_next/*`, `/favicon.ico`) to the UI, and
everything else to the backend.

## Architecture

### AWS (`terraform/litellm/aws/`)

```
                        ┌───────────────────────────────────────┐
                        │            Public Internet            │
                        └─────────────────┬─────────────────────┘
                                          │ HTTP/80
                          ┌───────────────▼───────────────┐
                          │   Application Load Balancer   │
                          │   (path-routing listener)     │
                          └─┬─────────────┬─────────────┬─┘
                            │             │             │
            UI assets, /    │  /v1/chat,  │   /key/*    │
            /_next/*, …     │  /v1/embed, │   /user/*   │
                            │  …          │   …         │
              ┌─────────────▼───┐  ┌──────▼──────┐  ┌───▼──────────────┐
              │    ECS Service  │  │ ECS Service │  │   ECS Service    │
              │       (ui)      │  │  (gateway)  │  │    (backend)     │
              │   Fargate :3000 │  │ Fargate:4000│  │  Fargate :4001   │
              └─────────────────┘  └──────┬──────┘  └────────┬─────────┘
                                          │                  │
              ┌─── private subnets (one per AZ) ──────────────────────┐
              │                                                       │
              │   ┌────────────────────────┐    ┌────────────────┐   │
              │   │  Aurora Postgres       │    │  ElastiCache   │   │
              │   │  cluster (IAM auth)    │    │  Redis (1 node)│   │
              │   │  ┌───────┐  ┌───────┐  │    └────────────────┘   │
              │   │  │writer │  │reader │  │                         │
              │   │  └───────┘  └───────┘  │    ┌────────────────┐   │
              │   └────────────────────────┘    │  S3 bucket     │   │
              │                                  │  (versioned)   │   │
              │   ┌────────────────────────┐    └────────────────┘   │
              │   │  Secrets Manager       │                         │
              │   │  • LITELLM_MASTER_KEY  │    ┌────────────────┐   │
              │   │  • DB master password  │    │ One-off ECS    │   │
              │   │  • user-supplied API   │    │ task: prisma   │   │
              │   │    keys (referenced)   │    │ migrate deploy │   │
              │   └────────────────────────┘    └────────────────┘   │
              │                                                       │
              └─── VPC ───────────────────────────────────────────────┘
                          │ NAT gateway in one public subnet
                          ▼
                    egress to LLM providers
```

### GCP (`terraform/litellm/gcp/`)

```
                        ┌───────────────────────────────────────┐
                        │            Public Internet            │
                        └─────────────────┬─────────────────────┘
                                          │ HTTP/80
                          ┌───────────────▼───────────────┐
                          │ External HTTPS Load Balancer  │
                          │   (global, URL map routing)   │
                          └─┬─────────────┬─────────────┬─┘
                            │             │             │
                            │ Serverless NEGs (one per service)
                            │             │             │
              ┌─────────────▼───┐  ┌──────▼──────┐  ┌───▼──────────────┐
              │   Cloud Run     │  │  Cloud Run  │  │    Cloud Run     │
              │      (ui)       │  │  (gateway)  │  │    (backend)     │
              │      :3000      │  │   :4000     │  │      :4001       │
              └─────────────────┘  └──────┬──────┘  └────────┬─────────┘
                                          │                  │
                                          │ Serverless VPC Access connector
              ┌─── VPC (private services access range) ──────────────────┐
              │                                                          │
              │   ┌────────────────────────┐    ┌──────────────────┐    │
              │   │  Cloud SQL Postgres    │    │  Memorystore     │    │
              │   │  ┌───────┐  ┌───────┐  │    │  Redis           │    │
              │   │  │writer │  │reader │  │    └──────────────────┘    │
              │   │  └───────┘  └───────┘  │                            │
              │   └────────────────────────┘    ┌──────────────────┐    │
              │                                  │  GCS bucket      │    │
              │   ┌────────────────────────┐    │  (versioned)     │    │
              │   │  Secret Manager        │    └──────────────────┘    │
              │   │  • LITELLM_MASTER_KEY  │                            │
              │   │  • DB password         │    ┌──────────────────┐    │
              │   │  • user-supplied API   │    │ Cloud Run Job:   │    │
              │   │    keys (referenced)   │    │ prisma migrate   │    │
              │   └────────────────────────┘    │ deploy           │    │
              │                                  └──────────────────┘    │
              └──────────────────────────────────────────────────────────┘
```

## Images

Both stacks take per-component image references as variables. The defaults
point at the public `ghcr.io/berriai/litellm-<component>:main-stable`
images, so the stack is runnable end-to-end without pre-flight setup —
pin to a specific tag for production:

- **AWS** can pull from any registry the task execution role can reach.
  The role gets `AmazonECSTaskExecutionRolePolicy` attached, which grants
  ECR pull permissions for repositories in the same account.

- **GCP Cloud Run** can only pull from Artifact Registry or
  `gcr.io`-style registries. To use images hosted elsewhere, mirror them
  into Artifact Registry first.

## Migrations

LiteLLM's proxy runs `prisma migrate deploy` at startup, but on first apply
the gateway/backend can race the empty database. Both stacks expose a
one-off migration task that runs `python litellm/proxy/prisma_migration.py`
against the backend image:

- AWS: an `aws_ecs_task_definition` (`litellm-migrations`). Run with
  `aws ecs run-task` — the command is printed in `terraform output`.
- GCP: a `google_cloud_run_v2_job` (`litellm-migrations`). Run with
  `gcloud run jobs execute` — the command is printed in `terraform output`.

Run the migration job once after the first `terraform apply` and before the
gateway/backend services start serving traffic.

## What's not included

- TLS certificates / custom domains. Both stacks expose plain-HTTP load
  balancers; bring your own ACM cert (AWS) or managed cert (GCP) and wire
  it into the LB resource.
- Remote state backends. Default local state — add an `s3` or `gcs`
  backend block to `versions.tf` when graduating to a team environment.
- Observability beyond the cloud provider's defaults (CloudWatch logs on
  AWS, Cloud Logging on GCP). Wire your own Prometheus / Datadog / Langfuse
  via the `*_extra_env` variables.
