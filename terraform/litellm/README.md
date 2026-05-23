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
# Direct from this monorepo (works today, no registry needed):
module "litellm" {
  source = "github.com/BerriAI/litellm//terraform/litellm/aws?ref=<tag>"
  # ... inputs ...
}

# Or, once published (see "Publishing to the Terraform Registry" below):
module "litellm" {
  source  = "BerriAI/litellm/aws"   # registry shorthand
  version = "~> 1.86"
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

## One-click deploy

The `examples/default/` roots are **zero-config trial deploys**: a bare
`terraform apply` with no tfvars brings up a working (HTTP-only) instance —
sensible region/zone defaults, an auto-generated master key, and (on GCP) an
auto-created Artifact Registry proxy so Cloud Run can pull the images. Launch
straight from your browser:

### GCP — Open in Cloud Shell

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https://github.com/BerriAI/litellm&cloudshell_workspace=terraform/litellm/gcp/examples/default&cloudshell_tutorial=tutorial.md)

Opens Cloud Shell (Terraform is pre-installed), clones the repo, and starts a
guided walkthrough that enables the APIs and runs `terraform apply` against
your active project.

### AWS — Open in CloudShell

[![Open in AWS CloudShell](https://img.shields.io/badge/Open%20in-AWS%20CloudShell-FF9900?logo=amazonaws&logoColor=white)](https://console.aws.amazon.com/cloudshell/home)

CloudShell already has your AWS credentials. Once it opens, paste:

```bash
git clone --depth 1 https://github.com/BerriAI/litellm.git
cd litellm/terraform/litellm/aws/examples/default && ./deploy.sh
```

`deploy.sh` installs a pinned, checksum-verified Terraform (CloudShell doesn't
ship one) and applies the stack. Full steps:
[aws walkthrough](aws/examples/default/tutorial.md).

> **Trial vs. production.** The one-click roots serve **plain HTTP** and
> register no models — fine for kicking the tires, not for production. For a
> real deploy, add TLS (`acm_certificate_arn` on AWS, `lb_domains` on GCP),
> register models via `proxy_config`, and supply your own master key. See the
> per-stack READMEs.

## Components

The proxy is split into three deployables:

| Component | Default image                            | Port | Role                                                                 |
| --------- | ---------------------------------------- | ---- | -------------------------------------------------------------------- |
| `gateway` | `ghcr.io/berriai/litellm-gateway:v1.86.0-dev` | 4000 | LLM data plane (`/v1/chat/completions`, `/v1/embeddings`, …)         |
| `backend` | `ghcr.io/berriai/litellm-backend:v1.86.0-dev` | 4001 | Management API (`/key/*`, `/user/*`, `/team/*`, `/model/*`, …)       |
| `ui`      | `ghcr.io/berriai/litellm-ui:v1.86.0-dev`      | 3000 | Static Next.js dashboard served by nginx                             |

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
point at the public `ghcr.io/berriai/litellm-<component>:v1.86.0-dev`
images, so the stack is runnable end-to-end without pre-flight setup —
pin to a specific tag for production:

- **AWS** can pull from any registry the task execution role can reach.
  The role gets `AmazonECSTaskExecutionRolePolicy` attached, which grants
  ECR pull permissions for repositories in the same account. GHCR is
  anonymous-readable, so the defaults work as-is.

- **GCP Cloud Run** can only pull from Artifact Registry or
  `gcr.io`-style registries — it rejects `ghcr.io`. By default the GCP
  stack auto-creates an Artifact Registry **remote repository** that
  proxies `https://ghcr.io` (`create_image_proxy_repo = true`), so the
  upstream images pull with no manual mirroring. Set
  `create_image_proxy_repo = false` and supply your own `image_registry`
  to opt out.

## Migrations

LiteLLM's proxy runs `prisma migrate deploy` at startup, but on first apply
the gateway/backend can race the empty database. Both stacks expose a
one-off migration task that runs `python3 /app/run.py` (assembles
`DATABASE_URL` from the `DATABASE_*` env vars, then `prisma migrate deploy`)
from the dedicated `ghcr.io/berriai/litellm-migrations` image:

- AWS: an `aws_ecs_task_definition` (`litellm-migrations`). Run with
  `aws ecs run-task` — the command is printed in `terraform output`.
- GCP: a `google_cloud_run_v2_job` (`litellm-migrations`). Run with
  `gcloud run jobs execute` — the command is printed in `terraform output`.

Run the migration job once after the first `terraform apply` and before the
gateway/backend services start serving traffic.

## What's not included

- Custom domains / DNS. Both stacks support TLS out of the box — an ACM
  cert (`acm_certificate_arn`) on AWS, a Google-managed cert (`lb_domains`)
  on GCP — and `terraform plan` refuses to provision a plaintext LB unless
  you explicitly opt in (`allow_plaintext_alb` / `allow_plaintext_lb`,
  which the one-click trial roots default to true). You still bring your
  own DNS name and point it at the LB; see the per-stack "TLS" sections.
- Remote state backends. Default local state — add an `s3` or `gcs`
  backend block to `versions.tf` when graduating to a team environment.
- Observability beyond the cloud provider's defaults (CloudWatch logs on
  AWS, Cloud Logging on GCP). Wire your own Prometheus / Datadog / Langfuse
  via the `*_extra_env` variables.

## Publishing to the Terraform Registry

These modules are registry-conformant — each is self-contained, declares no
`provider` block, ships a `README.md` + `examples/default/`, and documents
every variable/output. The public registry only indexes a module at the
**root** of a repo named `terraform-<PROVIDER>-<NAME>`, so the two stacks
here are mirrored out to dedicated repos rather than published in place:

| Module                  | Mirror repo                         | Registry source        |
| ----------------------- | ----------------------------------- | ---------------------- |
| `terraform/litellm/aws` | `BerriAI/terraform-aws-litellm`     | `BerriAI/litellm/aws`    |
| `terraform/litellm/gcp` | `BerriAI/terraform-google-litellm`  | `BerriAI/litellm/google` |

The [`Publish Terraform modules`](../../.github/workflows/terraform-modules-publish.yml)
GitHub Actions workflow does the mirroring: it `git subtree split`s each
module subdirectory into its mirror repo and tags it with the version you
pass. Run it manually (Actions → Publish Terraform modules → enter `vX.Y.Z`)
after a release. One-time setup (create the mirror repos, connect them to the
registry, add the `TERRAFORM_REGISTRY_SYNC_TOKEN` secret) is documented in
the workflow header.

Until a version is published to the registry, consume the modules straight
from this repo with the `github.com/BerriAI/litellm//terraform/litellm/<stack>?ref=<tag>`
source shown at the top.
