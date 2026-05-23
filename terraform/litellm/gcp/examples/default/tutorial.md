# Deploy LiteLLM on Google Cloud

<walkthrough-tutorial-duration duration="20"></walkthrough-tutorial-duration>

This guided walkthrough deploys the **LiteLLM AI Gateway** into your own
Google Cloud project using Terraform. You get the full componentized proxy —
gateway, backend, and dashboard on Cloud Run, fronted by an external HTTP(S)
load balancer, backed by Cloud SQL (Postgres), Memorystore (Redis), and a
GCS bucket.

Everything runs from this Cloud Shell session — Terraform is already
installed here, so there's nothing to set up on your machine.

## Choose your project

Pick the Google Cloud project to deploy into. Everything the stack creates is
billed to and lives in this project.

<walkthrough-project-setup></walkthrough-project-setup>

Set it as the active project for this session:

```bash
gcloud config set project <walkthrough-project-id/>
```

## Enable the required APIs

The stack uses Cloud Run, Cloud SQL, Memorystore, Secret Manager, Serverless
VPC Access, Compute, Service Networking, Cloud Storage, and Artifact Registry.
Enable them all in one call (this can take a minute):

```bash
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  redis.googleapis.com \
  secretmanager.googleapis.com \
  vpcaccess.googleapis.com \
  compute.googleapis.com \
  servicenetworking.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com
```

## Deploy

Tell Terraform which project to use, then initialize and apply. The apply
provisions everything, runs the database migration, and only then starts the
services — so it takes **15-20 minutes** on the first run (Cloud SQL and the
load balancer are the slow parts).

```bash
export TF_VAR_project=$(gcloud config get-value project)
terraform init
terraform apply
```

Review the plan and type `yes` to proceed.

<walkthrough-footnote>This trial deploy serves plain HTTP and auto-generates
a master key. For a production deploy, set `lb_domains` for a managed TLS
cert and supply your own master key — see the module README.</walkthrough-footnote>

## You're live

Print the proxy URL (the dashboard is at `/`, the OpenAI-compatible API at
`/v1/*`):

```bash
terraform output lb_url
```

Fetch the auto-generated admin / master key — log into the dashboard with
username `admin` and this value:

```bash
gcloud secrets versions access latest \
  --secret="$(terraform output -raw master_key_secret_id)"
```

Send a test request (replace `URL` and `KEY` with the two values above):

```bash
curl "$(terraform output -raw lb_url)/v1/models" \
  -H "Authorization: Bearer $(gcloud secrets versions access latest --secret="$(terraform output -raw master_key_secret_id)")"
```

## Add a model

Edit `terraform.tfvars` to register models and provider keys, then re-apply.
Store provider API keys in Secret Manager and reference them, e.g.:

```bash
echo -n "sk-proj-..." | gcloud secrets create openai-api-key --data-file=-
```

```hcl
proxy_config = {
  model_list = [{
    model_name     = "gpt-4o"
    litellm_params = { model = "openai/gpt-4o", api_key = "os.environ/OPENAI_API_KEY" }
  }]
}
gateway_extra_secrets = {
  OPENAI_API_KEY = "projects/<walkthrough-project-id/>/secrets/openai-api-key"
}
```

Then `terraform apply` again.

## Clean up

To tear everything down when you're done:

```bash
terraform destroy
```

<walkthrough-footnote>Cloud SQL has deletion protection on by default, so
destroy will refuse until you set `cloudsql_deletion_protection = false` (and
`gcs_force_destroy = true` for a non-empty bucket) and re-apply. That's a
guard against accidental data loss.</walkthrough-footnote>

## Done

<walkthrough-conclusion-trophy></walkthrough-conclusion-trophy>

You've deployed LiteLLM on Google Cloud. For configuration options (models,
TLS, sizing, multi-tenant deploys), see the
[module README](https://github.com/BerriAI/litellm/blob/main/terraform/litellm/gcp/README.md).
