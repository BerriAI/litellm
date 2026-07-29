# Deploy LiteLLM on GCP

<walkthrough-tutorial-duration duration="25"></walkthrough-tutorial-duration>

This walkthrough provisions the full LiteLLM stack on GCP via Cloud Run, Cloud SQL, Memorystore Redis, and an external HTTPS load balancer. You'll answer a few prompts; DeployStack writes a `terraform.tfvars` and runs `terraform apply` against the project you select.

## Prerequisites

<walkthrough-project-billing-setup></walkthrough-project-billing-setup>

Pick the GCP project you want to deploy into, then make sure billing is enabled on it. The stack provisions paid resources (Cloud SQL, Memorystore, an LB anycast IP).

## Enable required APIs

The stack needs these APIs enabled in the target project. Click to enable, or run the gcloud command below.

<walkthrough-enable-apis apis="run.googleapis.com,sqladmin.googleapis.com,redis.googleapis.com,secretmanager.googleapis.com,vpcaccess.googleapis.com,compute.googleapis.com,servicenetworking.googleapis.com,storage.googleapis.com,artifactregistry.googleapis.com"></walkthrough-enable-apis>

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

## Create the Artifact Registry passthrough to GHCR

Cloud Run only pulls from Artifact Registry, `gcr.io`, or `docker.io`; it rejects `ghcr.io` URIs at apply time. The four LiteLLM images live on GHCR, so the stack needs a remote Artifact Registry repo pointed at GHCR. This is a one-time setup per project.

```bash
gcloud artifacts repositories create litellm \
  --repository-format=docker \
  --location=<walkthrough-watcher-constant key="region" default="us-central1"/> \
  --mode=remote-repository \
  --remote-repo-config-desc="GitHub Container Registry passthrough" \
  --remote-docker-repo=https://ghcr.io
```

If the repo already exists, this command exits with a clear error and you can move on. When `deploystack install` prompts for `image_registry`, enter `<region>-docker.pkg.dev/<your-project>/litellm/berriai` (substituting your region and project). The shipped default contains a `PROJECT_ID` placeholder that will fail at apply time if left unedited.

## (Optional) Set tenant secrets

The stack auto-generates a `LITELLM_MASTER_KEY` if you don't supply one. If you have an enterprise license or want a pre-chosen master key, export them as `TF_VAR_*` env vars before running the installer so they end up in Secret Manager but not in `terraform.tfvars`.

```bash
export TF_VAR_litellm_master_key="sk-..."   # optional; auto-generated if omitted
export TF_VAR_litellm_license="lic-..."     # optional; OSS-only without it
export TF_VAR_ui_password="..."             # optional; falls back to master_key for UI login
```

Skip this step entirely for a trial deploy.

## Run the installer

DeployStack will prompt for project, region, tenant, env, image tag, `image_registry`, and TLS posture, then run `terraform apply`. Open `<walkthrough-editor-open-file filePath="terraform/litellm/gcp/examples/default/deploystack.json">deploystack.json</walkthrough-editor-open-file>` if you want to see the prompt definitions first.

```bash
deploystack install
```

The first apply takes 20-25 minutes; most of that is Cloud SQL provisioning. The migration Cloud Run Job runs automatically once the database is ready, and only then do gateway, backend, and UI start.

## Grab the LB URL

```bash
terraform output lb_url
```

For trial deploys (`allow_plaintext_lb=true`), this is `http://<lb-ip>`. The UI lives at `/ui`; sign in with username `admin` and the master key:

```bash
gcloud secrets versions access latest \
  --secret="$(terraform output -raw master_key_secret_id)"
```

## Going to TLS

If you picked `allow_plaintext_lb=true` to bootstrap but want HTTPS for real, point a DNS A record at the LB IP, then re-run terraform with `lb_domains` set and `allow_plaintext_lb` removed:

```bash
terraform apply \
  -var 'lb_domains=["proxy.example.com"]'
```

Google-managed certs sit in `PROVISIONING` for 15-60 minutes after DNS propagates. You can watch the state with `gcloud compute ssl-certificates describe <tenant>-litellm-<env>-cert`.

## Adding provider API keys

Provider keys (OpenAI, Anthropic, etc.) belong in Secret Manager, not in `terraform.tfvars`. Create the secret first, then reference its resource ID from `gateway_extra_secrets` and re-apply:

```bash
echo -n "sk-proj-..." | gcloud secrets create openai-api-key --data-file=-
```

Edit `terraform.tfvars`:

```hcl
gateway_extra_secrets = {
  OPENAI_API_KEY = "projects/<your-project>/secrets/openai-api-key"
}
proxy_config = {
  model_list = [
    {
      model_name = "gpt-4o"
      litellm_params = {
        model   = "openai/gpt-4o"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
  ]
}
```

Then `terraform apply`.

## Tearing it all down

```bash
deploystack uninstall
```

`cloudsql_deletion_protection` is `true` by default; flip it to `false` in `terraform.tfvars` and apply before uninstalling if you actually want the DB gone. Same goes for `gcs_force_destroy` on the bucket.

## You're done

<walkthrough-conclusion-trophy></walkthrough-conclusion-trophy>

Full configuration reference is in `<walkthrough-editor-open-file filePath="terraform/litellm/gcp/README.md">README.md</walkthrough-editor-open-file>`, and every input variable on the underlying module lives in `<walkthrough-editor-open-file filePath="terraform/litellm/gcp/variables.tf">variables.tf</walkthrough-editor-open-file>`.
