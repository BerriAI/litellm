# LiteLLM on Azure (Container Apps + App Gateway)

Deploys the componentized LiteLLM proxy on Azure:

- **VNet** with three isolated subnets: containers (delegated to Azure
  Container Apps), private endpoints (Postgres / Redis / Storage / Key
  Vault), Application Gateway
- **Azure Database for PostgreSQL Flexible Server**, single instance,
  with Entra (Azure AD) authentication enabled. Password auth is also
  enabled for bootstrap
- **Azure Cache for Redis** (Basic by default; Standard / Premium
  recommended for production)
- **Storage Account** (private, versioned, TLS 1.2 minimum) with a
  `proxy` blob container for cache / proxy_config + a `files` blob
  container for /v1/files passthrough storage
- **Key Vault** holding `LITELLM_MASTER_KEY`, the optional license, the
  optional UI password, and the bootstrap Postgres admin password
- **Container Apps Environment** running three apps (`gateway`, port
  4000; `backend`, port 4001; `ui`, port 3000) plus a one-off
  `migrations` Container Apps Job that runs `prisma migrate deploy`
  from the dedicated `ghcr.io/berriai/litellm-migrations` image
- **User-assigned managed identity** shared across the three apps + the
  migrations job, granted: Storage Blob Data Contributor on the storage
  account, Key Vault Secrets User on the vault
- **Application Gateway v2** with path-based routing:
  - LLM data-plane prefixes (`/v1/chat/*`, `/v1/embeddings`, ...) ->
    gateway
  - UI asset paths (`/_next/*`, `/assets/*`, ...) -> ui
  - Everything else (management API: `/key/*`, `/user/*`, ...) ->
    backend

## Quick start

```bash
cd terraform/litellm/azure/examples/default
cp terraform.tfvars.example terraform.tfvars   # edit location/tenant/env
terraform init && terraform apply
```

After the apply:

1. As the Entra admin (matches the principal running terraform), run
   the `db_bootstrap_sql` output once against the Postgres Flexible
   Server. This creates the `var.db_username` user the proxy will
   authenticate as via Entra-managed-identity tokens at runtime.
2. Run the `migration_run_command` output to trigger the one-off
   prisma migration Container App Job. The gateway / backend revisions
   do not auto-redeploy after this; that's intentional so traffic is
   not cut over mid-migration.
3. Point DNS at `app_gateway_fqdn` output. For TLS, set
   `key_vault_certificate_id` after the initial apply and re-apply; the
   HTTPS listener picks up the cert automatically.

## Components

### `proxy_config` (preferred)

Mirrors the helm chart's `gateway.config.proxy_config`. The map is
YAML-encoded and uploaded to the storage account blob
`config/litellm-config.yaml`; the gateway and backend container
start scripts download it to `/tmp/litellm-config.yaml` and set
`CONFIG_FILE_PATH` to match. Editing the value produces a new blob
content and a rolling redeploy of both services.

```hcl
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
  general_settings = {
    master_key   = "os.environ/LITELLM_MASTER_KEY"
    database_url = "os.environ/DATABASE_URL"
  }
}
```

LiteLLM resolves `os.environ/<NAME>` references in the YAML against
the container's environment. That means provider API keys belong in
`*_extra_secrets` (next section), and your YAML just references them
by name.

### Extra env vars

Non-sensitive plaintext (feature flags, observability hosts, etc.) via
`gateway_extra_env` / `backend_extra_env`:

```hcl
gateway_extra_env = {
  LITELLM_LOG     = "INFO"
  LANGFUSE_PUBLIC_KEY = "pk-lf-..."
}

backend_extra_env = {
  UI_USERNAME = "admin"
}
```

### Extra secrets

Sensitive env vars (provider API keys) via `gateway_extra_secrets` /
`backend_extra_secrets`:

```hcl
gateway_extra_secrets = {
  OPENAI_API_KEY = azurerm_key_vault_secret.openai.id
}
```

The Container Apps managed identity has `Get` / `List` on the Key Vault,
so it can read the referenced secret URIs. Create those `*_secret`
resources separately and grant the same access policy.

## Database authentication

The Container Apps managed identity uses Entra (Azure AD) tokens at
runtime against the Postgres Flexible Server. The proxy assembles
`DATABASE_URL` from `DATABASE_HOST/PORT/USER/NAME` + a short-lived
AAD token minted via `azure-identity`'s `DefaultAzureCredential`. The
gateway / backend / migration Container Apps have
`DATABASE_AZURE_AUTH=true` in their environment.

**Break-glass.** The local `litellm_admin` user (password lives in
Key Vault as `db-admin-password`) is kept for break-glass repairs. To
rotate, use the Azure Portal -> "Reset password" or pass a new
`random_password.db_admin_password` resource via a maintenance PR.

**Prerequisite.** A first-time `terraform apply` requires either
`var.litellm_master_key != ""` (the secret is set by the caller) OR
that the current terraform principal is granted `Set` / `Get` on the
Key Vault (the access policy is created automatically by
`keyvault.tf`).

## Container Apps pull

By default the stack pulls the four component images from
`ghcr.io/berriai/litellm-{gateway,backend,ui,migrations}:<tag>`.

Azure Container Apps can pull from `ghcr.io` directly without mirror
configuration (the AWS stack pulls from ECR post-push, the GCP stack
needs an Artifact Registry remote repo, Azure is the closest to a
direct pull-from-public-registry experience).

If you want to pin to a private Azure Container Registry (ACR) instead,
push the images there first and override the `image_*` variables:

```hcl
image_registry   = "myregistry.azurecr.io"
image_namespace  = "litellm"
image_tag        = "v1.86.0"
```

The same role assignment (`AcrPull` for the Container Apps managed
identity) is granted automatically at apply time when the registry is
an `*.azurecr.io` host.

## Observability

Set `otel_endpoint` to an OTLP collector URL to turn on OpenTelemetry
v2 instrumentation in both gateway and backend. Service names are
stamped per component:

```
${local.name}-gateway
${local.name}-backend
```

The current terraform principal can also grant `Monitoring Metrics
Publisher` on the same resource group if you want Container Apps
metrics to feed to Application Insights.

## Component sizing defaults

Defaults work for a small team / production-trail pattern:

| Component | CPU | Memory | Min replicas | Max replicas |
|-----------|-----|--------|--------------|--------------|
| gateway   | 1.0 | 2Gi    | 2            | 10           |
| backend   | 1.0 | 2Gi    | 1            | 5            |
| ui        | 0.5 | 1Gi    | 1            | 3            |
| migrations| 1.0 | 2Gi    | (job)        | (job)        |

Bump these via `gateway_cpu`, `gateway_memory`, etc. The full input list
is on the (forthcoming) Terraform Registry page; until then, see
`variables.tf` in this directory.

## State and CI

Apply from CI? Then pin the constructor's
`role_definition_name` to `Owner` or `Contributor + User Access
Administrator` on the subscription so the role assignments and
Key Vault access policies land. The current terraform principal's
object ID is read via `data.azurerm_client_config`; CI runners need to
be authenticated as the AAD admin for the bootstrap pattern to work
without manual steps.

## Known gaps vs AWS / GCP

- Azure Container Apps does not natively integrate with managed
  Postgres IAM tokens the way Aurora + RDS-iam-token does on AWS.
  Entra-managed-identity tokens at the application layer are the
  equivalent. The `proxy_config` blob is the only state shared
  between the two write paths; everything else is per-component env.
- Application Gateway WAF v2 is not enabled in this baseline. To turn
  it on, swap the SKU `Standard_v2` to `WAF_v2` and add WAF policy
  resources. The path rule structure does not change.
- Redis SSL on Azure: the `rediss://` URL is built from the SSL port
  (default 6380). The proxy uses `REDIS_SSL=true` to flip the scheme,
  matching AWS module's `transit_encryption_enabled = true`.
- Migration ordering: the AWS stack uses `depends_on` chains via
  ECS task ordering; Azure Container Apps Jobs are not part of ACA
  service ordering. The migration is run as a manual one-shot AFTER
  the gateway / backend services are deployed. The README's
  Quick-start ordering covers this.
