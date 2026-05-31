# LiteLLM Componentized Helm Chart

This chart deploys LiteLLM as separate gateway, backend, and UI workloads.

Use this chart when you intentionally want the split-service deployment model:

- `gateway`: LiteLLM data-plane service on port 4000.
- `backend`: UI / management API service on port 4001.
- `ui`: static UI service on port 3000.
- `migrationJob`: pre-install / pre-upgrade Prisma migration job.

If you want the existing single-proxy chart with optional bundled Postgres and
Redis dependency wiring, use `deploy/charts/litellm-helm` instead.

## Required External Resources

Before installing this chart, create or provide:

- a Secret containing the proxy master key, referenced by `masterKey`;
- a writer Postgres database, referenced by `database.writer`;
- a Secret containing the writer database username and password, unless using
  IAM authentication;
- optionally, a Redis endpoint and password Secret, referenced by `redis`.

This chart does not accept inline secret values. Sensitive values are passed by
Kubernetes Secret references.

## Minimal Install Shape

Set the required values in a file such as `values.local.yaml`:

```yaml
masterKey:
  secretName: litellm-master-key-secret
  secretKey: master-key

database:
  writer:
    host: postgres.example.internal
    port: 5432
    dbname: litellm
    passwordSecret:
      name: litellm-writer-secret
      usernameKey: username
      passwordKey: password
```

Then install with:

```bash
helm install litellm ./helm/litellm -f values.local.yaml
```

Enable `ingress.enabled=true` when you want one L7 entrypoint that routes to
the UI, gateway, and backend services.

## Relationship To `deploy/charts/litellm-helm`

`deploy/charts/litellm-helm` is the existing documented chart for the
single-proxy deployment path. This `helm/litellm` chart is for operators who
want to evaluate or adopt separate gateway, backend, and UI workloads. Pick one
chart per release; they are not meant to be installed over the same Helm
release name.
