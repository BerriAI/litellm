# ArgoCD Validation Report: Standalone DB Secret Consistency

This document records the validation process and results for the `litellm-helm` chart fix that unifies DB credential secret behavior in standalone DB mode.

It is intended to be reused before opening a PR.

## Scope

- Chart: `deploy/charts/litellm-helm`
- Focus: standalone DB mode secret consistency
- Environment used for this run:
  - Host: `47.118.19.219`
  - Kubernetes: `k3s`
  - GitOps: `ArgoCD`
  - Chart distribution: in-cluster OCI registry

## What Was Changed (Code)

The following templates were updated:

- `deploy/charts/litellm-helm/templates/_helpers.tpl`
  - Added `litellm.dbCredentialsSecretName` helper with precedence:
    1. `db.dbCredentialsSecretName`
    2. `postgresql.auth.existingSecret`
    3. `<fullname>-dbcredentials` fallback
  - Added `litellm.shouldCreateDbCredentialsSecret` helper.

- `deploy/charts/litellm-helm/templates/deployment.yaml`
  - Standalone DB env secret refs now use `litellm.dbCredentialsSecretName`.
  - `DATABASE_USERNAME`/`DATABASE_PASSWORD` keys are sourced from values with defaults.

- `deploy/charts/litellm-helm/templates/secret-dbcredentials.yaml`
  - Fallback secret is created only when no external secret is configured.

- `deploy/charts/litellm-helm/templates/migrations-job.yaml`
  - Standalone mode secret usage aligned with deployment behavior.

## ArgoCD Application YAML Used

Validation was done with this file:

- `/root/code/k8s-at-home/build/litellm.app.yaml`

Key values used in that file:

- `db.deployStandalone: true`
- `db.useExisting: false`
- `db.dbCredentialsSecretName: litellm-credentials`
- `postgresql.auth.existingSecret: litellm-credentials`
- `postgresql.auth.secretKeys.userPasswordKey: postgres-password`
- `postgresql.auth.secretKeys.adminPasswordKey: postgres-password`

## Environment Setup Procedure

1. Install `k3s` on host `47.118.19.219`.
2. Install ArgoCD (`v3.3.9`) in namespace `argocd`.
3. Create in-cluster registry namespace and service:
   - namespace: `registry`
   - service: `registry` NodePort (this run used `31666`)
4. Package chart:
   - `helm package /root/code/litellm/deploy/charts/litellm-helm --destination /root/code/k8s-at-home/build`
5. Push package to registry from remote host:
   - `helm push /root/litellm-helm-1.1.0.tgz oci://127.0.0.1:<NODEPORT>/helm-charts --plain-http`
6. Create runtime namespace/secrets:
   - namespace: `agents`
   - secret: `litellm-credentials`
   - secret: `litellm-env-secret`
7. Apply ArgoCD Application:
   - `kubectl apply -n argocd -f /root/litellm.app.yaml`

## Operational Notes (Important)

The following image/network behaviors were observed in this environment:

- Some pulls from upstream registries are very slow or timeout.
- For stability, pre-pull and retag images using `m.daocloud.io` as needed.
- `local-path` helper pods require `rancher/mirrored-library-busybox:1.37.0`.

Recommended prepull examples:

```bash
k3s ctr -n k8s.io images pull m.daocloud.io/docker.io/rancher/mirrored-library-busybox:1.37.0
k3s ctr -n k8s.io images tag m.daocloud.io/docker.io/rancher/mirrored-library-busybox:1.37.0 docker.io/rancher/mirrored-library-busybox:1.37.0
```

For long pulls, run scripts in background and log to file (`nohup ... > logfile 2>&1 &`).

## Secret Requirements for This Configuration

When `deployment.yaml` reads DB credentials from `litellm-credentials`, ensure these keys exist:

- `username`
- `password`
- `postgres-password`
- `redis-password`

In this run, missing `username` caused:

- Pod status: `CreateContainerConfigError`
- Event: `couldn't find key username in Secret agents/litellm-credentials`

Fix:

```bash
kubectl patch secret -n agents litellm-credentials --type merge -p '{"stringData":{"username":"litellm"}}'
kubectl delete pod -n agents -l app.kubernetes.io/name=litellm
```

## Validation Results

After deployment stabilized:

- `litellm-postgresql-0`: `Running`
- `litellm-redis-master-0`: `Running`
- `litellm`: `Running`

Critical check for this fix:

- `litellm-dbcredentials` was **not created** when external secret configuration was provided.

Observed runtime behavior:

- LiteLLM started successfully.
- Health endpoints returned `200` repeatedly.
- No Prisma `P1000` auth failure observed.

## Clean Revalidation (Delete and Recreate)

To verify the fix is stable and not dependent on old cluster state, we performed a full revalidation:

1. Deleted ArgoCD application `litellm`.
2. Deleted namespace `agents`.
3. Repackaged chart locally.
4. Pushed rebuilt chart package again to in-cluster OCI registry.
5. Recreated `agents` namespace and required secrets.
6. Reapplied ArgoCD Application YAML and forced refresh.

Observed results after redeploy:

- `litellm-postgresql-0`: `Running`
- `litellm-redis-master-0`: `Running`
- `litellm`: `Running`
- `litellm-dbcredentials`: **not found**
- LiteLLM logs show normal startup and readiness checks.
- No `P1000` database authentication error.

Note: ArgoCD status during this run appeared as `OutOfSync Healthy`; runtime workload health and secret behavior matched expected fix behavior.

## Revalidation Checklist (Before PR)

1. Re-run `helm template` checks for:
   - deployment secret name resolution
   - fallback secret creation gating
2. Re-deploy on clean environment with ArgoCD Application YAML.
3. Confirm no `litellm-dbcredentials` secret exists when external secret is configured.
4. Confirm `litellm`, `postgresql`, `redis` all reach `Running`.
5. Confirm LiteLLM logs do not include DB authentication errors.
6. Run functional API checks (`/v1/chat/completions` non-stream and stream) and capture outputs.

## Suggested PR Attachment Contents

- This validation document.
- The ArgoCD Application YAML used for reproduction.
- Command snippets used for:
  - chart package/push
  - app apply/sync
  - key verification checks
- Final pod status and key log evidence.
