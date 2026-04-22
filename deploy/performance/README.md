# Performance-related deploy assets

## Full Kustomize stack on EKS (Neon + in-memory cache)

For **LiteLLM + mock LLM + mock files API + Prometheus** with **Neon** (`DATABASE_URL` Secret) and **in-memory cache**, see **`../kubernetes/performance-eks/`**.

## LiteLLM on Kubernetes (EKS-friendly, Helm chart)

`litellm-eks/` mirrors the repo root `docker-compose.yml` layout: single replica, bundled Postgres, `STORE_MODEL_IN_DB`, and Helm migration hooks for plain `helm install`.

### AWS EKS prerequisites

- `kubectl` context points at your cluster (`aws eks update-kubeconfig ...`).
- A **default StorageClass** (EKS often ships with `gp2` / `gp3` via EBS CSI driver) so the Bitnami Postgres chart can bind PVCs.
- For a public proxy URL, either **`kubectl port-forward`**, set **`service.type: LoadBalancer`** in `litellm-eks/values.yaml` (NLB/CLB), or add an **Ingress** + AWS Load Balancer Controller / ALB—see the main Helm chart `ingress` values.

### Deploy with Helm only

From the repository root:

```bash
helm dependency update deploy/charts/litellm-helm
chmod +x deploy/performance/litellm-eks/install.sh
MASTER_KEY='sk-your-key' ./deploy/performance/litellm-eks/install.sh
```

Optional environment variables: `LITELLM_NAMESPACE`, `LITELLM_RELEASE`, `MASTER_KEY`.

### Deploy with Kustomize + Helm (chart inflator)

Kustomize renders the same chart; you then apply YAML (no Helm release stored in the cluster).

```bash
helm dependency update deploy/charts/litellm-helm
kubectl kustomize deploy/performance/litellm-eks --enable-helm | kubectl apply -f -
```

To pin the proxy master key when using this path, add a local values file and reference it from `litellm-eks/kustomization.yaml` under `helmCharts[].additionalValuesFiles` (keep that file out of git).

**Note:** `helmCharts[].version` in `kustomization.yaml` must match `version` in `deploy/charts/litellm-helm/Chart.yaml` after chart bumps.

### Prometheus

The compose file scrapes `litellm:4000`. In-cluster, point Prometheus at `http://<release>-litellm.<namespace>.svc.cluster.local:4000` (default install: `http://litellm-litellm.litellm.svc.cluster.local:4000`).
