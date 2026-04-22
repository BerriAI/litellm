# Performance stack (EKS)

**Primary path: Helm + Argo CD** — `deploy/charts/litellm-helm` with values in `helm/litellm-values.yaml` (bundled **Bitnami Postgres + Redis**, shared **Redis** cache, LoadBalancer, EKS `nodeSelector`). Observability: optional **kube-prometheus-stack** Application (Helm chart in cluster, not Bitnami node pools).

**Legacy path:** in-cluster Postgres + Redis + static Prometheus under [`legacy/kustomize/`](./legacy/kustomize/) (manual `kubectl kustomize` only; legacy Postgres uses **emptyDir**).

---

### What runs where

| Piece | Role |
|--------|------|
| **EKS** | Cluster from `eksctl-cluster.yaml` (or console). |
| **Helm** | Packages LiteLLM, **Postgres + Redis** subcharts, migration Job, mock via `extraResources`. |
| **Argo CD** | Watches Git and syncs `Application` manifests in `argocd/`. |

---

## Runbook: Helm + Argo (recommended)

| Step | Action |
|------|--------|
| **1–3** | AWS CLI, `kubectl`, `eksctl` — see [AWS CLI + kubectl](#aws-cli--kubectl-first-time-setup). |
| **4** | `eksctl create cluster -f deploy/performance/eksctl-cluster.yaml` |
| **5** | `aws eks update-kubeconfig --region us-east-1 --name litellm-performance` (or your `metadata.name` / region from `eksctl-cluster.yaml`) |
| **6** | **EBS CSI** — Bitnami Postgres (and often Redis) use PVCs. Install the [EKS EBS CSI add-on](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html) and ensure a default `StorageClass` exists. |
| **7** | **Install Argo CD** — `kubectl create namespace argocd` then `kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml` |
| **8** | **Register Git repo** if private (Argo **Settings → Repositories**). |
| **9** | **Apply Applications** (pin `targetRevision` in each file to your branch if needed): |

```bash
kubectl apply -n argocd -f deploy/performance/argocd/application-litellm.yaml
kubectl apply -n argocd -f deploy/performance/argocd/application-kube-prometheus-stack.yaml
```

| **10** | Argo UI → sync **litellm-helm**, then **kube-prometheus-stack** (large first sync). |
| **11** | **Postgres / Redis** — override defaults in `helm/litellm-values.yaml` (`postgresql.auth`, `redis.auth`) for non-dev clusters; re-sync after edits. |
| **12** | **Master key** — chart creates `litellm-masterkey` by default; override with `masterkeySecretName` + your Secret if required. |
| **13** | Verify: `kubectl -n performance get pods,svc` — `litellm` Service **LoadBalancer** on port **4000**. |

---

## Layout

```
deploy/performance/
  argocd/
    application-litellm.yaml
    application-kube-prometheus-stack.yaml
  helm/
    litellm-values.yaml
  eksctl-cluster.yaml
  legacy/
    dev-config.yaml                 # Proxy YAML for legacy Kustomize only
    kustomize/                      # Raw Postgres + Redis + Prometheus (optional)
```

---

## Helm values

- **Source of truth:** `helm/litellm-values.yaml` → `proxy_config`. Legacy overlay uses `legacy/dev-config.yaml` only if you still run Kustomize.
- **Argo** must sync a `targetRevision` that contains this file (paths in `application-litellm.yaml` are relative to `deploy/charts/litellm-helm`).

---

## Troubleshooting reference

### Stuck Postgres `Pending` / PVC `FailedBinding`

Install **EBS CSI** and a **StorageClass**; see [EKS EBS CSI](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html).

### Stuck LiteLLM `Init:*` / old `wait-db`

That was an **older** raw Deployment spec. This Helm chart does not use those init containers.

### EBS CSI / IRSA / OIDC

The EBS CSI controller must use **IRSA** (`eks.amazonaws.com/role-arn` on the ServiceAccount). Typical fixes:

- Associate OIDC: `eksctl utils associate-iam-oidc-provider --cluster <name> --region <region> --approve`
- Create role + annotate SA: `eksctl create iamserviceaccount ... --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy`
- Flag: **`--override-existing-serviceaccounts`** (plural)
- CloudFormation stack **`ROLLBACK_COMPLETE`**: disable termination protection, delete stack, retry
- Set add-on role: `aws eks update-addon --cluster-name <name> --addon-name aws-ebs-csi-driver --service-account-role-arn <role-arn> --resolve-conflicts OVERWRITE`

## Legacy: Kustomize (Postgres + Redis + Prometheus)

Create the namespace and Secret **first** (password must match `DATABASE_URL`):

```bash
kubectl apply -f deploy/performance/legacy/kustomize/namespace.yaml
kubectl create secret generic litellm-dotenv \
  --namespace performance \
  --from-literal=DATABASE_PASSWORD='choose-a-password' \
  --from-literal=DATABASE_URL='postgresql://litellm:choose-a-password@postgres:5432/litellm' \
  --from-literal=PROXY_MASTER_KEY='sk-your-admin-key' \
  --dry-run=client -o yaml | kubectl apply -f -
```

Then apply the overlay (from repo root):

```bash
kubectl kustomize deploy/performance/legacy/kustomize \
  --load-restrictor=LoadRestrictionsNone \
  | kubectl apply -f -
```

Postgres in this overlay is **emptyDir** (ephemeral). For PVC-backed DBs, use the **Helm** chart.

---

## TODOs / next steps

- [ ] **Pin Argo `targetRevision`** to a tag or SHA for reproducibility.
- [ ] **kube-prometheus-stack:** Add scrape config for LiteLLM metrics via chart values.
- [ ] **Secrets:** External Secrets / SSM for `postgresql.auth`, `redis.auth`, and `PROXY_MASTER_KEY` in production.

---

## AWS CLI + kubectl (first-time setup)

### 1. Install AWS CLI v2

Follow the [official install guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html). Verify:

```bash
aws --version   # aws-cli/2.x
```

### 2. Credentials

Use `aws configure`, SSO, or roles your org supports. Verify:

```bash
aws sts get-caller-identity
```

### 3. kubectl

Install a [kubectl version compatible with your EKS control plane](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html).

### 4. eksctl

Install [eksctl](https://eksctl.io/installation/). Create/delete cluster:

```bash
eksctl create cluster -f deploy/performance/eksctl-cluster.yaml
# eksctl delete cluster -f deploy/performance/eksctl-cluster.yaml
```

---

## Argo CD UI (quick login)

```bash
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo
kubectl -n argocd port-forward svc/argocd-server 8080:443
```

Open **https://localhost:8080**, user **`admin`**. Docs: [Argo CD getting started](https://argo-cd.readthedocs.io/en/stable/getting_started/).

---

## Teardown

**Argo-managed:** delete the Applications (or let prune), then remove workloads/namespaces as needed.

**Legacy Kustomize:**

```bash
kubectl kustomize deploy/performance/legacy/kustomize \
  --load-restrictor=LoadRestrictionsNone \
  | kubectl delete -f -
```

### After removing `mock-files-api` from Git

```bash
kubectl -n performance delete deploy,svc mock-files-api --ignore-not-found
kubectl -n performance delete configmap mock-files-api-code --ignore-not-found
```
